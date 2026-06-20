#include "model.h"
#include "convert.h"
#include "readout.h"
#include "field.h"
#include "backward.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>

void init_all()
{
    size_t H = g_hidden_dim, N = ::N, P = g_proj_dim;

    // 确保结构/工作区都已分配（不含权重随机化，由 init_*_weights 负责）
    init_convert_workspace();
    init_propagation();
    init_readout_workspace();

    // 梯度缓冲（注意 g_out_weight_grad/bias_grad 现在是 [H×P]/[P]）
    g_embed_grad.resize(g_vocab_size * H, 0.0f);
    g_in_weight_grad.resize(H * N, 0.0f);
    g_in_bias_grad.resize(N, 0.0f);
    g_out_weight_grad.resize(H * P, 0.0f);
    g_out_bias_grad.resize(P, 0.0f);
    g_prop_grad.resize(N, 0.0f);
    g_prop_act.resize(g_prop_steps);
    for (auto& v : g_prop_act)
        v.resize(N);
}

// ============ 读 h：field → 固定随机投影 R → tmp[128] → W_out → h[192] ============
static void readout_h(const float* flat_net, float* h_out)
{
    size_t H = g_hidden_dim, P = g_proj_dim, N = ::N;

    // tmp = R · field（固定随机投影，保距嵌入）
    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(P); ++k) {
        float sum = 0.0f;
        const float* row = g_proj_fixed.data() + static_cast<size_t>(k) * N;
        for (size_t n = 0; n < N; ++n)
            sum += row[n] * flat_net[n];
        g_proj_tmp[static_cast<size_t>(k)] = sum;
    }

    // g_diff = tmp - bias
    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(P); ++k)
        g_diff[k] = g_proj_tmp[k] - g_out_bias[k];

    // h = W_out · (tmp - bias)，W_out 仅 [192×128] = 24K 参数，当不了 FFN
    #pragma omp parallel for
    for (int j = 0; j < static_cast<int>(H); ++j) {
        float sum = 0.0f;
        const float* row = g_out_weight.data() + static_cast<size_t>(j) * P;
        for (size_t k = 0; k < P; ++k)
            sum += row[k] * g_diff[k];
        h_out[static_cast<size_t>(j)] = sum;
    }
}

// ============ h 反向：d_h → W_out^T → d_g_diff → R_fixed^T → d_network ============
static void backward_readout_h(const float* d_h, float* d_network)
{
    size_t H = g_hidden_dim, P = g_proj_dim, N = ::N;

    // d_g_diff = W_out^T · d_h
    static std::vector<float> d_g_diff;
    d_g_diff.resize(P);
    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(P); ++k) {
        float sum = 0.0f;
        size_t uk = static_cast<size_t>(k);
        for (size_t j = 0; j < H; ++j)
            sum += d_h[j] * g_out_weight[j * P + uk];
        d_g_diff[uk] = sum;
    }

    // d_network = R_fixed^T · d_g_diff（梯度回传入场）
    #pragma omp parallel for
    for (int n = 0; n < static_cast<int>(N); ++n) {
        float sum = 0.0f;
        size_t un = static_cast<size_t>(n);
        for (size_t k = 0; k < P; ++k)
            sum += g_proj_fixed[k * N + un] * d_g_diff[k];
        d_network[un] = sum;
    }

    // g_out_weight_grad[j, k] += d_h[j] · g_diff[k]
    #pragma omp parallel for
    for (int j = 0; j < static_cast<int>(H); ++j) {
        float dhj = d_h[static_cast<size_t>(j)];
        if (dhj == 0.0f) continue;
        float* row = g_out_weight_grad.data() + static_cast<size_t>(j) * P;
        for (size_t k = 0; k < P; ++k)
            row[k] += dhj * g_diff[k];
    }

    // g_out_bias_grad = -d_g_diff
    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(P); ++k)
        g_out_bias_grad[k] -= d_g_diff[k];
}

// ============ 采样 Softmax + Loss ============
// 只算 target + g_num_negatives 个随机负样本，避免全词表 128256 的 O(VH) 计算
static float sampled_softmax_loss(const float* h, size_t target_id, float* d_h)
{
    size_t H = g_hidden_dim, V = g_vocab_size;
    size_t K = g_num_negatives;
    size_t C = K + 1;  // target + negatives

    // RNG + 负样本候选集
    static std::mt19937 s_rng(std::random_device{}());
    static std::uniform_int_distribution<size_t> s_sampler(0, V - 1);

    static std::vector<size_t> candidates;
    candidates.resize(C);
    candidates[0] = target_id;
    for (size_t i = 1; i < C; ++i) {
        size_t neg;
        do { neg = s_sampler(s_rng); } while (neg == target_id);
        candidates[i] = neg;
    }

    // 只用候选集算 logits
    static std::vector<float> logits;
    logits.resize(C);
    #pragma omp parallel for
    for (int ci = 0; ci < static_cast<int>(C); ++ci) {
        const float* row = g_embed_weight.data() + candidates[ci] * H;
        float sum = 0.0f;
        for (size_t k = 0; k < H; ++k) sum += row[k] * h[k];
        logits[ci] = sum;
    }

    // Softmax（max-subtraction 防溢出）
    float max_val = logits[0];
    for (size_t i = 1; i < C; ++i)
        if (logits[i] > max_val) max_val = logits[i];

    static std::vector<float> probs;
    probs.resize(C);
    float sum_exp = 0.0f;
    for (size_t i = 0; i < C; ++i) {
        float e = std::exp(logits[i] - max_val);
        probs[i] = e;
        sum_exp += e;
    }
    float inv_sum = 1.0f / sum_exp;
    for (size_t i = 0; i < C; ++i)
        probs[i] *= inv_sum;

    // Loss = -log(prob[target])
    float loss = -std::log(probs[0] + 1e-30f);

    // d_h 反向：d_h[k] = Σ_t d_logits[t] · embed[t][k]
    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(H); ++k) {
        float sum = 0.0f;
        size_t uk = static_cast<size_t>(k);
        for (size_t ci = 0; ci < C; ++ci) {
            const float* row = g_embed_weight.data() + candidates[ci] * H;
            float d = probs[ci] - (ci == 0 ? 1.0f : 0.0f);
            sum += d * row[uk];
        }
        d_h[uk] = sum;
    }

    // Embedding 梯度（只对候选集内的 token 累加）
    for (size_t ci = 0; ci < C; ++ci) {
        float d = probs[ci] - (ci == 0 ? 1.0f : 0.0f);
        if (d == 0.0f) continue;
        float* row = g_embed_grad.data() + candidates[ci] * H;
        for (size_t k = 0; k < H; ++k)
            row[k] += d * h[k];
    }

    return loss;
}

// ============ 训练步骤 ============
// 注入 → 传播 → 读 h → LM Head → loss → 反向
float train_token(size_t input_id, size_t target_id,
                  float* flat_net, float* incoming_buf)
{
    size_t H = g_hidden_dim, N = ::N;

    // ---- 前向 ----

    // 1. 注入
    const float* emb = g_embed_weight.data() + input_id * H;
    tokens_to_field(emb, 1, flat_net);

    // 2. 传播（propagate_step 将 tanh 值写入预分配的 g_prop_act[s]）
    for (int s = 0; s < g_prop_steps; ++s)
        propagate_step(flat_net, incoming_buf, g_prop_act[s].data());

    // 3. 读 h → 采样 Softmax Loss
    std::vector<float> h_cur(H);
    static std::vector<float> d_h_cur;
    d_h_cur.assign(H, 0.0f);
    readout_h(flat_net, h_cur.data());
    float loss = sampled_softmax_loss(h_cur.data(), target_id, d_h_cur.data());

    // ---- 反向 ----

    // 1. backward_readout_h
    static std::vector<float> d_network;
    d_network.resize(N);
    backward_readout_h(d_h_cur.data(), d_network.data());

    // 2. backward_propagate
    std::fill(incoming_buf, incoming_buf + N, 0.0f);
    for (int s = g_prop_steps - 1; s >= 0; --s)
        backward_propagate(d_network.data(), incoming_buf, g_prop_act[s].data());

    // 3. backward_inject
    static std::vector<float> d_emb;
    d_emb.assign(H, 0.0f);
    backward_inject(d_network.data(), 1, emb, d_emb.data());

    // 4. Embedding 梯度
    float* row = g_embed_grad.data() + input_id * H;
    for (size_t k = 0; k < H; ++k) row[k] += d_emb[k];

    return loss;
}

// ============ 生成步（前向-only） ============
size_t generate_token(size_t input_id, float* flat_net, float* incoming_buf)
{
    size_t H = g_hidden_dim;

    // 1. 注入
    const float* emb = g_embed_weight.data() + input_id * H;
    tokens_to_field(emb, 1, flat_net);

    // 2. 传播
    for (int s = 0; s < g_prop_steps; ++s)
        propagate_step(flat_net, incoming_buf);

    // 3. 读 h
    std::vector<float> h_cur(H);
    readout_h(flat_net, h_cur.data());

    // 4. LM Head → logits
    std::vector<float> logits(g_vocab_size);
    #pragma omp parallel for
    for (int t = 0; t < static_cast<int>(g_vocab_size); ++t) {
        float sum = 0.0f;
        const float* row = g_embed_weight.data() + static_cast<size_t>(t) * H;
        for (size_t k = 0; k < H; ++k)
            sum += row[k] * h_cur[k];
        logits[static_cast<size_t>(t)] = sum;
    }

    // 5. Argmax
    auto it = std::max_element(logits.begin(), logits.end());
    return static_cast<size_t>(std::distance(logits.begin(), it));
}
