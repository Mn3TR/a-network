#include "model.h"
#include "convert.h"
#include "readout.h"
#include "field.h"
#include "backward.h"
#include <cmath>
#include <cstring>
#include <algorithm>

void init_all()
{
    size_t H = g_hidden_dim, N = ::N;

    // 确保结构/工作区都已分配（不含权重随机化，由 init_*_weights 负责）
    init_convert_workspace();
    init_propagation();
    init_readout_workspace();

    // 梯度缓冲 + 前向传播激活值保存
    g_embed_grad.resize(g_vocab_size * H, 0.0f);
    g_in_weight_grad.resize(H * N, 0.0f);
    g_in_bias_grad.resize(N, 0.0f);
    g_out_weight_grad.resize(H * N, 0.0f);
    g_out_bias_grad.resize(N, 0.0f);
    g_prop_grad.resize(N, 0.0f);
    g_prop_act.resize(g_prop_steps);
    for (auto& v : g_prop_act)
        v.resize(N);
}

// ============ 读 h：field → W_out → h ============
static void readout_h(const float* flat_net, float* h_out)
{
    size_t H = g_hidden_dim, N = ::N;
    #pragma omp parallel for
    for (int n = 0; n < static_cast<int>(N); ++n)
        g_diff[n] = flat_net[n] - g_out_bias[n];
    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(H); ++k) {
        float sum = 0.0f;
        const float* row = g_out_weight.data() + static_cast<size_t>(k) * N;
        for (size_t n = 0; n < N; ++n) sum += row[n] * g_diff[n];
        h_out[static_cast<size_t>(k)] = sum;
    }
}

// ============ h 反向：d_h → W_out^T → d_network ============
static void backward_readout_h(const float* d_h, float* d_network)
{
    size_t H = g_hidden_dim, N = ::N;
    #pragma omp parallel for
    for (int n = 0; n < static_cast<int>(N); ++n) {
        float sum = 0.0f;
        for (size_t k = 0; k < H; ++k)
            sum += d_h[k] * g_out_weight[k * N + static_cast<size_t>(n)];
        d_network[n] = sum;
    }
    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(H); ++k) {
        float dhk = d_h[static_cast<size_t>(k)];
        if (dhk == 0.0f) continue;
        float* row = g_out_weight_grad.data() + static_cast<size_t>(k) * N;
        for (size_t n = 0; n < N; ++n) row[n] += dhk * g_diff[n];
    }
    #pragma omp parallel for
    for (int n = 0; n < static_cast<int>(N); ++n)
        g_out_bias_grad[n] -= d_network[n];
}

// ============ 全词表 Softmax + Loss ============
static float lm_head_loss(const float* h, size_t target_id, float* d_h)
{
    size_t H = g_hidden_dim, V = g_vocab_size;
    static std::vector<float> logits, d_logits;
    logits.resize(V);
    d_logits.resize(V);

    #pragma omp parallel for
    for (int t = 0; t < static_cast<int>(V); ++t) {
        float sum = 0.0f;
        const float* row = g_embed_weight.data() + static_cast<size_t>(t) * H;
        for (size_t k = 0; k < H; ++k) sum += row[k] * h[k];
        logits[static_cast<size_t>(t)] = sum;
    }

    float loss = cross_entropy_loss(logits.data(), target_id, d_logits.data());

    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(H); ++k) {
        float sum = 0.0f;
        for (size_t t = 0; t < V; ++t)
            sum += d_logits[t] * g_embed_weight[t * H + static_cast<size_t>(k)];
        d_h[static_cast<size_t>(k)] = sum;
    }

    #pragma omp parallel for
    for (int t = 0; t < static_cast<int>(V); ++t) {
        float dv = d_logits[t];
        if (dv == 0.0f) continue;
        float* row = g_embed_grad.data() + static_cast<size_t>(t) * H;
        for (size_t k = 0; k < H; ++k) row[k] += dv * h[k];
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
    static thread_local std::vector<float> h_cur;
    h_cur.assign(H, 0.0f);
    static thread_local std::vector<float> d_h_cur;
    d_h_cur.assign(H, 0.0f);
    readout_h(flat_net, h_cur.data());
    float loss = lm_head_loss(h_cur.data(), target_id, d_h_cur.data());

    // ---- 反向 ----

    // 1. backward_readout_h
    static thread_local std::vector<float> d_network;
    d_network.assign(N, 0.0f);
    backward_readout_h(d_h_cur.data(), d_network.data());

    // 2. backward_propagate
    std::fill(incoming_buf, incoming_buf + N, 0.0f);
    for (int s = g_prop_steps - 1; s >= 0; --s)
        backward_propagate(d_network.data(), incoming_buf, g_prop_act[s].data());

    // 3. backward_inject
    static thread_local std::vector<float> d_emb;
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
    static thread_local std::vector<float> h_cur;
    h_cur.assign(H, 0.0f);
    readout_h(flat_net, h_cur.data());

    // 4. LM Head → logits
    static thread_local std::vector<float> logits;
    logits.assign(g_vocab_size, 0.0f);  // 复用,避免每步 512KB 堆分配
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
