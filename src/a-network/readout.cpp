#include "readout.h"
#include "config.h"     // g_proj_dim
#include "convert.h"    // g_embed_weight
#include <cmath>
#include <algorithm>
#include <random>

// 固定随机投影矩阵（不训练）
std::vector<float> g_proj_fixed;  // [P, N]

// 可学习的小读出矩阵
std::vector<float> g_out_weight;   // [H, P]
std::vector<float> g_out_bias;     // [P]

// 工作区
std::vector<float> g_diff;         // [P] — tmp - bias
std::vector<float> g_proj_tmp;     // [P] — 投影后的 tmp

void init_random_projection()
{
    static bool once = false;
    if (once) return;
    once = true;

    size_t P = g_proj_dim, N = ::N;

    // 固定种子确保可复现
    std::mt19937 gen(42);
    // U(-√(3/P), √(3/P))，使 E[||R·x||²] = ||x||²
    float proj_scale = std::sqrt(3.0f / static_cast<float>(P));
    std::uniform_real_distribution<float> proj_dist(-proj_scale, proj_scale);
    for (auto& r : g_proj_fixed) r = proj_dist(gen);
}

void init_readout_workspace()
{
    size_t H = g_hidden_dim;
    size_t P = g_proj_dim;
    size_t N = ::N;

    g_proj_fixed.resize(P * N);
    g_out_weight.resize(H * P);
    g_out_bias.resize(P);
    g_diff.resize(P);
    g_proj_tmp.resize(P);

    // 首次调用时填充投影矩阵（幂等：后续调用不再覆盖）
    init_random_projection();
}

void init_readout_weights()
{
    size_t H = g_hidden_dim;
    size_t P = g_proj_dim;

    init_readout_workspace();  // 分配 + 填充 R

    // 随机初始化 W_out
    std::random_device rd;
    std::mt19937 gen(rd());

    float out_scale = std::sqrt(6.0f / static_cast<float>(H + P));
    std::uniform_real_distribution<float> out_dist(-out_scale, out_scale);
    for (auto& w : g_out_weight) w = out_dist(gen);
    std::fill(g_out_bias.begin(), g_out_bias.end(), 0.0f);
}

// ============ Cross Entropy Loss（不变） ============
float cross_entropy_loss(const float* logits, size_t target_id, float* d_logits)
{
    float max_val = logits[0];
    for (size_t t = 1; t < g_vocab_size; ++t)
        if (logits[t] > max_val) max_val = logits[t];

    float sum_exp = 0.0f;
    #pragma omp parallel for reduction(+:sum_exp)
    for (int t = 0; t < static_cast<int>(g_vocab_size); ++t) {
        float e = std::exp(logits[t] - max_val);
        d_logits[t] = e;
        sum_exp += e;
    }

    float inv_sum = 1.0f / sum_exp;
    float loss = 0.0f;
    #pragma omp parallel for reduction(+:loss)
    for (int t = 0; t < static_cast<int>(g_vocab_size); ++t) {
        d_logits[t] *= inv_sum;
        if (static_cast<size_t>(t) == target_id)
            loss = -std::log(d_logits[t] + 1e-30f);
        d_logits[t] -= (static_cast<size_t>(t) == target_id ? 1.0f : 0.0f);
    }
    return loss;
}
