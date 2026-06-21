#include "readout.h"
#include "convert.h"    // g_embed_weight
#include <cmath>
#include <algorithm>
#include <random>

std::vector<float> g_out_weight;   // [H, N]
std::vector<float> g_out_bias;     // [N]

std::vector<float> g_diff;         // [N]

void init_readout_workspace()
{
    size_t H = g_hidden_dim;
    size_t N = ::N;

    g_out_weight.resize(H * N);
    g_out_bias.resize(N);
    g_diff.resize(N);
}

void init_readout_weights()
{
    size_t H = g_hidden_dim;
    size_t N = ::N;

    init_readout_workspace();

    std::random_device rd;
    std::mt19937 gen(rd());

    float out_scale = std::sqrt(6.0f / static_cast<float>(H + N));
    std::uniform_real_distribution<float> out_dist(-out_scale, out_scale);
    for (auto& w : g_out_weight) w = out_dist(gen);
    std::fill(g_out_bias.begin(), g_out_bias.end(), 0.0f);
}

// ============ Cross Entropy Loss ============
float cross_entropy_loss(const float* logits, size_t target_id, float* d_logits)
{
    float max_val = logits[0];
    #pragma omp parallel for reduction(max:max_val)
    for (int t = 0; t < static_cast<int>(g_vocab_size); ++t)
        if (logits[t] > max_val) max_val = logits[t];

    float sum_exp = 0.0f;
    #pragma omp parallel for reduction(+:sum_exp)
    for (int t = 0; t < static_cast<int>(g_vocab_size); ++t) {
        float e = std::exp(logits[t] - max_val);
        d_logits[t] = e;
        sum_exp += e;
    }

    float inv_sum = 1.0f / sum_exp;
    // 纯并行归一化(reduction(+:loss) 无必要:仅 target 项贡献 loss)
    #pragma omp parallel for
    for (int t = 0; t < static_cast<int>(g_vocab_size); ++t)
        d_logits[t] *= inv_sum;

    // 串行处理 target:归一化后值已定
    float loss = -std::log(d_logits[target_id] + 1e-30f);
    d_logits[target_id] -= 1.0f;
    return loss;
}
