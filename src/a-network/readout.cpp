#include "readout.h"
#include "convert.h"  // g_embed_weight, g_proj_weight, g_proj_bias
#include <cmath>
#include <algorithm>

// Readout 工作区
std::vector<float> g_diff;
std::vector<float> g_logits_d;

void init_readout_workspace()
{
    g_diff.resize(N);
    g_logits_d.resize(g_hidden_dim);
}

// ============ Readout 前向 ============
void forward_readout(const float* flat_net, float* logits_v_out)
{
    #pragma omp parallel for
    for (int j = 0; j < static_cast<int>(N); ++j)
        g_diff[j] = flat_net[j] - g_proj_bias[j];

    std::fill(g_logits_d.begin(), g_logits_d.end(), 0.0f);
    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(g_hidden_dim); ++k) {
        float sum = 0.0f;
        const float* row = g_proj_weight.data() + static_cast<size_t>(k) * N;
        for (size_t j = 0; j < N; ++j)
            sum += g_diff[j] * row[j];
        g_logits_d[static_cast<size_t>(k)] = sum;
    }

    #pragma omp parallel for
    for (int t = 0; t < static_cast<int>(g_vocab_size); ++t) {
        float sum = 0.0f;
        const float* row = g_embed_weight.data() + static_cast<size_t>(t) * g_hidden_dim;
        for (size_t k = 0; k < g_hidden_dim; ++k)
            sum += g_logits_d[k] * row[k];
        logits_v_out[static_cast<size_t>(t)] = sum;
    }
}

// ============ Cross Entropy Loss ============
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
