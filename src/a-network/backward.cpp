#include "backward.h"
#include "convert.h"
#include "field.h"
#include "readout.h"
#include <cmath>

// 梯度缓存
std::vector<float> g_embed_grad;
std::vector<float> g_proj_grad;
std::vector<float> g_bias_grad;
std::vector<float> g_prop_grad;

// 工作区
std::vector<float> g_signal;
std::vector<std::vector<float>> g_prop_act;

// ============ Readout 反向 ============
void backward_readout(const float* d_logits_v, float* d_network)
{
    std::vector<float> d_logits_d(g_hidden_dim, 0.0f);
    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(g_hidden_dim); ++k) {
        float sum = 0.0f;
        for (size_t t = 0; t < g_vocab_size; ++t)
            sum += d_logits_v[t] * g_embed_weight[t * g_hidden_dim + static_cast<size_t>(k)];
        d_logits_d[static_cast<size_t>(k)] = sum;
    }

    #pragma omp parallel for
    for (int t = 0; t < static_cast<int>(g_vocab_size); ++t) {
        float dv = d_logits_v[t];
        if (dv == 0.0f) continue;
        float* w_row = g_embed_grad.data() + static_cast<size_t>(t) * g_hidden_dim;
        for (size_t k = 0; k < g_hidden_dim; ++k)
            w_row[k] += dv * g_logits_d[k];
    }

    std::vector<float> d_diff(N, 0.0f);
    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(g_hidden_dim); ++k) {
        float dk = d_logits_d[static_cast<size_t>(k)];
        if (dk == 0.0f) continue;
        const float* row = g_proj_weight.data() + static_cast<size_t>(k) * N;
        for (size_t j = 0; j < N; ++j)
            d_diff[j] += dk * row[j];
    }

    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(g_hidden_dim); ++k) {
        float dk = d_logits_d[static_cast<size_t>(k)];
        if (dk == 0.0f) continue;
        float* w_row = g_proj_grad.data() + static_cast<size_t>(k) * N;
        for (size_t j = 0; j < N; ++j)
            w_row[j] += dk * g_diff[j];
    }

    #pragma omp parallel for
    for (int j = 0; j < static_cast<int>(N); ++j) {
        d_network[j] = d_diff[j];
        g_bias_grad[j] -= d_diff[j];
    }
}

// ============ 传播反向 ============
void backward_propagate(float* d_network, float* d_incoming,
                         const float* phase2_act)
{
    std::vector<float> d_combined(N);
    #pragma omp parallel for
    for (int j = 0; j < static_cast<int>(N); ++j)
        d_combined[j] = d_network[j];

    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(N); ++idx) {
        size_t x = static_cast<size_t>(idx) / (network_y * network_z);
        size_t yz = static_cast<size_t>(idx) % (network_y * network_z);
        size_t y = yz / network_z;
        size_t z = yz % network_z;

        // ---- 本地 26 邻域梯度汇聚 ----
        float grad_local = 0.0f;
        for (size_t n = 0; n < g_num_neighbors; ++n) {
            int nx = static_cast<int>(x) + g_neighbors_26[n].dx;
            int ny = static_cast<int>(y) + g_neighbors_26[n].dy;
            int nz = static_cast<int>(z) + g_neighbors_26[n].dz;
            if (nx >= 0 && nx < static_cast<int>(network_x) &&
                ny >= 0 && ny < static_cast<int>(network_y) &&
                nz >= 0 && nz < static_cast<int>(network_z)) {
                size_t ni = static_cast<size_t>(nx) * network_y * network_z
                          + static_cast<size_t>(ny) * network_z
                          + static_cast<size_t>(nz);
                grad_local += d_incoming[ni];
            }
        }

        // ---- 长程跳跃连接梯度汇聚 ----
        float grad_skip = 0.0f;
        for (uint32_t e = g_skip_ptr[idx]; e < g_skip_ptr[idx + 1]; ++e) {
            grad_skip += d_incoming[g_skip_dst[e]] * g_skip_weight[e];
        }

        float scaled = phase2_act[idx] * 0.5f;
        float tanh_a = std::tanh(scaled);
        float dtanh = 1.0f - tanh_a * tanh_a;
        float act_a = tanh_a * 2.0f;  // 前向激活值 a = tanh(x*0.5)*2

        d_combined[idx] += (grad_local * g_prop_weight[idx] + grad_skip) * dtanh;
        #pragma omp atomic
        g_prop_grad[idx] += grad_local * act_a;

        // 跳过连接权重梯度：d_incoming[target] * activation
        for (uint32_t e = g_skip_ptr[idx]; e < g_skip_ptr[idx + 1]; ++e) {
            #pragma omp atomic
            g_skip_grad[e] += d_incoming[g_skip_dst[e]] * act_a;
        }
    }

    #pragma omp parallel for
    for (int j = 0; j < static_cast<int>(N); ++j) {
        float d_total = d_combined[j];
        d_incoming[j] = d_total;
        d_network[j] = d_total * g_time_decay;
    }
}

// ============ 注入反向 ============
void backward_inject(size_t token_id, const float* d_network)
{
    #pragma omp parallel for
    for (int j = 0; j < static_cast<int>(N); ++j)
        g_bias_grad[j] += d_network[j];

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(g_hidden_dim); ++i) {
        float sum = 0.0f;
        const float* row = g_proj_weight.data() + static_cast<size_t>(i) * N;
        for (size_t j = 0; j < N; ++j)
            sum += d_network[j] * row[j];
        g_embed_grad[token_id * g_hidden_dim + static_cast<size_t>(i)] += sum;
    }

    const float* embed = g_embed_weight.data() + token_id * g_hidden_dim;
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(g_hidden_dim); ++i) {
        float ei = embed[static_cast<size_t>(i)];
        if (ei == 0.0f) continue;
        float* w_row = g_proj_grad.data() + static_cast<size_t>(i) * N;
        for (size_t j = 0; j < N; ++j)
            w_row[j] += ei * d_network[j];
    }
}
