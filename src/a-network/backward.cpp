#include "backward.h"
#include "convert.h"    // g_embed_weight, g_in_weight, g_in_bias
#include "readout.h"    // g_out_weight, g_out_bias, g_diff
#include "field.h"      // g_prop_weight, g_skip_*
#include <cmath>

// ============ 梯度缓存 ============
std::vector<float> g_embed_grad;
std::vector<float> g_in_weight_grad;
std::vector<float> g_in_bias_grad;
std::vector<float> g_out_weight_grad;
std::vector<float> g_out_bias_grad;
std::vector<float> g_prop_grad;

// ============ 工作区 ============
std::vector<std::vector<float>> g_prop_act;

// ============ 序列注入反向 ============
void backward_inject(const float* d_network, size_t S,
                     const float* hidden, float* d_hidden)
{
    size_t H = g_hidden_dim;
    size_t N = ::N;
    size_t chunk = N / S;
    float inv_sqrt_S = 1.0f / std::sqrt(static_cast<float>(S));

    #pragma omp parallel for
    for (int ii = 0; ii < static_cast<int>(S); ++ii) {
        size_t i = static_cast<size_t>(ii);
        const float* h_i = hidden + i * H;
        float* d_h_i = d_hidden + i * H;
        size_t off = i * chunk;

        for (size_t n = 0; n < chunk; ++n)
            g_in_bias_grad[off + n] += d_network[off + n] * inv_sqrt_S;

        for (size_t k = 0; k < H; ++k) {
            float hk = h_i[k];
            if (hk == 0.0f) continue;
            float* row = g_in_weight_grad.data() + k * N + off;
            for (size_t n = 0; n < chunk; ++n)
                row[n] += hk * d_network[off + n] * inv_sqrt_S;
        }

        for (size_t k = 0; k < H; ++k) {
            float sum = 0.0f;
            const float* row = g_in_weight.data() + k * N + off;
            for (size_t n = 0; n < chunk; ++n)
                sum += row[n] * d_network[off + n];
            d_h_i[k] += sum * inv_sqrt_S;
        }
    }
}

// ============ 传播反向 ============
void backward_propagate(float* d_network, float* d_incoming,
                        const float* phase2_act)
{
    static std::vector<float> d_combined;
    d_combined.resize(N);
    #pragma omp parallel for
    for (int j = 0; j < static_cast<int>(N); ++j)
        d_combined[j] = d_network[j];

    // 使用预计算坐标查找表和预存 tanh 值，无需原子操作
    // 每个细胞只写自己的 g_prop_grad[idx] 和自己的 edges g_skip_grad[e]，
    // 不同细胞对应不同的 idx/e，没有竞争
    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(N); ++idx) {
        int x = static_cast<int>(g_cell_x[idx]);
        int y = static_cast<int>(g_cell_y[idx]);
        int z = static_cast<int>(g_cell_z[idx]);

        float grad_local = 0.0f;
        for (size_t n = 0; n < g_num_neighbors; ++n) {
            int nx = x + g_neighbors_26[n].dx;
            int ny = y + g_neighbors_26[n].dy;
            int nz = z + g_neighbors_26[n].dz;
            if (nx >= 0 && nx < static_cast<int>(network_x) &&
                ny >= 0 && ny < static_cast<int>(network_y) &&
                nz >= 0 && nz < static_cast<int>(network_z)) {
                size_t ni = static_cast<size_t>(nx) * network_y * network_z
                          + static_cast<size_t>(ny) * network_z
                          + static_cast<size_t>(nz);
                grad_local += d_incoming[ni];
            }
        }

        float grad_skip = 0.0f;
        for (uint32_t e = g_skip_ptr[idx]; e < g_skip_ptr[idx + 1]; ++e) {
            grad_skip += d_incoming[g_skip_dst[e]] * g_skip_weight[e];
        }

        // phase2_act[idx] 是前向预存的 tanh(network[idx] * 0.5)，避免重算
        float tanh_a = phase2_act[idx];
        float dtanh = 1.0f - tanh_a * tanh_a;
        float act_a = tanh_a * 2.0f;

        d_combined[idx] += (grad_local * g_prop_weight[idx] + grad_skip) * dtanh;
        // 无竞争：每个 idx 对应唯一的 g_prop_grad[idx]
        g_prop_grad[idx] += grad_local * act_a;

        for (uint32_t e = g_skip_ptr[idx]; e < g_skip_ptr[idx + 1]; ++e) {
            // 无竞争：每个 edge e 属于唯一的源细胞 idx
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
