#include "a_network.h"
#include <cmath>

// ==================== 注入反向 ====================

void ANetwork::backward_inject(const float* d_network, size_t S,
                                const float* hidden, float* d_hidden)
{
    size_t H = m_H, N = m_N;
    size_t chunk = N / S;
    float inv_sqrt_S = 1.0f / std::sqrt(static_cast<float>(S));

    for (size_t i = 0; i < S; ++i) {
        const float* h_i = hidden + i * H;
        float* d_h_i = d_hidden + i * H;
        size_t off = i * chunk;

        #pragma omp parallel for
        for (int n = 0; n < static_cast<int>(chunk); ++n) {
            float dn = d_network[off + n] * inv_sqrt_S;
            m_in_bias_grad[off + n] += dn;
            for (size_t k = 0; k < H; ++k) {
                float hk = h_i[k];
                if (hk != 0.0f)
                    m_in_weight_grad[k * N + off + n] += hk * dn;
            }
        }

        #pragma omp parallel for
        for (int k = 0; k < static_cast<int>(H); ++k) {
            float sum = 0.0f;
            const float* row = m_in_weight.data() + static_cast<size_t>(k) * N + off;
            for (size_t n = 0; n < chunk; ++n)
                sum += row[n] * d_network[off + n];
            d_h_i[k] += sum * inv_sqrt_S;
        }
    }
}

// ==================== 传播反向 ====================

void ANetwork::backward_propagate(float* d_network, float* d_incoming,
                                   const float* phase2_act)
{
    size_t N = m_N;
    float decay = m_cfg.time_decay;
    int net_x = static_cast<int>(m_cfg.network_x);
    int net_y = static_cast<int>(m_cfg.network_y);
    int net_z = static_cast<int>(m_cfg.network_z);

    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(N); ++idx) {
        int x = static_cast<int>(m_cell_x[idx]);
        int y = static_cast<int>(m_cell_y[idx]);
        int z = static_cast<int>(m_cell_z[idx]);

        float grad_local = 0.0f;
        for (size_t n = 0; n < kNumNeighbors; ++n) {
            int nx = x + kNeighbors26[n].dx;
            int ny = y + kNeighbors26[n].dy;
            int nz = z + kNeighbors26[n].dz;
            if (nx >= 0 && nx < net_x &&
                ny >= 0 && ny < net_y &&
                nz >= 0 && nz < net_z) {
                size_t ni = static_cast<size_t>(nx) * m_cfg.network_y * m_cfg.network_z
                          + static_cast<size_t>(ny) * m_cfg.network_z
                          + static_cast<size_t>(nz);
                grad_local += d_incoming[ni];
            }
        }

        float grad_skip = 0.0f;
        for (uint32_t e = m_skip_ptr[idx]; e < m_skip_ptr[idx + 1]; ++e) {
            grad_skip += d_incoming[m_skip_dst[e]] * m_skip_weight[e];
        }

        float tanh_a = phase2_act[idx];
        float dtanh = 1.0f - tanh_a * tanh_a;
        float act_a = tanh_a * 2.0f;

        d_network[idx] += (grad_local * m_prop_weight[idx] + grad_skip) * dtanh;
        m_prop_grad[idx] += grad_local * act_a;

        for (uint32_t e = m_skip_ptr[idx]; e < m_skip_ptr[idx + 1]; ++e) {
            m_skip_grad[e] += d_incoming[m_skip_dst[e]] * act_a;
        }
    }

    #pragma omp parallel for
    for (int j = 0; j < static_cast<int>(N); ++j) {
        float d_total = d_network[j];
        d_incoming[j] = d_total;
        d_network[j] = d_total * decay;
    }
}

// ==================== 读出头反向 ====================

void ANetwork::backward_readout_h(const float* d_h, float* d_network)
{
    size_t H = m_H, N = m_N;

    // d_network = W_out^T * d_h
    #pragma omp parallel for
    for (int n = 0; n < static_cast<int>(N); ++n) {
        float sum = 0.0f;
        for (size_t k = 0; k < H; ++k)
            sum += d_h[k] * m_out_weight[k * N + static_cast<size_t>(n)];
        d_network[n] = sum;
    }

    // W_out grad: d_out_weight += d_h * diff^T
    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(H); ++k) {
        float dhk = d_h[static_cast<size_t>(k)];
        if (dhk == 0.0f) continue;
        float* row = m_out_weight_grad.data() + static_cast<size_t>(k) * N;
        for (size_t n = 0; n < N; ++n)
            row[n] += dhk * m_diff[n];
    }

    // b_out grad
    #pragma omp parallel for
    for (int n = 0; n < static_cast<int>(N); ++n)
        m_out_bias_grad[n] -= d_network[n];
}
