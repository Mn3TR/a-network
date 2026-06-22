#include "a_network.h"
#include <cmath>
#include <algorithm>

// ==================== 一步传播 ====================

void ANetwork::propagate_step(float* network, float* incoming, float* act_tanh)
{
    size_t N = m_N;
    float decay = m_cfg.time_decay;

    // 确保有激活值缓冲区（未传入时用成员缓冲区）
    float* act = act_tanh;
    if (!act) {
        if (m_act_work.size() != N) m_act_work.resize(N);
        act = m_act_work.data();
    }

    // Phase 1+2 合并: 衰减、接收 incoming、清零 incoming、计算激活
    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(N); ++idx) {
        float v = network[idx];
        network[idx] = v * decay + incoming[idx];
        incoming[idx] = 0.0f;
        act[idx] = std::tanh(network[idx] * 0.5f);
    }

    // Phase 3: scatter 模式 — 每个细胞广播激活值到邻域 + 跳跃连接
    int net_x = static_cast<int>(m_cfg.network_x);
    int net_y = static_cast<int>(m_cfg.network_y);
    int net_z = static_cast<int>(m_cfg.network_z);

    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(N); ++idx) {
        int x = static_cast<int>(m_cell_x[idx]);
        int y = static_cast<int>(m_cell_y[idx]);
        int z = static_cast<int>(m_cell_z[idx]);
        float a = act[idx] * 2.0f;
        float w = m_prop_weight[idx];

        // 本地 26 邻域发送
        if (a != 0.0f && w != 0.0f) {
            float sig = a * w;
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
                    #pragma omp atomic
                    incoming[ni] += sig;
                }
            }
        }

        // 长程跳跃连接发送
        if (a != 0.0f) {
            for (uint32_t e = m_skip_ptr[idx]; e < m_skip_ptr[idx + 1]; ++e) {
                #pragma omp atomic
                incoming[m_skip_dst[e]] += a * m_skip_weight[e];
            }
        }
    }
}
