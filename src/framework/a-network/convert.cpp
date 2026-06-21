#include "a_network.h"
#include <cmath>
#include <cstring>
#include <algorithm>

// ==================== 注入：hidden → field ====================

void ANetwork::tokens_to_field(const float* hidden, size_t S, float* field)
{
    size_t H = m_H;
    size_t N = m_N;
    size_t chunk = N / S;
    float inv_sqrt_S = 1.0f / std::sqrt(static_cast<float>(S));

    for (size_t i = 0; i < S; ++i) {
        const float* h_i = hidden + i * H;
        size_t off = i * chunk;

        #pragma omp parallel for
        for (int n = 0; n < static_cast<int>(chunk); ++n) {
            float s = m_in_bias[off + n];
            for (size_t k = 0; k < H; ++k) {
                float hk = h_i[k];
                if (hk != 0.0f)
                    s += hk * m_in_weight[k * N + off + n];
            }
            field[off + n] += s * inv_sqrt_S;
        }
    }
}
