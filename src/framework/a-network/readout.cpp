#include "a_network.h"
#include <cmath>
#include <algorithm>

// ==================== field → h ====================

void ANetwork::readout_h(const float* flat_net, float* h_out)
{
    size_t H = m_H, N = m_N;

    // diff = field - bias
    #pragma omp parallel for
    for (int n = 0; n < static_cast<int>(N); ++n)
        m_diff[n] = flat_net[n] - m_out_bias[n];

    // h = W_out * diff
    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(H); ++k) {
        float sum = 0.0f;
        const float* row = m_out_weight.data() + static_cast<size_t>(k) * N;
        for (size_t n = 0; n < N; ++n)
            sum += row[n] * m_diff[n];
        h_out[static_cast<size_t>(k)] = sum;
    }
}
