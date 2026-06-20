#include "convert.h"
#include "backward.h"
#include <random>
#include <cmath>
#include <algorithm>

std::vector<float> g_embed_weight;  // [V, H]
std::vector<float> g_in_weight;     // [H, N]
std::vector<float> g_in_bias;       // [N]

void init_convert_workspace()
{
    size_t H = g_hidden_dim;
    size_t N = ::N;
    g_embed_weight.resize(g_vocab_size * H);
    g_in_weight.resize(H * N);
    g_in_bias.resize(N);
}

void init_convert_weights()
{
    static bool once = false;
    if (once) return;
    once = true;

    init_convert_workspace();

    size_t H = g_hidden_dim;
    size_t N = ::N;

    std::random_device rd;
    std::mt19937 gen(rd());

    float emb_scale = std::sqrt(6.0f / static_cast<float>(H));
    std::uniform_real_distribution<float> emb_dist(-emb_scale, emb_scale);
    for (auto& w : g_embed_weight) w = emb_dist(gen);

    float in_scale = std::sqrt(2.0f / static_cast<float>(H));
    std::uniform_real_distribution<float> in_dist(-in_scale, in_scale);
    for (auto& w : g_in_weight) w = in_dist(gen);

    std::fill(g_in_bias.begin(), g_in_bias.end(), 0.0f);
}

void tokens_to_field(const float* hidden, size_t S, float* field)
{
    size_t H = g_hidden_dim;
    size_t N = ::N;
    size_t chunk = N / S;
    float inv_sqrt_S = 1.0f / std::sqrt(static_cast<float>(S));

    #pragma omp parallel for
    for (int ii = 0; ii < static_cast<int>(S); ++ii) {
        size_t i = static_cast<size_t>(ii);
        const float* h_i = hidden + i * H;
        size_t off = i * chunk;

        // 每个线程独立的栈缓冲，消除原 g_signal 的数据竞争
        std::vector<float> signal(chunk);
        for (size_t n = 0; n < chunk; ++n)
            signal[n] = g_in_bias[off + n];
        for (size_t k = 0; k < H; ++k) {
            float hk = h_i[k];
            if (hk == 0.0f) continue;
            const float* row = g_in_weight.data() + k * N + off;
            for (size_t n = 0; n < chunk; ++n)
                signal[n] += hk * row[n];
        }
        for (size_t n = 0; n < chunk; ++n)
            field[off + n] += signal[n] * inv_sqrt_S;
    }
}
