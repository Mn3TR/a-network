#include "convert.h"
#include <random>
#include <algorithm>
#include <cmath>

std::vector<float> g_embed_weight;
std::vector<float> g_proj_weight;
std::vector<float> g_proj_bias;

void init_convert_layer()
{
    static bool once = false;
    if (once) return;
    once = true;

    g_embed_weight.resize(g_vocab_size * g_hidden_dim);
    g_proj_weight.resize(g_hidden_dim * N);
    g_proj_bias.resize(N);

    std::random_device rd;
    std::mt19937 gen(rd());

    float emb_scale = std::sqrt(6.0f / g_hidden_dim);
    std::uniform_real_distribution<float> emb_dist(-emb_scale, emb_scale);
    for (auto& w : g_embed_weight) w = emb_dist(gen);

    float proj_scale = std::sqrt(6.0f / (g_hidden_dim + N));
    std::uniform_real_distribution<float> proj_dist(-proj_scale, proj_scale);
    for (auto& w : g_proj_weight) w = proj_dist(gen);

    std::fill(g_proj_bias.begin(), g_proj_bias.end(), 0.0f);
}

void token_to_signal(size_t token_id, NetworkView network)
{
    if (token_id >= g_vocab_size) return;

    const float* embed = g_embed_weight.data() + token_id * g_hidden_dim;

    std::vector<float> signal(N);
    for (size_t j = 0; j < N; ++j) signal[j] = g_proj_bias[j];

    for (size_t i = 0; i < g_hidden_dim; ++i) {
        float ei = embed[i];
        const float* row = g_proj_weight.data() + i * N;
        #pragma omp parallel for
        for (int j = 0; j < static_cast<int>(N); ++j)
            signal[j] += ei * row[j];
    }

    auto tensor = std::mdspan<const float, std::extents<size_t, network_x, network_y, network_z>>(signal.data());
    for (size_t x = 0; x < network_x; ++x)
        for (size_t y = 0; y < network_y; ++y)
            for (size_t z = 0; z < network_z; ++z)
                network[x, y, z] += tensor[x, y, z];
}
