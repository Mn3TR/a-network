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

    // 外层按 i 串行(为将来 S>1 留接口),内层按 n 并行直写 field
    // S=1 时外层只一轮,内层把 N 维度并行化(原版按 S 并行在 S=1 时退化成单线程)
    for (size_t i = 0; i < S; ++i) {
        const float* h_i = hidden + i * H;
        size_t off = i * chunk;

        #pragma omp parallel for
        for (int n = 0; n < static_cast<int>(chunk); ++n) {
            float s = g_in_bias[off + n];
            for (size_t k = 0; k < H; ++k) {
                float hk = h_i[k];
                if (hk != 0.0f)
                    s += hk * g_in_weight[k * N + off + n];
            }
            field[off + n] += s * inv_sqrt_S;  // 保持 += 累加语义(caller 依赖)
        }
    }
}
