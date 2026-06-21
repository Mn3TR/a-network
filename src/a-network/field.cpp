#include "field.h"
#include "config.h"   // g_skip_density
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>

const NeighborOffset g_neighbors_26[] = {
    {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
    {-1, -1, 0}, {-1, 1, 0}, {1, -1, 0}, {1, 1, 0},
    {-1, 0, -1}, {-1, 0, 1}, {1, 0, -1}, {1, 0, 1},
    {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1},
    {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
    {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1},
};
const size_t g_num_neighbors = 26;

std::vector<float> g_prop_weight;
std::vector<float> g_incoming;

std::vector<uint32_t> g_cell_x;
std::vector<uint32_t> g_cell_y;
std::vector<uint32_t> g_cell_z;

std::vector<uint32_t> g_skip_ptr;
std::vector<uint32_t> g_skip_dst;
std::vector<float>    g_skip_weight;
std::vector<float>    g_skip_grad;

std::vector<uint32_t> g_rev_skip_ptr;
std::vector<uint32_t> g_rev_skip_src;

// 检查两个细胞是否为 26-邻域邻居（含自身）
static bool is_too_close(uint32_t a, uint32_t b)
{
    if (a == b) return true;
    size_t ax = a / (network_y * network_z);
    size_t ayz = a % (network_y * network_z);
    size_t ay = ayz / network_z;
    size_t az = ayz % network_z;
    size_t bx = b / (network_y * network_z);
    size_t byz = b % (network_y * network_z);
    size_t by = byz / network_z;
    size_t bz = byz % network_z;
    int dx = static_cast<int>(ax) - static_cast<int>(bx);
    int dy = static_cast<int>(ay) - static_cast<int>(by);
    int dz = static_cast<int>(az) - static_cast<int>(bz);
    return std::abs(dx) <= 1 && std::abs(dy) <= 1 && std::abs(dz) <= 1;
}

// 构建坐标查找表（消除内层循环的除法和取模）。幂等，所有模式调用。
static void build_coord_tables()
{
    g_cell_x.resize(N);
    g_cell_y.resize(N);
    g_cell_z.resize(N);
    for (size_t idx = 0; idx < N; ++idx) {
        uint32_t yz = static_cast<uint32_t>(idx % (network_y * network_z));
        g_cell_x[idx] = static_cast<uint32_t>(idx / (network_y * network_z));
        g_cell_y[idx] = yz / static_cast<uint32_t>(network_z);
        g_cell_z[idx] = yz % static_cast<uint32_t>(network_z);
    }
}

// 生成随机长程跳跃连接拓扑 + 随机权重。仅在训练新模型时调用（once 守卫）。
static void build_skip_connections(std::mt19937& gen,
                                   std::uniform_real_distribution<float>& dist)
{
    if (g_skip_density <= 0.0f) return;

    std::vector<std::pair<uint32_t, uint32_t>> edges;
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> coord_dist(0, static_cast<int>(network_x - 1));

    for (uint32_t idx = 0; idx < N; ++idx) {
        if (prob_dist(gen) >= g_skip_density) continue;

        // 随机选择一个「远处」目标（排斥自身及 26 邻域）
        for (int attempt = 0; attempt < 20; ++attempt) {
            uint32_t tx = static_cast<uint32_t>(coord_dist(gen));
            uint32_t ty = static_cast<uint32_t>(coord_dist(gen));
            uint32_t tz = static_cast<uint32_t>(coord_dist(gen));
            uint32_t target = tx * network_y * network_z
                            + ty * network_z + tz;
            if (!is_too_close(idx, target)) {
                edges.emplace_back(idx, target);
                break;
            }
        }
    }

    // 按 src 排序，构建正向 CSR
    std::sort(edges.begin(), edges.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    size_t nc = edges.size();
    g_skip_dst.resize(nc);
    g_skip_weight.resize(nc);
    for (size_t i = 0; i < nc; ++i)
        g_skip_dst[i] = edges[i].second;

    // 权重初始化：与本地传播相同的量级
    for (auto& w : g_skip_weight) w = dist(gen);

    // 构建正向 ptr
    size_t e = 0;
    for (uint32_t idx = 0; idx < N; ++idx) {
        g_skip_ptr[idx] = static_cast<uint32_t>(e);
        while (e < nc && edges[e].first == idx) ++e;
    }
    g_skip_ptr[N] = static_cast<uint32_t>(nc);

    g_skip_grad.assign(nc, 0.0f);

    std::cout << "Skip connections generated: " << nc
              << " (density=" << (g_skip_density * 100.0f) << "%)" << std::endl;

    // ============ 构建反向 CSR（gather 模式用） ============
    // 按 dst 排序，然后构建反向 ptr+src
    std::vector<std::pair<uint32_t, uint32_t>> rev_edges;
    rev_edges.reserve(nc);
    for (size_t i = 0; i < nc; ++i)
        rev_edges.emplace_back(g_skip_dst[i], edges[i].first);  // (dst, src)

    std::sort(rev_edges.begin(), rev_edges.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    g_rev_skip_ptr.assign(N + 1, 0);
    g_rev_skip_src.resize(nc);

    size_t re = 0;
    for (uint32_t idx = 0; idx < N; ++idx) {
        g_rev_skip_ptr[idx] = static_cast<uint32_t>(re);
        while (re < nc && rev_edges[re].first == idx) {
            g_rev_skip_src[re] = rev_edges[re].second;
            ++re;
        }
    }
    g_rev_skip_ptr[N] = static_cast<uint32_t>(nc);
}

void init_propagation()
{
    static bool once = false;

    // 工作区分配（幂等，可重复调用）
    g_prop_weight.resize(N);
    g_incoming.resize(N);
    build_coord_tables();

    if (once) return;
    once = true;

    g_skip_ptr.assign(N + 1, 0);
    g_skip_dst.clear();
    g_skip_weight.clear();
    g_skip_grad.clear();
    g_rev_skip_ptr.clear();
    g_rev_skip_src.clear();

    std::random_device rd;
    std::mt19937 gen(rd());
    // 原版初始化：1/26 ≈ 0.0385，使信号进入 tanh 饱和区产生非线性竞争
    float scale = 1.0f / static_cast<float>(g_num_neighbors);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (auto& w : g_prop_weight) w = dist(gen);

    // 跳跃连接同量级初始化
    build_skip_connections(gen, dist);
}

void propagate_step(float* network, float* incoming, float* act_tanh)
{
    // 确保有激活值缓冲区：如果外部提供了就用外部，否则用内部的
    static std::vector<float> s_act_work;
    float* act = act_tanh;
    if (!act) {
        s_act_work.resize(N);
        act = s_act_work.data();
    }

    // Phase 1: 衰减 + 接收（每细胞独立，安全并行）
    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(N); ++idx) {
        float v = network[idx];
        network[idx] = v * g_time_decay + incoming[idx];
        incoming[idx] = 0.0f;
    }

    // Phase 2: 计算 tanh 激活值（全部读出，无竞争）
    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(N); ++idx) {
        act[idx] = std::tanh(network[idx] * 0.5f);
    }

    // Phase 3: scatter 模式 — 每个细胞广播激活值×权重到 26 邻域 + 跳跃连接
    // 使用原子操作避免写竞争（原版 scatter 语义，产生几何分化）
    #pragma omp parallel for
    for (int idx = 0; idx < static_cast<int>(N); ++idx) {
        int x = static_cast<int>(g_cell_x[idx]);
        int y = static_cast<int>(g_cell_y[idx]);
        int z = static_cast<int>(g_cell_z[idx]);
        float a = act[idx] * 2.0f;
        float w = g_prop_weight[idx];

        // 本地 26 邻域发送
        if (a != 0.0f && w != 0.0f) {
            float sig = a * w;
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
                    #pragma omp atomic
                    incoming[ni] += sig;
                }
            }
        }

        // 长程跳跃连接发送
        if (a != 0.0f) {
            for (uint32_t e = g_skip_ptr[idx]; e < g_skip_ptr[idx + 1]; ++e) {
                #pragma omp atomic
                incoming[g_skip_dst[e]] += a * g_skip_weight[e];
            }
        }
    }
}

void propagate_network(NetworkView network)
{
    propagate_step(network.data_handle(), g_incoming.data());
}
