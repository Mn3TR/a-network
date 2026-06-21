#include "a_network.h"
#include "framework/tokenizer.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>

// ==================== 构造 ====================

ANetwork::ANetwork(const ANetworkConfig& cfg)
    : m_cfg(cfg)
    , m_N(cfg.N())
    , m_H(cfg.hidden_dim)
    , m_V(g_vocab_size)
{
}

// ==================== 坐标表 ====================

void ANetwork::build_coord_tables()
{
    m_cell_x.resize(m_N);
    m_cell_y.resize(m_N);
    m_cell_z.resize(m_N);
    size_t yz = m_cfg.network_y * m_cfg.network_z;
    for (size_t idx = 0; idx < m_N; ++idx) {
        uint32_t yz_val = static_cast<uint32_t>(idx % yz);
        m_cell_x[idx] = static_cast<uint32_t>(idx / yz);
        m_cell_y[idx] = yz_val / static_cast<uint32_t>(m_cfg.network_z);
        m_cell_z[idx] = yz_val % static_cast<uint32_t>(m_cfg.network_z);
    }
}

// ==================== 跳跃连接构建 ====================

static bool is_too_close(uint32_t a, uint32_t b,
                         size_t nx, size_t ny, size_t nz)
{
    if (a == b) return true;
    size_t yz = ny * nz;
    size_t ax = a / yz;
    size_t ayz = a % yz;
    size_t ay = ayz / nz;
    size_t az = ayz % nz;
    size_t bx = b / yz;
    size_t byz = b % yz;
    size_t by = byz / nz;
    size_t bz = byz % nz;
    int dx = static_cast<int>(ax) - static_cast<int>(bx);
    int dy = static_cast<int>(ay) - static_cast<int>(by);
    int dz = static_cast<int>(az) - static_cast<int>(bz);
    return std::abs(dx) <= 1 && std::abs(dy) <= 1 && std::abs(dz) <= 1;
}

void ANetwork::build_skip_connections()
{
    float density = m_cfg.skip_density;
    if (density <= 0.0f) {
        m_skip_ptr.assign(m_N + 1, 0);
        return;
    }

    std::vector<std::pair<uint32_t, uint32_t>> edges;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> coord_dist(
        0, static_cast<int>(m_cfg.network_x - 1));

    for (uint32_t idx = 0; idx < m_N; ++idx) {
        if (prob_dist(gen) >= density) continue;
        for (int attempt = 0; attempt < 20; ++attempt) {
            uint32_t tx = static_cast<uint32_t>(coord_dist(gen));
            uint32_t ty = static_cast<uint32_t>(coord_dist(gen));
            uint32_t tz = static_cast<uint32_t>(coord_dist(gen));
            uint32_t target = tx * m_cfg.network_y * m_cfg.network_z
                            + ty * m_cfg.network_z + tz;
            if (!is_too_close(idx, target,
                              m_cfg.network_x, m_cfg.network_y, m_cfg.network_z)) {
                edges.emplace_back(idx, target);
                break;
            }
        }
    }

    std::sort(edges.begin(), edges.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    size_t nc = edges.size();
    m_skip_dst.resize(nc);
    m_skip_weight.resize(nc);

    float scale = 1.0f / static_cast<float>(kNumNeighbors);
    std::uniform_real_distribution<float> weight_dist(-scale, scale);
    for (auto& w : m_skip_weight) w = weight_dist(gen);

    for (size_t i = 0; i < nc; ++i)
        m_skip_dst[i] = edges[i].second;

    m_skip_ptr.assign(m_N + 1, 0);
    size_t e = 0;
    for (uint32_t idx = 0; idx < m_N; ++idx) {
        m_skip_ptr[idx] = static_cast<uint32_t>(e);
        while (e < nc && edges[e].first == idx) ++e;
    }
    m_skip_ptr[m_N] = static_cast<uint32_t>(nc);
    m_skip_grad.assign(nc, 0.0f);

    std::cout << "Skip connections generated: " << nc
              << " (density=" << (density * 100.0f) << "%)" << std::endl;

    // 反向 CSR
    std::vector<std::pair<uint32_t, uint32_t>> rev_edges;
    rev_edges.reserve(nc);
    for (size_t i = 0; i < nc; ++i)
        rev_edges.emplace_back(m_skip_dst[i], edges[i].first);

    std::sort(rev_edges.begin(), rev_edges.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    m_rev_skip_ptr.assign(m_N + 1, 0);
    m_rev_skip_src.resize(nc);
    size_t re = 0;
    for (uint32_t idx = 0; idx < m_N; ++idx) {
        m_rev_skip_ptr[idx] = static_cast<uint32_t>(re);
        while (re < nc && rev_edges[re].first == idx) {
            m_rev_skip_src[re] = rev_edges[re].second;
            ++re;
        }
    }
    m_rev_skip_ptr[m_N] = static_cast<uint32_t>(nc);
}

// ==================== 参数组 ====================

std::vector<ParamGroup> ANetwork::param_groups()
{
    return {
        { &m_embed_weight,  &m_embed_grad,     "embed"   },
        { &m_in_weight,     &m_in_weight_grad,  "in_w"    },
        { &m_in_bias,       &m_in_bias_grad,    "in_b"    },
        { &m_out_weight,    &m_out_weight_grad, "out_w"   },
        { &m_out_bias,      &m_out_bias_grad,   "out_b"   },
        { &m_prop_weight,   &m_prop_grad,       "prop"    },
        { &m_skip_weight,   &m_skip_grad,       "skip"    },
    };
}

// ==================== 初始化权重 ====================

void ANetwork::init_weights()
{
    // ---- 分配工作区（幂等） ----
    m_embed_weight.resize(m_V * m_H);
    m_in_weight.resize(m_H * m_N);
    m_in_bias.resize(m_N);
    m_out_weight.resize(m_H * m_N);
    m_out_bias.resize(m_N);
    m_prop_weight.resize(m_N);

    // 梯度
    m_embed_grad.resize(m_V * m_H, 0.0f);
    m_in_weight_grad.resize(m_H * m_N, 0.0f);
    m_in_bias_grad.resize(m_N, 0.0f);
    m_out_weight_grad.resize(m_H * m_N, 0.0f);
    m_out_bias_grad.resize(m_N, 0.0f);
    m_prop_grad.resize(m_N, 0.0f);

    // 工作区
    m_diff.resize(m_N);
    m_incoming.resize(m_N);
    m_flat_net.resize(m_N, 0.0f);

    m_prop_act.resize(m_cfg.prop_steps);
    for (auto& v : m_prop_act)
        v.resize(m_N);

    // 坐标表
    build_coord_tables();

    // ---- 随机化权重 ----
    std::random_device rd;
    std::mt19937 gen(rd());

    // Embedding
    float emb_scale = std::sqrt(6.0f / static_cast<float>(m_H));
    std::uniform_real_distribution<float> emb_dist(-emb_scale, emb_scale);
    for (auto& w : m_embed_weight) w = emb_dist(gen);

    // In weight
    float in_scale = std::sqrt(2.0f / static_cast<float>(m_H));
    std::uniform_real_distribution<float> in_dist(-in_scale, in_scale);
    for (auto& w : m_in_weight) w = in_dist(gen);
    std::fill(m_in_bias.begin(), m_in_bias.end(), 0.0f);

    // Out weight
    float out_scale = std::sqrt(6.0f / static_cast<float>(m_H + m_N));
    std::uniform_real_distribution<float> out_dist(-out_scale, out_scale);
    for (auto& w : m_out_weight) w = out_dist(gen);
    std::fill(m_out_bias.begin(), m_out_bias.end(), 0.0f);

    // 传播权重
    float prop_scale = 1.0f / static_cast<float>(kNumNeighbors);
    std::uniform_real_distribution<float> prop_dist(-prop_scale, prop_scale);
    for (auto& w : m_prop_weight) w = prop_dist(gen);

    // 跳跃连接
    build_skip_connections();
}

// ==================== 重置场状态 ====================

void ANetwork::reset_state()
{
    std::fill(m_flat_net.begin(), m_flat_net.end(), 0.0f);
    std::fill(m_incoming.begin(), m_incoming.end(), 0.0f);
}

// ==================== 清零梯度 ====================

void ANetwork::zero_grad()
{
    auto zero = [](std::vector<float>& v) {
        if (v.empty()) return;
        std::fill(v.begin(), v.end(), 0.0f);
    };
    zero(m_embed_grad);
    zero(m_in_weight_grad);
    zero(m_in_bias_grad);
    zero(m_out_weight_grad);
    zero(m_out_bias_grad);
    zero(m_prop_grad);
    zero(m_skip_grad);
}

// ==================== 场快照 ====================

void ANetwork::dump_field(const std::string& path)
{
    std::ofstream f(path, std::ios::binary);
    if (f)
        f.write(reinterpret_cast<const char*>(m_flat_net.data()),
                m_flat_net.size() * sizeof(float));
}

// ==================== 序列化 ====================

static constexpr uint32_t WEIGHTS_MAGIC = 0x31574E52u;
static constexpr uint32_t WEIGHTS_VERSION = 1u;

void ANetwork::save(const std::string& path)
{
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "ERROR: Cannot write " << path << std::endl;
        return;
    }

    auto write_tensor = [&](const std::vector<float>& t) {
        uint64_t n = t.size();
        file.write(reinterpret_cast<const char*>(&n), sizeof(n));
        file.write(reinterpret_cast<const char*>(t.data()), n * sizeof(float));
    };

    file.write(reinterpret_cast<const char*>(&WEIGHTS_MAGIC), sizeof(WEIGHTS_MAGIC));
    uint32_t ver = WEIGHTS_VERSION;
    file.write(reinterpret_cast<const char*>(&ver), sizeof(ver));

    // 按 param_groups 顺序写张量
    auto groups = param_groups();
    for (auto& g : groups) {
        if (g.weights) write_tensor(*g.weights);
    }

    // 跳跃连接拓扑
    uint64_t nc = (m_skip_ptr.empty() ? 0 : m_skip_ptr[m_N]);
    file.write(reinterpret_cast<const char*>(&nc), sizeof(nc));
    for (uint32_t src = 0; src < m_N; ++src) {
        for (uint32_t e = m_skip_ptr[src]; e < m_skip_ptr[src + 1]; ++e) {
            uint32_t le_src = src;
            uint32_t le_dst = m_skip_dst[e];
            float    le_w   = m_skip_weight[e];
            file.write(reinterpret_cast<const char*>(&le_src), sizeof(le_src));
            file.write(reinterpret_cast<const char*>(&le_dst), sizeof(le_dst));
            file.write(reinterpret_cast<const char*>(&le_w),   sizeof(le_w));
        }
    }

    std::cout << "Weights saved to " << path << std::endl;
}

void ANetwork::load(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "ERROR: Cannot open " << path << std::endl;
        return;
    }

    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (!file || magic != WEIGHTS_MAGIC) {
        std::cerr << "ERROR: Invalid weights file (bad magic: 0x"
                  << std::hex << magic << std::dec << ")." << std::endl;
        return;
    }

    uint32_t version = 0;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!file || version != WEIGHTS_VERSION) {
        std::cerr << "ERROR: Unsupported weights version (" << version
                  << "), expected " << WEIGHTS_VERSION << "." << std::endl;
        return;
    }

    auto read_tensor = [&](std::vector<float>& t) -> bool {
        uint64_t n = 0;
        file.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (!file) return false;
        t.resize(static_cast<size_t>(n));
        file.read(reinterpret_cast<char*>(t.data()), n * sizeof(float));
        return file.good();
    };

    // 按 param_groups 顺序读张量
    auto groups = param_groups();
    for (auto& g : groups) {
        if (g.weights) {
            if (!read_tensor(*g.weights)) {
                std::cerr << "ERROR: Failed to read tensor '" << g.name
                          << "' from " << path << std::endl;
                return;
            }
        }
    }

    // 场状态和工作区
    m_flat_net.assign(m_N, 0.0f);
    m_incoming.assign(m_N, 0.0f);
    m_diff.resize(m_N);

    m_prop_act.resize(m_cfg.prop_steps);
    for (auto& v : m_prop_act)
        v.resize(m_N);

    // 梯度缓存
    auto init_grad = [](std::vector<float>& g, size_t sz) {
        g.assign(sz, 0.0f);
    };
    init_grad(m_embed_grad, m_V * m_H);
    init_grad(m_in_weight_grad, m_H * m_N);
    init_grad(m_in_bias_grad, m_N);
    init_grad(m_out_weight_grad, m_H * m_N);
    init_grad(m_out_bias_grad, m_N);
    init_grad(m_prop_grad, m_N);

    // 跳跃连接拓扑
    m_skip_ptr.assign(m_N + 1, 0);
    m_skip_dst.clear();
    m_skip_weight.clear();

    uint64_t nc = 0;
    file.read(reinterpret_cast<char*>(&nc), sizeof(nc));
    if (!file) {
        std::cerr << "ERROR: Failed to read skip connection count." << std::endl;
        return;
    }
    m_skip_dst.resize(static_cast<size_t>(nc));
    m_skip_weight.resize(static_cast<size_t>(nc));
    m_skip_grad.assign(static_cast<size_t>(nc), 0.0f);

    for (uint64_t i = 0; i < nc; ++i) {
        uint32_t src = 0, dst = 0;
        float w = 0.0f;
        file.read(reinterpret_cast<char*>(&src), sizeof(src));
        file.read(reinterpret_cast<char*>(&dst), sizeof(dst));
        file.read(reinterpret_cast<char*>(&w), sizeof(w));
        if (!file || src >= m_N) {
            std::cerr << "ERROR: Failed to read skip connection " << i
                      << "/" << nc << std::endl;
            return;
        }
        m_skip_dst[static_cast<size_t>(i)] = dst;
        m_skip_weight[static_cast<size_t>(i)] = w;
        m_skip_ptr[src + 1] += 1;
    }

    // 前缀和
    for (uint32_t i = 0; i < m_N; ++i)
        m_skip_ptr[i + 1] += m_skip_ptr[i];

    // 反向 CSR
    m_rev_skip_ptr.assign(m_N + 1, 0);
    m_rev_skip_src.resize(static_cast<size_t>(nc));
    std::vector<std::pair<uint32_t, uint32_t>> rev_pairs;
    rev_pairs.reserve(static_cast<size_t>(nc));
    for (uint32_t src = 0; src < m_N; ++src) {
        for (uint32_t e = m_skip_ptr[src]; e < m_skip_ptr[src + 1]; ++e)
            rev_pairs.emplace_back(m_skip_dst[static_cast<size_t>(e)], src);
    }
    std::sort(rev_pairs.begin(), rev_pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    size_t re = 0;
    for (uint32_t idx = 0; idx < m_N; ++idx) {
        m_rev_skip_ptr[idx] = static_cast<uint32_t>(re);
        while (re < nc && rev_pairs[static_cast<size_t>(re)].first == idx) {
            m_rev_skip_src[static_cast<size_t>(re)] = rev_pairs[static_cast<size_t>(re)].second;
            ++re;
        }
    }
    m_rev_skip_ptr[m_N] = static_cast<uint32_t>(nc);

    // 坐标表
    build_coord_tables();

    std::cout << "Weights loaded from " << path << std::endl;
}

// ==================== Cross-Entropy Loss ====================

float ANetwork::cross_entropy_loss(const float* logits, size_t target_id,
                                    float* d_logits)
{
    size_t V = m_V;
    float max_val = logits[0];
    for (int t = 0; t < static_cast<int>(V); ++t)
        if (logits[t] > max_val) max_val = logits[t];

    float sum_exp = 0.0f;
    for (int t = 0; t < static_cast<int>(V); ++t) {
        float e = std::exp(logits[t] - max_val);
        d_logits[t] = e;
        sum_exp += e;
    }

    float inv_sum = 1.0f / sum_exp;
    for (int t = 0; t < static_cast<int>(V); ++t)
        d_logits[t] *= inv_sum;

    float loss = -std::log(d_logits[target_id] + 1e-30f);
    d_logits[target_id] -= 1.0f;
    return loss;
}
