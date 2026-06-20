#include "checkpoint.h"
#include "core/config.h"
#include "core/types.h"
#include "a-network/config.h"  // g_proj_dim
#include "a-network/convert.h"
#include "a-network/field.h"
#include "a-network/readout.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <utility>
#include <algorithm>

void save_weights()
{
    std::string path = g_weights_path;
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

    write_tensor(g_embed_weight);
    write_tensor(g_in_weight);
    write_tensor(g_in_bias);
    write_tensor(g_out_weight);
    write_tensor(g_out_bias);
    write_tensor(g_prop_weight);

    uint64_t nc = (g_skip_ptr.empty() ? 0 : g_skip_ptr[N]);
    file.write(reinterpret_cast<const char*>(&nc), sizeof(nc));
    // 保存按 src 升序遍历（与 init_propagation 中 std::sort 后的顺序一致）
    for (uint32_t src = 0; src < N; ++src) {
        for (uint32_t e = g_skip_ptr[src]; e < g_skip_ptr[src + 1]; ++e) {
            uint32_t le_src = src;
            uint32_t le_dst = g_skip_dst[e];
            float    le_w   = g_skip_weight[e];
            file.write(reinterpret_cast<const char*>(&le_src), sizeof(le_src));
            file.write(reinterpret_cast<const char*>(&le_dst), sizeof(le_dst));
            file.write(reinterpret_cast<const char*>(&le_w),   sizeof(le_w));
        }
    }

    std::cout << "Weights saved to " << path << std::endl;
}

void load_weights()
{
    std::string path = g_weights_path;
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

    if (!read_tensor(g_embed_weight) ||
        !read_tensor(g_in_weight) ||
        !read_tensor(g_in_bias) ||
        !read_tensor(g_out_weight) ||
        !read_tensor(g_out_bias) ||
        !read_tensor(g_prop_weight)) {
        std::cerr << "ERROR: Failed to read weight tensors from "
                  << path << " (file truncated or corrupted)." << std::endl;
        return;
    }

    g_incoming.resize(N, 0.0f);

    // 初始化投影工作区（固定种子重建，不覆盖已加载的 W_out/bias）
    g_proj_fixed.resize(g_proj_dim * N);
    g_diff.resize(g_proj_dim);
    g_proj_tmp.resize(g_proj_dim);
    init_random_projection();

    g_skip_ptr.assign(N + 1, 0);
    g_skip_dst.clear();
    g_skip_weight.clear();

    uint64_t nc = 0;
    file.read(reinterpret_cast<char*>(&nc), sizeof(nc));
    if (!file) {
        std::cerr << "ERROR: Failed to read skip connection count." << std::endl;
        return;
    }
    g_skip_dst.resize(nc);
    g_skip_weight.resize(nc);
    g_skip_grad.assign(nc, 0.0f);

    // g_skip_ptr[src+1] 先用作「该 src 的出边计数」，最后前缀和为偏移量
    for (uint64_t i = 0; i < nc; ++i) {
        uint32_t src = 0, dst = 0;
        float w = 0.0f;
        file.read(reinterpret_cast<char*>(&src), sizeof(src));
        file.read(reinterpret_cast<char*>(&dst), sizeof(dst));
        file.read(reinterpret_cast<char*>(&w), sizeof(w));
        if (!file) {
            std::cerr << "ERROR: Failed to read skip connection " << i
                      << "/" << nc << " from " << path << std::endl;
            return;
        }
        if (src >= N) {
            std::cerr << "ERROR: skip connection " << i
                      << " has out-of-range src=" << src << std::endl;
            return;
        }
        g_skip_dst[i] = dst;
        g_skip_weight[i] = w;
        g_skip_ptr[src + 1] += 1;  // 累加出边数（修复：原代码为赋值，丢失多出边）
    }

    // 前缀和：g_skip_ptr[i+1] 从「出边数」变为「累计偏移量」
    // 要求保存时边已按 src 升序排列（save_weights 中按 src 自然升序遍历保证）
    for (uint32_t i = 0; i < N; ++i)
        g_skip_ptr[i + 1] += g_skip_ptr[i];

    // 重建反向 CSR（gather 模式）与坐标查找表
    // 这些结构与场拓扑耦合，需与正向 ptr 一并恢复
    g_rev_skip_ptr.assign(N + 1, 0);
    g_rev_skip_src.resize(nc);

    // 反向 CSR 要求按 dst 升序，遍历正向边构造 (dst, src) 后排序
    std::vector<std::pair<uint32_t, uint32_t>> rev_pairs;
    rev_pairs.reserve(nc);
    for (uint32_t src = 0; src < N; ++src) {
        for (uint32_t e = g_skip_ptr[src]; e < g_skip_ptr[src + 1]; ++e)
            rev_pairs.emplace_back(g_skip_dst[e], src);
    }
    std::sort(rev_pairs.begin(), rev_pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    size_t re = 0;
    for (uint32_t idx = 0; idx < N; ++idx) {
        g_rev_skip_ptr[idx] = static_cast<uint32_t>(re);
        while (re < nc && rev_pairs[re].first == idx) {
            g_rev_skip_src[re] = rev_pairs[re].second;
            ++re;
        }
    }
    g_rev_skip_ptr[N] = static_cast<uint32_t>(nc);

    // 重建坐标查找表（消除内层循环的除法和取模，前向/反向均依赖）
    g_cell_x.resize(N);
    g_cell_y.resize(N);
    g_cell_z.resize(N);
    for (size_t idx = 0; idx < N; ++idx) {
        uint32_t yz = static_cast<uint32_t>(idx % (network_y * network_z));
        g_cell_x[idx] = static_cast<uint32_t>(idx / (network_y * network_z));
        g_cell_y[idx] = yz / static_cast<uint32_t>(network_z);
        g_cell_z[idx] = yz % static_cast<uint32_t>(network_z);
    }

    std::cout << "Weights loaded from " << path << std::endl;
}
