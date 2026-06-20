#include "checkpoint.h"
#include "core/config.h"
#include "a-network/convert.h"
#include "a-network/field.h"
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

    // 魔数 + 版本
    file.write(reinterpret_cast<const char*>(&WEIGHTS_MAGIC), sizeof(WEIGHTS_MAGIC));

    // 写入一个张量：长度(uint64) + 数据(float[])
    auto write_tensor = [&](const std::vector<float>& t) {
        uint64_t n = t.size();
        file.write(reinterpret_cast<const char*>(&n), sizeof(n));
        file.write(reinterpret_cast<const char*>(t.data()), n * sizeof(float));
    };

    write_tensor(g_embed_weight);
    write_tensor(g_proj_weight);
    write_tensor(g_proj_bias);
    write_tensor(g_prop_weight);

    // ============ 保存跳过连接拓扑 + 权重 ============
    uint64_t nc = (g_skip_ptr.empty() ? 0 : g_skip_ptr[N]);
    file.write(reinterpret_cast<const char*>(&nc), sizeof(nc));
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

    std::cout << "Weights saved to " << path
              << " (" << (g_embed_weight.size() + g_proj_weight.size()
                        + g_proj_bias.size() + g_prop_weight.size()) * 4 / 1048576
              << " MB, " << nc << " skip edges)" << std::endl;
}

void load_weights()
{
    std::string path = g_weights_path;
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "ERROR: Cannot open " << path << std::endl;
        return;
    }

    // 验证魔数
    uint32_t magic = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != WEIGHTS_MAGIC) {
        std::cerr << "ERROR: Invalid weights file (bad magic: 0x"
                  << std::hex << magic << std::dec
                  << "). File may be corrupted or from an older version."
                  << std::endl;
        return;
    }

    // 读取一个张量：长度(uint64) + 数据(float[])
    auto read_tensor = [&](std::vector<float>& t) {
        uint64_t n = 0;
        file.read(reinterpret_cast<char*>(&n), sizeof(n));
        t.resize(static_cast<size_t>(n));
        file.read(reinterpret_cast<char*>(t.data()), n * sizeof(float));
    };

    read_tensor(g_embed_weight);
    read_tensor(g_proj_weight);
    read_tensor(g_proj_bias);
    read_tensor(g_prop_weight);

    // 确保接收缓冲区已分配（加载模式不调用 init_propagation）
    g_incoming.resize(N, 0.0f);

    // ============ 加载跳过连接拓扑 + 权重 ============
    // 旧版本权重文件可能没有跳过连接数据，通过 peeking 判断
    g_skip_ptr.assign(N + 1, 0);
    g_skip_dst.clear();
    g_skip_weight.clear();
    g_skip_grad.clear();

    if (file.peek() != EOF) {
        uint64_t nc = 0;
        file.read(reinterpret_cast<char*>(&nc), sizeof(nc));
        if (nc > 0) {
            std::vector<std::pair<uint32_t, uint32_t>> edges(nc);
            g_skip_weight.resize(nc);
            for (uint64_t i = 0; i < nc; ++i) {
                uint32_t src, dst;
                float w;
                file.read(reinterpret_cast<char*>(&src), sizeof(src));
                file.read(reinterpret_cast<char*>(&dst), sizeof(dst));
                file.read(reinterpret_cast<char*>(&w),   sizeof(w));
                edges[i] = {src, dst};
                g_skip_weight[i] = w;
            }

            // 按 src 排序构建 CSR
            std::sort(edges.begin(), edges.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

            g_skip_dst.resize(nc);
            for (size_t i = 0; i < nc; ++i)
                g_skip_dst[i] = edges[i].second;

            size_t e = 0;
            for (uint32_t idx = 0; idx < N; ++idx) {
                g_skip_ptr[idx] = static_cast<uint32_t>(e);
                while (e < nc && edges[e].first == idx) ++e;
            }
            g_skip_ptr[N] = static_cast<uint32_t>(nc);
            g_skip_grad.resize(nc, 0.0f);

            std::cout << "Loaded " << nc << " skip connections" << std::endl;
        }
    }

    std::cout << "Weights loaded from " << path << std::endl;
}
