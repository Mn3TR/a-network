#pragma once
#include "core/types.h"
#include <vector>
#include <cstdint>

// 26 邻域偏移量
struct NeighborOffset { int dx, dy, dz; };
extern const NeighborOffset g_neighbors_26[26];
extern const size_t g_num_neighbors;

extern std::vector<float> g_prop_weight;  // [N]
extern std::vector<float> g_incoming;     // [N]

// ============ 长程跳跃连接（CSR 格式） ============
extern std::vector<uint32_t> g_skip_ptr;       // [N+1] — 每个细胞的出边起始偏移量
extern std::vector<uint32_t> g_skip_dst;       // [NC]  — 目标细胞索引
extern std::vector<float>    g_skip_weight;    // [NC]  — 可学习的连接权重
extern std::vector<float>    g_skip_grad;      // [NC]  — 梯度缓存

void init_propagation();

// 在任意缓冲区上执行一步传播（前向 + 反向共用同一核心）
void propagate_step(float* network, float* incoming);

// 在 NetworkView 上执行一步传播（供 generate 使用）
void propagate_network(NetworkView network);
