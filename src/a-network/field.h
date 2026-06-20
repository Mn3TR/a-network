#pragma once
#include "core/types.h"
#include <vector>
#include <cstdint>

// 26 邻域偏移量
struct NeighborOffset { int dx, dy, dz; };
extern const NeighborOffset g_neighbors_26[26];
extern const size_t g_num_neighbors;

// 传播权重
extern std::vector<float> g_prop_weight;  // [N]
extern std::vector<float> g_incoming;     // [N]

// ============ 预计算的坐标查找表（避免除法和取模） ============
extern std::vector<uint32_t> g_cell_x;  // [N]
extern std::vector<uint32_t> g_cell_y;  // [N]
extern std::vector<uint32_t> g_cell_z;  // [N]

// ============ 长程跳跃连接（CSR 格式） ============
extern std::vector<uint32_t> g_skip_ptr;       // [N+1] — 每个细胞的出边起始偏移量
extern std::vector<uint32_t> g_skip_dst;       // [NC]  — 目标细胞索引
extern std::vector<float>    g_skip_weight;    // [NC]  — 可学习的连接权重
extern std::vector<float>    g_skip_grad;      // [NC]  — 梯度缓存

// ============ 反向跳跃连接 CSR（gather 模式用） ============
extern std::vector<uint32_t> g_rev_skip_ptr;   // [N+1] — 每个细胞作为目标的入边起始偏移
extern std::vector<uint32_t> g_rev_skip_src;   // [NC]  — 源细胞索引

void init_propagation();

// 在任意缓冲区上执行一步传播（gather 模式，无原子操作）
// network: 场状态（读/写），incoming: 输入信号缓冲区（读/写）
// act_tanh: 如果非空，存储 tanh(network[idx]*0.5) 到该缓冲区（供反向传播使用）
void propagate_step(float* network, float* incoming, float* act_tanh = nullptr);

// 在 NetworkView 上执行一步传播（供 generate 使用）
void propagate_network(NetworkView network);
