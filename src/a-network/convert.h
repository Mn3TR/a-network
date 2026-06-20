#pragma once
#include "core/types.h"
#include <vector>

// Embedding / LM Head 权重（输入查找和输出投影共享同一矩阵）
extern std::vector<float> g_embed_weight;  // [V, H]

// 投影入场
extern std::vector<float> g_in_weight;     // [H, N]
extern std::vector<float> g_in_bias;       // [N]

// 分配权重缓冲（不重置已有值，幂等）。所有模式调用。
void init_convert_workspace();

// 随机初始化权重（仅训练新模型时调用）
void init_convert_weights();

// 序列级注入：将 hidden 状态 [S, H] 累积到场 [N]
// field 会被 += 结果
void tokens_to_field(const float* hidden, size_t S, float* field);
