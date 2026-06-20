#pragma once
#include "core/types.h"
#include <vector>

// 从场读出权重
extern std::vector<float> g_out_weight;  // [H, N]
extern std::vector<float> g_out_bias;    // [N]

// 工作区
extern std::vector<float> g_diff;          // [N] — field - b_out

// ============ 初始化 ============
// 分配工作区（g_out_weight/g_out_bias/g_diff）。在所有模式下调用。
// 幂等：可重复调用，只保证 buffer 大小正确，不重置已加载的权重值。
void init_readout_workspace();

// 随机初始化权重（仅训练新模型时调用；load/gen 模式切勿调用，否则覆盖 checkpoint）
void init_readout_weights();

// Cross Entropy Loss + 梯度（保持不变）
float cross_entropy_loss(const float* logits, size_t target_id, float* d_logits);
