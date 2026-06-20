#pragma once
#include "core/types.h"
#include <vector>

// ============ 随机投影读出 ============
// field → 固定随机投影 R → tmp[128] → 可学习 W_out → h[192]

// 固定随机投影矩阵（不训练，初始化后不变）
extern std::vector<float> g_proj_fixed;  // [P, N]

// 可学习的小读出矩阵
extern std::vector<float> g_out_weight;  // [H, P]
extern std::vector<float> g_out_bias;    // [P]

// 工作区
extern std::vector<float> g_diff;        // [P] — tmp - bias
extern std::vector<float> g_proj_tmp;    // [P] — 投影后的 tmp

// ============ 初始化 ============
// 分配工作区（g_proj_fixed/g_out_weight/g_out_bias/g_diff/g_proj_tmp）
// 幂等：可重复调用，只保证 buffer 大小正确，不重置已加载的权重值。
void init_readout_workspace();

// 用固定种子填充 g_proj_fixed（所有模式一致，不存盘，load 后重造）
void init_random_projection();

// 随机初始化可学习权重（仅训练新模型时调用；load/gen 模式切勿调用，否则覆盖 checkpoint）
void init_readout_weights();

// Cross Entropy Loss + 梯度（保持不变）
float cross_entropy_loss(const float* logits, size_t target_id, float* d_logits);
