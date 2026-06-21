#pragma once
#include "core/types.h"
#include <vector>

// 从场读出权重
extern std::vector<float> g_out_weight;  // [H, N]
extern std::vector<float> g_out_bias;    // [N]

// 工作区
extern std::vector<float> g_diff;          // [N] — field - b_out

// ============ 初始化 ============
void init_readout_workspace();
void init_readout_weights();

// Cross Entropy Loss
float cross_entropy_loss(const float* logits, size_t target_id, float* d_logits);
