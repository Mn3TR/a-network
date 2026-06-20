#pragma once
#include "core/types.h"
#include <vector>

// ============ 梯度缓存 ============

// Embedding / LM Head
extern std::vector<float> g_embed_grad;  // [V, H]

// 投影入场 (W_in, b_in)
extern std::vector<float> g_in_weight_grad;  // [H, N]
extern std::vector<float> g_in_bias_grad;    // [N]

// 从场读出 (W_out, b_out)
extern std::vector<float> g_out_weight_grad;  // [H, N]
extern std::vector<float> g_out_bias_grad;    // [N]

// 传播
extern std::vector<float> g_prop_grad;  // [N]

// ============ 工作区 ============
extern std::vector<std::vector<float>> g_prop_act;

// ============ 反向函数声明 ============

// 序列注入反向
void backward_inject(const float* d_network, size_t S,
                     const float* hidden, float* d_hidden);

// 传播反向
void backward_propagate(float* d_network, float* d_incoming,
                        const float* phase2_act);
