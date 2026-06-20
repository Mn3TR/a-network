#pragma once
#include "core/types.h"
#include <vector>

// 梯度缓存（供 optimizer 使用）
extern std::vector<float> g_embed_grad;
extern std::vector<float> g_proj_grad;
extern std::vector<float> g_bias_grad;
extern std::vector<float> g_prop_grad;

// 工作区（训练专用，跨函数复用）
extern std::vector<float> g_signal;
extern std::vector<std::vector<float>> g_prop_act;

void backward_readout(const float* d_logits_v, float* d_network);
void backward_propagate(float* d_network, float* d_incoming, const float* phase2_act);
void backward_inject(size_t token_id, const float* d_network);
