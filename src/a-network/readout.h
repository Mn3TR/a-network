#pragma once
#include "core/types.h"
#include <vector>

// Readout 工作区缓冲区（跨函数复用，避免反复分配）
extern std::vector<float> g_diff;       // [N]
extern std::vector<float> g_logits_d;   // [g_hidden_dim]

void init_readout_workspace();

// Readout 前向：network → logits
// flat_net[N], logits_v_out[g_vocab_size]
void forward_readout(const float* flat_net, float* logits_v_out);

// Cross Entropy Loss + 梯度
// logits[g_vocab_size], target_id, d_logits[g_vocab_size]（输出）
float cross_entropy_loss(const float* logits, size_t target_id, float* d_logits);
