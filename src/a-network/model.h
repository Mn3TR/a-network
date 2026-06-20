#pragma once
#include "core/types.h"
#include "core/config.h"
#include "config.h"   // A-Network 专有超参
#include <vector>

// 初始化整个 A Network 组件（权重 + 传播 + 训练状态）
void init_all();

// 单个 token 的训练步骤（前向 + 反向 + loss）
float train_token(size_t input_id, size_t target_id, float* flat_net,
                  float* incoming_buf);
