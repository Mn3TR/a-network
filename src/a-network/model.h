#pragma once
#include "core/types.h"
#include "core/config.h"
#include "config.h"
#include <vector>

void init_all();

// 单 token 训练：注入 → 传播 → 读出 h → LM Head → loss
float train_token(size_t input_id, size_t target_id,
                  float* flat_net, float* incoming_buf);

// 生成模式：注入 → 传播 → 读出 h → LM Head → argmax
size_t generate_token(size_t input_id, float* flat_net, float* incoming_buf);
