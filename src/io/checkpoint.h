#pragma once
#include <cstdint>

// 权重文件魔数 "RNW1" (Reasonix Network Weights v1)
constexpr uint32_t WEIGHTS_MAGIC = 0x31574E52u;  // little-endian: 'R','N','W','1'

// 权重文件格式版本（魔数之后写入 uint32_t）
// v1: 初始版本（embed, in_weight, in_bias, out_weight, out_bias, prop_weight,
//            skip connections）
constexpr uint32_t WEIGHTS_VERSION = 1u;

void save_weights();
void load_weights();
