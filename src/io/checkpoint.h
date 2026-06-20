#pragma once
#include <cstdint>

// 权重文件魔数 "RNW1" (Reasonix Network Weights v1)
constexpr uint32_t WEIGHTS_MAGIC = 0x31574E52u;  // little-endian: 'R','N','W','1'

void save_weights();
void load_weights();
