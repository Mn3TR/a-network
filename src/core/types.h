#pragma once
#include <cstddef>

// ============ 张量维度 ============
constexpr size_t network_x = 80;
constexpr size_t network_y = 80;
constexpr size_t network_z = 80;

// ============ ConvertLayer 常量 ============
constexpr size_t g_vocab_size = 128256;
constexpr size_t g_hidden_dim = 48;

// ============ 传播常量 ============
constexpr float g_time_decay = 0.9f;

// 张量大小
constexpr size_t N = network_x * network_y * network_z;
