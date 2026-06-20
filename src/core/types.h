#pragma once
#include <mdspan>
#include <cstddef>

// ============ 张量维度 ============
constexpr size_t network_x = 64;
constexpr size_t network_y = 64;
constexpr size_t network_z = 64;

using NetworkView = std::mdspan<float, std::extents<size_t, network_x, network_y, network_z>>;

// ============ ConvertLayer 常量 ============
constexpr size_t g_vocab_size = 128256;
constexpr size_t g_hidden_dim = 192;

// ============ 传播常量 ============
constexpr float g_time_decay = 0.9f;

// 张量大小
constexpr size_t N = network_x * network_y * network_z;
