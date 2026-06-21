#pragma once
#include <cstddef>

// ============ A-Network 专有超参 ============

struct ANetworkConfig {
    // 张量维度
    size_t network_x = 80;
    size_t network_y = 80;
    size_t network_z = 80;
    size_t hidden_dim = 48;

    // 物理传播
    float time_decay = 0.9f;
    size_t prop_steps = 20;

    // 跳跃连接
    float skip_density = 0.20f;

    // 派生值
    size_t N() const { return network_x * network_y * network_z; }
};
