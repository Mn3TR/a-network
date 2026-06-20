#pragma once

// ============ A Network 专有超参数 ============

// 长程跳跃连接密度（每个细胞概率）
constexpr float g_skip_density = 0.20f;

// 采样 softmax 负样本数（训练时只算 target + 这些负样本，替代全词表 128256）
constexpr size_t g_num_negatives = 200;
