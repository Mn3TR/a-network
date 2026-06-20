#pragma once
#include <cstddef>

// ============ 训练超参数 ============

// 每 token 注入后的传播步数
constexpr size_t g_prop_steps = 8;

// 梯度累积步数（累积多少步再更新一次权重）
constexpr size_t g_grad_accum = 4;

// 优化器参数
constexpr float g_lr = 0.0001f;       // 初始学习率
constexpr float g_lr_min = 0.000001f; // 最低学习率（退火终点）
constexpr float g_mu = 0.9f;          // 动量系数
constexpr float g_clip_norm = 50.0f;    // 梯度裁剪阈值

// 训练终止条件
constexpr int g_max_epochs = 100;       // 最多 epoch 数（训练终止条件之一）
constexpr float g_min_loss = 0.01f;    // loss 低于此值则提前停止

// 路径
constexpr const char* g_weights_path = "output/weights.bin";
constexpr const char* g_tokenizer_path = "tokenizer/tokenizer.json";
constexpr const char* g_data_dir = "dataset/";
constexpr const char* g_log_dir = "log/";
