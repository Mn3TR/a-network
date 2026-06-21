#pragma once
#include <cstddef>
#include <string>

// ============ 训练超参数 ============

// 每 token 注入后的传播步数
constexpr size_t g_prop_steps = 20;

// 梯度累积步数（累积多少步再更新一次权重）
constexpr size_t g_grad_accum = 4;

// 优化器参数
constexpr float g_lr = 0.0001f;       // 初始学习率
constexpr float g_lr_min = 0.000001f; // 最低学习率（退火终点）

// Adam 超参数
constexpr float g_beta1 = 0.9f;       // 一阶矩衰减系数
constexpr float g_beta2 = 0.999f;     // 二阶矩衰减系数
constexpr float g_eps = 1e-8f;        // 数值稳定性常数

// 训练终止条件
constexpr int g_max_epochs = 100;       // 最多 epoch 数（训练终止条件之一）
constexpr float g_min_loss = 0.0f;     // 设为 0 禁用提前终止

// 路径（运行时可变——run_train 会改为时间戳目录）
extern std::string g_weights_path;
extern std::string g_log_dir;
constexpr const char* g_tokenizer_path = "tokenizer/tokenizer.json";
constexpr const char* g_data_dir = "dataset/";
