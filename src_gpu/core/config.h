#pragma once
// GPU 版训练配置（从 CPU 版同步）
constexpr int g_prop_steps = 20;
constexpr int g_grad_accum = 4;
constexpr float g_lr = 0.0001f;
constexpr float g_lr_min = 0.000001f;
constexpr float g_mu = 0.9f;
constexpr int g_max_epochs = 100;
constexpr float g_min_loss = 0.0f;
constexpr int g_vocab_size = 128256;
constexpr int g_hidden_dim = 48;
