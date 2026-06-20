#pragma once
#include "core/types.h"
#include "core/config.h"
#include <vector>

// SGD + Momentum 优化器
struct SGDMomentum {
    float lr = g_lr;
    float mu = g_mu;
    float clip_norm = g_clip_norm;
    std::vector<float> buf_embed;
    std::vector<float> buf_proj;
    std::vector<float> buf_bias;
    std::vector<float> buf_prop;
    std::vector<float> buf_skip;

    void init();
    void step();
    void zero_grad();
    void log_grad(const char* path);
    void set_lr(float lr) { this->lr = lr; }
};

extern SGDMomentum g_optim;

// 学习率调度（退火）
void lr_schedule(int epoch, int total_epochs);
