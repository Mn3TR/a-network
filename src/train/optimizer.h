#pragma once
#include "core/types.h"
#include "core/config.h"
#include <vector>
#include <string>

struct SGDMomentum {
    float lr = g_lr;
    float mu = g_mu;

    std::vector<float> buf_embed;
    std::vector<float> buf_in_weight;
    std::vector<float> buf_in_bias;
    std::vector<float> buf_out_weight;
    std::vector<float> buf_out_bias;
    std::vector<float> buf_prop;
    std::vector<float> buf_skip;

    // 梯度日志缓冲（避免每 N 步写文件）
    std::string log_buffer;
    int log_line_count = 0;
    static constexpr int log_flush_interval = 100;

    // step() 计算的梯度范数缓存，供 log_grad() 复用
    float m_cache_embed  = 0.0f;
    float m_cache_in_w   = 0.0f;
    float m_cache_in_b   = 0.0f;
    float m_cache_out_w  = 0.0f;
    float m_cache_out_b  = 0.0f;
    float m_cache_prop   = 0.0f;
    float m_cache_skip   = 0.0f;

    // 自适应梯度裁剪（EMA 追踪全局梯度范数）
    int m_step_count = 0;
    float m_ema_norm = 0.0f;
    static constexpr float m_ema_decay = 0.99f;
    static constexpr float m_clip_factor = 3.0f;

    void init();
    void step();
    void zero_grad();
    void log_grad(const char* path);
    void flush_log(const char* path);
    void set_lr(float lr) { this->lr = lr; }
};

extern SGDMomentum g_optim;
void lr_schedule(int epoch, int total_epochs);
