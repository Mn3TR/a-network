#pragma once
#include "core/types.h"
#include "core/config.h"
#include <vector>
#include <string>

struct Adam {
    float lr = g_lr;
    float beta1 = g_beta1;
    float beta2 = g_beta2;
    float eps = g_eps;

    // 一阶矩（动量） m
    std::vector<float> m_embed;
    std::vector<float> m_in_weight;
    std::vector<float> m_in_bias;
    std::vector<float> m_out_weight;
    std::vector<float> m_out_bias;
    std::vector<float> m_prop;
    std::vector<float> m_skip;

    // 二阶矩（自适应学习率） v
    std::vector<float> v_embed;
    std::vector<float> v_in_weight;
    std::vector<float> v_in_bias;
    std::vector<float> v_out_weight;
    std::vector<float> v_out_bias;
    std::vector<float> v_prop;
    std::vector<float> v_skip;

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

    int m_step_count = 0;   // 已更新步数（用于偏置校正）

    void init();
    void step();
    void zero_grad();
    void log_grad(const char* path);
    void flush_log(const char* path);
    void set_lr(float lr) { this->lr = lr; }
};

extern Adam g_optim;
void lr_schedule(int epoch, int total_epochs);
