#pragma once
#include "framework/network.h"
#include <vector>
#include <string>

// ============ Adam 优化器（ParamGroup 模式） ============
// 不再直接引用全局变量；通过 Network::param_groups() 动态遍历参数组。

class Adam {
public:
    float lr;
    float beta1, beta2, eps;

    Adam(float lr = 0.001f, float beta1 = 0.9f,
         float beta2 = 0.999f, float eps = 1e-8f);

    // 根据 param_groups 初始化动量/自适应率缓冲区
    void init(const std::vector<ParamGroup>& groups);

    // 一步 Adam 更新：遍历所有参数组
    void step(const std::vector<ParamGroup>& groups);

    // 梯度日志（每步调用，内部缓冲，每 log_flush_interval 行写入文件）
    void log_grad(const char* path);
    void flush_log(const char* path);

private:
    struct State {
        std::vector<float> m;  // 一阶矩
        std::vector<float> v;  // 二阶矩
    };
    std::vector<State> m_states;

    int m_step_count = 0;

    // 梯度范数缓存（log_grad 用）
    std::vector<float> m_grad_norms;
    std::vector<std::string> m_group_names;

    // 日志缓冲
    std::string log_buffer;
    int log_line_count = 0;
    static constexpr int kLogFlushInterval = 100;
};
