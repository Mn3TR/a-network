#pragma once
#include <cstddef>
#include <string>

// ============ 框架级通用配置 ============
// 网络专有参数由各自的 Network 子类自己定义。

struct Config {
    // ======== 训练 ========
    size_t grad_accum = 4;
    int max_epochs = 100;
    float min_loss = 0.0f;

    // ======== 优化器 (Adam) ========
    float lr = 0.0001f;
    float lr_min = 0.000001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;

    // ======== 路径 ========
    std::string tokenizer_path = "tokenizer/tokenizer.json";
    std::string data_dir = "dataset/";
    std::string weights_path = "output/weights.bin";
    std::string log_dir = "log/";
};
