#pragma once
#include <vector>
#include <string>
#include <cstddef>

// ============ 参数组 ============
// 一个 ParamGroup 暴露一组权重及其对应梯度。
// Trainer/Optimizer/Checkpoint 通过它统一操作所有参数，
// 不依赖具体子类。

struct ParamGroup {
    std::vector<float>* weights;     // 权重指针
    std::vector<float>* gradients;   // 梯度指针
    std::string         name;        // 日志用名称
};

// ============ 网络抽象基类 ============
// 所有物理网络实现继承此类。

class Network {
public:
    virtual ~Network() = default;

    // === 参数暴露 ===
    // 返回所有可训练参数组（权重 + 梯度），供优化器和 checkpoint 遍历
    virtual std::vector<ParamGroup> param_groups() = 0;

    // === 生命周期 ===
    virtual void init_weights() = 0;    // 随机初始化所有权重
    virtual void reset_state() = 0;     // 重置场状态（新 epoch / 新序列）

    // === 训练 ===
    virtual void zero_grad() = 0;               // 清零梯度缓冲
    virtual float train_step(size_t input_id, size_t target_id) = 0;

    // === 推理 ===
    virtual size_t generate_step(size_t input_id) = 0;

    // === 序列化 ===
    virtual void save(const std::string& path) = 0;
    virtual void load(const std::string& path) = 0;

    // === 调试 ===
    virtual void dump_field(const std::string& path) {}
};
