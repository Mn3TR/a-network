#pragma once
#include "framework/network.h"
#include <memory>
#include <vector>

// ============ 复合网络基类 ============
// 持有多个子 Network，自动处理参数聚合、生命周期、序列化等样板代码。
//
// 子类只需重写 train_step() / generate_step() 来编排自定义的前向/反向流程，
// 例如：级联、多头读出、Transformer 控制低层网络等。

class CompositeNetwork : public Network {
public:
    ~CompositeNetwork() override = default;

    // ---- 子网管理 ----

    // 添加子网络，返回其索引
    size_t add_child(std::unique_ptr<Network> child);

    // 获取子网（返回裸指针，调用者不应接管所有权）
    Network* child(size_t idx);
    const Network* child(size_t idx) const;
    size_t child_count() const { return m_children.size(); }

    // ---- Network 接口（默认委托所有子网） ----

    std::vector<ParamGroup> param_groups() override;

    void init_weights() override;
    void reset_state() override;
    void zero_grad() override;

    // 序列化：每个子网保存到 path.0, path.1, … 文件
    void save(const std::string& path) override;
    void load(const std::string& path) override;

    // ---- 可重写的默认行为 ----

    // 默认：所有子网分别对同一对 (input, target) 做 train_step，返回损失之和（简单 ensemble）
    // 子类可重写以实现级联、分层控制等复杂编排
    float train_step(size_t input_id, size_t target_id) override;

    // 默认：返回第一个子网的生成结果
    size_t generate_step(size_t input_id) override;

protected:
    std::vector<std::unique_ptr<Network>> m_children;
};
