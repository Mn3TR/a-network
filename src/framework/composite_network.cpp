#include "framework/composite_network.h"
#include <algorithm>
#include <string>

// ==================== 子网管理 ====================

size_t CompositeNetwork::add_child(std::unique_ptr<Network> child)
{
    size_t idx = m_children.size();
    m_children.push_back(std::move(child));
    return idx;
}

Network* CompositeNetwork::child(size_t idx)
{
    return (idx < m_children.size()) ? m_children[idx].get() : nullptr;
}

const Network* CompositeNetwork::child(size_t idx) const
{
    return (idx < m_children.size()) ? m_children[idx].get() : nullptr;
}

// ==================== 参数聚合 ====================

std::vector<ParamGroup> CompositeNetwork::param_groups()
{
    std::vector<ParamGroup> all;
    for (size_t i = 0; i < m_children.size(); ++i) {
        auto child_groups = m_children[i]->param_groups();
        std::string prefix = std::to_string(i) + ":";
        for (auto& g : child_groups)
            all.push_back({g.weights, g.gradients, prefix + g.name});
    }
    return all;
}

// ==================== 生命周期 ====================

void CompositeNetwork::init_weights()
{
    for (auto& c : m_children)
        c->init_weights();
}

void CompositeNetwork::reset_state()
{
    for (auto& c : m_children)
        c->reset_state();
}

void CompositeNetwork::zero_grad()
{
    for (auto& c : m_children)
        c->zero_grad();
}

// ==================== 序列化 ====================

void CompositeNetwork::save(const std::string& path)
{
    for (size_t i = 0; i < m_children.size(); ++i)
        m_children[i]->save(path + "." + std::to_string(i));
}

void CompositeNetwork::load(const std::string& path)
{
    for (size_t i = 0; i < m_children.size(); ++i)
        m_children[i]->load(path + "." + std::to_string(i));
}

// ==================== 默认训练/推理 ====================

float CompositeNetwork::train_step(size_t input_id, size_t target_id)
{
    if (m_children.empty()) return 0.0f;

    float total_loss = 0.0f;
    for (auto& c : m_children)
        total_loss += c->train_step(input_id, target_id);
    return total_loss;
}

size_t CompositeNetwork::generate_step(size_t input_id)
{
    if (m_children.empty()) return 0;
    return m_children[0]->generate_step(input_id);
}
