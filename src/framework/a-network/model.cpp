#include "a_network.h"
#include <cmath>
#include <algorithm>

// ==================== LM Head: h → logits → loss ====================

float ANetwork::lm_head_loss(const float* h, size_t target_id, float* d_h)
{
    size_t H = m_H, V = m_V;
    float* logits = m_head_logits.data();
    float* d_logits = m_head_d_logits.data();

    // logits = embed_weight * h
    #pragma omp parallel for
    for (int t = 0; t < static_cast<int>(V); ++t) {
        float sum = 0.0f;
        const float* row = m_embed_weight.data() + static_cast<size_t>(t) * H;
        for (size_t k = 0; k < H; ++k)
            sum += row[k] * h[k];
        logits[static_cast<size_t>(t)] = sum;
    }

    float loss = cross_entropy_loss(logits, target_id, d_logits);

    // d_h = embed_weight^T * d_logits
    #pragma omp parallel for
    for (int k = 0; k < static_cast<int>(H); ++k) {
        float sum = 0.0f;
        for (size_t t = 0; t < V; ++t)
            sum += d_logits[t] * m_embed_weight[t * H + static_cast<size_t>(k)];
        d_h[static_cast<size_t>(k)] = sum;
    }

    // embed_grad += d_logits * h^T
    #pragma omp parallel for
    for (int t = 0; t < static_cast<int>(V); ++t) {
        float dv = d_logits[t];
        if (dv == 0.0f) continue;
        float* row = m_embed_grad.data() + static_cast<size_t>(t) * H;
        for (size_t k = 0; k < H; ++k)
            row[k] += dv * h[k];
    }

    return loss;
}

// ==================== 训练步 ====================

float ANetwork::train_step(size_t input_id, size_t target_id)
{
    size_t H = m_H, prop_steps = m_cfg.prop_steps;

    // ---- 前向 ----

    // 1. 注入
    const float* emb = m_embed_weight.data() + input_id * H;
    tokens_to_field(emb, 1, m_flat_net.data());

    // 2. 传播（将 tanh 值预存到 m_prop_act）
    for (size_t s = 0; s < prop_steps; ++s)
        propagate_step(m_flat_net.data(), m_incoming.data(), m_prop_act[s].data());

    // 3. 读 h → LM Head → loss
    std::fill(m_head_h_cur.begin(), m_head_h_cur.end(), 0.0f);
    std::fill(m_head_d_h_cur.begin(), m_head_d_h_cur.end(), 0.0f);
    readout_h(m_flat_net.data(), m_head_h_cur.data());
    float loss = lm_head_loss(m_head_h_cur.data(), target_id, m_head_d_h_cur.data());

    // ---- 反向 ----

    // 1. 读出头反向
    std::fill(m_head_d_network.begin(), m_head_d_network.end(), 0.0f);
    backward_readout_h(m_head_d_h_cur.data(), m_head_d_network.data());

    // 2. 传播反向
    std::fill(m_incoming.begin(), m_incoming.end(), 0.0f);
    for (int s = static_cast<int>(prop_steps) - 1; s >= 0; --s)
        backward_propagate(m_head_d_network.data(), m_incoming.data(), m_prop_act[s].data());

    // 3. 注入反向
    std::fill(m_head_d_emb.begin(), m_head_d_emb.end(), 0.0f);
    backward_inject(m_head_d_network.data(), 1, emb, m_head_d_emb.data());

    // 4. Embedding 梯度
    float* row = m_embed_grad.data() + input_id * H;
    for (size_t k = 0; k < H; ++k) row[k] += m_head_d_emb[k];

    return loss;
}

// ==================== 生成步 ====================

size_t ANetwork::generate_step(size_t input_id)
{
    size_t H = m_H, V = m_V, prop_steps = m_cfg.prop_steps;

    // 1. 注入
    const float* emb = m_embed_weight.data() + input_id * H;
    tokens_to_field(emb, 1, m_flat_net.data());

    // 2. 传播
    for (size_t s = 0; s < prop_steps; ++s)
        propagate_step(m_flat_net.data(), m_incoming.data());

    // 3. 读 h
    std::fill(m_head_h_cur.begin(), m_head_h_cur.end(), 0.0f);
    readout_h(m_flat_net.data(), m_head_h_cur.data());

    // 4. LM Head → logits
    #pragma omp parallel for
    for (int t = 0; t < static_cast<int>(V); ++t) {
        float sum = 0.0f;
        const float* row = m_embed_weight.data() + static_cast<size_t>(t) * H;
        for (size_t k = 0; k < H; ++k)
            sum += row[k] * m_head_h_cur[k];
        m_head_logits[static_cast<size_t>(t)] = sum;
    }

    // 5. Argmax
    auto it = std::max_element(m_head_logits.begin(), m_head_logits.end());
    return static_cast<size_t>(std::distance(m_head_logits.begin(), it));
}
