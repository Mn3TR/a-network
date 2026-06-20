#include "model.h"
#include "convert.h"
#include "field.h"
#include "readout.h"
#include "backward.h"
#include <cmath>

// ============ 初始化 ============
void init_all()
{
    g_embed_grad.resize(g_vocab_size * g_hidden_dim, 0.0f);
    g_proj_grad.resize(g_hidden_dim * N, 0.0f);
    g_bias_grad.resize(N, 0.0f);
    g_prop_grad.resize(N, 0.0f);
    g_signal.resize(N);
    init_readout_workspace();
    g_prop_act.clear();
}

// ============ 训练步骤 ============
float train_token(size_t input_id, size_t target_id, float* flat_net,
                  float* incoming_buf)
{
    // ---- 前向 ----

    // 1. 计算注入信号：signal = bias + embed[input_id] · proj_weight
    const float* embed = g_embed_weight.data() + input_id * g_hidden_dim;
    #pragma omp parallel for
    for (int j = 0; j < static_cast<int>(N); ++j) g_signal[j] = g_proj_bias[j];

    for (size_t i = 0; i < g_hidden_dim; ++i) {
        float ei = embed[i];
        const float* row = g_proj_weight.data() + i * N;
        #pragma omp parallel for
        for (int j = 0; j < static_cast<int>(N); ++j)
            g_signal[j] += ei * row[j];
    }

    // 2. 注入
    #pragma omp parallel for
    for (int j = 0; j < static_cast<int>(N); ++j)
        flat_net[j] += g_signal[j];

    // 3. 传播 N 步，存快照供反向使用
    g_prop_act.clear();
    for (int s = 0; s < g_prop_steps; ++s) {
        propagate_step(flat_net, incoming_buf);
        g_prop_act.emplace_back(flat_net, flat_net + N);
    }

    // 4. 读取 logits
    std::vector<float> logits_v(g_vocab_size);
    forward_readout(flat_net, logits_v.data());

    // 5. Loss + d_logits_v
    std::vector<float> d_logits_v(g_vocab_size);
    float loss = cross_entropy_loss(logits_v.data(), target_id, d_logits_v.data());

    // ---- 反向 ----

    // 6. Readout 反向
    std::vector<float> d_network(N);
    backward_readout(d_logits_v.data(), d_network.data());

    // 7. 传播反向（逆序）
    std::vector<float> d_incoming_buf(N, 0.0f);
    for (int s = g_prop_steps - 1; s >= 0; --s) {
        backward_propagate(d_network.data(), d_incoming_buf.data(),
                           g_prop_act[s].data());
    }

    // 8. 注入反向
    backward_inject(input_id, d_network.data());

    return loss;
}
