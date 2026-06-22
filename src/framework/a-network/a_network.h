#pragma once
#include "framework/a-network/a_network_config.h"
#include "framework/network.h"
#include <vector>
#include <cstdint>

// ============ A-Network — 物理神经网络实现 ============
// 以 3D 张量场作为可微分的「物理」计算介质。
// Token 以信号形式注入场中，传播 N 步后读出为 logits。

class ANetwork : public Network {
public:
    explicit ANetwork(const ANetworkConfig& cfg);
    ~ANetwork() override = default;

    // ============== Network 接口 ==============

    std::vector<ParamGroup> param_groups() override;
    void init_weights() override;
    void reset_state() override;
    float train_step(size_t input_id, size_t target_id) override;
    size_t generate_step(size_t input_id) override;
    void save(const std::string& path) override;
    void load(const std::string& path) override;
    void zero_grad() override;
    void dump_field(const std::string& path) override;

    // ============== 内部方法 ==============

private:
    ANetworkConfig m_cfg;
    size_t m_N;   // N = network_x * network_y * network_z
    size_t m_H;   // hidden_dim
    size_t m_V;   // g_vocab_size（来自 tokenizer.h）

    // ---- 权重 ----

    // Embedding / LM Head（输入查找和输出投影共享同一矩阵）
    std::vector<float> m_embed_weight;   // [V, H]

    // 投影入场
    std::vector<float> m_in_weight;      // [H, N]
    std::vector<float> m_in_bias;        // [N]

    // 从场读出
    std::vector<float> m_out_weight;     // [H, N]
    std::vector<float> m_out_bias;       // [N]

    // 传播权重
    std::vector<float> m_prop_weight;    // [N]

    // ---- 梯度 ----

    std::vector<float> m_embed_grad;     // [V, H]
    std::vector<float> m_in_weight_grad; // [H, N]
    std::vector<float> m_in_bias_grad;   // [N]
    std::vector<float> m_out_weight_grad;// [H, N]
    std::vector<float> m_out_bias_grad;  // [N]
    std::vector<float> m_prop_grad;      // [N]

    // ---- 场状态 ----

    std::vector<float> m_flat_net;       // [N]，场当前值
    std::vector<float> m_incoming;       // [N]，邻域输入缓冲区

    // ---- 工作区 ----

    std::vector<float> m_diff;           // [N]，field - b_out
    std::vector<std::vector<float>> m_prop_act; // [prop_steps][N]，前向 tanh 激活值

    // ---- LM Head / train_step 工作区（替换 static thread_local，避免 OMP 冲突） ----

    std::vector<float> m_head_logits;    // [V]  LM Head logits
    std::vector<float> m_head_d_logits;  // [V]  LM Head logits 梯度
    std::vector<float> m_head_h_cur;     // [H]  当前 hidden
    std::vector<float> m_head_d_h_cur;   // [H]  hidden 梯度
    std::vector<float> m_head_d_network; // [N]  场梯度
    std::vector<float> m_head_d_emb;     // [H]  embedding 梯度
    std::vector<float> m_act_work;       // [N]  propagate_step 备用激活缓冲区

    // ---- 坐标表 ----

    std::vector<uint32_t> m_cell_x;      // [N]
    std::vector<uint32_t> m_cell_y;      // [N]
    std::vector<uint32_t> m_cell_z;      // [N]

    // ---- 长程跳跃连接（CSR 格式） ----

    std::vector<uint32_t> m_skip_ptr;     // [N+1]
    std::vector<uint32_t> m_skip_dst;     // [NC]
    std::vector<float>    m_skip_weight;  // [NC]
    std::vector<float>    m_skip_grad;    // [NC]
    std::vector<uint32_t> m_rev_skip_ptr; // [N+1]
    std::vector<uint32_t> m_rev_skip_src; // [NC]

    // ========== 内部初始化方法 ==========

    void build_coord_tables();
    void build_skip_connections();

    // ========== 前向方法 ==========

    // 注入：hidden → field (in_weight, in_bias)
    void tokens_to_field(const float* hidden, size_t S, float* field);

    // 传播：field 一步演化（邻域 scatter + 跳跃连接 + tanh）
    void propagate_step(float* network, float* incoming,
                        float* act_tanh = nullptr);

    // field → h (out_weight, out_bias)
    void readout_h(const float* flat_net, float* h_out);

    // Softmax Cross-Entropy Loss
    float cross_entropy_loss(const float* logits, size_t target_id,
                              float* d_logits);

    // h → logits → loss
    float lm_head_loss(const float* h, size_t target_id, float* d_h);

    // ========== 反向方法 ==========

    // d_network → d_in_weight / d_in_bias / d_hidden
    void backward_inject(const float* d_network, size_t S,
                         const float* hidden, float* d_hidden);

    // d_incoming → d_prop_weight / d_skip_grad + 更新 d_network
    void backward_propagate(float* d_network, float* d_incoming,
                            const float* phase2_act);

    // d_h → d_out_weight / d_out_bias / d_network
    void backward_readout_h(const float* d_h, float* d_network);

    // ========== 常量 ==========

    // 26 邻域偏移量（静态，所有实例共享）
    struct NeighborOffset { int dx, dy, dz; };
    static constexpr NeighborOffset kNeighbors26[26] = {
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
        {-1, -1, 0}, {-1, 1, 0}, {1, -1, 0}, {1, 1, 0},
        {-1, 0, -1}, {-1, 0, 1}, {1, 0, -1}, {1, 0, 1},
        {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1},
        {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
        {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1},
    };
    static constexpr size_t kNumNeighbors = 26;
};
