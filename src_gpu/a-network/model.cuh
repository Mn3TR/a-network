#pragma once
#include "field.cuh"
#include "../core/config.h"

// ============ 权重（永久驻留 GPU） ============
extern float*       d_W_embed;     // [V, H]
extern float*       d_W_in;        // [H, N]
extern float*       d_b_in;        // [N]
extern float*       d_W_out;       // [H, N]
extern float*       d_b_out;       // [N]

// ============ 梯度缓冲 ============
extern float*       d_W_embed_grad;
extern float*       d_W_in_grad;
extern float*       d_b_in_grad;
extern float*       d_W_out_grad;
extern float*       d_b_out_grad;

// ============ 动量缓冲 ============
extern float*       d_m_embed;
extern float*       d_m_in_w;
extern float*       d_m_in_b;
extern float*       d_m_out_w;
extern float*       d_m_out_b;
extern float*       d_m_prop;
extern float*       d_m_skip;

// ============ 辅助缓冲 ============
extern float*       d_diff;        // [N]  — field - b_out
extern float*       d_logits;      // [V]

// ============ 初始化 ============
void gpu_init_model();   // cudaMalloc 所有权重/梯度/动量/辅助
void gpu_free_model();
void gpu_upload_weights();  // CPU → GPU
void gpu_download_weights(float* cpu_embed, float* cpu_w_in, float* cpu_b_in,
                           float* cpu_w_out, float* cpu_b_out);
void gpu_upload_skip_csr(const uint32_t* rev_skip_ptr,
                          const uint32_t* rev_skip_src,
                          const float* skip_weight, uint32_t num_edges);

// ============ Kernel 启动函数 ============

// 注入: field += h × W_in + b_in
void gpu_inject(float* field, const float* h);

// 读出: field → h （含 diff 写入）
void gpu_readout_h(const float* field, float* h);

// LM Head: h → logits
void gpu_lm_head(const float* h);

// Softmax + CE Loss: logits → loss + d_logits
float gpu_softmax_loss(int target_id);

// LM Head 反向: d_logits → d_h + d_W_embed_grad
void gpu_lm_head_backward(const float* h, int target_id);

// 读出反向: d_h → d_field + d_W_out_grad + d_b_out_grad
void gpu_readout_backward(const float* d_h);

// 注入反向: d_field → d_emb + d_W_in_grad + d_b_in_grad
void gpu_inject_backward(const float* h);

// 优化器: SGD + Momentum + 自适应梯度裁剪
void gpu_optimizer_step(float lr, float mu);

// 梯度清零
void gpu_zero_all_gradients();
