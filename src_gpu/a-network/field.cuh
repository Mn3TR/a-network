#pragma once
#include <cstdint>

// ============ 网络维度 ============
constexpr int NET_X = 80;
constexpr int NET_Y = 80;
constexpr int NET_Z = 80;
constexpr int N = NET_X * NET_Y * NET_Z;  // 512000

// 26-邻域
struct NeighborOffset { int dx, dy, dz; };
extern const NeighborOffset g_neighbors_26[26];
extern const int g_num_neighbors;

// ============ 设备内存 ============
extern float*       d_network;
extern float*       d_incoming;
extern float*       d_act;
extern float*       d_prop_weight;
extern uint32_t*    d_rev_skip_ptr;   // [N+1]
extern uint32_t*    d_rev_skip_src;   // [num_edges]
extern float*       d_skip_weight;    // [num_edges]

// ============ 初始化 / 清理 ============
void gpu_init_field();
void gpu_free_field();
void gpu_upload_prop_weights(const float* cpu_prop_weight);
void gpu_upload_skip_csr(const uint32_t* rev_skip_ptr,
                         const uint32_t* rev_skip_src,
                         const float* skip_weight,
                         uint32_t num_edges);

// ============ 传播 ============
void gpu_propagate_step(float* host_network  = nullptr,
                         float* host_incoming = nullptr,
                         float* host_act      = nullptr);
