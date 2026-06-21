#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n",              \
                err, __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(1);                                                      \
    }                                                                 \
} while(0)

constexpr int NET_X = 80, NET_Y = 80, NET_Z = 80;
constexpr int N = NET_X * NET_Y * NET_Z;
constexpr int g_prop_steps = 20;

struct NeighborOffset { int dx, dy, dz; };
extern const NeighborOffset g_neighbors_26[26];
extern const int g_num_neighbors;

// 设备指针
extern float*       d_field;       // [N] — 场状态（前向用，与 d_network 分离）
extern float*       d_network;     // [N] — 梯度缓冲（反向用）
extern float*       d_incoming;
extern float*       d_act;
extern float*       d_prop_weight;
extern float*       d_prop_grad;
extern uint32_t*    d_rev_skip_ptr;
extern uint32_t*    d_rev_skip_src;
extern float*       d_skip_weight;
extern float*       d_skip_grad;

extern float**      d_act_buf;  // [g_prop_steps][N]
extern uint32_t     gpu_num_edges;

// 初始化/清理
void gpu_init_field();
void gpu_free_field();

// 上传/下载
void gpu_upload_network(const float* src);
void gpu_download_network(float* dst);
void gpu_download_incoming(float* dst);
void gpu_download_gradients(float* prop_grad, float* skip_grad, uint32_t num_edges);
void gpu_upload_prop_weights(const float* src);
void gpu_upload_incoming(const float* src);
void gpu_upload_skip_csr(const uint32_t* rev_skip_ptr,
                         const uint32_t* rev_skip_src,
                         const float* skip_weight, uint32_t num_edges);

void gpu_zero_network();
void gpu_zero_field();
void gpu_zero_incoming();
void gpu_clear_gradients();

// 前向/反向传播（无 PCIe，数据常驻设备）
void gpu_forward_propagate();
void gpu_backward_propagate();
