#include "field.cuh"
#include <cuda_runtime.h>
#include <cstdio>

// ============ 26-邻域（与 CPU 版一致） ============
const NeighborOffset g_neighbors_26[] = {
    {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
    {-1, -1, 0}, {-1, 1, 0}, {1, -1, 0}, {1, 1, 0},
    {-1, 0, -1}, {-1, 0, 1}, {1, 0, -1}, {1, 0, 1},
    {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1},
    {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
    {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1},
};
const int g_num_neighbors = 26;

// ============ 设备内存 ============
float*       d_network      = nullptr;
float*       d_incoming     = nullptr;
float*       d_act          = nullptr;
float*       d_prop_weight  = nullptr;
uint32_t*    d_rev_skip_ptr = nullptr;
uint32_t*    d_rev_skip_src = nullptr;
float*       d_skip_weight  = nullptr;

// ============ 常量内存 ============
__constant__ int      c_net_x = NET_X;
__constant__ int      c_net_y = NET_Y;
__constant__ int      c_net_z = NET_Z;
__constant__ float    c_time_decay = 0.9f;
__constant__ int      c_num_neighbors = 26;
__constant__ NeighborOffset c_neighbors_26[26];

// ============ 检查 CUDA 错误 ============
#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n",              \
                err, __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(1);                                                      \
    }                                                                 \
} while(0)

// ============ Kernel 1: 衰减 + 接收 + 激活 ============
// 结果: network[idx] = v*decay + incoming, incoming[idx]=0, act[idx]=tanh
__global__ void decay_kernel(float* network, float* incoming, float* act)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= NET_X || y >= NET_Y || z >= NET_Z) return;

    int idx = z * NET_Y * NET_X + y * NET_X + x;
    float v = network[idx] * c_time_decay + incoming[idx];
    network[idx] = v;
    incoming[idx] = 0.0f;
    act[idx] = tanhf(v * 0.5f);
}

// ============ Kernel 2: 26-邻域 gather + 跳跃连接 gather ============
// 结果: incoming[idx] = Σ act[neighbor]*2*w + Σ act[skip_src]*2*skip_w
__global__ void gather_kernel(
    float*       incoming,
    const float* act,
    const float* prop_weight,
    const uint32_t* rev_skip_ptr,
    const uint32_t* rev_skip_src,
    const float* skip_weight)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= NET_X || y >= NET_Y || z >= NET_Z) return;

    int idx = z * NET_Y * NET_X + y * NET_X + x;

    // Phase 3: 26-邻域 gather
    float sum = 0.0f;
    #pragma unroll
    for (int n = 0; n < 26; ++n) {
        int nx = x + c_neighbors_26[n].dx;
        int ny = y + c_neighbors_26[n].dy;
        int nz = z + c_neighbors_26[n].dz;
        if (nx >= 0 && nx < NET_X &&
            ny >= 0 && ny < NET_Y &&
            nz >= 0 && nz < NET_Z) {
            int ni = nz * NET_Y * NET_X + ny * NET_X + nx;
            sum += act[ni] * 2.0f * prop_weight[ni];
        }
    }

    // Phase 4: 长程跳跃连接（反向 CSR gather）
    for (uint32_t e = rev_skip_ptr[idx]; e < rev_skip_ptr[idx + 1]; ++e) {
        uint32_t src = rev_skip_src[e];
        sum += act[src] * 2.0f * skip_weight[e];
    }

    incoming[idx] += sum;
}

// ============ 宿主 API ============

void gpu_init_field()
{
    size_t bytes = N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_network,      bytes));
    CUDA_CHECK(cudaMalloc(&d_incoming,     bytes));
    CUDA_CHECK(cudaMalloc(&d_act,          bytes));
    CUDA_CHECK(cudaMalloc(&d_prop_weight,  bytes));

    CUDA_CHECK(cudaMemcpyToSymbol(c_neighbors_26, g_neighbors_26,
                                  sizeof(g_neighbors_26)));

    // skip CSR 由 gpu_upload_skip_csr 分配
}

void gpu_free_field()
{
    cudaFree(d_network);      d_network      = nullptr;
    cudaFree(d_incoming);     d_incoming     = nullptr;
    cudaFree(d_act);          d_act          = nullptr;
    cudaFree(d_prop_weight);  d_prop_weight  = nullptr;
    cudaFree(d_rev_skip_ptr); d_rev_skip_ptr = nullptr;
    cudaFree(d_rev_skip_src); d_rev_skip_src = nullptr;
    cudaFree(d_skip_weight);  d_skip_weight  = nullptr;
}

void gpu_upload_prop_weights(const float* cpu_prop_weight)
{
    CUDA_CHECK(cudaMemcpy(d_prop_weight, cpu_prop_weight,
                          N * sizeof(float), cudaMemcpyHostToDevice));
}

void gpu_upload_skip_csr(const uint32_t* rev_skip_ptr,
                          const uint32_t* rev_skip_src,
                          const float*    skip_weight,
                          uint32_t        num_edges)
{
    cudaFree(d_rev_skip_ptr);
    cudaFree(d_rev_skip_src);
    cudaFree(d_skip_weight);

    CUDA_CHECK(cudaMalloc(&d_rev_skip_ptr, (N + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rev_skip_src, num_edges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_skip_weight,  num_edges * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rev_skip_ptr, rev_skip_ptr,
                          (N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rev_skip_src, rev_skip_src,
                          num_edges * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_skip_weight, skip_weight,
                          num_edges * sizeof(float), cudaMemcpyHostToDevice));
}

void gpu_propagate_step(float* host_network,
                         float* host_incoming,
                         float* host_act)
{
    if (host_network) {
        CUDA_CHECK(cudaMemcpy(d_network, host_network,
                              N * sizeof(float), cudaMemcpyHostToDevice));
    }
    if (host_incoming) {
        CUDA_CHECK(cudaMemcpy(d_incoming, host_incoming,
                              N * sizeof(float), cudaMemcpyHostToDevice));
    }

    dim3 block(8, 8, 8);   // 256 线程
    dim3 grid(10, 10, 10); // 1000 块, 共 256k 线程

    // Kernel 1: 衰减 + tanh（所有线程写完 act 后才进入 kernel 2）
    decay_kernel<<<grid, block>>>(d_network, d_incoming, d_act);
    CUDA_CHECK(cudaGetLastError());

    // Kernel 2: gather（kernel 1 的 act 已全部就绪）
    gather_kernel<<<grid, block>>>(
        d_incoming, d_act,
        d_prop_weight, d_rev_skip_ptr, d_rev_skip_src, d_skip_weight);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    if (host_network) {
        CUDA_CHECK(cudaMemcpy(host_network, d_network,
                              N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (host_incoming) {
        CUDA_CHECK(cudaMemcpy(host_incoming, d_incoming,
                              N * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (host_act) {
        CUDA_CHECK(cudaMemcpy(host_act, d_act,
                              N * sizeof(float), cudaMemcpyDeviceToHost));
    }
}
