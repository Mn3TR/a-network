#include "field.cuh"
#include <cuda_runtime.h>
#include <cstdio>

// ============ 26-邻域 ============
const NeighborOffset g_neighbors_26[] = {
    {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
    {-1, -1, 0}, {-1, 1, 0}, {1, -1, 0}, {1, 1, 0},
    {-1, 0, -1}, {-1, 0, 1}, {1, 0, -1}, {1, 0, 1},
    {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1},
    {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
    {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1},
};
const int g_num_neighbors = 26;

// ============ 设备指针 ============
float*       d_network      = nullptr;
float*       d_incoming     = nullptr;
float*       d_act          = nullptr;
float*       d_prop_weight  = nullptr;
uint32_t*    d_rev_skip_ptr = nullptr;
uint32_t*    d_rev_skip_src = nullptr;
float*       d_skip_weight  = nullptr;

// ============ 常量内存 ============
__constant__ float    c_time_decay = 0.9f;
__constant__ int      c_net_x      = NET_X;
__constant__ int      c_net_y      = NET_Y;
__constant__ int      c_net_z      = NET_Z;
__constant__ NeighborOffset c_neighbors_26[26];

// ============ 块/片元配置 ============
// 每个 block 处理 8×8×8 = 512 个细胞
// 共享内存片元覆盖 (8+2)×(8+2)×(8+2) = 10×10×10 = 1000 细胞（含 halo）
constexpr int BW = 8, BH = 8, BD = 8;   // 计算块大小
constexpr int SW = BW + 2, SH = BH + 2, SD = BD + 2;  // 片元大小

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n",              \
                err, __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(1);                                                      \
    }                                                                 \
} while(0)

// ============ Kernel 1: 衰减 + 接收 + tanh（逐元素，无共享内存需要） ============
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
// 共享内存做 10×10×10 tile，缓存 act 和 prop_weight
// 仅内部 8×8×8 线程做计算，边界线程加载 halo
__global__ void gather_kernel(
    float*       incoming,
    const float* act,
    const float* prop_weight,
    const uint32_t* rev_skip_ptr,
    const uint32_t* rev_skip_src,
    const float* skip_weight)
{
    __shared__ float s_act[SD][SH][SW];     // 10×10×10 = 4KB
    __shared__ float s_w[SD][SH][SW];       // 同上 = 4KB (共 8KB)

    int tx = threadIdx.x;  // 0..SW-1
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    int gx = blockIdx.x * BW + tx - 1;  // 全局 x, 可能 -1 或 NET_X（越界 halo）
    int gy = blockIdx.y * BH + ty - 1;
    int gz = blockIdx.z * BD + tz - 1;

    // 加载片元（含 halo，越界填 0）
    float a = 0.0f, w = 0.0f;
    if (gx >= 0 && gx < NET_X && gy >= 0 && gy < NET_Y && gz >= 0 && gz < NET_Z) {
        int idx = gz * NET_Y * NET_X + gy * NET_X + gx;
        a = act[idx];
        w = prop_weight[idx];
    }
    s_act[tz][ty][tx] = a;
    s_w[tz][ty][tx]   = w;

    __syncthreads();

    // 仅内部线程（1..8）做计算
    if (tx >= 1 && tx <= BW && ty >= 1 && ty <= BH && tz >= 1 && tz <= BD) {
        int gx2 = blockIdx.x * BW + (tx - 1);
        int gy2 = blockIdx.y * BH + (ty - 1);
        int gz2 = blockIdx.z * BD + (tz - 1);
        int idx = gz2 * NET_Y * NET_X + gy2 * NET_X + gx2;

        // 26-邻域：全部从共享内存读（分支无关，越界 halo 为 0）
        float sum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 26; ++n) {
            int nx = tx + c_neighbors_26[n].dx;  // 始终在 [0, SW-1]
            int ny = ty + c_neighbors_26[n].dy;
            int nz = tz + c_neighbors_26[n].dz;
            sum += s_act[nz][ny][nx] * 2.0f * s_w[nz][ny][nx];
        }

        // 跳跃连接：不可避免的全局随机读，用 __ldg 走只读缓存
        for (uint32_t e = rev_skip_ptr[idx]; e < rev_skip_ptr[idx + 1]; ++e) {
            uint32_t src = rev_skip_src[e];
            sum += __ldg(&act[src]) * 2.0f * __ldg(&skip_weight[e]);
        }

        incoming[idx] += sum;
    }
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

void gpu_upload_network(const float* cpu_network)
{
    CUDA_CHECK(cudaMemcpy(d_network, cpu_network,
                          N * sizeof(float), cudaMemcpyHostToDevice));
}

void gpu_download_network(float* cpu_network)
{
    CUDA_CHECK(cudaMemcpy(cpu_network, d_network,
                          N * sizeof(float), cudaMemcpyDeviceToHost));
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

// GPU-only 传播（无 PCIe 拷贝，数据始终驻留设备）
void gpu_propagate_step_device()
{
    dim3 block(8, 8, 8);
    dim3 grid(10, 10, 10);

    decay_kernel<<<grid, block>>>(d_network, d_incoming, d_act);
    CUDA_CHECK(cudaGetLastError());

    gather_kernel<<<grid, block>>>(
        d_incoming, d_act,
        d_prop_weight, d_rev_skip_ptr, d_rev_skip_src, d_skip_weight);
    CUDA_CHECK(cudaGetLastError());
}

// 带同步的传播（等待 GPU 完成后返回）
void gpu_propagate_step_sync()
{
    gpu_propagate_step_device();
    CUDA_CHECK(cudaDeviceSynchronize());
}
