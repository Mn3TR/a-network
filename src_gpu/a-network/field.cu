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
float*       d_field        = nullptr;  // 场状态（前向用）
float*       d_network      = nullptr;  // 梯度缓冲（反向用）
float*       d_incoming     = nullptr;
float*       d_act          = nullptr;
float*       d_prop_weight  = nullptr;
float*       d_prop_grad    = nullptr;
uint32_t*    d_rev_skip_ptr = nullptr;
uint32_t*    d_rev_skip_src = nullptr;
float*       d_skip_weight  = nullptr;
float*       d_skip_grad    = nullptr;

// 激活值快照 [g_prop_steps][N]
float**      d_act_buf      = nullptr;

// 跳跃连接边数（给梯度清零用）
uint32_t     gpu_num_edges  = 0;

// ============ 常量内存 ============
__constant__ float    c_time_decay = 0.9f;
__constant__ int      c_num_neighbors = 26;
__constant__ NeighborOffset c_neighbors_26[26];

// ============ 块/片元配置 ============
constexpr int BW = 8, BH = 8, BD = 8;
constexpr int SW = BW + 2, SH = BH + 2, SD = BD + 2;

#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n",              \
                err, __FILE__, __LINE__, cudaGetErrorString(err));    \
        exit(1);                                                      \
    }                                                                 \
} while(0)

// ============ Kernel 1: 衰减 + 接收 + tanh ============
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

// ============ Kernel 2: 前向 gather（共享内存 tile） ============
__global__ void gather_kernel(
    float*       incoming,
    const float* act,
    const float* prop_weight,
    const uint32_t* rev_skip_ptr,
    const uint32_t* rev_skip_src,
    const float* skip_weight)
{
    __shared__ float s_act[SD][SH][SW];
    __shared__ float s_w[SD][SH][SW];

    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int gx = blockIdx.x * BW + tx - 1;
    int gy = blockIdx.y * BH + ty - 1;
    int gz = blockIdx.z * BD + tz - 1;

    float a = 0.0f, w = 0.0f;
    if (gx >= 0 && gx < NET_X && gy >= 0 && gy < NET_Y && gz >= 0 && gz < NET_Z) {
        int idx = gz * NET_Y * NET_X + gy * NET_X + gx;
        a = act[idx];
        w = prop_weight[idx];
    }
    s_act[tz][ty][tx] = a;
    s_w[tz][ty][tx]   = w;
    __syncthreads();

    if (tx >= 1 && tx <= BW && ty >= 1 && ty <= BH && tz >= 1 && tz <= BD) {
        int gx2 = blockIdx.x * BW + (tx - 1);
        int gy2 = blockIdx.y * BH + (ty - 1);
        int gz2 = blockIdx.z * BD + (tz - 1);
        int idx = gz2 * NET_Y * NET_X + gy2 * NET_X + gx2;

        float sum = 0.0f;
        #pragma unroll
        for (int n = 0; n < 26; ++n) {
            int nx = tx + c_neighbors_26[n].dx;
            int ny = ty + c_neighbors_26[n].dy;
            int nz = tz + c_neighbors_26[n].dz;
            sum += s_act[nz][ny][nx] * 2.0f * s_w[nz][ny][nx];
        }
        for (uint32_t e = rev_skip_ptr[idx]; e < rev_skip_ptr[idx + 1]; ++e)
            sum += __ldg(&act[rev_skip_src[e]]) * 2.0f * __ldg(&skip_weight[e]);
        incoming[idx] += sum;
    }
}

// ============ Kernel 3: 反向 gather（共享内存 tile） ============
// 与前向对称，读 d_incoming[neighbor] 而非 act[neighbor]
__global__ void backward_gather_kernel(
    float*       d_network,
    float*       d_incoming,
    const float* phase2_act,
    const float* prop_weight,
    float*       prop_grad,
    const uint32_t* rev_skip_ptr,
    const uint32_t* rev_skip_src,
    const float* skip_weight,
    float*       skip_grad)
{
    __shared__ float s_dinc[SD][SH][SW];
    __shared__ float s_w[SD][SH][SW];

    int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
    int gx = blockIdx.x * BW + tx - 1;
    int gy = blockIdx.y * BH + ty - 1;
    int gz = blockIdx.z * BD + tz - 1;

    float di = 0.0f, w = 0.0f;
    if (gx >= 0 && gx < NET_X && gy >= 0 && gy < NET_Y && gz >= 0 && gz < NET_Z) {
        int idx = gz * NET_Y * NET_X + gy * NET_X + gx;
        di = d_incoming[idx];
        w = prop_weight[idx];
    }
    s_dinc[tz][ty][tx] = di;
    s_w[tz][ty][tx]    = w;
    __syncthreads();

    if (tx >= 1 && tx <= BW && ty >= 1 && ty <= BH && tz >= 1 && tz <= BD) {
        int gx2 = blockIdx.x * BW + (tx - 1);
        int gy2 = blockIdx.y * BH + (ty - 1);
        int gz2 = blockIdx.z * BD + (tz - 1);
        int idx = gz2 * NET_Y * NET_X + gy2 * NET_X + gx2;

        // 从 26 邻域 gather d_incoming
        float grad_local = 0.0f;
        #pragma unroll
        for (int n = 0; n < 26; ++n) {
            int nx = tx + c_neighbors_26[n].dx;
            int ny = ty + c_neighbors_26[n].dy;
            int nz = tz + c_neighbors_26[n].dz;
            grad_local += s_dinc[nz][ny][nx];  // d_incoming[neighbor]
        }

        // 从跳跃连接 gather d_incoming
        float grad_skip = 0.0f;
        for (uint32_t e = rev_skip_ptr[idx]; e < rev_skip_ptr[idx + 1]; ++e)
            grad_skip += __ldg(&d_incoming[rev_skip_src[e]]) * __ldg(&skip_weight[e]);

        float tanh_a = phase2_act[idx];
        float dtanh  = 1.0f - tanh_a * tanh_a;
        float act_a  = tanh_a * 2.0f;

        // 当前 d_network 已有反向传回的梯度，累加本地梯度
        d_network[idx] += (grad_local * prop_weight[idx] + grad_skip) * dtanh;
        prop_grad[idx] += grad_local * act_a;

        for (uint32_t e = rev_skip_ptr[idx]; e < rev_skip_ptr[idx + 1]; ++e)
            skip_grad[e] += __ldg(&d_incoming[rev_skip_src[e]]) * act_a;
    }

    __syncthreads();

    // 将 d_network 传回 d_incoming 并衰减
    if (tx >= 1 && tx <= BW && ty >= 1 && ty <= BH && tz >= 1 && tz <= BD) {
        int gx2 = blockIdx.x * BW + (tx - 1);
        int gy2 = blockIdx.y * BH + (ty - 1);
        int gz2 = blockIdx.z * BD + (tz - 1);
        int idx = gz2 * NET_Y * NET_X + gy2 * NET_X + gx2;

        float total = d_network[idx];
        d_incoming[idx] = total;
        d_network[idx]  = total * c_time_decay;
    }
}

// ============ 宿主 API ============

void gpu_init_field()
{
    size_t bytes = N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_field,       bytes));
    CUDA_CHECK(cudaMalloc(&d_network,     bytes));
    CUDA_CHECK(cudaMalloc(&d_incoming,    bytes));
    CUDA_CHECK(cudaMalloc(&d_act,          bytes));
    CUDA_CHECK(cudaMalloc(&d_prop_weight,  bytes));
    CUDA_CHECK(cudaMalloc(&d_prop_grad,    bytes));

    // 激活快照: g_prop_steps 个 N-vector
    CUDA_CHECK(cudaMalloc(&d_act_buf, g_prop_steps * sizeof(float*)));
    for (int s = 0; s < g_prop_steps; ++s) {
        float* slot;
        CUDA_CHECK(cudaMalloc(&slot, bytes));
        CUDA_CHECK(cudaMemcpy(d_act_buf + s, &slot, sizeof(float*), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_neighbors_26, g_neighbors_26, sizeof(g_neighbors_26)));
}

void gpu_free_field()
{
    // 释放快照
    if (d_act_buf) {
        for (int s = 0; s < g_prop_steps; ++s) {
            float* slot;
            CUDA_CHECK(cudaMemcpy(&slot, d_act_buf + s, sizeof(float*), cudaMemcpyDeviceToHost));
            cudaFree(slot);
        }
        cudaFree(d_act_buf);
    }
    cudaFree(d_field);        d_field        = nullptr;
    cudaFree(d_network);      d_network      = nullptr;
    cudaFree(d_incoming);     d_incoming     = nullptr;
    cudaFree(d_act);          d_act          = nullptr;
    cudaFree(d_prop_weight);  d_prop_weight  = nullptr;
    cudaFree(d_prop_grad);    d_prop_grad    = nullptr;
    cudaFree(d_rev_skip_ptr); d_rev_skip_ptr = nullptr;
    cudaFree(d_rev_skip_src); d_rev_skip_src = nullptr;
    cudaFree(d_skip_weight);  d_skip_weight  = nullptr;
    cudaFree(d_skip_grad);    d_skip_grad    = nullptr;
}

void gpu_upload_network(const float* cpu_network)
{
    CUDA_CHECK(cudaMemcpy(d_network, cpu_network, N * sizeof(float), cudaMemcpyHostToDevice));
}

void gpu_download_network(float* cpu_network)
{
    CUDA_CHECK(cudaMemcpy(cpu_network, d_network, N * sizeof(float), cudaMemcpyDeviceToHost));
}

void gpu_download_incoming(float* cpu_incoming)
{
    CUDA_CHECK(cudaMemcpy(cpu_incoming, d_incoming, N * sizeof(float), cudaMemcpyDeviceToHost));
}

void gpu_download_gradients(float* cpu_prop_grad, float* cpu_skip_grad, uint32_t num_edges)
{
    CUDA_CHECK(cudaMemcpy(cpu_prop_grad, d_prop_grad, N * sizeof(float), cudaMemcpyDeviceToHost));
    if (cpu_skip_grad && num_edges > 0)
        CUDA_CHECK(cudaMemcpy(cpu_skip_grad, d_skip_grad, num_edges * sizeof(float), cudaMemcpyDeviceToHost));
}

void gpu_upload_prop_weights(const float* cpu_prop_weight)
{
    CUDA_CHECK(cudaMemcpy(d_prop_weight, cpu_prop_weight, N * sizeof(float), cudaMemcpyHostToDevice));
}

void gpu_upload_incoming(const float* cpu_incoming)
{
    CUDA_CHECK(cudaMemcpy(d_incoming, cpu_incoming, N * sizeof(float), cudaMemcpyHostToDevice));
}

void gpu_upload_skip_csr(const uint32_t* rev_skip_ptr,
                          const uint32_t* rev_skip_src,
                          const float*    skip_weight,
                          uint32_t        num_edges)
{
    cudaFree(d_rev_skip_ptr); cudaFree(d_rev_skip_src);
    cudaFree(d_skip_weight);  cudaFree(d_skip_grad);

    CUDA_CHECK(cudaMalloc(&d_rev_skip_ptr, (N + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_rev_skip_src, num_edges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_skip_weight,  num_edges * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_skip_grad,    num_edges * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_rev_skip_ptr, rev_skip_ptr, (N + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rev_skip_src, rev_skip_src, num_edges * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_skip_weight,  skip_weight,  num_edges * sizeof(float),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_skip_grad, 0, num_edges * sizeof(float)));
    gpu_num_edges = num_edges;
}

void gpu_clear_gradients()
{
    CUDA_CHECK(cudaMemset(d_prop_grad, 0, N * sizeof(float)));
    if (d_skip_grad && gpu_num_edges > 0)
        CUDA_CHECK(cudaMemset(d_skip_grad, 0, gpu_num_edges * sizeof(float)));
}

void gpu_zero_field()
{
    CUDA_CHECK(cudaMemset(d_field, 0, N * sizeof(float)));
}

void gpu_zero_network()
{
    CUDA_CHECK(cudaMemset(d_network, 0, N * sizeof(float)));
}

void gpu_zero_incoming()
{
    CUDA_CHECK(cudaMemset(d_incoming, 0, N * sizeof(float)));
}

// ============ 前向传播 N 步，存 act 快照 ============
void gpu_forward_propagate()
{
    dim3 block(8, 8, 8);
    dim3 grid(10, 10, 10);

    for (int s = 0; s < g_prop_steps; ++s) {
        // 取当前步的 act 快照地址
        float* slot;
        cudaMemcpy(&slot, d_act_buf + s, sizeof(float*), cudaMemcpyDeviceToHost);

        decay_kernel<<<grid, block>>>(d_field, d_incoming, slot);
        CUDA_CHECK(cudaGetLastError());

        gather_kernel<<<grid, block>>>(
            d_incoming, slot, d_prop_weight,
            d_rev_skip_ptr, d_rev_skip_src, d_skip_weight);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============ 反向传播 N 步 ============
void gpu_backward_propagate()
{
    dim3 block(8, 8, 8);
    dim3 grid(10, 10, 10);

    for (int s = g_prop_steps - 1; s >= 0; --s) {
        float* slot;
        cudaMemcpy(&slot, d_act_buf + s, sizeof(float*), cudaMemcpyDeviceToHost);

        backward_gather_kernel<<<grid, block>>>(
            d_network, d_incoming, slot,
            d_prop_weight, d_prop_grad,
            d_rev_skip_ptr, d_rev_skip_src, d_skip_weight, d_skip_grad);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}
