#include "../a-network/field.cuh"
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>

// ============ CPU 参考实现（验证用） ============
// 直接从 CPU 版简化：手动展开 26 邻域和 skip CSR

static float cpu_time_decay = 0.9f;

static void cpu_propagate_step(
    float* network, float* incoming,
    const float* prop_weight,
    const uint32_t* rev_skip_ptr,
    const uint32_t* rev_skip_src,
    const float* skip_weight)
{
    // Phase 1+2: 衰减 + 接收 + 激活
    for (int idx = 0; idx < N; ++idx) {
        float v = network[idx] * cpu_time_decay + incoming[idx];
        network[idx] = v;
        incoming[idx] = 0.0f;
    }

    // 预计算 tanh 激活值
    std::vector<float> act(N);
    for (int i = 0; i < N; ++i)
        act[i] = tanhf(network[i] * 0.5f);

    // Phase 3+4: gather
    for (int idx = 0; idx < N; ++idx) {
        int x = idx / (NET_Y * NET_Z);
        int yz = idx % (NET_Y * NET_Z);
        int y = yz / NET_Z;
        int z = yz % NET_Z;

        float sum = 0.0f;
        for (int n = 0; n < 26; ++n) {
            int nx = x + g_neighbors_26[n].dx;
            int ny = y + g_neighbors_26[n].dy;
            int nz = z + g_neighbors_26[n].dz;
            if (nx >= 0 && nx < NET_X &&
                ny >= 0 && ny < NET_Y &&
                nz >= 0 && nz < NET_Z) {
                int ni = nz * NET_Y * NET_X + ny * NET_X + nx;
                sum += act[ni] * 2.0f * prop_weight[ni];
            }
        }

        // 跳跃连接
        for (uint32_t e = rev_skip_ptr[idx]; e < rev_skip_ptr[idx + 1]; ++e) {
            uint32_t src = rev_skip_src[e];
            sum += act[src] * 2.0f * skip_weight[e];
        }
        incoming[idx] += sum;
    }
}

int main()
{
    printf("=== A-Network GPU Test ===\n");
    printf("Grid: %dx%dx%d = %d cells\n", NET_X, NET_Y, NET_Z, N);

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "No CUDA device found!\n");
        return 1;
    }
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);

    // 分配宿主内存
    std::vector<float> h_network(N, 0.0f);
    std::vector<float> h_incoming(N, 0.0f);
    std::vector<float> h_prop_weight(N);
    std::vector<float> h_act_gpu(N);   // GPU 版激活暂存
    std::vector<float> h_act_cpu(N);

    // 随机初始化场和传播权重
    srand(42);
    for (int i = 0; i < N; ++i) {
        h_network[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_prop_weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }

    // 跳过连接（模拟少量随机边）
    // 注意：CPU 版有完整随机拓扑，这里造一个简版
    uint32_t num_edges = 1024;
    std::vector<uint32_t> h_rev_skip_ptr(N + 1, 0);
    std::vector<uint32_t> h_rev_skip_src(num_edges);
    std::vector<float> h_skip_weight(num_edges);

    for (uint32_t e = 0; e < num_edges; ++e) {
        uint32_t dst = (uint32_t)((float)rand() / RAND_MAX * N);
        uint32_t src = (uint32_t)((float)rand() / RAND_MAX * N);
        h_rev_skip_src[e] = src;
        h_skip_weight[e] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
        h_rev_skip_ptr[dst + 1]++;
    }
    // 前缀和
    for (uint32_t i = 0; i < N; ++i)
        h_rev_skip_ptr[i + 1] += h_rev_skip_ptr[i];

    printf("Skip edges: %u\n", num_edges);

    // 复制一份给 CPU
    std::vector<float> cpu_network = h_network;
    std::vector<float> cpu_incoming(N, 0.0f);

    // ====== GPU 初始化 ======
    gpu_init_field();
    gpu_upload_prop_weights(h_prop_weight.data());
    gpu_upload_skip_csr(h_rev_skip_ptr.data(), h_rev_skip_src.data(),
                        h_skip_weight.data(), num_edges);

    // ====== 预热 ======
    gpu_propagate_step(h_network.data(), h_incoming.data(), h_act_gpu.data());
    printf("Warmup done\n");

    // ====== GPU benchmark ======
    int steps = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        gpu_propagate_step(h_network.data(), h_incoming.data(), h_act_gpu.data());
    }
    auto t1 = std::chrono::steady_clock::now();
    float gpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    printf("GPU: %d steps in %.1f ms  (%.1f ms/step)\n",
           steps, gpu_ms, gpu_ms / steps);

    // ====== CPU benchmark ======
    t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) {
        cpu_propagate_step(cpu_network.data(), cpu_incoming.data(),
                           h_prop_weight.data(),
                           h_rev_skip_ptr.data(), h_rev_skip_src.data(),
                           h_skip_weight.data());
    }
    t1 = std::chrono::steady_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    printf("CPU: %d steps in %.1f ms  (%.1f ms/step)\n",
           steps, cpu_ms, cpu_ms / steps);

    printf("Speedup: %.1fx\n", cpu_ms / gpu_ms);

    // ====== 正确性检查 ======
    double max_diff = 0.0, sum_diff = 0.0;
    for (int i = 0; i < N; ++i) {
        double diff = fabs((double)h_network[i] - (double)cpu_network[i]);
        if (diff > max_diff) max_diff = diff;
        sum_diff += diff;
    }
    printf("Max difference: %e  Mean difference: %e\n",
           max_diff, sum_diff / N);

    // 清理
    gpu_free_field();
    return 0;
}
