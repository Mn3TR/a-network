#include "../a-network/field.cuh"
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>

// ============ CPU 参考实现 ============
static float cpu_time_decay = 0.9f;

static void cpu_propagate_step(
    float* network, float* incoming,
    const float* prop_weight,
    const uint32_t* rev_skip_ptr,
    const uint32_t* rev_skip_src,
    const float* skip_weight)
{
    for (int idx = 0; idx < N; ++idx) {
        float v = network[idx] * cpu_time_decay + incoming[idx];
        network[idx] = v;
        incoming[idx] = 0.0f;
    }

    std::vector<float> act(N);
    for (int i = 0; i < N; ++i) act[i] = tanhf(network[i] * 0.5f);

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
        for (uint32_t e = rev_skip_ptr[idx]; e < rev_skip_ptr[idx + 1]; ++e) {
            uint32_t src = rev_skip_src[e];
            sum += act[src] * 2.0f * skip_weight[e];
        }
        incoming[idx] += sum;
    }
}

int main()
{
    printf("=== A-Network GPU Benchmark ===\n");
    printf("Grid: %dx%dx%d = %d cells\n", NET_X, NET_Y, NET_Z, N);

    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) { fprintf(stderr, "No CUDA device!\n"); return 1; }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s  (SM %d.%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // ====== 宿主数据 ======
    std::vector<float> h_network(N, 0.0f);
    std::vector<float> h_incoming(N, 0.0f);
    std::vector<float> h_prop_weight(N);

    srand(42);
    for (int i = 0; i < N; ++i) {
        h_network[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
        h_prop_weight[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.01f;
    }

    // 随机跳跃连接（1024 条）
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
    for (uint32_t i = 0; i < N; ++i)
        h_rev_skip_ptr[i + 1] += h_rev_skip_ptr[i];
    printf("Skip edges: %u\n", num_edges);

    std::vector<float> cpu_net = h_network;
    std::vector<float> cpu_inc(N, 0.0f);

    // ====== GPU 初始化 ======
    gpu_init_field();
    gpu_upload_prop_weights(h_prop_weight.data());
    gpu_upload_skip_csr(h_rev_skip_ptr.data(), h_rev_skip_src.data(),
                        h_skip_weight.data(), num_edges);
    gpu_upload_network(h_network.data());

    // ====== GPU 预热 ======
    gpu_propagate_step_sync();
    printf("Warmup done\n");

    // ====== GPU 基准（纯设备端，无 PCIe 开销） ======
    int steps = 20;
    auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s)
        gpu_propagate_step_device();          // 异步 launch（不阻塞）
    CUDA_CHECK(cudaDeviceSynchronize());       // 等全部完成
    auto t1 = std::chrono::steady_clock::now();

    float gpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    printf("GPU: %d steps in %.2f ms  (%.3f ms/step)\n",
           steps, gpu_ms, gpu_ms / steps);

    // ====== CPU 基准 ======
    t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s)
        cpu_propagate_step(cpu_net.data(), cpu_inc.data(),
                           h_prop_weight.data(),
                           h_rev_skip_ptr.data(), h_rev_skip_src.data(),
                           h_skip_weight.data());
    t1 = std::chrono::steady_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    printf("CPU: %d steps in %.2f ms  (%.3f ms/step)\n",
           steps, cpu_ms, cpu_ms / steps);

    printf("Speedup: %.1fx\n", cpu_ms / gpu_ms);

    // ====== 正确性检查 ======
    gpu_download_network(h_network.data());

    double max_diff = 0.0, sum_diff = 0.0;
    for (int i = 0; i < N; ++i) {
        double d = fabs((double)h_network[i] - (double)cpu_net[i]);
        if (d > max_diff) max_diff = d;
        sum_diff += d;
    }
    printf("Max diff: %e  Mean diff: %e\n", max_diff, sum_diff / N);

    gpu_free_field();
    return 0;
}
