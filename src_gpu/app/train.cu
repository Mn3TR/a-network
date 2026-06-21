#include "field.cuh"
#include "../core/config.h"
#include <cstdio>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <ctime>

// CPU 组件
#include "../../src/core/types.h"
#include "../../src/core/config.h"
#include "../../src/a-network/convert.h"
#include "../../src/a-network/field.h"
#include "../../src/a-network/model.h"
#include "../../src/a-network/readout.h"
#include "../../src/a-network/backward.h"
#include "../../src/tokenizer/bpe.h"
#include "../../src/train/optimizer.h"
#include "../../src/io/data.h"
#include "../../src/io/progress.h"
#include "../../src/io/checkpoint.h"

// ============ GPU 加速的 train_token ============
// CPU: 注入 → 上传 → GPU: 前向 20 步 → 下载 → CPU: 读出+Loss
// CPU: 反向读出 → 上传 → GPU: 反向 20 步 → 下载 → CPU: 反向注入
static float gpu_train_token(size_t input_id, size_t target_id,
                              float* flat_net, float* incoming_buf)
{
    size_t H = g_hidden_dim;

    // ---- 1. CPU 注入 ----
    const float* emb = g_embed_weight.data() + input_id * H;
    tokens_to_field(emb, 1, flat_net);

    // ---- 2. 上传 → GPU 前向传播 ----
    CUDA_CHECK(cudaMemcpy(d_network,  flat_net,    N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_incoming, incoming_buf, N * sizeof(float), cudaMemcpyHostToDevice));
    gpu_forward_propagate();
    CUDA_CHECK(cudaMemcpy(flat_net, d_network, N * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 3. CPU 读出 ----
    for (size_t n = 0; n < N; ++n) g_diff[n] = flat_net[n] - g_out_bias[n];
    std::vector<float> h_cur(H);
    for (size_t k = 0; k < H; ++k) {
        float sum = 0.0f;
        for (size_t n = 0; n < N; ++n) sum += g_out_weight[k * N + n] * g_diff[n];
        h_cur[k] = sum;
    }

    // ---- 4. CPU LM Head + Loss ----
    std::vector<float> logits(g_vocab_size);
    std::vector<float> d_logits(g_vocab_size);
    for (size_t t = 0; t < g_vocab_size; ++t) {
        float sum = 0.0f;
        for (size_t k = 0; k < H; ++k) sum += g_embed_weight[t * H + k] * h_cur[k];
        logits[t] = sum;
    }
    float loss = cross_entropy_loss(logits.data(), target_id, d_logits.data());

    // ---- 5. CPU 反向读出 ----
    std::vector<float> d_h(H, 0.0f);
    for (size_t k = 0; k < H; ++k)
        for (size_t t = 0; t < g_vocab_size; ++t)
            d_h[k] += d_logits[t] * g_embed_weight[t * H + k];

    // embed grad
    for (size_t t = 0; t < g_vocab_size; ++t) {
        float dv = d_logits[t];
        if (dv == 0.0f) continue;
        for (size_t k = 0; k < H; ++k)
            g_embed_grad[t * H + k] += dv * h_cur[k];
    }

    // W_out, b_out grad
    static std::vector<float> d_net(N);
    for (size_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (size_t k = 0; k < H; ++k) sum += d_h[k] * g_out_weight[k * N + n];
        d_net[n] = sum;
    }
    for (size_t k = 0; k < H; ++k) {
        float dhk = d_h[k];
        if (dhk == 0.0f) continue;
        for (size_t n = 0; n < N; ++n)
            g_out_weight_grad[k * N + n] += dhk * g_diff[n];
    }
    for (size_t n = 0; n < N; ++n) g_out_bias_grad[n] -= d_net[n];

    // ---- 6. 上传 → GPU 反向传播 ----
    CUDA_CHECK(cudaMemcpy(d_network,  d_net.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    gpu_backward_propagate();
    CUDA_CHECK(cudaMemcpy(d_net.data(), d_network, N * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- 7. CPU 反向注入 ----
    static std::vector<float> d_emb;
    d_emb.assign(H, 0.0f);
    backward_inject(d_net.data(), 1, emb, d_emb.data());
    for (size_t k = 0; k < H; ++k)
        g_embed_grad[input_id * H + k] += d_emb[k];

    return loss;
}

// ============ 主函数 ============
int main(int argc, char* argv[])
{
    printf("=== A-Network GPU Training ===\n");

    int dev = 0;
    cudaGetDeviceCount(&dev);
    if (!dev) { fprintf(stderr, "No CUDA device\n"); return 1; }
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s  (%d SMs, %.1f GB)\n", p.name, p.multiProcessorCount, p.totalGlobalMem / 1e9f);

    // ---- 初始化 ----
    load_tokenizer(g_tokenizer_path);
    bool load_mode = (argc > 1 && std::string(argv[1]) == "load");
    if (load_mode) { load_weights(); }
    else { init_convert_weights(); init_propagation(); init_readout_weights(); }
    init_all();
    g_optim.init();

    // GPU
    gpu_init_field();
    gpu_upload_prop_weights(g_prop_weight.data());
    size_t num_skip = g_skip_ptr.empty() ? 0 : g_skip_ptr[N];
    if (num_skip > 0)
        gpu_upload_skip_csr(g_rev_skip_ptr.data(), g_rev_skip_src.data(),
                            g_skip_weight.data(), (uint32_t)num_skip);

    // 数据
    DataLoader data;
    data.load_dir(g_data_dir);
    if (data.tokens.empty()) return 1;
    auto& tokens = data.tokens;
    printf("Tokens: %zu, Skip edges: %zu\n", tokens.size(), num_skip);

    // 时间戳
    {
        auto now = std::chrono::system_clock::now();
        time_t tt = std::chrono::system_clock::to_time_t(now);
        std::tm tm; localtime_r(&tt, &tm);
        char buf[32]; strftime(buf, sizeof(buf), "%Y-%m-%d-%H%M", &tm);
        std::string stamp(buf), od = "output_gpu/output-" + stamp + "/";
        g_weights_path = od + "weights.bin";
        system(("mkdir -p " + od).c_str());
        printf("Output: %s\n", od.c_str());
    }

    // ---- 训练 ----
    std::vector<float> buffer(N, 0.0f), incoming(N, 0.0f);
    ProgressBar pb;

    for (int epoch = 0; epoch < g_max_epochs; ++epoch) {
        lr_schedule(epoch, g_max_epochs);
        std::fill(buffer.begin(), buffer.end(), 0.0f);
        std::fill(incoming.begin(), incoming.end(), 0.0f);

        float total_loss = 0.0f;
        int total_steps = 0;
        auto ep_start = std::chrono::steady_clock::now();

        pb.start_epoch(epoch, g_max_epochs, (int)tokens.size() - 1);

        for (size_t t = 0; t + 1 < tokens.size(); ++t) {
            float loss = gpu_train_token(tokens[t], tokens[t+1],
                                          buffer.data(), incoming.data());
            total_loss += loss;
            ++total_steps;

            if (total_steps % g_grad_accum == 0) {
                gpu_download_gradients(g_prop_grad.data(), g_skip_grad.data(), (uint32_t)num_skip);
                g_optim.step();
                g_optim.zero_grad();
                gpu_clear_gradients();
            }
            pb.step(loss);
        }

        if (total_steps % g_grad_accum != 0) {
            gpu_download_gradients(g_prop_grad.data(), g_skip_grad.data(), (uint32_t)num_skip);
            g_optim.step();
            g_optim.zero_grad();
            gpu_clear_gradients();
        }

        float avg = total_loss / total_steps;
        pb.end_epoch(avg);
        float es = std::chrono::duration<float>(std::chrono::steady_clock::now() - ep_start).count();
        printf("  %.1fs  avg=%.4f  lr=%.6f\n", es, avg, g_optim.lr);

        if (avg < g_min_loss) break;
    }

    save_weights();
    gpu_free_field();
    printf("Done. Weights: %s\n", g_weights_path.c_str());
    return 0;
}
