#include "field.cuh"
#include "model.cuh"
#include "../core/config.h"
#include <cstdio>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <ctime>

// CPU: 分词器、数据加载、进度条、权重 save/load
#include "../../src/tokenizer/bpe.h"
#include "../../src/io/data.h"
#include "../../src/io/progress.h"
#include "../../src/io/checkpoint.h"

// ============ 纯 GPU 训练（零 PCIe 拷贝） ============
static float gpu_train_token_gpu(int input_id, int target_id)
{
    // 1. 注入 → d_field
    gpu_inject(d_field, d_W_embed + input_id * g_hidden_dim);

    // 2. 前向传播 20 步（使用 d_field + d_incoming）
    gpu_forward_propagate();

    // 3. 读出 h（从 d_field）
    gpu_readout_h(d_field, d_h);

    // 4. LM Head
    gpu_lm_head(d_h);

    // 5. Softmax + Loss
    float loss = gpu_softmax_loss(target_id);

    // 6. LM Head 反向 → d_h + d_W_embed_grad
    gpu_lm_head_backward(d_h);

    // 7. 读出反向 → d_network + d_W_out_grad + d_b_out_grad
    gpu_readout_backward(d_h);

    // 8. 清零 d_incoming（反向传播前提）
    gpu_zero_incoming();

    // 9. 反向传播 20 步（使用 d_network + d_incoming）
    gpu_backward_propagate();

    // 10. 注入反向（从 d_network 读梯度）
    gpu_inject_backward(d_W_embed + input_id * g_hidden_dim);

    return loss;
}

// ============ 主函数 ============
int main(int argc, char* argv[])
{
    printf("=== A-Network Full GPU Training ===\n");

    int dev = 0;
    cudaGetDeviceCount(&dev);
    if (!dev) { fprintf(stderr, "No CUDA device\n"); return 1; }
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s  (%d SMs)\n", p.name, p.multiProcessorCount);

    // ---- CPU 初始化 ----
    load_tokenizer(g_tokenizer_path);
    bool load_mode = (argc > 1 && std::string(argv[1]) == "load");
    if (load_mode) { load_weights(); }
    else { init_convert_weights(); init_propagation(); init_readout_weights(); }
    init_all();

    // ---- GPU 初始化 ----
    gpu_init_field();
    gpu_init_model();

    gpu_upload_weights(
        g_embed_weight.data(), g_in_weight.data(), g_in_bias.data(),
        g_out_weight.data(), g_out_bias.data());
    gpu_upload_prop_weights(g_prop_weight.data());

    uint32_t num_skip = g_skip_ptr.empty() ? 0 : (uint32_t)g_skip_ptr[N];
    if (num_skip > 0)
        gpu_upload_skip_csr(g_rev_skip_ptr.data(), g_rev_skip_src.data(),
                            g_skip_weight.data(), num_skip);

    // 动量清零
    size_t NH = N * g_hidden_dim, VH = g_vocab_size * g_hidden_dim;
    CUDA_CHECK(cudaMemset(d_m_embed, 0, VH * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_in_w,  0, NH * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_in_b,  0, N  * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_out_w, 0, NH * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_out_b, 0, N  * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_m_prop,  0, N  * sizeof(float)));
    if (num_skip > 0) {
        cudaFree(d_m_skip);
        CUDA_CHECK(cudaMalloc(&d_m_skip, num_skip * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_m_skip, 0, num_skip * sizeof(float)));
    }

    // ---- 数据 ----
    DataLoader data;
    data.load_dir(g_data_dir);
    if (data.tokens.empty()) return 1;
    auto& tokens = data.tokens;
    printf("Tokens: %zu, Skip edges: %u\n", tokens.size(), num_skip);

    // ---- 时间戳 ----
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
    ProgressBar pb;
    float lr = g_lr;

    for (int epoch = 0; epoch < g_max_epochs; ++epoch) {
        // 余弦退火
        {
            float p = (float)epoch / (g_max_epochs - 1);
            float factor = (1.0f + cos(3.14159265f * p)) * 0.5f;
            lr = g_lr_min + (g_lr - g_lr_min) * factor;
        }

        gpu_zero_field();
        gpu_zero_incoming();

        float total_loss = 0.0f;
        int total_steps = 0;
        auto ep_start = std::chrono::steady_clock::now();

        pb.start_epoch(epoch, g_max_epochs, (int)tokens.size() - 1);

        for (size_t t = 0; t + 1 < tokens.size(); ++t) {
            float loss = gpu_train_token_gpu((int)tokens[t], (int)tokens[t + 1]);
            total_loss += loss;
            ++total_steps;

            if (total_steps % g_grad_accum == 0) {
                gpu_optimizer_step(lr, g_mu);
                gpu_zero_all_gradients();
            }
            pb.step(loss);
        }

        if (total_steps % g_grad_accum != 0) {
            gpu_optimizer_step(lr, g_mu);
            gpu_zero_all_gradients();
        }

        float avg = total_loss / total_steps;
        pb.end_epoch(avg);
        float es = std::chrono::duration<float>(std::chrono::steady_clock::now() - ep_start).count();
        printf("  %.1fs  avg=%.4f  lr=%.6f\n", es, avg, lr);
        if (avg < g_min_loss) break;
    }

    // ---- 下载权重 + 保存 ----
    gpu_download_weights(g_embed_weight.data(), g_in_weight.data(), g_in_bias.data(),
                          g_out_weight.data(), g_out_bias.data());
    CUDA_CHECK(cudaMemcpy(g_prop_weight.data(), d_prop_weight,
                          N * sizeof(float), cudaMemcpyDeviceToHost));
    if (num_skip > 0)
        CUDA_CHECK(cudaMemcpy(g_skip_weight.data(), d_skip_weight,
                              num_skip * sizeof(float), cudaMemcpyDeviceToHost));

    save_weights();
    gpu_free_model();
    gpu_free_field();
    printf("Done. Weights: %s\n", g_weights_path.c_str());
    return 0;
}
