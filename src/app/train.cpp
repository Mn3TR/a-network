#include "core/types.h"
#include "core/config.h"
#include "a-network/convert.h"
#include "a-network/field.h"
#include "a-network/model.h"
#include "a-network/readout.h"
#include "tokenizer/bpe.h"
#include "train/optimizer.h"
#include "io/data.h"
#include "io/progress.h"
#include "io/checkpoint.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <vector>
#include <cstring>
#include <chrono>

// ============ 生成模式 ============
// 注入 → 传播 → 读出 h → LM Head → argmax

static std::string token_str(size_t id)
{
    if (id >= g_id_to_token.size()) return "<?>";
    std::string s = g_id_to_token[id];
    for (auto& c : s) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (uc < 32 && uc != '\n') c = '.';
    }
    return s;
}

static int run_generate(int argc, char* argv[])
{
    load_weights();                       // 恢复所有权重（含 skip 拓扑/坐标表/反向CSR）
    load_tokenizer(g_tokenizer_path);
    init_all();

    std::string seed = (argc > 2) ? argv[2] : "Time";
    auto seed_tokens = sentencepiece_to_tokens(seed);
    std::cout << "Seed: \"" << seed << "\"" << std::endl;

    std::vector<size_t> generated = seed_tokens;
    std::vector<float> net(N, 0.0f);
    std::vector<float> incoming(N, 0.0f);

    // 用 seed tokens 构建场状态 + h-window
    for (size_t t = 0; t < seed_tokens.size(); ++t) {
        // 前向注入 + 传播 + 读出 h + 推入窗口（忽略此时的预测）
        generate_token(seed_tokens[t], net.data(), incoming.data());
    }

    // 自回归生成
    for (int i = 0; i < 100; ++i) {
        size_t pred = generate_token(generated.back(), net.data(), incoming.data());

        if (pred == 1) {
            std::cout << "[EOS]";
            break;
        }

        generated.push_back(pred);

        std::string s = token_str(pred);
        if (pred >= 3 && s.find("<|") == std::string::npos)
            std::cout << s;
    }
    std::cout << "\"" << std::endl;

    std::string full = tokens_to_sentence(generated);
    std::cout << "\nFull text:\n" << full << std::endl;

    return 0;
}

// ============ 训练模式 ============

static int run_train(int argc, char* argv[])
{
    std::vector<float> buffer(N, 0.0f);
    std::vector<float> incoming(N, 0.0f);

    bool load_mode = (argc > 1 && std::string(argv[1]) == "load");

    if (load_mode) {
        load_weights();                   // 恢复权重 + 重建拓扑结构
        std::cout << "Loaded weights from " << g_weights_path << std::endl;
    } else {
        init_convert_weights();           // 仅新训练随机化权重
        init_propagation();               // 工作区 + 随机 skip 拓扑（内部 once 守卫）
        init_readout_weights();
    }
    load_tokenizer(g_tokenizer_path);
    init_all();                           // 工作区（幂等）+ 梯度缓冲
    g_optim.init();

    std::cout << "=== Initialization complete ===" << std::endl;
    std::cout << "Hyperparams: lr=" << g_lr
              << " mu=" << g_mu
              << " prop_steps=" << g_prop_steps
              << " grad_accum=" << g_grad_accum
              << " hidden_dim=" << g_hidden_dim
              << " epochs=" << g_max_epochs
              << std::endl;

    DataLoader data;
    data.load_dir(g_data_dir);

    if (data.tokens.empty()) {
        std::cerr << "No training data!" << std::endl;
        return 1;
    }
    auto& tokens = data.tokens;
    std::cout << "Total tokens: " << tokens.size() << std::endl;

    std::cout << "\n========== Training start ==========" << std::endl;

    std::string log_train_path = std::string(g_log_dir) + "train_log.csv";
    std::ofstream log_train(log_train_path, std::ios::trunc);
    log_train << "epoch,avg_loss,avg_step_ms,epoch_s,lr\n";

    ProgressBar pb;
    int total_epochs = g_max_epochs;

    for (int epoch = 0; epoch < total_epochs; ++epoch) {
        lr_schedule(epoch, total_epochs);

        // 重置场状态
        std::fill(buffer.begin(), buffer.end(), 0.0f);
        std::fill(incoming.begin(), incoming.end(), 0.0f);

        float total_loss = 0.0f;
        int total_steps = 0;
        auto epoch_start = std::chrono::steady_clock::now();

        pb.start_epoch(epoch, total_epochs,
                       static_cast<int>(tokens.size()) - 1);

        for (size_t t = 0; t + 1 < tokens.size(); ++t) {
            float loss = train_token(tokens[t], tokens[t + 1],
                                     buffer.data(), incoming.data());
            total_loss += loss;
            ++total_steps;

            if (total_steps % g_grad_accum == 0) {
                g_optim.step();
                g_optim.log_grad((std::string(g_log_dir) + "grad_log.csv").c_str());
                g_optim.zero_grad();
            }

            pb.step(loss);
        }

        if (total_steps % g_grad_accum != 0) {
            g_optim.step();
            g_optim.log_grad((std::string(g_log_dir) + "grad_log.csv").c_str());
            g_optim.zero_grad();
        }
        g_optim.flush_log((std::string(g_log_dir) + "grad_log.csv").c_str());

        float avg_loss = total_loss / static_cast<float>(total_steps);
        pb.end_epoch(avg_loss);

        // 场快照
        {
            std::string fname = std::string(g_log_dir) + "field_e"
                              + std::to_string(epoch) + "_t"
                              + std::to_string(total_steps) + ".bin";
            std::ofstream f(fname, std::ios::binary);
            if (f) f.write(reinterpret_cast<const char*>(buffer.data()),
                          buffer.size() * sizeof(float));
        }

        float epoch_s = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - epoch_start).count();
        float avg_step_ms = pb.avg_step_ms();

        log_train << epoch << ","
                  << avg_loss << ","
                  << avg_step_ms << ","
                  << epoch_s << ","
                  << g_optim.lr << "\n";
        log_train.flush();

        if (avg_loss < g_min_loss) {
            std::cout << ">>> Early stop" << std::endl;
            break;
        }
    }
    log_train.close();

    std::cout << "\n========== Training end ==========" << std::endl;
    save_weights();
    return 0;
}

// ============ 入口分发 ============

int main(int argc, char* argv[])
{
    if (argc > 1 && std::string(argv[1]) == "gen")
        return run_generate(argc, argv);
    else
        return run_train(argc, argv);
}
