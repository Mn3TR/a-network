#include "trainer.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>

// ============ Cosine 学习率退火 ============
static constexpr float PI = 3.14159265358979323846f;

static void lr_schedule(float& lr, float lr_min, float lr_max,
                         int epoch, int total_epochs)
{
    if (total_epochs <= 1) return;
    float progress = static_cast<float>(epoch) / (total_epochs - 1);
    float factor = (1.0f + std::cos(PI * progress)) * 0.5f;
    lr = lr_min + (lr_max - lr_min) * factor;
}

// ==================== 构造 ====================

Trainer::Trainer(Network& net, const Config& cfg)
    : m_net(net)
    , m_cfg(cfg)
    , m_optim(cfg.lr, cfg.beta1, cfg.beta2, cfg.eps)
{
}

// ==================== 初始化 ====================

void Trainer::setup()
{
    load_tokenizer(m_cfg.tokenizer_path);
    m_optim.init(m_net.param_groups());
    m_logger.create_timestamp_dirs(m_cfg);

    // 数据加载
    m_data.load_dir(m_cfg.data_dir);
    if (m_data.tokens.empty()) {
        std::cerr << "No training data!" << std::endl;
        return;
    }
    std::cout << "Total tokens: " << m_data.tokens.size() << std::endl;
}

// ==================== 训练主循环 ====================

void Trainer::train()
{
    auto& tokens = m_data.tokens;
    if (tokens.empty()) return;

    std::cout << "\n========== Training start ==========" << std::endl;
    std::cout << "lr=" << m_cfg.lr
              << " grad_accum=" << m_cfg.grad_accum
              << " epochs=" << m_cfg.max_epochs
              << std::endl;

    int total_epochs = m_cfg.max_epochs;

    for (int epoch = 0; epoch < total_epochs; ++epoch) {
        lr_schedule(m_optim.lr, m_cfg.lr_min, m_cfg.lr, epoch, total_epochs);
        m_net.reset_state();

        float total_loss = 0.0f;
        int total_steps = 0;
        auto epoch_start = std::chrono::steady_clock::now();

        size_t steps_per_epoch = tokens.size() - 1;
        m_pb.start_epoch(epoch, total_epochs,
                         static_cast<int>(steps_per_epoch));

        for (size_t t = 0; t + 1 < tokens.size(); ++t) {
            float loss = m_net.train_step(tokens[t], tokens[t + 1]);
            total_loss += loss;
            ++total_steps;

            if (total_steps % static_cast<int>(m_cfg.grad_accum) == 0) {
                m_optim.step(m_net.param_groups());
                m_optim.log_grad((m_logger.log_dir + "grad_log.csv").c_str());
                m_net.zero_grad();
            }

            m_pb.step(loss);
        }

        if (total_steps % static_cast<int>(m_cfg.grad_accum) != 0) {
            m_optim.step(m_net.param_groups());
            m_optim.log_grad((m_logger.log_dir + "grad_log.csv").c_str());
            m_net.zero_grad();
        }
        m_optim.flush_log((m_logger.log_dir + "grad_log.csv").c_str());

        float avg_loss = total_loss / static_cast<float>(total_steps);
        m_pb.end_epoch(avg_loss);

        m_logger.snapshot_field(m_net, epoch, total_steps);

        float epoch_s = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - epoch_start).count();
        float avg_step_ms = m_pb.avg_step_ms();

        m_logger.log_epoch(epoch, avg_loss, epoch_s, avg_step_ms, m_optim.lr);

        if (avg_loss < m_cfg.min_loss) {
            std::cout << ">>> Early stop (loss=" << avg_loss
                      << " < min_loss=" << m_cfg.min_loss << ")" << std::endl;
            break;
        }
    }

    std::cout << "\n========== Training end ==========" << std::endl;
    m_net.save(m_logger.weights_path);
}
