#include "optimizer.h"
#include <cmath>
#include <fstream>
#include <sstream>

Adam::Adam(float lr, float beta1, float beta2, float eps)
    : lr(lr), beta1(beta1), beta2(beta2), eps(eps)
{
}

void Adam::init(const std::vector<ParamGroup>& groups)
{
    m_states.clear();
    m_grad_norms.clear();
    m_group_names.clear();
    for (auto& g : groups) {
        size_t sz = g.weights ? g.weights->size() : 0;
        m_states.push_back({
            std::vector<float>(sz, 0.0f),
            std::vector<float>(sz, 0.0f)
        });
        m_grad_norms.push_back(0.0f);
        m_group_names.push_back(g.name);
    }
    m_step_count = 0;
}

void Adam::step(const std::vector<ParamGroup>& groups)
{
    m_step_count++;
    float inv_1mb1 = 1.0f / (1.0f - std::pow(beta1, static_cast<float>(m_step_count)));
    float inv_1mb2 = 1.0f / (1.0f - std::pow(beta2, static_cast<float>(m_step_count)));

    for (size_t gi = 0; gi < groups.size(); ++gi) {
        auto& g = groups[gi];
        if (!g.weights || g.weights->empty()) {
            m_grad_norms[gi] = 0.0f;
            continue;
        }

        auto& w = *g.weights;
        auto& gr = *g.gradients;
        auto& m = m_states[gi].m;
        auto& v = m_states[gi].v;

        // 梯度范数
        double n2 = 0.0;
        #pragma omp parallel for reduction(+:n2)
        for (int i = 0; i < static_cast<int>(gr.size()); ++i)
            n2 += static_cast<double>(gr[i]) * gr[i];
        m_grad_norms[gi] = static_cast<float>(std::sqrt(n2));

        // Adam 更新
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(w.size()); ++i) {
            float gi_val = gr[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * gi_val;
            v[i] = beta2 * v[i] + (1.0f - beta2) * gi_val * gi_val;
            float m_hat = m[i] * inv_1mb1;
            float v_hat = v[i] * inv_1mb2;
            w[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    }
}

void Adam::log_grad(const char* path)
{
    static int step_counter = 0;
    std::ostringstream line;
    line << (step_counter++);
    for (size_t gi = 0; gi < m_grad_norms.size(); ++gi)
        line << "," << m_grad_norms[gi];
    line << "\n";
    log_buffer += line.str();
    ++log_line_count;

    if (log_line_count >= kLogFlushInterval)
        flush_log(path);
}

void Adam::flush_log(const char* path)
{
    if (log_buffer.empty()) return;

    // 检查文件是否存在（决定是否写表头）
    std::ifstream test(path);
    bool need_header = !test.good();
    test.close();

    std::ofstream file(path, std::ios::app);
    if (need_header) {
        file << "step";
        for (size_t i = 0; i < m_group_names.size(); ++i)
            file << "," << m_group_names[i];
        file << "\n";
    }
    file << log_buffer;
    log_buffer.clear();
    log_line_count = 0;
}
