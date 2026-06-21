#include "optimizer.h"
#include "a-network/convert.h"     // g_embed_weight, g_in_weight, g_in_bias
#include "a-network/readout.h"     // g_out_weight, g_out_bias
#include "a-network/field.h"       // g_prop_weight, g_skip_weight
#include "a-network/backward.h"    // gradient buffers
#include <cmath>
#include <fstream>
#include <sstream>

Adam g_optim;

void Adam::init()
{
    // 一阶矩
    m_embed.resize(g_embed_weight.size(), 0.0f);
    m_in_weight.resize(g_in_weight.size(), 0.0f);
    m_in_bias.resize(g_in_bias.size(), 0.0f);
    m_out_weight.resize(g_out_weight.size(), 0.0f);
    m_out_bias.resize(g_out_bias.size(), 0.0f);
    m_prop.resize(g_prop_weight.size(), 0.0f);
    if (!g_skip_weight.empty())
        m_skip.resize(g_skip_weight.size(), 0.0f);

    // 二阶矩
    v_embed.resize(g_embed_weight.size(), 0.0f);
    v_in_weight.resize(g_in_weight.size(), 0.0f);
    v_in_bias.resize(g_in_bias.size(), 0.0f);
    v_out_weight.resize(g_out_weight.size(), 0.0f);
    v_out_bias.resize(g_out_bias.size(), 0.0f);
    v_prop.resize(g_prop_weight.size(), 0.0f);
    if (!g_skip_weight.empty())
        v_skip.resize(g_skip_weight.size(), 0.0f);

    m_step_count = 0;
}

void Adam::step()
{
    m_step_count++;
    float inv_1mb1 = 1.0f / (1.0f - std::pow(beta1, static_cast<float>(m_step_count)));
    float inv_1mb2 = 1.0f / (1.0f - std::pow(beta2, static_cast<float>(m_step_count)));

    // 对每个参数组执行 Adam 更新
    // lambda: w=权重, m=一阶矩, v=二阶矩, g=梯度, out_norm=输出梯度范数
    auto upd = [&](std::vector<float>& w, std::vector<float>& m,
                   std::vector<float>& v, const std::vector<float>& g,
                   float* out_norm) {
        if (w.empty() || g.empty()) { if (out_norm) *out_norm = 0.0f; return; }

        // 1) 计算梯度 L2 范数（用于日志）
        double n2 = 0.0;
        #pragma omp parallel for reduction(+:n2)
        for (int i = 0; i < static_cast<int>(g.size()); ++i)
            n2 += static_cast<double>(g[i]) * g[i];
        float norm = static_cast<float>(std::sqrt(n2));
        if (out_norm) *out_norm = norm;

        // 2) Adam 更新
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(w.size()); ++i) {
            float gi = g[i];
            // 一阶矩（动量）
            m[i] = beta1 * m[i] + (1.0f - beta1) * gi;
            // 二阶矩（自适应学习率）
            v[i] = beta2 * v[i] + (1.0f - beta2) * gi * gi;
            // 偏置校正 + 参数更新
            float m_hat = m[i] * inv_1mb1;
            float v_hat = v[i] * inv_1mb2;
            w[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
        }
    };

    upd(g_embed_weight,  m_embed,     v_embed,     g_embed_grad,     &m_cache_embed);
    upd(g_in_weight,     m_in_weight, v_in_weight, g_in_weight_grad, &m_cache_in_w);
    upd(g_in_bias,       m_in_bias,   v_in_bias,   g_in_bias_grad,   &m_cache_in_b);
    upd(g_out_weight,    m_out_weight, v_out_weight,g_out_weight_grad,&m_cache_out_w);
    upd(g_out_bias,      m_out_bias,  v_out_bias,  g_out_bias_grad,  &m_cache_out_b);
    upd(g_prop_weight,   m_prop,      v_prop,      g_prop_grad,      &m_cache_prop);
    if (!g_skip_weight.empty())
        upd(g_skip_weight, m_skip, v_skip, g_skip_grad, &m_cache_skip);
    else
        m_cache_skip = 0.0f;
}

void Adam::zero_grad()
{
    auto zero = [](std::vector<float>& v) {
        if (v.empty()) return;
        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(v.size()); ++i)
            v[i] = 0.0f;
    };

    zero(g_embed_grad);
    zero(g_in_weight_grad);
    zero(g_in_bias_grad);
    zero(g_out_weight_grad);
    zero(g_out_bias_grad);
    zero(g_prop_grad);
    zero(g_skip_grad);
}

void Adam::flush_log(const char* path)
{
    if (log_buffer.empty()) return;
    static bool first = true;
    std::ofstream file;
    if (first) {
        file.open(path, std::ios::trunc);
        file << "step,embed,in_w,in_b,out_w,out_b,prop,skip\n";
        first = false;
    } else {
        file.open(path, std::ios::app);
    }
    file << log_buffer;
    log_buffer.clear();
    log_line_count = 0;
}

void Adam::log_grad(const char* path)
{
    static int step_counter = 0;
    std::ostringstream line;
    line << (step_counter++) << ","
         << m_cache_embed << ","
         << m_cache_in_w << ","
         << m_cache_in_b << ","
         << m_cache_out_w << ","
         << m_cache_out_b << ","
         << m_cache_prop << ","
         << m_cache_skip << "\n";
    log_buffer += line.str();
    ++log_line_count;

    if (log_line_count >= log_flush_interval)
        flush_log(path);
}

static constexpr float PI = 3.14159265358979323846f;

void lr_schedule(int epoch, int total_epochs)
{
    if (total_epochs <= 1) return;
    float progress = static_cast<float>(epoch) / (total_epochs - 1);
    float factor = (1.0f + std::cos(PI * progress)) * 0.5f;
    float new_lr = g_lr_min + (g_lr - g_lr_min) * factor;
    g_optim.set_lr(new_lr);
}
