#include "optimizer.h"
#include "a-network/convert.h"     // g_embed_weight, g_in_weight, g_in_bias
#include "a-network/readout.h"     // g_out_weight, g_out_bias
#include "a-network/field.h"       // g_prop_weight, g_skip_weight
#include "a-network/backward.h"    // gradient buffers
#include <cmath>
#include <fstream>
#include <sstream>

SGDMomentum g_optim;

void SGDMomentum::init()
{
    buf_embed.resize(g_embed_weight.size(), 0.0f);
    buf_in_weight.resize(g_in_weight.size(), 0.0f);
    buf_in_bias.resize(g_in_bias.size(), 0.0f);
    buf_out_weight.resize(g_out_weight.size(), 0.0f);
    buf_out_bias.resize(g_out_bias.size(), 0.0f);
    buf_prop.resize(g_prop_weight.size(), 0.0f);
    if (!g_skip_weight.empty())
        buf_skip.resize(g_skip_weight.size(), 0.0f);
}

void SGDMomentum::step()
{
    // 自适应裁剪阈值 = EMA(全局梯度范数) × clip_factor
    float effective_clip = (m_step_count == 0) ?
        100.0f : m_ema_norm * m_clip_factor;

    auto upd = [&](std::vector<float>& w, std::vector<float>& b,
                   const std::vector<float>& g, float* out_norm) {
        if (w.empty() || g.empty()) { if (out_norm) *out_norm = 0.0f; return; }
        double n2 = 0.0;
        #pragma omp parallel for reduction(+:n2)
        for (int i = 0; i < static_cast<int>(g.size()); ++i)
            n2 += static_cast<double>(g[i]) * g[i];
        float norm = static_cast<float>(std::sqrt(n2));
        if (out_norm) *out_norm = norm;
        float s = 1.0f;
        if (n2 > static_cast<double>(effective_clip) * effective_clip)
            s = effective_clip / norm;

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(w.size()); ++i) {
            float gi = g[i] * s;
            b[i] = mu * b[i] + gi;
            w[i] -= lr * b[i];
        }
    };

    upd(g_embed_weight,  buf_embed,       g_embed_grad,       &m_cache_embed);
    upd(g_in_weight,     buf_in_weight,   g_in_weight_grad,   &m_cache_in_w);
    upd(g_in_bias,       buf_in_bias,     g_in_bias_grad,     &m_cache_in_b);
    upd(g_out_weight,    buf_out_weight,  g_out_weight_grad,  &m_cache_out_w);
    upd(g_out_bias,      buf_out_bias,    g_out_bias_grad,    &m_cache_out_b);
    upd(g_prop_weight,   buf_prop,        g_prop_grad,        &m_cache_prop);
    if (!g_skip_weight.empty())
        upd(g_skip_weight, buf_skip, g_skip_grad, &m_cache_skip);
    else
        m_cache_skip = 0.0f;

    // 更新 EMA（从缓存的各张量范数合并为全局范数）
    m_step_count++;
    float total_norm = std::sqrt(
        m_cache_embed * m_cache_embed +
        m_cache_in_w  * m_cache_in_w  +
        m_cache_in_b  * m_cache_in_b  +
        m_cache_out_w * m_cache_out_w +
        m_cache_out_b * m_cache_out_b +
        m_cache_prop  * m_cache_prop  +
        m_cache_skip  * m_cache_skip
    );
    if (m_step_count == 1) m_ema_norm = total_norm;
    else m_ema_norm = m_ema_decay * m_ema_norm + (1.0f - m_ema_decay) * total_norm;
}

void SGDMomentum::zero_grad()
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

void SGDMomentum::flush_log(const char* path)
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

void SGDMomentum::log_grad(const char* path)
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
