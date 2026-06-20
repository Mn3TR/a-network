#include "optimizer.h"
#include "a-network/convert.h"   // g_embed_weight, g_proj_weight, g_proj_bias
#include "a-network/field.h"     // g_prop_weight
#include "a-network/backward.h"  // gradient buffers
#include <cmath>
#include <fstream>

SGDMomentum g_optim;

// ============ 初始化 ============
void SGDMomentum::init()
{
    buf_embed.resize(g_embed_weight.size(), 0.0f);
    buf_proj.resize(g_proj_weight.size(), 0.0f);
    buf_bias.resize(g_proj_bias.size(), 0.0f);
    buf_prop.resize(g_prop_weight.size(), 0.0f);
    buf_skip.resize(g_skip_weight.size(), 0.0f);
}

// ============ 参数更新 ============
void SGDMomentum::step()
{
    auto upd = [&](std::vector<float>& w, std::vector<float>& b,
                   const std::vector<float>& g) {
        // 逐层梯度裁剪：每层独立 norm、独立 clip
        double n2 = 0.0;
        for (auto v : g) n2 += static_cast<double>(v) * v;
        float s = 1.0f;
        if (n2 > static_cast<double>(clip_norm) * clip_norm)
            s = clip_norm / static_cast<float>(std::sqrt(n2));

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(w.size()); ++i) {
            float gi = g[i] * s;
            b[i] = mu * b[i] + gi;
            w[i] -= lr * b[i];
        }
    };
    upd(g_embed_weight, buf_embed, g_embed_grad);
    upd(g_proj_weight, buf_proj, g_proj_grad);
    upd(g_proj_bias, buf_bias, g_bias_grad);
    upd(g_prop_weight, buf_prop, g_prop_grad);
    if (!g_skip_weight.empty())
        upd(g_skip_weight, buf_skip, g_skip_grad);
}

void SGDMomentum::zero_grad()
{
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(g_embed_grad.size()); ++i)
        g_embed_grad[i] = 0.0f;
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(g_proj_grad.size()); ++i)
        g_proj_grad[i] = 0.0f;
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(g_bias_grad.size()); ++i)
        g_bias_grad[i] = 0.0f;
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(g_prop_grad.size()); ++i)
        g_prop_grad[i] = 0.0f;
    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(g_skip_grad.size()); ++i)
        g_skip_grad[i] = 0.0f;
}

// ============ 梯度日志 ============
static float grad_norm(const std::vector<float>& g)
{
    double sum = 0.0;
    for (auto v : g) sum += static_cast<double>(v) * v;
    return static_cast<float>(std::sqrt(sum));
}

void SGDMomentum::log_grad(const char* path)
{
    static bool first = true;
    std::ofstream file;
    if (first) {
        file.open(path, std::ios::trunc);
        file << "step,embed_norm,proj_norm,bias_norm,prop_norm,skip_norm\n";
        first = false;
    } else {
        file.open(path, std::ios::app);
    }
    static int step_counter = 0;
    file << (step_counter++) << ","
         << grad_norm(g_embed_grad) << ","
         << grad_norm(g_proj_grad) << ","
         << grad_norm(g_bias_grad) << ","
         << grad_norm(g_prop_grad) << ","
         << grad_norm(g_skip_grad) << "\n";
}

// ============ 学习率退火 ============
static constexpr float PI = 3.14159265358979323846f;

// 余弦退火：从 g_lr 平滑降到 g_lr_min
void lr_schedule(int epoch, int total_epochs)
{
    if (total_epochs <= 1) return;
    float progress = static_cast<float>(epoch) / (total_epochs - 1);
    float factor = (1.0f + std::cos(PI * progress)) * 0.5f;
    float new_lr = g_lr_min + (g_lr - g_lr_min) * factor;
    g_optim.set_lr(new_lr);
}
