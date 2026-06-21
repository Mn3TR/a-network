#include "model.cuh"
#include <cstdio>
#include <cmath>

// ============ 权重（host 端设备指针） ============
float* d_W_embed      = nullptr;
float* d_W_in         = nullptr;
float* d_b_in         = nullptr;
float* d_W_out        = nullptr;
float* d_b_out        = nullptr;

// ============ 梯度 ============
float* d_W_embed_grad = nullptr;
float* d_W_in_grad    = nullptr;
float* d_b_in_grad    = nullptr;
float* d_W_out_grad   = nullptr;
float* d_b_out_grad   = nullptr;

// ============ 动量 ============
float* d_m_embed = nullptr;
float* d_m_in_w  = nullptr;
float* d_m_in_b  = nullptr;
float* d_m_out_w = nullptr;
float* d_m_out_b = nullptr;
float* d_m_prop  = nullptr;
float* d_m_skip  = nullptr;

// ============ 辅助 ============
float* d_diff   = nullptr;
float* d_logits = nullptr;
float* d_softmax_buf = nullptr;
float* d_h_emb  = nullptr;  // [H] — d_h from inject backward
float* d_h      = nullptr;  // [H] — hidden state buffer

constexpr int REDUCE_BLOCKS = 512;

// 简单的 grid-stride zero kernel
__global__ void zero_kernel(float* buf, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) buf[i] = 0.0f;
}

// ============ 1. 注入 ============
__global__ void inject_kernel(
    float* field, const float* h,
    const float* W_in, const float* b_in, float inv_sqrt_S)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float s = b_in[n];
    #pragma unroll
    for (int k = 0; k < g_hidden_dim; ++k) {
        float hk = h[k];
        if (hk != 0.0f) s += hk * W_in[k * N + n];
    }
    field[n] += s * inv_sqrt_S;
}

// ============ 2. diff = field - b_out ============
__global__ void diff_kernel(const float* field, const float* b_out, float* diff) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n < N) diff[n] = field[n] - b_out[n];
}

// ============ 3. 读出 h[k] = Σ W_out[k][n] × diff[n] ============
__global__ void readout_h_kernel(const float* diff, const float* W_out, float* h) {
    int k = blockIdx.x;
    if (k >= g_hidden_dim) return;
    extern __shared__ float s_buf[];
    float sum = 0.0f;
    for (int n = threadIdx.x; n < N; n += blockDim.x)
        sum += W_out[k * N + n] * diff[n];
    s_buf[threadIdx.x] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_buf[threadIdx.x] += s_buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) h[k] = s_buf[0];
}

// ============ 4. LM Head: logits[t] = Σ W_embed[t][k] × h[k] ============
__global__ void lm_head_kernel(const float* h, const float* W_embed, float* logits) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= g_vocab_size) return;
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < g_hidden_dim; ++k)
        sum += W_embed[t * g_hidden_dim + k] * h[k];
    logits[t] = sum;
}

// ============ 5a. Softmax max ============
__global__ void smax_max_kernel(const float* logits, float* block_out) {
    extern __shared__ float s_buf[];
    float mx = -1e30f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < g_vocab_size; i += gridDim.x * blockDim.x)
        if (logits[i] > mx) mx = logits[i];
    s_buf[threadIdx.x] = mx; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_buf[threadIdx.x] = fmaxf(s_buf[threadIdx.x], s_buf[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) block_out[blockIdx.x] = s_buf[0];
}

// ============ 5b. Softmax sum_exp ============
__global__ void smax_sum_kernel(const float* logits, float max_val, float* block_out) {
    extern __shared__ float s_buf[];
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < g_vocab_size; i += gridDim.x * blockDim.x)
        sum += expf(logits[i] - max_val);
    s_buf[threadIdx.x] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_buf[threadIdx.x] += s_buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) block_out[blockIdx.x] = s_buf[0];
}

// ============ 5c. Softmax 归一化 + loss ============
__global__ void smax_norm_kernel(
    const float* logits, float* d_logits,
    float max_val, float inv_sum, int target_id, float* loss_out)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= g_vocab_size) return;
    float p = expf(logits[t] - max_val) * inv_sum;
    d_logits[t] = p - (t == target_id ? 1.0f : 0.0f);
    if (t == target_id) *loss_out = -logf(fmaxf(p, 1e-30f));
}

// ============ 6. LM Head 反向: d_W_embed_grad + d_h ============
// 每个 t 更新 W_embed_grad[t][*]; d_h 用另一个归约 kernel
__global__ void lm_head_bwd_grad_kernel(
    const float* d_logits, const float* h,
    float* d_W_embed_grad)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= g_vocab_size) return;
    float dv = d_logits[t];
    if (dv == 0.0f) return;
    for (int k = 0; k < g_hidden_dim; ++k)
        d_W_embed_grad[t * g_hidden_dim + k] += dv * h[k];
}

__global__ void lm_head_bwd_dh_kernel(
    const float* d_logits, const float* W_embed, float* d_h)
{
    int k = blockIdx.x;
    if (k >= g_hidden_dim) return;
    extern __shared__ float s_buf[];
    float sum = 0.0f;
    for (int t = threadIdx.x; t < g_vocab_size; t += blockDim.x) {
        float dv = d_logits[t];
        if (dv != 0.0f) sum += dv * W_embed[t * g_hidden_dim + k];
    }
    s_buf[threadIdx.x] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_buf[threadIdx.x] += s_buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) d_h[k] += s_buf[0];
}

// ============ 7. 读出反向 ============
__global__ void readout_bwd_kernel(
    const float* d_h, const float* diff, const float* W_out,
    float* d_field, float* d_W_out_grad, float* d_b_out_grad)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < g_hidden_dim; ++k) sum += d_h[k] * W_out[k * N + n];
    d_field[n] = sum;
    d_b_out_grad[n] -= sum;
    for (int k = 0; k < g_hidden_dim; ++k) {
        float dhk = d_h[k];
        if (dhk != 0.0f) d_W_out_grad[k * N + n] += dhk * diff[n];
    }
}

// ============ 8. 注入反向 ============
__global__ void inject_bwd_kernel(
    const float* d_field, const float* h,
    const float* W_in, float* d_W_in_grad, float* d_b_in_grad, float inv_sqrt_S)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float dn = d_field[n] * inv_sqrt_S;
    d_b_in_grad[n] += dn;
    for (int k = 0; k < g_hidden_dim; ++k) {
        float hk = h[k];
        if (hk != 0.0f) d_W_in_grad[k * N + n] += hk * dn;
    }
}

__global__ void inject_bwd_dh_kernel(
    const float* d_field, const float* W_in, float* d_h_emb, float inv_sqrt_S)
{
    int k = blockIdx.x;
    if (k >= g_hidden_dim) return;
    extern __shared__ float s_buf[];
    float sum = 0.0f;
    for (int n = threadIdx.x; n < N; n += blockDim.x)
        sum += W_in[k * N + n] * d_field[n];
    s_buf[threadIdx.x] = sum; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_buf[threadIdx.x] += s_buf[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) d_h_emb[k] += sum * inv_sqrt_S;
}

// ============ 9. SGD Momentum ============
__global__ void momentum_kernel(float* w, float* m, const float* g, int size, float lr, float mu) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float v = mu * m[i] + g[i];
    m[i] = v;
    w[i] -= lr * v;
}

// ============ 宿主 API ============

void gpu_init_model()
{
    size_t NH = N * g_hidden_dim;
    size_t VH = g_vocab_size * g_hidden_dim;
    auto al = [](auto& p, size_t b) { CUDA_CHECK(cudaMalloc(&p, b)); };
    al(d_W_embed,      VH * sizeof(float));
    al(d_W_in,         NH * sizeof(float));
    al(d_b_in,         N  * sizeof(float));
    al(d_W_out,        NH * sizeof(float));
    al(d_b_out,        N  * sizeof(float));
    al(d_W_embed_grad, VH * sizeof(float));
    al(d_W_in_grad,    NH * sizeof(float));
    al(d_b_in_grad,    N  * sizeof(float));
    al(d_W_out_grad,   NH * sizeof(float));
    al(d_b_out_grad,   N  * sizeof(float));
    al(d_m_embed,      VH * sizeof(float));
    al(d_m_in_w,       NH * sizeof(float));
    al(d_m_in_b,       N  * sizeof(float));
    al(d_m_out_w,      NH * sizeof(float));
    al(d_m_out_b,      N  * sizeof(float));
    al(d_m_prop,       N  * sizeof(float));
    al(d_m_skip,       1 * sizeof(float));  // 占位，upload_skip 时 realloc
    al(d_diff,         N  * sizeof(float));
    al(d_logits,       g_vocab_size * sizeof(float));
    al(d_softmax_buf,  REDUCE_BLOCKS * 2 * sizeof(float));
    al(d_h_emb,        g_hidden_dim * sizeof(float));
    al(d_h,            g_hidden_dim * sizeof(float));
}

void gpu_free_model()
{
    auto fr = [](auto& p) { cudaFree(p); p = nullptr; };
    fr(d_W_embed); fr(d_W_in); fr(d_b_in); fr(d_W_out); fr(d_b_out);
    fr(d_W_embed_grad); fr(d_W_in_grad); fr(d_b_in_grad); fr(d_W_out_grad); fr(d_b_out_grad);
    fr(d_m_embed); fr(d_m_in_w); fr(d_m_in_b); fr(d_m_out_w); fr(d_m_out_b);
    fr(d_m_prop); fr(d_m_skip);
    fr(d_diff); fr(d_logits); fr(d_softmax_buf); fr(d_h_emb); fr(d_h);
}

void gpu_upload_weights(const float* cpu_embed, const float* cpu_w_in,
                         const float* cpu_b_in, const float* cpu_w_out,
                         const float* cpu_b_out)
{
    size_t NH = N * g_hidden_dim, VH = g_vocab_size * g_hidden_dim;
    CUDA_CHECK(cudaMemcpy(d_W_embed, cpu_embed, VH * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_in,    cpu_w_in,  NH * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_in,    cpu_b_in,  N  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_W_out,   cpu_w_out, NH * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b_out,   cpu_b_out, N  * sizeof(float), cudaMemcpyHostToDevice));
}

void gpu_download_weights(float* cpu_embed, float* cpu_w_in, float* cpu_b_in,
                           float* cpu_w_out, float* cpu_b_out)
{
    size_t NH = N * g_hidden_dim, VH = g_vocab_size * g_hidden_dim;
    CUDA_CHECK(cudaMemcpy(cpu_embed, d_W_embed, VH * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu_w_in,  d_W_in,    NH * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu_b_in,  d_b_in,    N  * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu_w_out, d_W_out,   NH * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu_b_out, d_b_out,   N  * sizeof(float), cudaMemcpyDeviceToHost));
}

void gpu_download_skip_grad(float* cpu_skip_grad)
{
    if (cpu_skip_grad && d_skip_grad && gpu_num_edges > 0)
        CUDA_CHECK(cudaMemcpy(cpu_skip_grad, d_skip_grad, gpu_num_edges * sizeof(float), cudaMemcpyDeviceToHost));
}

void gpu_zero_all_gradients()
{
    size_t NH = N * g_hidden_dim, VH = g_vocab_size * g_hidden_dim;
    CUDA_CHECK(cudaMemset(d_W_embed_grad, 0, VH * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_W_in_grad,    0, NH * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b_in_grad,    0, N  * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_W_out_grad,   0, NH * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b_out_grad,   0, N  * sizeof(float)));
    gpu_clear_gradients();  // prop_grad + skip_grad
}

// ============ Launch wrappers ============
#define LAUNCH_ZERO(buf, size) do {                                     \
    int n = (size);                                                     \
    zero_kernel<<<(n+255)/256, 256>>>(buf, n);                          \
    CUDA_CHECK(cudaGetLastError());                                     \
} while(0)

void gpu_inject(float* field, const float* h)
{
    inject_kernel<<<(N+255)/256, 256>>>(field, h, d_W_in, d_b_in, 1.0f);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_readout_h(const float* field, float* h)
{
    LAUNCH_ZERO(d_h, g_hidden_dim);  // 显式清零
    diff_kernel<<<(N+255)/256, 256>>>(field, d_b_out, d_diff);
    readout_h_kernel<<<g_hidden_dim, 256, 256*sizeof(float)>>>(d_diff, d_W_out, h);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_lm_head(const float* h)
{
    int b = 256, g = (g_vocab_size + b - 1) / b;
    lm_head_kernel<<<g, b>>>(h, d_W_embed, d_logits);
    CUDA_CHECK(cudaGetLastError());
}

float gpu_softmax_loss(int target_id)
{
    int b = 256, g = (g_vocab_size + b - 1) / b;
    if (g > REDUCE_BLOCKS) g = REDUCE_BLOCKS;

    smax_max_kernel<<<g, b, b*sizeof(float)>>>(d_logits, d_softmax_buf);
    CUDA_CHECK(cudaGetLastError());

    static std::vector<float> hbuf(REDUCE_BLOCKS);
    CUDA_CHECK(cudaMemcpy(hbuf.data(), d_softmax_buf, g * sizeof(float), cudaMemcpyDeviceToHost));
    float maxv = -1e30f;
    for (int i = 0; i < g; ++i) if (hbuf[i] > maxv) maxv = hbuf[i];

    float* sum_out = d_softmax_buf + REDUCE_BLOCKS;
    smax_sum_kernel<<<g, b, b*sizeof(float)>>>(d_logits, maxv, sum_out);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(hbuf.data(), sum_out, g * sizeof(float), cudaMemcpyDeviceToHost));
    float total = 0.0f;
    for (int i = 0; i < g; ++i) total += hbuf[i];

    float inv = 1.0f / total;
    float loss = 0.0f;
    float* d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    smax_norm_kernel<<<g, b>>>(d_logits, d_logits, maxv, inv, target_id, d_loss);
    CUDA_CHECK(cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    cudaFree(d_loss);
    return loss;
}

void gpu_lm_head_backward(const float* h)
{
    int b = 256, g = (g_vocab_size + b - 1) / b;
    lm_head_bwd_grad_kernel<<<g, b>>>(d_logits, h, d_W_embed_grad);
    lm_head_bwd_dh_kernel<<<g_hidden_dim, b, b*sizeof(float)>>>(d_logits, d_W_embed, d_h);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_readout_backward(const float* d_h)
{
    readout_bwd_kernel<<<(N+255)/256, 256>>>(d_h, d_diff, d_W_out,
        d_network, d_W_out_grad, d_b_out_grad);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_backward_propagate_full()
{
    gpu_backward_propagate();  // 使用 field.cu 中的实现
}

void gpu_inject_backward(const float* h)
{
    inject_bwd_kernel<<<(N+255)/256, 256>>>(d_network, h, d_W_in,
        d_W_in_grad, d_b_in_grad, 1.0f);
    LAUNCH_ZERO(d_h_emb, g_hidden_dim);
    inject_bwd_dh_kernel<<<g_hidden_dim, 256, 256*sizeof(float)>>>(d_network, d_W_in,
        d_h_emb, 1.0f);
    CUDA_CHECK(cudaGetLastError());
}

void gpu_optimizer_step(float lr, float mu)
{
    auto apply = [&](float* w, float* m, float* g, int sz) {
        if (!w || !m || !g || sz <= 0) return;
        momentum_kernel<<<(sz+255)/256, 256>>>(w, m, g, sz, lr, mu);
    };
    size_t NH = N * g_hidden_dim, VH = g_vocab_size * g_hidden_dim;
    apply(d_W_embed,     d_m_embed,  d_W_embed_grad, VH);
    apply(d_W_in,        d_m_in_w,   d_W_in_grad,    NH);
    apply(d_b_in,        d_m_in_b,   d_b_in_grad,    N);
    apply(d_W_out,       d_m_out_w,  d_W_out_grad,   NH);
    apply(d_b_out,       d_m_out_b,  d_b_out_grad,   N);
    apply(d_prop_weight, d_m_prop,   d_prop_grad,    N);
    if (d_skip_grad && d_m_skip && gpu_num_edges > 0)
        apply(d_skip_weight, d_m_skip, d_skip_grad, gpu_num_edges);
    CUDA_CHECK(cudaGetLastError());
}
