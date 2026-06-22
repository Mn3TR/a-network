# A-Network (test_network) — Python 实现

物理神经网络模拟：用 PyTorch 实现一个可训练的 3D 张量场（80×80×80）作为可微分"物理"计算介质。Token 作为信号注入场中，经过 20 步 26-邻域扩散 + 随机长程跳跃连接传播后，读出为 logits。

从 C++23 版改写而来，利用 PyTorch autograd 消除手写反向传播（-400 行），HuggingFace tokenizers 替代手写 BPE（360→3 行），总量从 ~2400 行 C++ 降至 ~780 行 Python。

## 项目

| 项 | 值 |
|---|---|
| **语言** | Python 3.11+ (PyTorch 2.0+) |
| **入口** | `python -m src.anetwork.train` |
| **栈** | PyTorch (`nn.Module`, `F.conv3d`, `autograd`), HuggingFace `tokenizers` |
| **Tokenizer** | HuggingFace-format `tokenizer/tokenizer.json` (vocab=128815) |
| **权重** | `output/weights.pt`（PyTorch `state_dict`） |
| **可视化** | `utils/viz_field.py` — matplotlib 场切片；`utils/viewer/index.html` — 拖放式 web 查看器 |

## 命令

| 命令 | 操作 |
|---|---|
| `npm run train` | `python -m src.anetwork.train` — 全新训练 |
| `npm run train:load` | 从 checkpoint 继续训练 |
| `npm run gen` | 生成模式 |
| `npm run viz` | 可视化场快照 |
| `pip install -r requirements.txt` | 安装依赖 |

## 快速开始

```bash
pip install -r requirements.txt
python -m src.anetwork.train          # 训练
python -m src.anetwork.train load     # 继续
python -m src.anetwork.train gen      # 生成
```

## 架构

```
src/
  anetwork/
    __init__.py          包标记
    config.py            ANetworkConfig + TrainConfig 数据类
    model.py             ANetwork — 核心 nn.Module（注入→传播→读出→lm_head）
    tokenizer.py         HuggingFace tokenizer 薄包装
    data.py              从 .txt 文件加载并编码数据
    train.py             训练循环 + 生成模式入口
utils/
  viz_field.py           场状态可视化（三切片 PNG）
  viewer/
    index.html           浏览器交互场查看器
```

## 数据流

```
Token ID → Embedding → 注入 (in_weight + in_bias) → [N 场]
  → 20× propagate_step (conv3d 26-邻域 gather + skip scatter)
  → 读出 (out_weight + out_bias) → h → LM Head (embed_weight, 权值共享)
  → logits → CrossEntropy loss
```

反向传播由 `loss.backward()` 自动完成（autograd）。

## 核心算法

### 传播步 (`_propagate`)

```
Phase 1+2: field = field * time_decay + incoming
           incoming = 0; act = tanh(field / 2)

Phase 3:   signal = 2 * act * prop_weight
           new_incoming = conv3d(signal, kernel_of_ones_center_zero, padding=1)
           new_incoming += scatter_add(skip_signal, skip_dst)
```

使用 `F.conv3d` (3×3×3 kernel, center=0) 实现 26-邻域 gather。由于邻域图对称，gather 等效于 C++ 版本的 atomic scatter，但利用了高度优化的 MKL/cuDNN 卷积实现。

### 梯度管理

场状态 (`field`, `incoming`) 是 `nn.Buffer`，在 `train_step()` 开头 `detach()`，确保梯度不跨 token 传播（无 BPTT）。每步的 loss 仅对应单个 token 的 next-token-prediction。

## 训练参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `network_x/y/z` | 80 | 3D 场维度，共 512K 细胞 |
| `hidden_dim` | 48 | 隐向量维度 |
| `prop_steps` | 20 | 传播步数 |
| `time_decay` | 0.9 | 每步衰减系数 |
| `skip_density` | 0.20 | 跳跃连接密度 |
| `lr` | 1e-4 | Adam 学习率 |
| `grad_accum` | 4 | 梯度积累步数 |

## 资源

| 项 | 值 |
|---|---|
| 模型参数 | 56.9M（~228 MB float32） |
| Adam 状态 | ~456 MB（momentum + variance） |
| 训练内存 (CPU) | ~1 GB |
| 训练内存 (GPU) | ~1.5 GB（显存 ≥ 4GB 可用，推荐 8GB+） |
| 场大小 | 512K floats = 2 MB |
| 词表 | 128,815 |

## 与 C++ 版的差异

| 方面 | C++ 版 | Python 版 |
|---|---|---|
| 反向传播 | 手写 backward_inject/backward_propagate/backward_readout_h | `loss.backward()` autograd |
| BPE tokenizer | 手写 JSON 解析器 + BPE 合并（tokenizer.cpp, 360 行） | `from tokenizers import Tokenizer`（3 行） |
| 优化器 | 手写 Adam + 学习率调度 | `torch.optim.Adam` + `CosineAnnealingLR` |
| 邻域传播 | `#pragma omp atomic` scatter（26 原子写 × 512K） | `F.conv3d(ones_kernel)` gather |
| 跳跃连接 | CSR 格式（forward + reverse） | COO 格式 + `scatter_add_` |
| 权重格式 | 自定义 magic (`0x31574E52`) + 手动序列化 | `torch.save(state_dict)` |
| 构建 | C++23 编译为 train.exe | 无需编译，即时运行 |
| 代码量 | ~2400 行 | ~780 行 |

## 约定

- **类**: PascalCase (`ANetwork`, `ANetworkConfig`, `TokenizerWrapper`)
- **函数/方法**: snake_case (`train_step`, `generate_step`, `_propagate`)
- **私有方法**: `_` 前缀 (`_inject`, `_readout`, `_init_weights`)
- **参数**: `nn.Parameter` (`embed_weight`, `prop_weight` 等)
- **状态**: `nn.Buffer` (`field`, `incoming`, `neighbor_kernel`, `skip_src`/`dst`)
- **数据类型**: `float32` 贯穿
- **Loss**: `F.cross_entropy`（label smoothing 为 0）
- **无 BPTT**: 每步 `field.detach()` 切断梯度流
- **设备**: 自动选择 CUDA（若可用）或 CPU（`torch.device("cuda" if torch.cuda.is_available() else "cpu")`）
- **梯度裁剪**: `clip_grad_norm_(all_params, max_norm=1.0)`
- **注释**: 中文描述意图，英文解释代码细节
