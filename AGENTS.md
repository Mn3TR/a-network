# A-Network (test_network) — Python 实现

物理神经网络模拟：用 PyTorch 实现一个可训练的 3D 张量场（80×80×80）作为可微分"物理"计算介质。Token 作为信号注入场中，经过 20 步 26-邻域扩散 + 随机长程跳跃连接传播后，读出为 logits。

从 C++23 版改写而来，利用 PyTorch autograd 消除手写反向传播，HuggingFace tokenizers 替代手写 BPE。

## 项目

| 项 | 值 |
|---|---|
| **语言** | Python 3.11+ (PyTorch 2.0+) |
| **入口** | `src/anetwork/train.py` → `python -m src.anetwork.train` |
| **栈** | PyTorch, HuggingFace tokenizers, conv3d 加速邻域传播 |
| **Tokenizer** | HuggingFace-format `tokenizer/tokenizer.json` (128256 vocab) |
| **权重** | `output/weights.pt`（PyTorch state_dict） |
| **Python 工具** | `utils/viz_field.py` — matplotlib 场可视化 |
| **Web 查看器** | `utils/viewer/index.html` — 拖放式 `.bin` 场查看器 |

## 命令

| 命令 | 操作 |
|---|---|
| `npm run train` | 全新训练 |
| `npm run train:load` | 从 checkpoint 继续训练 |
| `npm run gen` | 生成模式 |
| `npm run viz` | 可视化场快照 |
| `npm run install:deps` | 安装 Python 依赖 |
| `pip install -r requirements.txt` | 安装 Python 依赖 |

## 快速开始

```bash
pip install -r requirements.txt
python -m src.anetwork.train
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

使用 `F.conv3d` (3×3×3 kernel, center=0) 实现 26-邻域 gather。由于邻域图对称，gather 等效于 C++ 版本的 atomic scatter。

### 梯度管理

场状态 (`field`, `incoming`) 是 `nn.Buffer`，在 `train_step()` 开头 `detach()`，确保梯度不跨 token 传播（无 BPTT）。

## 与 C++ 版的差异

| 方面 | C++ 版 | Python 版 |
|---|---|---|
| 反向传播 | 手写 400+ 行 | autograd 自动 |
| BPE tokenizer | 手写 360 行 | `tokenizers` 库 3 行 |
| 优化器 | 手写 Adam | `torch.optim.Adam` |
| 学习率调度 | 手写 | 余弦退火 (math.cos) |
| 邻域传播 | OpenMP atomic scatter | conv3d gather |
| 权重格式 | 自定义 magic 二进制 | torch.save state_dict |
| 代码量 | ~2400 行 | ~500 行 |

## 约定

- **类**: PascalCase (`ANetwork`, `ANetworkConfig`, `TokenizerWrapper`)
- **函数/方法**: snake_case (`train_step`, `generate_step`, `_propagate`)
- **私有方法**: `_` 前缀 (`_inject`, _readout`, `_init_weights`)
- **成员变量 (nn.Module)**: `self.field`, `self.incoming` 为 buffer；`self.embed_weight`, `self.prop_weight` 为 Parameter
- **所有网络值**: float32 贯穿
- **Loss**: CrossEntropy（替代原有 MSE）
- **无 BPTT**: 每步 field.detach() 切断梯度流
- **设备**: 自动选择 CUDA (若可用) 或 CPU
