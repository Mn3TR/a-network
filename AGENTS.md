AGENT八荣八耻
1.以瞎猜接口为耻，以认真查询为荣。
2.以模糊执行为耻，以寻求确认为荣。
3.以臆想业务为耻，以人类确认为荣。
4.以创造接口为耻，以复用现有为荣。
5.以跳过验证为耻，以主动测试为荣。
6.以破坏架构为耻，以遵循规范为荣。
7.以假装理解为耻，以诚实无知为荣。
8.以盲目修改为耻，以谨慎重构为荣。

# A-Network (test_network)

A physical-neural-network simulation: trains a 3D tensor field (80×80×80) as a differentiable "physical" computation medium. Tokens are injected as signals into the field, propagate for N steps, and are read out as logits. Built in C++23 with OpenMP, Adam optimizer, BPE tokenizer, polymorphic Network base class.

## Project

| Item | Value |
|---|---|
| **Language** | C++23 (clang++, `-target x86_64-pc-windows-msvc`) |
| **Entry point** | `src/app/train.cpp` → `train.exe` |
| **Stack** | C++23, OpenMP, `std::mdspan`, BPE tokenizer |
| **Tokenizer** | HuggingFace-format `tokenizer/tokenizer.json` (128256 vocab) |
| **Weights** | `output/weights.bin` (magic `0x31574E52` = "RNW1") |
| **Python tools** | `viz_field.py` — matplotlib field visualization |
| **Web viewer** | `viewer/index.html` — drag-and-drop `.bin` field viewer |

## Commands

| Command | Action |
|---|---|
| `npm run build` | Compile `train.exe` via `script\build.bat` |
| `npm run train` | Run `train.exe` (fresh training) |
| `npm run train:load` | Run `train.exe load` (continue from checkpoint) |
| `npm run gen` | Run `train.exe gen` (generate mode) |
| `npm run start` | Build + train |
| `python viz_field.py` | Generate 3-slice PNGs from `log/field_*.bin` |
| `python viz_field.py <dir>` | Same from custom directory |
| (open `viewer/index.html`) | Browser-based interactive field viewer |

## Architecture

```
src/
  app/train.cpp                         Entry — training loop + generate mode

  framework/
    config.h                            框架级通用配置（lr, beta1/beta2, 路径等）
    network.h                           Network 抽象基类 + ParamGroup
    optimizer.h/.cpp                    Adam 优化器（ParamGroup 模式）
    tokenizer.h/.cpp                    BPE tokenizer 类 + 全局兼容函数
    data.h/.cpp                         DataLoader — 读取 .txt 文件并 tokenize
    progress.h/.cpp                     ProgressBar — 控制台进度条
    logger.h/.cpp                       RunLogger — 训练日志记录（从 Trainer 拆出）
    trainer.h/.cpp                      Trainer — 训练循环编排
    generator.h/.cpp                    Generator — 自回归生成
    composite_network.h/.cpp            复合网络基类（多子网编排）

    a-network/                          A-Network 具体实现（继承 Network）
      a_network_config.h                A-Network 专有超参（80³, hidden_dim=48, prop_steps 等）
      a_network.h/.cpp                  主实现：权重/梯度/场状态/生命周期
      convert.h/.cpp                    Token ID → 信号注入
      field.h/.cpp                      3D 场操作（26 邻域传播核心）
      readout.h/.cpp                    前向读出：场 → hidden → logits
      backward.h/.cpp                   反向传播（通过传播步骤）
      model.h/.cpp                      train_step() / generate_step()
```

**数据流**: Token ID → `tokens_to_field()` 注入 hidden 向量到场 → `propagate_step()` 运行场前向传播（26 邻域 + 长程跳跃连接 + tanh）→ `forward_readout()` 读出 hidden → `lm_head_loss()` 计算交叉熵损失并反向传播。

## Conventions

- **Headers**: `#pragma once`，include 相对于 `src/`（编译时 `-Isrc`）
- **框架级常量**（Config 结构体）：驼峰命名，通过 `Config` 实例传递（`cfg.lr`, `cfg.beta1`）
- **网络专有参数**（ANetworkConfig）：同样驼峰命名，通过实例传递
- **类名**: PascalCase（`ANetwork`, `Trainer`, `Adam`, `RunLogger`, `DataLoader`）
- **函数**: snake_case（`tokens_to_field`, `propagate_step`, `readout_h`）
- **成员变量**: `m_` 前缀（`m_epoch`, `m_flat_net`, `m_prop_weight`）
- **注释**: 中文描述意图；英文写代码相关说明
- **所有权重/梯度**: `float` 类型
- **梯度**: 累积 `grad_accum` 步后更新权重
- **损失**: Softmax 交叉熵（替换了原 MSE）
- **Hot path 不用 STL 异常**
- **OpenMP**: 用于传播和反向传递（`#pragma omp parallel for`）

## 设计关键

1. **Network 抽象基类**: 所有物理网络继承自 `Network`，通过 `ParamGroup` 暴露权重/梯度，Trainer/Optimizer/Checkpoint 通过该接口统一操作，不依赖具体子类。
2. **Config 分层**: `Config` 只放框架级参数（lr、路径、beta）；网络专用超参放在对应子类的配置结构体中（如 `ANetworkConfig`）。
3. **CompositeNetwork**: 持有多子网，自动处理参数聚合、序列化等样板代码；子类重写 `train_step()`/`generate_step()` 实现级联、多头等自定义编排。
4. **Adam 优化器**: 替换原 SGD+Momentum，通过 `param_groups()` 动态遍历参数组，不直接引用全局变量。
5. **Tokenzier 封装为类**: 保留全局函数向后兼容；Trainer/Generator 通过 `setup()` 初始化。

## 已知问题

- **训练速度**: 80³ 网络每步约 1.7s（20 步传播 × 512k 细胞 × 26 邻域 atomic scatter），完整 100 epoch 训练约需 12 小时
- **避免 `static thread_local` + OMP**: `thread_local` 变量在 OpenMP 并行区域中会导致 Clang 运行时挂起。所有临时缓冲区请使用成员变量（`m_head_*`、`m_act_work`），详见 `a_network.h`

## Notes

(Add project-specific notes here as they arise.)
