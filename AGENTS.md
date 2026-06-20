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

A physical-neural-network simulation: trains a 3D tensor field (64×64×64) as a differentiable "physical" computation medium. Tokens are injected as signals into the field, propagate for N steps, and are read out as logits. Built in C++23 with OpenMP, BPE tokenizer, SGD+Momentum optimizer.

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
  app/train.cpp           Entry — training loop + generate mode
  core/
    types.h               64×64×64 tensor dimensions, vocab/hidden sizes
    config.h              Hyperparameters (lr, prop_steps, grad_accum, paths)
  a-network/
    convert.h/.cpp        Token ID → signal injection into field
    field.h/.cpp          3D field operations (propagation core)
    model.h/.cpp          init_all(), train_token() — one training step
    readout.h/.cpp        Forward readout: field → logits
    backward.h/.cpp       Backward pass through propagation
  tokenizer/
    bpe.h/.cpp            BPE tokenizer (load, encode, decode)
  io/
    data.h/.cpp           DataLoader — read .txt files, tokenize
    checkpoint.h/.cpp     save_weights() / load_weights() (binary with magic)
    progress.h/.cpp       ProgressBar — console training progress
  train/
    optimizer.h/.cpp      SGDMomentum — SGD + momentum + gradient clipping
```

**Data flow**: Token ID → `token_to_signal()` injects a pattern into the 64×64×64 field → `propagate_network()` runs the field forward (configurable steps) → `forward_readout()` extracts logits → `train_token()` computes loss and backpropagates.

## Conventions

- **Headers**: `#pragma once`, includes relative to `src/` (compiled with `-Isrc`)
- **Global constants**: `g_` prefix, `constexpr`, all lowercase snake_case (`g_vocab_size`, `g_prop_steps`)
- **Classes**: PascalCase (`ProgressBar`, `DataLoader`, `SGDMomentum`)
- **Functions**: snake_case (`load_tokenizer`, `save_weights`, `train_token`)
- **Member variables**: `m_` prefix (`m_epoch`, `m_current_loss`)
- **Comments**: Chinese for descriptions/intent; English for code-relevant notes
- **Extern globals**: declared in headers with `extern`, defined exactly once in a `.cpp`
- **All network values**: `float` throughout
- **Gradients**: accumulated over `g_grad_accum` steps before weight update
- **Loss**: MSE (mean squared error) between predicted logits and one-hot target
- **No STL exceptions** used in hot paths
- **OpenMP**: used in propagation and backward passes (`#pragma omp parallel for`)

## Notes

(Add project-specific notes here as they arise.)
