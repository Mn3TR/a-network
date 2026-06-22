#!/usr/bin/env python3
"""
A-Network 训练 / 生成入口
==========================
替代 src/app/train.cpp + src/framework/trainer.cpp + src/framework/generator.cpp

用法:
    python -m src.anetwork.train              # 全新训练
    python -m src.anetwork.train load         # 从 checkpoint 继续
    python -m src.anetwork.train gen [seed]   # 生成模式
"""

import math
import sys
import time
import csv
from pathlib import Path
from datetime import datetime

import torch

from .config import ANetworkConfig, TrainConfig
from .model import ANetwork
from .tokenizer import TokenizerWrapper
from .data import load_data


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  日志工具
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ProgressBar:
    """简易进度条"""

    def __init__(self, total: int):
        self.total = total
        self.start_time = time.time()
        self.last_time = self.start_time

    def step(self, current: int, loss: float):
        elapsed = time.time() - self.last_time
        self.last_time = time.time()
        pct = 100.0 * current / self.total
        bar = "█" * (current * 50 // self.total) + "░" * (50 - current * 50 // self.total)
        step_ms = elapsed * 1000 if current > 0 else 0
        print(f"\r  [{bar}] {pct:5.1f}% | loss {loss:.4f} | {step_ms:.1f}ms/step", end="")
        if current >= self.total:
            total_sec = time.time() - self.start_time
            print(f" | total {total_sec:.1f}s")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  训练
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train(net: ANetwork, tokens: list, tcfg: TrainConfig, device: torch.device):
    """训练主循环"""
    if not tokens:
        print("No training data!")
        return

    print(f"\n{'=' * 42}")
    print(f"Training start")
    print(f"  lr={tcfg.lr}  grad_accum={tcfg.grad_accum}  epochs={tcfg.max_epochs}")
    print(f"  tokens={len(tokens)}  device={device}")
    print(f"{'=' * 42}\n")

    # 优化器 + 余弦学习率退火
    optimizer = torch.optim.Adam(
        net.parameters(), lr=tcfg.lr,
        betas=(tcfg.beta1, tcfg.beta2), eps=tcfg.eps
    )

    total_epochs = tcfg.max_epochs
    total_steps = len(tokens) - 1

    # 日志目录
    log_dir = Path(tcfg.log_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # CSV 日志
    epoch_log = open(run_dir / "epoch_log.csv", "w", newline="")
    epoch_writer = csv.writer(epoch_log)
    epoch_writer.writerow(["epoch", "avg_loss", "avg_step_ms", "epoch_s", "lr"])

    best_loss = float("inf")

    for epoch in range(total_epochs):
        # 余弦学习率退火
        if total_epochs > 1:
            progress = epoch / (total_epochs - 1)
            factor = (1.0 + math.cos(math.pi * progress)) * 0.5
            lr = tcfg.lr_min + (tcfg.lr - tcfg.lr_min) * factor
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        else:
            lr = tcfg.lr

        net.reset_state()
        net.train()

        epoch_loss = 0.0
        epoch_start = time.time()

        # 进度条信息
        pb = ProgressBar(total_steps)

        optimizer.zero_grad()

        for step_idx in range(total_steps):
            input_id = torch.tensor(tokens[step_idx], device=device)
            target_id = torch.tensor(tokens[step_idx + 1], device=device)

            loss = net.train_step(input_id, target_id)
            loss.backward()

            epoch_loss += loss.item()

            # 梯度积累
            if (step_idx + 1) % tcfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            pb.step(step_idx + 1, loss.item())

        # 处理剩余的梯度积累
        if total_steps % tcfg.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        epoch_s = time.time() - epoch_start
        avg_loss = epoch_loss / total_steps
        avg_step_ms = (epoch_s / total_steps) * 1000

        pb.step(total_steps, avg_loss)
        print()

        # 日志
        epoch_writer.writerow([epoch, f"{avg_loss:.6f}", f"{avg_step_ms:.2f}",
                               f"{epoch_s:.1f}", f"{lr:.8f}"])
        epoch_log.flush()

        # 场快照
        field_path = run_dir / f"field_e{epoch}.bin"
        net.dump_field(str(field_path))

        print(f"  Epoch {epoch}: loss={avg_loss:.6f}  "
              f"step_ms={avg_step_ms:.1f}  "
              f"epoch_s={epoch_s:.1f}  "
              f"lr={lr:.8f}")

        # 早停
        if avg_loss < tcfg.min_loss:
            print(f">>> Early stop (loss={avg_loss:.6f} < min_loss={tcfg.min_loss})")
            break

        # 保存最佳权重
        if avg_loss < best_loss:
            best_loss = avg_loss
            net.save(str(run_dir / "best.pt"))

    epoch_log.close()

    # 保存最终权重
    net.save(str(run_dir / "final.pt"))
    # 同时保存到 output/weights.pt 以便加载
    net.save(tcfg.weights_path)

    print(f"\n{'=' * 42}")
    print(f"Training end  (best_loss={best_loss:.6f})")
    print(f"Logs saved to {run_dir}")
    print(f"{'=' * 42}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  生成
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate(net: ANetwork, tokenizer: TokenizerWrapper,
             seed_text: str, max_tokens: int):
    """生成模式"""
    seed_ids = tokenizer.encode(seed_text)
    print(f"Seed: \"{seed_text}\"")

    generated = net.generate(seed_ids, max_tokens=max_tokens)

    # 解码并打印输出
    output_text = tokenizer.decode(generated)
    print(f"\nFull text:\n{output_text}")

    return output_text


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    load_mode = (mode == "load")
    gen_mode = (mode == "gen")

    a_cfg = ANetworkConfig()
    tcfg = TrainConfig()

    # 加载 tokenizer（需要提前获取词表大小）
    tokenizer = TokenizerWrapper(tcfg.tokenizer_path)
    a_cfg.vocab_size = tokenizer.vocab_size

    # 创建网络
    net = ANetwork(a_cfg, device=device)

    if gen_mode:
        # ── 生成模式 ──
        net.to(device)
        if not net.load(tcfg.weights_path):
            print("ERROR: No trained weights found. Train first or specify a different path.")
            return

        seed_text = sys.argv[2] if len(sys.argv) > 2 else tcfg.seed_text
        generate(net, tokenizer, seed_text, tcfg.gen_max_tokens)
        return

    # ── 训练模式 ──
    net.to(device)

    if load_mode:
        if not net.load(tcfg.weights_path):
            print("Warning: No checkpoint found, starting fresh training.")
    else:
        print("Initializing fresh weights.")

    # 加载数据
    print(f"Loading data from '{tcfg.data_dir}' ...")
    tokens = load_data(tcfg.data_dir, tokenizer)

    # 训练
    train(net, tokens, tcfg, device)


if __name__ == "__main__":
    main()
