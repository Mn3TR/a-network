"""
A-Network — 物理神经网络 PyTorch 实现
======================================
以 3D 张量场作为可微分的「物理」计算介质。
Token 以信号形式注入场中，传播 N 步后读出为 logits。
"""

import math
import random
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ANetworkConfig


class ANetwork(nn.Module):
    """A-Network 核心模型"""

    # 26 邻域偏移量
    _NEIGHBORS_26 = [
        (-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1),
        (-1, -1, 0), (-1, 1, 0), (1, -1, 0), (1, 1, 0),
        (-1, 0, -1), (-1, 0, 1), (1, 0, -1), (1, 0, 1),
        (0, -1, -1), (0, -1, 1), (0, 1, -1), (0, 1, 1),
        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
    ]

    def __init__(self, cfg: ANetworkConfig, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.cfg = cfg
        self.device = device

        N = cfg.N
        H = cfg.hidden_dim
        V = cfg.vocab_size

        # ── 可学习参数 ──────────────────────────────────────

        # Embedding / LM Head（权值共享）
        self.embed_weight = nn.Parameter(torch.empty(V, H, device=device))

        # 投影入场
        self.in_weight = nn.Parameter(torch.empty(H, N, device=device))
        self.in_bias = nn.Parameter(torch.zeros(N, device=device))

        # 从场读出
        self.out_weight = nn.Parameter(torch.empty(H, N, device=device))
        self.out_bias = nn.Parameter(torch.zeros(N, device=device))

        # 传播权重（每个细胞一个标量）
        self.prop_weight = nn.Parameter(torch.empty(N, device=device))

        # ── 跳跃连接 ────────────────────────────────────────
        skip_src, skip_dst, skip_weight = self._build_skip_connections()
        if len(skip_src) > 0:
            self.register_buffer("skip_src", skip_src)    # [NC]
            self.register_buffer("skip_dst", skip_dst)    # [NC]
            self.skip_weight = nn.Parameter(skip_weight)  # [NC]
        else:
            self.register_buffer("skip_src", None)
            self.register_buffer("skip_dst", None)
            self.skip_weight = None

        # ── 26-邻域卷积核（conv3d 实现高效 gather） ──────────
        kernel = torch.ones(1, 1, 3, 3, 3, device=device)
        kernel[0, 0, 1, 1, 1] = 0.0  # 排除中心自身
        self.register_buffer("neighbor_kernel", kernel)

        # ── 权重初始化 ──────────────────────────────────────
        self._init_weights()

        # ── 场状态（训练时在 train_step 中管理） ────────────
        self.register_buffer("field", torch.zeros(N, device=device))
        self.register_buffer("incoming", torch.zeros(N, device=device))

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  权重初始化
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _init_weights(self):
        """按 C++ 版本的初始化策略初始化所有权重"""
        N, H, V = self.cfg.N, self.cfg.hidden_dim, self.cfg.vocab_size

        # Embedding: Uniform(-√(6/H), √(6/H))
        emb_scale = math.sqrt(6.0 / H)
        nn.init.uniform_(self.embed_weight, -emb_scale, emb_scale)

        # In weight: Uniform(-√(2/H), √(2/H))
        in_scale = math.sqrt(2.0 / H)
        nn.init.uniform_(self.in_weight, -in_scale, in_scale)
        # in_bias 初始为 0（默认）

        # Out weight: Uniform(-√(6/(H+N)), √(6/(H+N)))
        out_scale = math.sqrt(6.0 / (H + N))
        nn.init.uniform_(self.out_weight, -out_scale, out_scale)
        # out_bias 初始为 0（默认）

        # 传播权重: Uniform(-1/26, 1/26)
        prop_scale = 1.0 / 26.0
        nn.init.uniform_(self.prop_weight, -prop_scale, prop_scale)

        # 跳跃连接权重已由 _build_skip_connections 初始化

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  跳跃连接构建
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _is_too_close(a: int, b: int,
                      nx: int, ny: int, nz: int) -> bool:
        """判断两个细胞是否在曼哈顿距离 ≤1 以内"""
        if a == b:
            return True
        yz = ny * nz
        ax, ayz = a // yz, a % yz
        ay, az = ayz // nz, ayz % nz
        bx, byz = b // yz, b % yz
        by, bz = byz // nz, byz % nz
        return (abs(ax - bx) <= 1 and abs(ay - by) <= 1 and abs(az - bz) <= 1)

    def _build_skip_connections(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建随机长程跳跃连接（CSR → COO 格式）"""
        cfg = self.cfg
        N, density = cfg.N, cfg.skip_density

        if density <= 0.0:
            return torch.empty(0, dtype=torch.long, device=self.device), \
                   torch.empty(0, dtype=torch.long, device=self.device), \
                   torch.empty(0, device=self.device)

        edges = []  # [(src, dst), ...]
        rng = random.Random(42)  # 固定种子保证可复现

        for idx in range(N):
            if rng.random() >= density:
                continue
            for _ in range(20):  # 最多尝试 20 次
                tx = rng.randint(0, cfg.network_x - 1)
                ty = rng.randint(0, cfg.network_y - 1)
                tz = rng.randint(0, cfg.network_z - 1)
                target = tx * cfg.network_y * cfg.network_z + ty * cfg.network_z + tz
                if not self._is_too_close(idx, target,
                                          cfg.network_x, cfg.network_y, cfg.network_z):
                    edges.append((idx, target))
                    break

        if not edges:
            return torch.empty(0, dtype=torch.long, device=self.device), \
                   torch.empty(0, dtype=torch.long, device=self.device), \
                   torch.empty(0, device=self.device)

        # 按 src 排序
        edges.sort(key=lambda e: e[0])

        nc = len(edges)
        scale = 1.0 / 26.0
        rng_uniform = random.Random(43)

        skip_src = torch.tensor([e[0] for e in edges], dtype=torch.long, device=self.device)
        skip_dst = torch.tensor([e[1] for e in edges], dtype=torch.long, device=self.device)
        skip_weight = torch.tensor(
            [rng_uniform.uniform(-scale, scale) for _ in range(nc)],
            dtype=torch.float, device=self.device)

        print(f"Skip connections generated: {nc} (density={density * 100:.0f}%)")
        return skip_src, skip_dst, skip_weight

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  场状态管理
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def reset_state(self):
        """重置场状态（每个 epoch 开始调用）"""
        self.field.zero_()
        self.incoming.zero_()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  核心前向方法
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _inject(self, field: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """将隐藏向量注入到场中

        支持单条 (N,) 和 batch (B,N) 两种输入。

        Args:
            field: 当前场状态 [N] 或 [B, N]
            h: 隐藏向量 [H]（单条）或 [B, H]（batch）

        Returns:
            更新后的场
        """
        # injection = in_bias + h @ in_weight   -> (N,) or (B, N)
        injection = self.in_bias + h @ self.in_weight
        return field + injection

    def _propagate(self, field: torch.Tensor,
                   incoming: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """一步传播

        支持单条 (N,) 和 batch (B,N) 两种输入。
        conv3d 天然支持 batch 维，只需调整 reshape 维度即可。

        流程:
          1. field *= decay; field += incoming; incoming = 0
          2. act = tanh(field / 2)
          3. signal = 2 * act * prop_weight
          4. conv3d gather 26-邻域 → new_incoming
          5. index_add 跳跃连接 → new_incoming

        Args:
            field: [N] 或 [B, N]
            incoming: [N] 或 [B, N]

        Returns:
            (new_field, new_incoming)  形状与输入一致
        """
        cfg = self.cfg

        # Phase 1+2: 衰减 + 接收 + 清零 incoming + 计算激活
        new_field = field * cfg.time_decay + incoming
        act = torch.tanh(new_field * 0.5)

        # Phase 3: signal = 2 * act * prop_weight  广播自动处理 batch
        signal = 2.0 * act * self.prop_weight  # [N] or [B, N]

        # conv3d gather: reshape 为 (B, 1, X, Y, Z) 或 (1, 1, X, Y, Z)
        X, Y, Z = cfg.network_x, cfg.network_y, cfg.network_z
        ndim = signal.dim()  # 1=single, 2=batch
        signal_3d = signal.reshape(-1, 1, X, Y, Z) if ndim == 2 else signal.reshape(1, 1, X, Y, Z)
        new_incoming = F.conv3d(signal_3d, self.neighbor_kernel,
                                padding=1).reshape(signal.shape)

        # 跳跃连接：index_add 同时兼容 1D 和 2D
        if self.skip_src is not None and self.skip_src.numel() > 0:
            skip_signal = signal[..., self.skip_src] * self.skip_weight
            new_incoming.index_add_(-1, self.skip_dst, skip_signal)

        return new_field, new_incoming

    def _readout(self, field: torch.Tensor) -> torch.Tensor:
        """从场读出隐藏向量

        h = (field - out_bias) @ out_weight^T
        改写为矩阵乘形式以兼容 batch。

        Returns:
            h [H]（单条）或 [B, H]（batch）
        """
        diff = field - self.out_bias
        return diff @ self.out_weight.T  # [H] or [B, H]

    def _lm_head(self, h: torch.Tensor) -> torch.Tensor:
        """LM Head: 隐藏向量 → logits（权值共享）

        logits = h @ embed_weight^T
        改写为矩阵乘形式以兼容 batch。

        Returns:
            logits [V]（单条）或 [B, V]（batch）
        """
        return h @ self.embed_weight.T  # [V] or [B, V]

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  训练步
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def train_step(self, input_id: torch.Tensor,
                   target_id: torch.Tensor) -> torch.Tensor:
        """单序列训练步（保留兼容）

        内部委托给 train_step_batch, batch_size=1。

        Args:
            input_id: 标量 [0, V-1]
            target_id: 标量 [0, V-1]

        Returns:
            loss 标量
        """
        field = self.field.detach().unsqueeze(0)   # [1, N]
        incoming = torch.zeros_like(field)

        loss, new_field, _ = self.train_step_batch(
            field, incoming, input_id.unsqueeze(0), target_id.unsqueeze(0))

        # 恢复单序列状态
        with torch.no_grad():
            self.field.copy_(new_field[0])

        return loss

    def train_step_batch(self, fields: torch.Tensor,
                         incomings: torch.Tensor,
                         input_ids: torch.Tensor,
                         target_ids: torch.Tensor,
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched 训练步（前向 + 反向）

        多个独立序列共享网络参数，但各有独立的场状态。
        conv3d 和矩阵乘天然支持 batch，仅跳跃连接需注意。

        Args:
            fields: [B, N] 各序列的当前场状态
            incomings: [B, N] 各序列的 incoming 缓冲
            input_ids: [B] 各序列的当前 token
            target_ids: [B] 各序列的目标 token

        Returns:
            loss: 标量（batch 平均）
            new_fields: [B, N] 更新后的场状态（detach 后的）
            new_incomings: [B, N] 更新后的 incoming 缓冲
        """
        # 1. Embed → Inject
        h = self.embed_weight[input_ids]                  # [B, H]
        field = self._inject(fields, h)                   # [B, N]

        # 2. Propagate
        incoming = incomings
        for _ in range(self.cfg.prop_steps):
            field, incoming = self._propagate(field, incoming)

        # 3. Readout
        h_read = self._readout(field)                      # [B, H]

        # 4. LM Head
        logits = self._lm_head(h_read)                     # [B, V]

        # 5. Loss
        loss = F.cross_entropy(logits, target_ids)

        return loss, field.detach(), incoming.detach()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  生成
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @torch.no_grad()
    def generate_step(self, input_id: torch.Tensor) -> int:
        """单步自回归生成：注入 → 传播 → 读出 → argmax

        Args:
            input_id: 标量 [0, V-1]

        Returns:
            预测的下一 token id
        """
        cfg = self.cfg

        # 1. 注入
        h = self.embed_weight[input_id]
        self.field = self._inject(self.field, h)

        # 2. 传播（无梯度，直接用内部状态）
        for _ in range(cfg.prop_steps):
            self.field, self.incoming = self._propagate(self.field, self.incoming)

        # 3. 读出 → logits → argmax
        h_read = self._readout(self.field)
        logits = self._lm_head(h_read)
        return int(logits.argmax().item())

    @torch.no_grad()
    def generate(self, seed_ids: list, max_tokens: int = 100) -> list:
        """自回归生成

        Args:
            seed_ids: 种子 token ID 列表
            max_tokens: 最多生成多少 token

        Returns:
            生成的 token ID 列表（含种子）
        """
        self.reset_state()

        # 用前 N-1 个种子 token 构建场状态
        for tid in seed_ids[:-1]:
            self.generate_step(torch.tensor(tid, device=self.device))

        generated = list(seed_ids)

        # 自回归生成
        for _ in range(max_tokens):
            pred = self.generate_step(
                torch.tensor(generated[-1], device=self.device))
            generated.append(pred)
            if pred == 1:  # EOS
                break

        return generated

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #  序列化
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def save(self, path: str):
        """保存模型权重"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"Weights saved to {path}")

    def load(self, path: str):
        """加载模型权重"""
        path = Path(path)
        if not path.exists():
            print(f"ERROR: Cannot open {path}")
            return False
        state = torch.load(path, map_location=self.device,
                           weights_only=True)
        self.load_state_dict(state)
        print(f"Weights loaded from {path}")
        return True

    def dump_field(self, path: str):
        """将当前场状态保存为二进制文件（与 viz_field.py 兼容）"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.field.detach().cpu().numpy().astype("float32")
        data.tofile(str(path))
