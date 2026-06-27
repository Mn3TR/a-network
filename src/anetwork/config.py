"""
A-Network 超参数配置
====================
物理神经网络：3D 张量场作为可微分的「物理」计算介质。
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ANetworkConfig:
    """A-Network 专有超参"""
    # 张量维度
    network_x: int = 80
    network_y: int = 80
    network_z: int = 80
    hidden_dim: int = 48

    # 物理传播
    time_decay: float = 0.9
    prop_steps: int = 20

    # 跳跃连接
    skip_density: float = 0.20

    # 词表大小（加载 tokenizer 后更新）
    vocab_size: int = 128256

    @property
    def N(self) -> int:
        """场中细胞总数"""
        return self.network_x * self.network_y * self.network_z


@dataclass
class TrainConfig:
    """训练超参数"""
    # 梯度
    grad_accum: int = 4
    max_epochs: int = 100
    min_loss: float = 0.0

    # 优化器 (Adam)
    lr: float = 1e-4
    lr_min: float = 1e-6
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Batch
    batch_size: int = 1

    # 路径
    tokenizer_path: str = "tokenizer/tokenizer.json"
    data_dir: str = "dataset/"
    weights_path: str = "output/weights.pt"
    log_dir: str = "log/"

    # 数据集来源: "local" | "huggingface:数据集名" | "url:http://..."
    dataset_source: str = "local"

    # 生成
    gen_max_tokens: int = 100
    seed_text: str = "Time"
