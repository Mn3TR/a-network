"""
数据加载
========
支持三种数据来源:
  - local:       从 dataset/ 目录读取 .txt 文件
  - huggingface: 从 HuggingFace Datasets 加载
  - url:         从 URL 下载文本文件

同时提供 batch 打包工具。
"""

from pathlib import Path
from typing import Optional
import tempfile
import urllib.request

from .tokenizer import TokenizerWrapper


def _load_local(data_dir: str, tokenizer: TokenizerWrapper) -> list:
    """加载目录下所有 .txt 文件并 tokenize"""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"WARNING: Data directory '{data_dir}' not found.")
        return []

    all_tokens = []
    files = sorted(data_path.glob("*.txt"))

    if not files:
        print(f"WARNING: No .txt files in '{data_dir}'.")
        return []

    for file in files:
        text = file.read_text(encoding="utf-8", errors="replace")
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        print(f"  Loaded {file.name}: {len(tokens)} tokens")

    return all_tokens


def _load_huggingface(dataset_spec: str, tokenizer: TokenizerWrapper) -> list:
    """从 HuggingFace Datasets 加载

    dataset_spec 格式: "用户名/数据集名" 或 "用户名/数据集名,split=名称"
    例如: "roneneldan/TinyStories" 或 "roneneldan/TinyStories,split=train"
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets 未安装。运行: pip install datasets")

    # 解析 spec: "dataset_name" 或 "dataset_name,split=xxx"
    parts = dataset_spec.split(",")
    ds_name = parts[0]
    split = None
    for p in parts[1:]:
        if p.startswith("split="):
            split = p.split("=", 1)[1]

    print(f"  Downloading HuggingFace dataset '{ds_name}' (split={split or 'default'}) ...")
    ds = load_dataset(ds_name, split=split)
    # 取文本列: 通常为 'text' 或 'content'
    text_col = "text" if "text" in ds.features else "content"
    texts = ds[text_col]

    total_tokens = []
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(str(text))
        total_tokens.extend(tokens)
        if (i + 1) % 1000 == 0:
            print(f"    Processed {i+1}/{len(texts)} texts")

    print(f"  Loaded {len(texts)} texts, {len(total_tokens)} tokens")
    return total_tokens


def _load_url(url: str, tokenizer: TokenizerWrapper) -> list:
    """从 URL 下载文本文件并 tokenize"""
    print(f"  Downloading from '{url}' ...")
    with urllib.request.urlopen(url, timeout=30) as response:
        text = response.read().decode("utf-8", errors="replace")
    tokens = tokenizer.encode(text)
    print(f"  Loaded {len(tokens)} tokens from URL")
    return tokens


def load_data(data_dir: str, tokenizer: TokenizerWrapper,
              dataset_source: str = "local") -> list:
    """加载数据集

    Args:
        data_dir: 本地数据目录（local 模式使用）
        tokenizer: TokenizerWrapper 实例
        dataset_source: "local" | "huggingface:数据集名" | "url:http://..."

    Returns:
        token ID 列表
    """
    if dataset_source == "local":
        return _load_local(data_dir, tokenizer)

    if dataset_source.startswith("huggingface:"):
        spec = dataset_source[len("huggingface:"):]
        return _load_huggingface(spec, tokenizer)

    if dataset_source.startswith("url:"):
        url = dataset_source[len("url:"):]
        return _load_url(url, tokenizer)

    print(f"WARNING: Unknown dataset_source '{dataset_source}', falling back to local.")
    return _load_local(data_dir, tokenizer)


def pack_batch(tokens: list, batch_size: int):
    """将 token 序列打包为 batch 数据

    将 tokens 切分为 batch_size 个连续的 chunk，
    每个 chunk 是一个独立序列（各有自己的场状态）。

    Args:
        tokens: token ID 列表
        batch_size: 并行序列数

    Yields:
        (input_ids, target_ids): 各为 [B] 的 LongTensor
    """
    B = batch_size
    if B <= 1 or len(tokens) < B + 1:
        # 退化为单序列
        for t in range(len(tokens) - 1):
            yield [tokens[t]], [tokens[t + 1]]
        return

    # 切分为 B 个等长 chunk
    chunk_len = len(tokens) // B

    for step in range(chunk_len - 1):
        inputs = [tokens[B * b + step] for b in range(B)]
        targets = [tokens[B * b + step + 1] for b in range(B)]
        yield inputs, targets


def count_batch_steps(tokens: list, batch_size: int) -> int:
    """计算 batch 模式下的总步数"""
    if batch_size <= 1:
        return len(tokens) - 1
    return len(tokens) // batch_size - 1
