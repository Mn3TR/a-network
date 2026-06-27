"""
数据加载
========
直用 datasets.load_dataset()，不手写遍历。
"""

from datasets import load_dataset
from .tokenizer import TokenizerWrapper


def load_data(dataset_source: str, tokenizer: TokenizerWrapper) -> list:
    """下载 HuggingFace 数据集，拼接文本后一次性 tokenize"""
    print(f"  Downloading '{dataset_source}' ...")
    ds = load_dataset(dataset_source, split="train")
    text_col = "text" if "text" in ds.features else "content"
    all_text = "\n".join(ds[text_col])
    tokens = tokenizer.encode(all_text)
    print(f"  {len(ds)} texts, {len(tokens)} tokens")
    return tokens


def pack_batch(tokens: list, batch_size: int):
    """将 token 序列打包为 batch 数据"""
    B = batch_size
    if B <= 1 or len(tokens) < B + 1:
        for t in range(len(tokens) - 1):
            yield [tokens[t]], [tokens[t + 1]]
        return
    chunk_len = len(tokens) // B
    for step in range(chunk_len - 1):
        inputs = [tokens[B * b + step] for b in range(B)]
        targets = [tokens[B * b + step + 1] for b in range(B)]
        yield inputs, targets


def count_batch_steps(tokens: list, batch_size: int) -> int:
    if batch_size <= 1:
        return len(tokens) - 1
    return len(tokens) // batch_size - 1
