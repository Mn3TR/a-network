"""
数据加载
========
直用 datasets.load_dataset()，不手写遍历。
"""

from datasets import load_dataset
from .tokenizer import TokenizerWrapper


def load_data(dataset_source: str, tokenizer: TokenizerWrapper,
              max_texts: int = 0) -> list:
    """下载 HuggingFace 数据集，拼接文本后一次性 tokenize

    支持格式:
      "用户名/数据集名"                          — 无 config
      "用户名/数据集名,config名"                  — 有 config
      "用户名/数据集名,config名,split=名称"       — config + split
      "用户名/数据集名,split=名称"                — 仅 split

    Args:
        dataset_source: 数据集描述
        tokenizer: TokenizerWrapper
        max_texts: 最多加载多少条文本（0=全部，TinyStories 建议 50000 防 OOM）
    """
    parts = dataset_source.split(",")
    ds_name = parts[0]
    config = None
    split = "train"
    for p in parts[1:]:
        if p.startswith("split="):
            split = p.split("=", 1)[1]
        elif p:
            config = p

    print(f"  Downloading '{ds_name}' (config={config}, split={split}) ...")
    ds = load_dataset(ds_name, config, split=split)
    text_col = "text" if "text" in ds.features else "content"

    # 截取子集避免 OOM
    texts = ds[text_col]
    if max_texts > 0 and len(texts) > max_texts:
        texts = texts[:max_texts]
        print(f"  Using first {max_texts}/{len(ds)} texts")

    all_text = "\n".join(texts)
    tokens = tokenizer.encode(all_text)
    print(f"  {len(texts)} texts, {len(tokens)} tokens")
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
