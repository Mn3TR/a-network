"""
数据加载
========
仅支持 HuggingFace Datasets 在线加载。
"""

from .tokenizer import TokenizerWrapper


def load_data(dataset_source: str, tokenizer: TokenizerWrapper) -> list:
    """从 HuggingFace Datasets 加载并 tokenize

    格式: "用户名/数据集名" 或 "用户名/数据集名,split=名称"
    例如: "roneneldan/TinyStories" 或 "roneneldan/TinyStories,split=train"
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("需要安装 datasets 库: pip install datasets")

    # 解析 spec: "dataset_name" 或 "dataset_name,split=xxx"
    parts = dataset_source.split(",")
    ds_name = parts[0]
    split = None
    for p in parts[1:]:
        if p.startswith("split="):
            split = p.split("=", 1)[1]

    print(f"  Downloading '{ds_name}' (split={split or 'default'}) ...")
    ds = load_dataset(ds_name, split=split)
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
        for t in range(len(tokens) - 1):
            yield [tokens[t]], [tokens[t + 1]]
        return

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
