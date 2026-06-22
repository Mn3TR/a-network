"""
数据加载
========
从 dataset/ 目录读取文本文件，然后用 tokenizer 编码。
"""

from pathlib import Path
from typing import Optional

from .tokenizer import TokenizerWrapper


def load_data(data_dir: str, tokenizer: TokenizerWrapper) -> list:
    """加载目录下所有 .txt 文件并 tokenize

    Args:
        data_dir: 数据集目录
        tokenizer: TokenizerWrapper 实例

    Returns:
        token ID 列表
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"WARNING: Data directory '{data_dir}' not found, using empty dataset.")
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

    print(f"Total tokens: {len(all_tokens)}")
    return all_tokens
