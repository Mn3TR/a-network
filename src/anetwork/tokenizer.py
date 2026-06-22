"""
HuggingFace Tokenizer 包装
==========================
使用 tokenizers 库加载 HuggingFace 格式的 tokenizer.json，
替换原有的手写 C++ BPE tokenizer。
"""

from pathlib import Path
from tokenizers import Tokenizer as HFTokenizer


class TokenizerWrapper:
    """HuggingFace tokenizer 的一层薄包装"""

    def __init__(self, path: str = "tokenizer/tokenizer.json"):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {path}")
        self.tokenizer = HFTokenizer.from_file(str(path))
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list:
        """将文本编码为 token ID 列表"""
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list) -> str:
        """将 token ID 列表解码为文本"""
        return self.tokenizer.decode(ids)

    def id_to_token(self, token_id: int) -> str:
        """将 token ID 转为字符串表示"""
        return self.tokenizer.id_to_token(token_id)

    def token_to_id(self, token: str) -> int:
        """将字符串转为 token ID"""
        return self.tokenizer.token_to_id(token)
