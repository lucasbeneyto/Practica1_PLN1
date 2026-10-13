"""Text corpus loading and fixed-length GPT-2 token batches."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset
import tiktoken


class TokenBlockDataset(Dataset):
    """Non-overlapping next-token examples encoded with the GPT-2 tokenizer."""

    def __init__(self, path: str | Path, context_length: int, encoding_name: str = "gpt2"):
        if context_length < 2:
            raise ValueError("context_length debe ser al menos 2")
        text = Path(path).read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"El corpus está vacío: {path}")
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text, allowed_special={"<|endoftext|>"})
        block = context_length + 1
        usable = len(tokens) // block * block
        if usable < block:
            raise ValueError(f"Se necesitan al menos {block} tokens en {path}; hay {len(tokens)}")
        self.tokens = torch.tensor(tokens[:usable], dtype=torch.long).view(-1, block)

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.tokens[index]
        return row[:-1], row[1:]
