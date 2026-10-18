"""Memory-bounded GPT-2 text tokenization and memory-mapped token blocks."""

from __future__ import annotations

import hashlib
import os
from array import array
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
import tiktoken


def _cache_path(path: Path, cache_dir: Path) -> Path:
    stat = path.stat()
    identity = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    digest = hashlib.sha256(identity).hexdigest()[:16]
    return cache_dir / f"{path.stem}-{digest}.u16"


def _build_token_cache(path: Path, cache_path: Path, encoding_name: str) -> None:
    """Tokenize line-delimited articles in small batches, never loading all text."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Hay una caché parcial que debe revisarse: {temporary}")

    encoding = tiktoken.get_encoding(encoding_name)
    batch_size = 256
    processed = 0
    with path.open("r", encoding="utf-8") as source, temporary.open("wb") as target:
        while True:
            texts = []
            for _ in range(batch_size):
                line = source.readline()
                if not line:
                    break
                text = line.strip()
                if text:
                    texts.append(text)
            if not texts:
                break

            encoded = encoding.encode_ordinary_batch(texts)
            token_ids = array("H")
            for article_tokens in encoded:
                if any(token >= 65536 for token in article_tokens):
                    raise ValueError("El vocabulario excede la capacidad de la caché uint16")
                token_ids.extend(article_tokens)
                token_ids.append(encoding.eot_token)
            token_ids.tofile(target)
            processed += len(texts)
            if processed % 10000 < batch_size:
                target.flush()
                print(f"Tokenizados {processed:,} artículos: {path.name}", flush=True)

        target.flush()
        os.fsync(target.fileno())

    if temporary.stat().st_size < 4:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"No se encontraron suficientes tokens en {path}")
    os.replace(temporary, cache_path)
    print(f"Caché de tokens creada: {cache_path} ({cache_path.stat().st_size:,} bytes)", flush=True)


class TokenBlockDataset(Dataset):
    """Fixed-length next-token blocks backed by a compact uint16 mmap cache.

    Articles are separated with GPT-2's end-of-text token. ``block_range`` is
    expressed in block indices and enables disjoint train/validation slices.
    """

    def __init__(
        self,
        path: str | Path,
        context_length: int,
        encoding_name: str = "gpt2",
        block_range: tuple[int, int] | None = None,
        cache_dir: str | Path = "gpt2_weights/token_cache",
    ) -> None:
        if context_length < 2:
            raise ValueError("context_length debe ser al menos 2")
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.block_size = context_length + 1
        cache_path = _cache_path(self.path, Path(cache_dir))
        if not cache_path.exists():
            _build_token_cache(self.path, cache_path, encoding_name)
        token_count = cache_path.stat().st_size // np.dtype(np.uint16).itemsize
        self.total_blocks = token_count // self.block_size
        if self.total_blocks < 2:
            raise ValueError(f"Se necesitan al menos dos bloques de {self.block_size} tokens en {self.path}")
        self.start_block, stop_block = block_range or (0, self.total_blocks)
        if not (0 <= self.start_block < stop_block <= self.total_blocks):
            raise ValueError(f"Rango de bloques inválido: {self.start_block}:{stop_block}/{self.total_blocks}")
        self.stop_block = stop_block
        self.tokens = np.memmap(cache_path, mode="r", dtype=np.uint16)

    def __len__(self) -> int:
        return self.stop_block - self.start_block

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = (self.start_block + index) * self.block_size
        row = self.tokens[start : start + self.block_size].astype(np.int64)
        return torch.from_numpy(row[:-1]), torch.from_numpy(row[1:])
