"""Bloque Transformer (pre-normalización) del modelo causal."""

from __future__ import annotations

import torch
import torch.nn as nn

from minigpt.attention import MultiHeadAttention
from minigpt.config import GPTConfig
from minigpt.layers import FeedForward, LayerNorm


class TransformerBlock(nn.Module):
    """Bloque de atención + *feed-forward* con conexiones residuales.

    Estructura pre-norm (normalizar antes de cada subcapa), idéntica a la de
    GPT-2.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.att = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        # Rama de atención con atajo / residual.
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, position_offset=position_offset)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Rama feed-forward con atajo / residual.
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
