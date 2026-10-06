"""Capas básicas del transformer: normalización, activación y *feed-forward*.

Implementación didáctica equivalente a los capítulos 3-4 de referencia
(Raschka et al.), con la misma estructura de atributos que espera la carga de
pesos de GPT-2 (:mod:`minigpt.weights`).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from minigpt.config import GPTConfig


class LayerNorm(nn.Module):
    """Normalización de capa con desplazamiento y escala aprendibles.

    Equivalente a ``nn.LayerNorm(emb_dim)`` con ``eps=1e-5``, expuesto con los
    nombres ``scale``/``shift`` para asemejarlo a la notación del libro.
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift


class GELU(nn.Module):
    """Aproximación tangente hiperbólica de la GELU (usada por GPT-2)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    """Red *feed-forward* de dos capas con expansión 4x (MLP del bloque)."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            GELU(),
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
