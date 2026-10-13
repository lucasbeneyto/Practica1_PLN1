"""Minimal LoRA wrapper for the project's own nn.Linear attention modules."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.05):
        super().__init__()
        if rank < 1:
            raise ValueError("rank debe ser positivo")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + update * (self.alpha / self.rank)


def inject_lora(model: nn.Module, targets: tuple[str, ...], rank: int, alpha: float, dropout: float) -> int:
    """Freeze backbone and wrap selected per-block Q/V (or Q/K/V) layers."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    count = 0
    for block in model.trf_blocks:
        for name in targets:
            if name not in {"query", "key", "value"}:
                raise ValueError(f"Objetivo LoRA no permitido: {name}")
            attr = f"W_{name}"
            setattr(block.att, attr, LoRALinear(getattr(block.att, attr), rank, alpha, dropout))
            count += 1
    return count


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()
            if "lora_A" in key or "lora_B" in key}


def load_adapter_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load adapter-only weights after injecting LoRA into the base model."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_adapters = [key for key in missing if "lora_A" in key or "lora_B" in key]
    if missing_adapters or unexpected:
        raise ValueError(f"Adaptador incompatible; faltan={missing_adapters}, sobran={unexpected}")
