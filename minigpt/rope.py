"""Rotary Position Embeddings (RoPE).

RoPE (Su et al., 2021) codifica la posición rotando los vectores de consulta
y de clave en el plano bidimensional asociado a cada par de dimensiones de la
cabeza.  Para la posición :math:`m` y la frecuencia :math:`\\theta_i`:

.. math::

    \\mathbf{q}_m = R(\\Theta_m)\\mathbf{q},\\qquad
    \\Theta_m = m\\,\\theta_i,\\qquad
    \\theta_i = \\text{base}^{-2i/d}

De modo que el producto punto entre una consulta en posición :math:`m` y una
clave en posición :math:`n` solo depende de la diferencia :math:`m-n`
(invariancia a traslación) y no incorpora parámetros entrenables.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def inverse_frequencies(head_dim: int, theta: float = 10_000.0, device=None) -> torch.Tensor:
    """Frecuencias inversas ``θ_i = theta^(-2i/d)`` de cada plano rotacional."""
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim ({head_dim}) debe ser par para aplicar RoPE")
    indices = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    return 1.0 / (theta ** (indices / head_dim))


def rope_angles(
    seq_len: int,
    head_dim: int,
    theta: float = 10_000.0,
    offset: int = 0,
    device=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosenos y senos de rotación para las posiciones ``offset .. offset+seq_len``.

    Returns:
        ``(cos, sin)`` con forma ``(seq_len, head_dim)``.
    """
    inv_freq = inverse_frequencies(head_dim, theta, device=device)
    positions = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
    angles = torch.outer(positions, inv_freq)          # (seq_len, head_dim/2)
    angles = torch.cat((angles, angles), dim=-1)        # (seq_len, head_dim)
    return angles.cos(), angles.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotación de 90 grados: ``(-x2, x1)`` con ``x = (x1, x2)``."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Aplica la rotación RoPE a ``x``.

    Args:
        x: Tensores de forma ``(batch, n_heads, seq_len, head_dim)``.
        cos: Cosenos de forma ``(seq_len, head_dim)``.
        sin: Senos de forma ``(seq_len, head_dim)``.

    Returns:
        El tensor rotado con la misma forma que ``x``.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class RoPECache(nn.Module):
    """Caché de cosenos/senos reutilizable y ampliable dinámicamente.

    Se registra como submódulo del bloque de atención para que ``.to(device)``
    mueva los tensores junto con el resto de parámetros.  Si se solicitan
    posiciones más allá de la capacidad actual (extrapolación de contexto),
    la caché se recalcula automáticamente.
    """

    def __init__(self, head_dim: int, max_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        cos, sin = rope_angles(max_len, head_dim, theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @property
    def max_len(self) -> int:
        return self.cos.shape[0]

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self.max_len:
            return
        new_len = max(needed, 2 * self.max_len)
        cos, sin = rope_angles(new_len, self.head_dim, self.theta, device=self.cos.device)
        self.cos = cos
        self.sin = sin

    def get(
        self, seq_len: int, offset: int = 0, device=None, dtype=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Devuelve ``(cos, sin)`` para las posiciones ``offset .. offset+seq_len``."""
        self._ensure_capacity(offset + seq_len)
        cos = self.cos[offset : offset + seq_len]
        sin = self.sin[offset : offset + seq_len]
        if device is not None:
            cos, sin = cos.to(device), sin.to(device)
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos, sin
