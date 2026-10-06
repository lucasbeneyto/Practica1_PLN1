"""Configuración de la arquitectura del modelo de lengua GPT.

Define :class:`GPTConfig`, el objeto que centraliza todos los hiperparámetros
de la arquitectura y el esquema de codificación posicional empleado
(``"learned"`` para las posiciones aprendidas absolutas de GPT-2 o ``"rope"``
para Rotary Position Embeddings).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

PositionalEncoding = Literal["learned", "rope"]

#: Esquemas posicionales admitidos.
POSITIONAL_ENCODINGS: tuple[str, ...] = ("learned", "rope")


@dataclass(frozen=True)
class GPTConfig:
    """Hiperparámetros del modelo causal.

    Attributes:
        vocab_size: Tamaño del vocabulario (BPE de GPT-2: 50257).
        context_length: Longitud máxima de contexto admitida por el modelo.
        emb_dim: Dimensión de las incrustaciones y de la red.
        n_heads: Número de cabezas de atención.
        n_layers: Número de bloques Transformer.
        drop_rate: Probabilidad de dropout.
        qkv_bias: Si se usa bias en las proyecciones de consulta, clave y valor.
        positional: Esquema posicional, ``"learned"`` (absoluto, entrenable)
            o ``"rope"`` (rotación relativa, sin parámetros).
        rope_theta: Base angular ``θ`` de RoPE (10000 en la publicación
            original; GPT-NeoX/Llama usan valores mayores).
    """

    vocab_size: int
    context_length: int
    emb_dim: int
    n_heads: int
    n_layers: int
    drop_rate: float
    qkv_bias: bool
    positional: PositionalEncoding = "learned"
    rope_theta: float = 10_000.0

    def __post_init__(self) -> None:
        if self.emb_dim % self.n_heads != 0:
            raise ValueError(
                f"emb_dim ({self.emb_dim}) debe ser divisible por n_heads ({self.n_heads})"
            )
        if self.positional not in POSITIONAL_ENCODINGS:
            raise ValueError(
                f"positional={self.positional!r} no es válido; use uno de {POSITIONAL_ENCODINGS}"
            )
        if self.context_length < 1:
            raise ValueError("context_length debe ser positivo")

    @property
    def head_dim(self) -> int:
        """Dimensión de cada cabeza de atención."""
        return self.emb_dim // self.n_heads

    @property
    def uses_rope(self) -> bool:
        """Indica si el modelo inyecta posición mediante RoPE."""
        return self.positional == "rope"

    def with_(self, **changes) -> "GPTConfig":
        """Devuelve una copia de la configuración con los campos indicados."""
        return replace(self, **changes)

    @classmethod
    def gpt2_124m(cls, positional: PositionalEncoding = "learned") -> "GPTConfig":
        """Configuración del GPT-2 pequeño (124 M de parámetros)."""
        return cls(
            vocab_size=50257,
            context_length=1024,
            emb_dim=768,
            n_heads=12,
            n_layers=12,
            drop_rate=0.1,
            qkv_bias=True,
            positional=positional,
        )

    @classmethod
    def tiny(cls, positional: PositionalEncoding = "learned", **overrides) -> "GPTConfig":
        """Configuración reducida para experimentos rápidos en CPU/GPU."""
        base = dict(
            vocab_size=32,
            context_length=256,
            emb_dim=64,
            n_heads=4,
            n_layers=2,
            drop_rate=0.0,
            qkv_bias=True,
            positional=positional,
        )
        base.update(overrides)
        return cls(**base)
