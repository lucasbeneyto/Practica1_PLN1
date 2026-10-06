"""Atención multi-cabeza causal, con posiciones aprendidas o con RoPE."""

from __future__ import annotations

import torch
import torch.nn as nn

from minigpt.config import GPTConfig
from minigpt.rope import RoPECache, apply_rotary


class MultiHeadAttention(nn.Module):
    """Atención multi-cabeza con máscara causal (autoregresiva).

    El esquema posicional se decide a partir de la configuración:

    * ``positional="learned"``: no hace nada aquí; la posición se suma antes
      de los bloques, como en GPT-2.
    * ``positional="rope"``: rota consultas y claves con :class:`RoPECache`.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        if cfg.emb_dim % cfg.n_heads != 0:
            raise ValueError("emb_dim debe ser divisible por n_heads")

        self.d_out = cfg.emb_dim
        self.num_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.context_length = cfg.context_length
        self.use_rope = cfg.uses_rope

        self.W_query = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_key = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.W_value = nn.Linear(cfg.emb_dim, cfg.emb_dim, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)
        self.dropout = nn.Dropout(cfg.drop_rate)

        # Máscara causal triangular superior registrada como buffer (no entrenable).
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg.context_length, cfg.context_length), diagonal=1),
        )
        self.rope_cache = (
            RoPECache(cfg.head_dim, cfg.context_length, cfg.rope_theta)
            if self.use_rope
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Proyecta ``x`` a consultas/claves/valores y calcula la atención.

        Args:
            x: Entrada de forma ``(batch, seq_len, emb_dim)``.
            position_offset: Índice de posición del primer token.  Permite
                evaluar la invariancia a traslación de RoPE y simular
                contextos que empiezan en una posición no nula.
            return_attn: Si es ``True``, devuelve también los pesos de
                atención (útiles para visualización).

        Returns:
            Vector de contexto ``(batch, seq_len, emb_dim)`` o
            ``(contexto, pesos_atencion)`` si ``return_attn=True``.
        """
        b, num_tokens, _ = x.shape
        if position_offset + num_tokens > self.context_length:
            raise IndexError(
                f"La secuencia (offset={position_offset} + longitud={num_tokens}) "
                f"supera context_length={self.context_length}"
            )

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # (batch, seq, emb) -> (batch, n_heads, seq, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rope_cache.get(
                num_tokens, position_offset, device=x.device, dtype=queries.dtype
            )
            queries = apply_rotary(queries, cos, sin)
            keys = apply_rotary(keys, cos, sin)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (self.head_dim**0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        if return_attn:
            return context_vec, attn_weights
        return context_vec
