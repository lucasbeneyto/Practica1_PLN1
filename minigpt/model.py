"""Modelo de lengua GPT con posicionamiento aprendido o RoPE."""

from __future__ import annotations

import torch
import torch.nn as nn

from minigpt.blocks import TransformerBlock
from minigpt.config import GPTConfig
from minigpt.layers import LayerNorm


class GPTModel(nn.Module):
    """Decodificador transformer causal (GPT).

    Con ``positional="learned"`` se suman incrustaciones posicionales
    entrenables (GPT-2); con ``positional="rope"`` no existe esa capa y la
    posición se inyecta dentro de la atención mediante RoPE, por lo que el
    modelo tiene menos parámetros y puede extrapolar a secuencias más largas
    que las vistas en entrenamiento.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

        # La incrustación posicional se inicializa la última: así, con la
        # misma semilla, las variantes "learned" y "rope" comparten
        # exactamente la inicialización de todos los pesos comunes.
        self.pos_emb = (
            nn.Embedding(cfg.context_length, cfg.emb_dim)
            if not cfg.uses_rope
            else None
        )
        if self.pos_emb is not None:
            # Mismo rango de inicialización que GPT-2 (N(0, 0.02)).
            nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    @property
    def positional_parameters(self) -> int:
        """Número de parámetros dedicados a la codificación posicional."""
        if self.pos_emb is None:
            return 0
        return self.pos_emb.weight.numel()

    def forward(
        self,
        in_idx: torch.Tensor,
        position_offset: int = 0,
        return_attns: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Calcula los logits para la secuencia ``in_idx``.

        Args:
            in_idx: Índices de tokens, forma ``(batch, seq_len)``.
            position_offset: Posición del primer token de la ventana.
            return_attns: Si es ``True``, devuelve también los pesos de
                atención de cada bloque.

        Returns:
            Logits ``(batch, seq_len, vocab_size)`` o ``(logits, attns)``.
        """
        _, seq_len = in_idx.shape
        if position_offset + seq_len > self.cfg.context_length:
            raise IndexError(
                f"La secuencia (offset={position_offset} + longitud={seq_len}) "
                f"supera context_length={self.cfg.context_length}"
            )

        x = self.tok_emb(in_idx)
        if self.pos_emb is not None:
            positions = torch.arange(
                position_offset, position_offset + seq_len, device=in_idx.device
            )
            x = x + self.pos_emb(positions)
        x = self.drop_emb(x)

        attns: list[torch.Tensor] = []
        for block in self.trf_blocks:
            if return_attns:
                # La atención solo está disponible en la primera capa de cada
                # bloque; se captura interceptando la salida del módulo.
                out, attn = block.att(
                    block.norm1(x),
                    position_offset=position_offset,
                    return_attn=True,
                )
                out = block.drop_shortcut(out)
                x = x + out
                attns.append(attn)
                shortcut = x
                x = block.drop_shortcut(block.ff(block.norm2(x))) + shortcut
            else:
                x = block(x, position_offset=position_offset)

        x = self.final_norm(x)
        logits = self.out_head(x)

        if return_attns:
            return logits, attns
        return logits
