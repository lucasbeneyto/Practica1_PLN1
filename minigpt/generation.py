"""Generación de texto autoregresiva (decodificación voraz o muestreo)."""

from __future__ import annotations

import torch


@torch.no_grad()
def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float | None = None,
    top_k: int | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    """Genera ``max_new_tokens`` tokens a partir de ``idx``.

    Por defecto (``temperature=None``) realiza decodificación voraz
    (argmax), igual que la función ``generate_text_simple`` de referencia.

    Args:
        model: Modelo causal con ``forward(idx) -> logits``.
        idx: Semilla de tokens, forma ``(batch, seq_len)``.
        max_new_tokens: Número de tokens a generar.
        context_size: Ventana máxima de contexto que se pasa al modelo.
        temperature: Si se indica, muestrea con ``logits / temperature``.
        top_k: Si se indica, restringe el muestreo a los ``k`` tokens más
            probables.
        seed: Semilla opcional para el muestreo.

    Returns:
        Secuencia completa ``(batch, seq_len + max_new_tokens)``.
    """
    if seed is not None:
        torch.manual_seed(seed)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)[:, -1, :]

        if temperature is None:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                topk_vals, _ = torch.topk(logits, top_k)
                logits = logits.masked_fill(logits < topk_vals[:, [-1]], -torch.inf)
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx
