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
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    seed: int | None = None,
    eos_id: int | None = None,
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
        top_p: Nucleus sampling opcional, entre 0 y 1.
        repetition_penalty: Penalización de tokens ya generados (>= 1).
        no_repeat_ngram_size: Evita repetir n-gramas de este tamaño.
        seed: Semilla opcional para el muestreo.
        eos_id: Token de fin opcional; al generarlo se detiene la secuencia.

    Returns:
        Secuencia completa ``(batch, seq_len + max_new_tokens)``.
    """
    if seed is not None:
        torch.manual_seed(seed)
    if temperature is not None and temperature <= 0:
        raise ValueError("temperature debe ser positivo")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k debe ser positivo")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p debe estar en (0, 1]")
    if repetition_penalty < 1:
        raise ValueError("repetition_penalty debe ser >= 1")
    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size no puede ser negativo")

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        logits = model(idx_cond)[:, -1, :]

        if repetition_penalty != 1:
            for row in range(idx.size(0)):
                seen = idx[row].unique()
                values = logits[row, seen]
                logits[row, seen] = torch.where(values < 0, values * repetition_penalty, values / repetition_penalty)
        if no_repeat_ngram_size and idx.size(1) >= no_repeat_ngram_size - 1:
            for row in range(idx.size(0)):
                prefix = idx[row, -(no_repeat_ngram_size - 1):].tolist() if no_repeat_ngram_size > 1 else []
                history = idx[row].tolist()
                for start in range(len(history) - no_repeat_ngram_size + 1):
                    ngram = history[start : start + no_repeat_ngram_size]
                    if not prefix or ngram[:-1] == prefix:
                        logits[row, ngram[-1]] = -torch.inf

        if temperature is None:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None:
                topk_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < topk_vals[:, [-1]], -torch.inf)
            if top_p is not None and top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[remove] = -torch.inf
                logits = torch.full_like(logits, -torch.inf).scatter(1, sorted_indices, sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)
        if eos_id is not None and torch.all(idx_next == eos_id):
            break

    return idx
