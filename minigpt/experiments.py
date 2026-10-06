"""Métricas de evaluación y comparación entre variantes del modelo."""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from minigpt.model import GPTModel


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Número de parámetros del modelo (entrenables por defecto)."""
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def format_parameters(n: int) -> str:
    """Formatea un número de parámetros con puntos de millar (estilo ES)."""
    return f"{n:,}".replace(",", ".")


@torch.no_grad()
def perplexity(
    model: GPTModel,
    tokens: torch.Tensor,
    context_length: int | None = None,
    stride: int | None = None,
    device: str | torch.device | None = None,
) -> float:
    """Perplejidad ``exp(H)`` sobre una secuencia de tokens.

    Se recorren ventanas deslizantes de ``context_length`` tokens con paso
    ``stride`` (por defecto, la mitad de la ventana).  Cada token se evalúa
    exactamente una vez y siempre con la mayor cantidad de contexto
    izquierdo disponible: a cada ventana se le asignan solo los tokens que
    aún no han sido puntuados por ventanas anteriores.

    Args:
        model: Modelo en modo evaluación.
        tokens: Secuencia 1D de índices de tokens.
        context_length: Ventana de contexto (por defecto, la del modelo).
        stride: Paso entre ventanas; debe ser menor que la ventana.
        device: Dispositivo de evaluación.
    """
    if device is None:
        device = next(model.parameters()).device
    if context_length is None:
        context_length = model.cfg.context_length
    context_length = min(context_length, model.cfg.context_length)
    stride = stride or max(context_length // 2, 1)
    if stride >= context_length:
        raise ValueError("stride debe ser menor que context_length")

    tokens = tokens.to(device)
    n_tokens = tokens.numel()
    if n_tokens < 2:
        raise ValueError("Se necesitan al menos 2 tokens para evaluar")

    total_nll = 0.0
    total_count = 0
    next_target = 1  # primer token sin evaluar (el índice 0 no tiene previo)

    for start in range(0, n_tokens, stride):
        end = min(start + context_length, n_tokens)
        window = tokens[start:end].unsqueeze(0)
        if window.shape[1] < 2:
            break

        logits = model(window)
        # nll[i] corresponde al objetivo tokens[start + 1 + i].
        nll = F.cross_entropy(
            logits[0, :-1].float(),
            window[0, 1:],
            reduction="none",
        )
        first = max(0, next_target - (start + 1))
        if first < nll.numel():
            total_nll += nll[first:].sum().item()
            total_count += nll.numel() - first
            next_target = start + 1 + nll.numel()

        if end == n_tokens:
            break

    if total_count == 0:
        raise ValueError("No se evaluó ningún token")
    return float(torch.exp(torch.tensor(total_nll / total_count)))


@torch.no_grad()
def forward_throughput(
    model: GPTModel,
    seq_len: int,
    batch_size: int = 1,
    device: str | torch.device | None = None,
    warmup: int = 20,
    iters: int = 50,
    rounds: int = 5,
) -> float:
    """Rendimiento del paso hacia adelante en tokens/segundo.

    Repite la medición ``rounds`` veces y devuelve el **mejor** resultado
    (menor tiempo), que es el estimador menos sensible al ruido del sistema
    (relojes dinámicos de la GPU, tareas en segundo plano).
    """
    device = torch.device(device) if device is not None else next(model.parameters()).device
    if seq_len > model.cfg.context_length:
        raise ValueError(
            f"seq_len={seq_len} supera context_length={model.cfg.context_length}"
        )

    was_training = model.training
    model.eval()
    x = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    best_elapsed = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        best_elapsed = min(best_elapsed, time.perf_counter() - start)

    if was_training:
        model.train()
    return batch_size * seq_len * iters / best_elapsed


@torch.no_grad()
def translation_invariance_error(
    model: GPTModel,
    input_ids: torch.Tensor,
    offset: int = 16,
) -> float:
    """Error máximo de logits al desplazar la ventana de posiciones.

    Para RoPE los logits deben ser prácticamente idénticos (la atención solo
    depende de la distancia relativa entre posiciones); para el
    posicionamiento aprendido absoluto cambian.
    """
    was_training = model.training
    model.eval()
    logits_0 = model(input_ids, position_offset=0)
    logits_k = model(input_ids, position_offset=offset)
    if was_training:
        model.train()
    return (logits_0 - logits_k).abs().max().item()
