"""Entrenamiento mínimo y reproducible para los experimentos controlados."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from minigpt.model import GPTModel


@dataclass
class TrainingConfig:
    """Hiperparámetros del bucle de entrenamiento.

    Attributes:
        max_steps: Número de pasos de optimización.
        batch_size: Secuencias por paso.
        lr: Tasa de aprendizaje del optimizador AdamW.
        eval_every: Frecuencia de evaluación en validación.
        ignore_first: Los primeros ``ignore_first`` tokens de cada secuencia
            no se usan como objetivo (son ruido impredecible por diseño).
        seed: Semilla del muestreo de lotes.
    """

    max_steps: int = 400
    batch_size: int = 32
    lr: float = 3e-4
    eval_every: int = 50
    ignore_first: int = 0
    seed: int = 123


@dataclass
class TrainingHistory:
    """Curvas de pérdida registradas durante el entrenamiento."""

    steps: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_steps: list[int] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    elapsed_seconds: float = 0.0


@torch.no_grad()
def next_token_loss(
    model: GPTModel,
    seqs: torch.Tensor,
    device: str | torch.device | None = None,
    batch_size: int = 64,
    ignore_first: int = 0,
) -> float:
    """Pérdida media de predicción del siguiente token sobre ``seqs``.

    Solo se puntúan los objetivos ``t >= max(ignore_first, 1)``.
    """
    if device is None:
        device = next(model.parameters()).device

    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_count = 0
    for start in range(0, seqs.shape[0], batch_size):
        batch = seqs[start : start + batch_size].to(device)
        logits = model(batch)
        nll = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            batch[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(batch.shape[0], -1)
        first = max(ignore_first, 1) - 1  # columna que corresponde al objetivo
        total_nll += nll[:, first:].sum().item()
        total_count += nll[:, first:].numel()

    if was_training:
        model.train()
    return total_nll / total_count


@torch.no_grad()
def _evaluate(model: GPTModel, seqs: torch.Tensor, cfg: TrainingConfig, device) -> float:
    return next_token_loss(
        model,
        seqs,
        device=device,
        ignore_first=cfg.ignore_first,
    )


def train_language_model(
    model: GPTModel,
    train_seqs: torch.Tensor,
    val_seqs: torch.Tensor,
    cfg: TrainingConfig | None = None,
    device: str | torch.device | None = None,
) -> TrainingHistory:
    """Entrena ``model`` sobre secuencias de longitud fija.

    Args:
        model: Modelo a entrenar (se pone en modo entrenamiento al final).
        train_seqs: Secuencias de entrenamiento, forma ``(n, seq_len)``.
        val_seqs: Secuencias de validación, misma longitud.
        cfg: Hiperparámetros; por defecto :class:`TrainingConfig`.
        device: Dispositivo de entrenamiento.

    Returns:
        Historial con las pérdidas y el tiempo empleado.
    """
    cfg = cfg or TrainingConfig()
    if device is None:
        device = next(model.parameters()).device

    generator = torch.Generator().manual_seed(cfg.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    history = TrainingHistory()
    model.to(device)
    model.train()
    start_time = time.perf_counter()

    for step in range(1, cfg.max_steps + 1):
        idx = torch.randint(
            train_seqs.shape[0], (cfg.batch_size,), generator=generator
        )
        batch = train_seqs[idx].to(device)

        logits = model(batch)
        nll = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            batch[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(batch.shape[0], -1)
        first = max(cfg.ignore_first, 1) - 1
        loss = nll[:, first:].mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            history.steps.append(step)
            history.train_losses.append(loss.item())
            history.val_steps.append(step)
            history.val_losses.append(_evaluate(model, val_seqs, cfg, device))
            model.train()

    history.elapsed_seconds = time.perf_counter() - start_time
    return history
