"""Generación sintética de datos para los experimentos controlados."""

from __future__ import annotations

import torch


def shifted_copy_permutation(vocab_size: int, seed: int = 0) -> torch.Tensor:
    """Permutación fija del vocabulario usada por :func:`shifted_copy_sequences`.

    Se genera aparte para que entrenamiento, validación y test compartan la
    misma regla; solo cambian las secuencias iniciales.
    """
    if vocab_size < 2:
        raise ValueError("vocab_size debe ser >= 2")
    return torch.randperm(vocab_size, generator=torch.Generator().manual_seed(seed))


def shifted_copy_sequences(
    n_seqs: int,
    seq_len: int,
    vocab_size: int,
    k: int = 2,
    seed: int = 0,
    perm: torch.Tensor | None = None,
) -> torch.Tensor:
    """Secuencias de un proceso ``seq[i] = perm(seq[i-k])``.

    Cada token depende de otro situado ``k`` posiciones atrás (tras aplicar
    una permutación fija del vocabulario).  La estructura es 100 %
    predecible salvo los primeros ``k`` tokens, de modo que el modelo debe:

    1. atender a una distancia relativa ``k`` (capacidad de atención) y
    2. memorizar la permutación (capacidad del MLP / *feed-forward*).

    El proceso no depende de la posición absoluta, por lo que un modelo con
    posicionamiento relativo (RoPE) debería extrapolarse a longitudes no
    vistas sin deterioro, mientras que uno con posiciones absolutas
    aprendidas choca con filas de incrustación nunca entrenadas.

    Args:
        n_seqs: Número de secuencias.
        seq_len: Longitud de cada secuencia.
        vocab_size: Tamaño del vocabulario.
        k: Retardo de la dependencia.
        seed: Semilla de las secuencias iniciales (no de la permutación).
        perm: Permutación compartida; si es ``None`` se genera con ``seed``.

    Returns:
        Tensor entero de forma ``(n_seqs, seq_len)``.
    """
    if k < 1 or k >= seq_len:
        raise ValueError("k debe estar en [1, seq_len)")
    if vocab_size < 2:
        raise ValueError("vocab_size debe ser >= 2")

    generator = torch.Generator().manual_seed(seed)
    if perm is None:
        perm = torch.randperm(vocab_size, generator=torch.Generator().manual_seed(seed))
    if perm.shape != (vocab_size,):
        raise ValueError(f"perm debe tener forma ({vocab_size},)")

    seqs = torch.empty(n_seqs, seq_len, dtype=torch.long)
    seqs[:, :k] = torch.randint(vocab_size, (n_seqs, k), generator=generator)
    for i in range(k, seq_len):
        seqs[:, i] = perm[seqs[:, i - k]]
    return seqs


def train_val_split(
    n_seqs: int, val_fraction: float = 0.1, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reparte los índices ``[0, n_seqs)`` en entrenamiento y validación."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction debe estar en (0, 1)")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n_seqs, generator=generator)
    n_val = max(1, int(n_seqs * val_fraction))
    val_idx, train_idx = order[:n_val], order[n_val:]
    return train_idx, val_idx
