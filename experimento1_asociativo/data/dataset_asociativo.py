"""Dataset sintético de *Associative Recall* para el Experimento 1.

Este módulo implementa la tarea del Experimento 1: memorización asociativa de
pares clave-valor y extrapolación a longitudes no vistas.  Se apoya en el
tokenizador BPE de GPT-2 (``tiktoken``) que ya usa en el proyecto, pero restringe
el vocabulario a los 41 tokens que aparecen realmente en la tarea.  De este
modo se conserva la tokenización real de GPT-2 y se obtiene un vocabulario
compacto que permite medir el *accuracy* exacto del token de respuesta.

Formato de una secuencia::

    " G: 4, T: 7, A: 2, Z: 9 QUERY A 2 ->"

Las claves son letras mayúsculas distintas y los valores dígitos.  El valor
de la consulta aparece **inmediatamente después** de la clave consultada; la
flecha ``->`` actúa como terminador de la secuencia.  Esta colocación es
deliberada: permite que una cabeza de inducción copie el token que sigue a la
aparición previa de la clave, sin necesidad de aprender un salto adicional a
través de un separador.  En el entrenamiento solo se puntúa el token de
respuesta (el valor), de forma que la tarea mide memoria asociativa pura y no
la predicción de contenido aleatorio de los pares.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch

import tiktoken

#: Claves posibles (26 letras mayúsculas).
KEYS: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
#: Valores posibles (dígitos decimales).
VALUES: tuple[str, ...] = tuple("0123456789")
#: Token de relleno a la derecha para agrupar secuencias de distinta longitud.
#: Se usa un id compacto propio, no un token de tiktoken.
PAD_TOKEN = "<pad>"

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parent
DATA_DIR = EXPERIMENT_DIR / "data"
WEIGHTS_ROOT = EXPERIMENT_DIR / "pesos"

#: Semillas fijas de todo el experimento.
DATA_SEED = 2026
MODEL_SEED = 42
TRAINING_SEED = 123

#: Número de pares de entrenamiento / validación in-distribution.
TRAIN_PAIRS = (4, 6)
#: Longitudes (en pares) de los conjuntos de extrapolación.
TEST_PAIR_LENGTHS = (6, 8, 12, 16, 20, 24)
#: Tamaños de los conjuntos.
N_TRAIN = 10_000
N_VAL = 1_000
N_TEST_PER_LENGTH = 1_000


class AssociativeTokenizer:
    """Tokenizador GPT-2 restringido al vocabulario de la tarea.

    Usa el BPE de GPT-2 de ``tiktoken`` y construye un mapa de los ids
    originales que aparecen en la tarea a índices compactos ``0..40``.  El
    índice ``41`` se reserva para el relleno.  El vocabulario resultante
    contiene exactamente las 26 claves, los 10 dígitos, la coma, los dos
    puntos, la flecha, las piezas ``" QU"``/``"ERY"`` y el relleno.
    """

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding("gpt2")

        original_ids: set[int] = set()
        # Claves y valores tal y como aparecen en el texto (con espacio
        # inicial, que el BPE absorbe en un único token).
        for key in KEYS:
            ids = self._encoding.encode(f" {key}")
            if len(ids) != 1:
                raise RuntimeError(f"La clave {key!r} no es un token único: {ids}")
            original_ids.update(ids)
        for value in VALUES:
            ids = self._encoding.encode(f" {value}")
            if len(ids) != 1:
                raise RuntimeError(f"El valor {value!r} no es un token único: {ids}")
            original_ids.update(ids)
        # Signos y prefijo de la consulta tal y como los produce el BPE en
        # una secuencia completa.
        for piece in (":", ",", " ->", " QU", "ERY"):
            original_ids.update(self._encoding.encode(piece))

        self.original_ids: list[int] = sorted(original_ids)
        self._original_to_compact = {
            original: compact for compact, original in enumerate(self.original_ids)
        }
        self.pad_id = len(self.original_ids)
        self.vocab_size = len(self.original_ids) + 1
        self.id_to_token = [
            self._encoding.decode([original]) for original in self.original_ids
        ] + [PAD_TOKEN]
        self.token_to_id = {token: i for i, token in enumerate(self.id_to_token)}

        # Id compacto del token de cada valor (siempre con espacio inicial).
        self.value_token_ids = {
            value: self.encode(f" {value}")[-1] for value in VALUES
        }
        # El token de la flecha: último token de " ->".
        self.arrow_token_id = self.encode(" ->")[-1]

    def encode(self, text: str) -> list[int]:
        """Tokeniza ``text`` y lo traduce al vocabulario compacto."""
        original = self._encoding.encode(text)
        try:
            return [self._original_to_compact[token] for token in original]
        except KeyError as exc:  # pragma: no cover - protección de cobertura
            piece = self._encoding.decode([exc.args[0]])
            raise ValueError(
                f"El token {piece!r} no pertenece al vocabulario compacto de la tarea"
            ) from exc

    def decode(self, ids: Sequence[int]) -> str:
        """Reconstruye el texto a partir de ids compactos (ignora el relleno)."""
        original = [
            self.original_ids[token]
            for token in ids
            if token != self.pad_id
        ]
        return self._encoding.decode(original)

    @property
    def vocab(self) -> list[str]:
        """Lista de tokens legibles en orden de id compacto."""
        return list(self.id_to_token)


@dataclass(frozen=True)
class RecallExample:
    """Un ejemplo de memoria asociativa."""

    text: str
    token_ids: tuple[int, ...]
    n_pairs: int
    query_key: str
    answer: str

    @property
    def answer_position(self) -> int:
        """Índice del token de respuesta (penúltimo: la flecha es el último)."""
        return len(self.token_ids) - 2

    @property
    def arrow_position(self) -> int:
        """Índice del token terminador ``->`` (siempre el último token)."""
        return len(self.token_ids) - 1

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "token_ids": list(self.token_ids),
            "n_pairs": self.n_pairs,
            "query_key": self.query_key,
            "answer": self.answer,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RecallExample":
        return cls(
            text=payload["text"],
            token_ids=tuple(payload["token_ids"]),
            n_pairs=payload["n_pairs"],
            query_key=payload["query_key"],
            answer=payload["answer"],
        )


def make_example(
    n_pairs: int,
    rng: random.Random,
    tokenizer: AssociativeTokenizer,
) -> RecallExample:
    """Genera un ejemplo con ``n_pairs`` pares clave-valor y una consulta."""
    if not 1 <= n_pairs <= len(KEYS):
        raise ValueError(f"n_pairs debe estar en [1, {len(KEYS)}]")

    keys = rng.sample(KEYS, n_pairs)
    values = [rng.choice(VALUES) for _ in range(n_pairs)]
    query_key = rng.choice(keys)
    answer = values[keys.index(query_key)]

    pairs_text = ", ".join(f"{key}: {value}" for key, value in zip(keys, values))
    # El valor sigue inmediatamente a la clave consultada; la flecha solo
    # termina la secuencia (no se interpone entre la clave y el valor).
    text = f" {pairs_text} QUERY {query_key} {answer} ->"
    token_ids = tokenizer.encode(text)

    expected_answer_token = tokenizer.value_token_ids[answer]
    if token_ids[-2] != expected_answer_token:
        raise RuntimeError(
            "El token de respuesta no coincide con el penúltimo token de la secuencia"
        )
    if token_ids[-1] != tokenizer.arrow_token_id:
        raise RuntimeError("El último token de la secuencia debe ser la flecha ' ->'")
    if len(token_ids) != 4 * n_pairs + 4:
        raise RuntimeError(
            f"Longitud inesperada: {len(token_ids)} tokens para {n_pairs} pares"
        )

    return RecallExample(
        text=text,
        token_ids=tuple(token_ids),
        n_pairs=n_pairs,
        query_key=query_key,
        answer=answer,
    )


def generate_recall_dataset(
    n_examples: int,
    min_pairs: int,
    max_pairs: int,
    seed: int,
    tokenizer: AssociativeTokenizer,
) -> list[RecallExample]:
    """Genera ``n_examples`` ejemplos con un número de pares en el rango dado."""
    if min_pairs > max_pairs:
        raise ValueError("min_pairs no puede ser mayor que max_pairs")
    rng = random.Random(seed)
    examples: list[RecallExample] = []
    for _ in range(n_examples):
        n_pairs = rng.randint(min_pairs, max_pairs)
        examples.append(make_example(n_pairs, rng, tokenizer))
    return examples


def save_examples(examples: Iterable[RecallExample], path: str | Path) -> None:
    """Guarda una lista de ejemplos en JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([example.to_dict() for example in examples], handle, ensure_ascii=False, indent=2)


def load_examples(path: str | Path) -> list[RecallExample]:
    """Carga una lista de ejemplos guardada con :func:`save_examples`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [RecallExample.from_dict(item) for item in payload]


def collate_examples(
    examples: Sequence[RecallExample],
    pad_id: int,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Agrupa ejemplos de distinta longitud con relleno a la derecha.

    Returns:
        ``(input_ids, answer_positions, answer_ids)``.  El relleno queda
        siempre después de la respuesta, por lo que la máscara causal impide
        que influya en la predicción.
    """
    max_len = max(len(example.token_ids) for example in examples)
    input_ids = torch.full(
        (len(examples), max_len), pad_id, dtype=torch.long, device=device
    )
    answer_positions = torch.empty(len(examples), dtype=torch.long, device=device)
    answer_ids = torch.empty(len(examples), dtype=torch.long, device=device)
    for row, example in enumerate(examples):
        ids = torch.tensor(example.token_ids, dtype=torch.long, device=device)
        input_ids[row, : ids.numel()] = ids
        answer_positions[row] = example.answer_position
        answer_ids[row] = example.token_ids[example.answer_position]
    return input_ids, answer_positions, answer_ids


def dataset_statistics(examples: Sequence[RecallExample]) -> dict[str, float]:
    """Estadísticas descriptivas de longitudes y pares."""
    lengths = [len(example.token_ids) for example in examples]
    pairs = [example.n_pairs for example in examples]
    return {
        "n_examples": len(examples),
        "min_pairs": min(pairs),
        "max_pairs": max(pairs),
        "mean_pairs": sum(pairs) / len(pairs),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
    }
