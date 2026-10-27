"""Ejecuta el Experimento 1 completo: datos, entrenamiento y evaluación.

Uso::

    python scripts/entrenamiento_exp1.py

Genera los conjuntos sintéticos en ``data/``, entrena desde cero las dos
variantes posicionales (``learned`` y ``rope``) con idéntica inicialización y
guarda los pesos en ``pesos/learned/`` y ``pesos/rope/``.  Finalmente evalúa la
extrapolación a longitudes no vistas y escribe un JSON con todos los resultados,
que el cuaderno consume para generar tablas y figuras.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

# La raíz del proyecto y la carpeta de datos deben estar en el path para
# importar ``minigpt`` y el módulo de dataset respectivamente.
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_DIR.parent
DATA_DIR = EXPERIMENT_DIR / "data"
WEIGHTS_ROOT = EXPERIMENT_DIR / "pesos"
for path in (PROJECT_ROOT, DATA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import dataset_asociativo as data  # noqa: E402
from minigpt.config import GPTConfig  # noqa: E402
from minigpt.model import GPTModel  # noqa: E402

#: Configuración de arquitectura común a ambos modelos.
MODEL_KWARGS = dict(
    context_length=256,
    emb_dim=128,
    n_heads=4,
    n_layers=4,
    drop_rate=0.0,
    qkv_bias=True,
)

#: Configuración de entrenamiento común.
TRAIN_KWARGS = dict(
    max_steps=4000,
    batch_size=64,
    lr=1e-3,
    weight_decay=0.01,
    grad_clip=1.0,
    warmup_steps=100,
    eval_every=100,
    target_accuracy=0.98,
    patience_evals=3,
    seed=data.TRAINING_SEED,
)


def set_global_seed(seed: int) -> None:
    """Fija las semillas de Python, NumPy (si existe) y PyTorch."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_datasets(tokenizer: data.AssociativeTokenizer) -> dict[str, list[data.RecallExample]]:
    """Genera y persiste todos los conjuntos del experimento."""
    data.DATA_DIR.mkdir(parents=True, exist_ok=True)

    train = data.generate_recall_dataset(
        data.N_TRAIN, data.TRAIN_PAIRS[0], data.TRAIN_PAIRS[1], data.DATA_SEED, tokenizer
    )
    val = data.generate_recall_dataset(
        data.N_VAL,
        data.TRAIN_PAIRS[0],
        data.TRAIN_PAIRS[1],
        data.DATA_SEED + 1,
        tokenizer,
    )
    data.save_examples(train, data.DATA_DIR / "train.json")
    data.save_examples(val, data.DATA_DIR / "val.json")

    tests: dict[str, list[data.RecallExample]] = {}
    for index, n_pairs in enumerate(data.TEST_PAIR_LENGTHS):
        examples = data.generate_recall_dataset(
            data.N_TEST_PER_LENGTH,
            n_pairs,
            n_pairs,
            data.DATA_SEED + 100 + index,
            tokenizer,
        )
        data.save_examples(examples, data.DATA_DIR / f"test_pairs_{n_pairs}.json")
        tests[n_pairs] = examples

    metadata = {
        "seeds": {
            "data": data.DATA_SEED,
            "model": data.MODEL_SEED,
            "training": data.TRAINING_SEED,
        },
        "train_pairs_range": list(data.TRAIN_PAIRS),
        "test_pair_lengths": list(data.TEST_PAIR_LENGTHS),
        "sizes": {
            "train": len(train),
            "val": len(val),
            "test_per_length": data.N_TEST_PER_LENGTH,
        },
        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
            "id_to_token": tokenizer.id_to_token,
        },
        "statistics": {
            "train": data.dataset_statistics(train),
            "val": data.dataset_statistics(val),
            **{
                f"test_pairs_{n_pairs}": data.dataset_statistics(examples)
                for n_pairs, examples in tests.items()
            },
        },
    }
    with (data.DATA_DIR / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    return {"train": train, "val": val, **{str(n): examples for n, examples in tests.items()}}


def common_initialization_check(
    tokenizer: data.AssociativeTokenizer, device: torch.device
) -> dict[str, float | bool]:
    """Comprueba que ambas variantes parten de los mismos pesos comunes."""
    configs = {}
    models = {}
    for positional in ("learned", "rope"):
        set_global_seed(data.MODEL_SEED)
        cfg = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            positional=positional,
            **MODEL_KWARGS,
        )
        configs[positional] = cfg
        models[positional] = GPTModel(cfg)

    learned_state = models["learned"].state_dict()
    rope_state = models["rope"].state_dict()
    common_keys = sorted(set(learned_state) & set(rope_state))
    max_diff = max(
        (learned_state[key] - rope_state[key]).abs().max().item()
        for key in common_keys
    )
    return {
        "n_common_tensors": len(common_keys),
        "max_abs_diff": max_diff,
        "identical": max_diff == 0.0,
    }


def answer_logits(
    model: GPTModel,
    input_ids: torch.Tensor,
    answer_positions: torch.Tensor,
) -> torch.Tensor:
    """Logits del token de respuesta para cada secuencia del lote."""
    logits = model(input_ids)
    batch = torch.arange(input_ids.shape[0], device=input_ids.device)
    # El logit en la posición ``answer_position - 1`` predice el token
    # situado en ``answer_position``.
    return logits[batch, answer_positions - 1]


@torch.no_grad()
def evaluate_recall(
    model: GPTModel,
    examples: Sequence[data.RecallExample],
    tokenizer: data.AssociativeTokenizer,
    batch_size: int = 256,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    """Evalúa pérdida y *accuracy* exacto en el token de respuesta."""
    if device is None:
        device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        input_ids, positions, targets = data.collate_examples(
            batch, tokenizer.pad_id, device=device
        )
        logits = answer_logits(model, input_ids, positions)
        loss = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == targets).sum().item()
        total += targets.numel()

    if was_training:
        model.train()
    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "n_examples": float(total),
    }


@dataclass
class RecallTrainingConfig:
    """Hiperparámetros del bucle de entrenamiento del Experimento 1."""

    max_steps: int = 6_000
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 100
    eval_every: int = 100
    target_accuracy: float = 0.99
    patience_evals: int = 4
    seed: int = data.TRAINING_SEED


@dataclass
class RecallHistory:
    """Curvas y metadatos de un entrenamiento."""

    steps: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    train_accuracies: list[float] = field(default_factory=list)
    val_steps: list[int] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accuracies: list[float] = field(default_factory=list)
    best_step: int = 0
    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0.0
    elapsed_seconds: float = 0.0
    stopped_early: bool = False
    steps_run: int = 0


def _lr_lambda(step: int, config: RecallTrainingConfig) -> float:
    """Calentamiento lineal + decaimiento coseno hasta el 10 % del LR."""
    if step < config.warmup_steps:
        return (step + 1) / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(
        config.max_steps - config.warmup_steps, 1
    )
    progress = min(max(progress, 0.0), 1.0)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train_recall_model(
    model: GPTModel,
    train_examples: Sequence[data.RecallExample],
    val_examples: Sequence[data.RecallExample],
    tokenizer: data.AssociativeTokenizer,
    config: RecallTrainingConfig | None = None,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> tuple[RecallHistory, dict[str, torch.Tensor]]:
    """Entrena un modelo y devuelve el historial y el mejor estado de pesos."""
    config = config or RecallTrainingConfig()
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_lambda(step, config)
    )
    generator = torch.Generator().manual_seed(config.seed)

    history = RecallHistory()
    best_state = _clone_state_dict(model)
    best_loss = float("inf")
    best_accuracy = 0.0
    best_step = 0
    early_counter = 0
    start_time = time.perf_counter()

    for step in range(1, config.max_steps + 1):
        indices = torch.randint(
            len(train_examples), (config.batch_size,), generator=generator
        ).tolist()
        batch = [train_examples[index] for index in indices]
        input_ids, positions, targets = data.collate_examples(
            batch, tokenizer.pad_id, device=device
        )

        logits = answer_logits(model, input_ids, positions)
        loss = F.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if config.grad_clip is not None and config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        scheduler.step()

        if step % config.eval_every == 0 or step == config.max_steps:
            val_metrics = evaluate_recall(
                model, val_examples, tokenizer, device=device
            )
            history.steps.append(step)
            history.train_losses.append(loss.item())
            history.train_accuracies.append(accuracy)
            history.val_steps.append(step)
            history.val_losses.append(val_metrics["loss"])
            history.val_accuracies.append(val_metrics["accuracy"])

            improved = val_metrics["accuracy"] > best_accuracy or (
                val_metrics["accuracy"] == best_accuracy
                and val_metrics["loss"] < best_loss
            )
            if improved:
                best_accuracy = val_metrics["accuracy"]
                best_loss = val_metrics["loss"]
                best_step = step
                best_state = _clone_state_dict(model)
                early_counter = 0
            if val_metrics["accuracy"] >= config.target_accuracy:
                early_counter += 1
            else:
                early_counter = 0

            if verbose:
                print(
                    f"  paso {step:>5d} | train loss {loss.item():.4f} "
                    f"acc {accuracy:5.1%} | val loss {val_metrics['loss']:.4f} "
                    f"acc {val_metrics['accuracy']:6.2%}"
                )

            if early_counter >= config.patience_evals:
                history.stopped_early = True
                if verbose:
                    print(
                        f"  parada temprana: val acc >= {config.target_accuracy:.1%} "
                        f"durante {config.patience_evals} evaluaciones"
                    )
                break

    history.best_step = best_step
    history.best_val_loss = best_loss
    history.best_val_accuracy = best_accuracy
    history.elapsed_seconds = time.perf_counter() - start_time
    history.steps_run = step

    # Restaura el mejor estado encontrado (según validación).
    model.load_state_dict(best_state)
    return history, best_state


def save_checkpoint(
    path: str | Path,
    model: GPTModel,
    history: RecallHistory,
    tokenizer: data.AssociativeTokenizer,
) -> None:
    """Guarda pesos, configuración e historial en un único ``.pt``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "config": asdict(model.cfg),
        "history": asdict(history),
        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "pad_id": tokenizer.pad_id,
            "id_to_token": tokenizer.id_to_token,
        },
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[GPTModel, RecallHistory, dict]:
    """Carga un checkpoint guardado con :func:`save_checkpoint`."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(path, map_location=device, weights_only=False)
    model = GPTModel(GPTConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    history_payload = payload["history"]
    history = RecallHistory(**history_payload)
    return model, history, payload


def run() -> dict:
    """Entrena ambos modelos, evalúa y devuelve el informe completo."""
    torch.backends.cudnn.benchmark = False
    set_global_seed(data.MODEL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = data.AssociativeTokenizer()
    print(f"Dispositivo: {device}")
    print(f"Vocabulario compacto: {tokenizer.vocab_size} tokens")

    print("Generando conjuntos sintéticos ...")
    datasets = build_datasets(tokenizer)
    train = datasets["train"]
    val = datasets["val"]

    init_check = common_initialization_check(tokenizer, device)
    print(
        "Inicialización común: "
        f"{init_check['n_common_tensors']} tensores, "
        f"diferencia máxima = {init_check['max_abs_diff']:.2e}"
    )

    # Construye ambos modelos con la misma semilla (pesos comunes idénticos).
    models: dict[str, GPTModel] = {}
    configs: dict[str, GPTConfig] = {}
    for positional in ("learned", "rope"):
        set_global_seed(data.MODEL_SEED)
        cfg = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            positional=positional,
            **MODEL_KWARGS,
        )
        configs[positional] = cfg
        models[positional] = GPTModel(cfg)

    histories: dict[str, RecallHistory] = {}
    evaluations: dict[str, dict] = {}
    training_times: dict[str, float] = {}

    for positional in ("learned", "rope"):
        print(f"\n=== Entrenando variante {positional} ===")
        model = models[positional]
        t0 = time.perf_counter()
        history, best_state = train_recall_model(
            model,
            train,
            val,
            tokenizer,
            RecallTrainingConfig(**TRAIN_KWARGS),
            device=device,
            verbose=True,
        )
        training_times[positional] = time.perf_counter() - t0
        histories[positional] = history

        weight_dir = data.WEIGHTS_ROOT / positional
        weight_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(weight_dir / "best_model.pt", model, history, tokenizer)
        # Guarda también el historial en JSON legible.
        with (weight_dir / "training_history.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(history), handle, ensure_ascii=False, indent=2)

        val_metrics = evaluate_recall(model, val, tokenizer, device=device)
        length_metrics = {}
        for n_pairs in data.TEST_PAIR_LENGTHS:
            test_examples = datasets[str(n_pairs)]
            metrics = evaluate_recall(model, test_examples, tokenizer, device=device)
            length_metrics[str(n_pairs)] = metrics
            print(
                f"  extrapolación {n_pairs:2d} pares "
                f"({4 * n_pairs + 5:3d} tokens): "
                f"loss {metrics['loss']:.4f} | acc {metrics['accuracy']:.2%}"
            )

        evaluations[positional] = {
            "val": val_metrics,
            "by_length": length_metrics,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "positional_parameters": model.positional_parameters,
            "training_seconds": training_times[positional],
        }
        print(
            f"  resumen {positional}: val acc {val_metrics['accuracy']:.2%} | "
            f"mejor paso {history.best_step} | "
            f"{training_times[positional]:.1f} s"
        )

    results = {
        "device": str(device),
        "model_config": {
            positional: asdict(config) for positional, config in configs.items()
        },
        "training_config": TRAIN_KWARGS,
        "initialization_check": init_check,
        "histories": {
            positional: asdict(history) for positional, history in histories.items()
        },
        "evaluations": evaluations,
        "dataset_metadata": {
            "train_size": len(train),
            "val_size": len(val),
            "test_per_length": data.N_TEST_PER_LENGTH,
            "test_pair_lengths": list(data.TEST_PAIR_LENGTHS),
            "train_pairs_range": list(data.TRAIN_PAIRS),
        },
    }

    results_path = data.EXPERIMENT_DIR / "resultados_experimento1.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"\nResultados guardados en {results_path}")

    print("\nTabla final de accuracy en el token de respuesta")
    header = f"{'pares':>6} {'tokens':>7} {'learned':>10} {'rope':>10}"
    print(header)
    print("-" * len(header))
    for n_pairs in data.TEST_PAIR_LENGTHS:
        row = f"{n_pairs:>6} {4 * n_pairs + 5:>7}"
        for positional in ("learned", "rope"):
            accuracy = evaluations[positional]["by_length"][str(n_pairs)]["accuracy"]
            row += f" {accuracy:>9.2%}"
        print(row)

    return results


if __name__ == "__main__":
    run()
