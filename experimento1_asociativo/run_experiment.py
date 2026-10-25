"""Ejecuta el Experimento 1 completo: datos, entrenamiento y evaluación.

Uso::

    python run_experiment.py

Genera los conjuntos sintéticos en ``data/``, entrena desde cero las dos
variantes posicionales (``learned`` y ``rope``) con idéntica inicialización y
guarda los pesos en ``experimento1_asociativo/exp1_associative_recall/``.  Finalmente
evalúa la extrapolación a longitudes no vistas y escribe un JSON con todos los
resultados, que el cuaderno consume para generar tablas y figuras.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

# Asegura que la raíz del proyecto esté en el path al ejecutar como script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

    histories: dict[str, data.RecallHistory] = {}
    evaluations: dict[str, dict] = {}
    training_times: dict[str, float] = {}

    for positional in ("learned", "rope"):
        print(f"\n=== Entrenando variante {positional} ===")
        model = models[positional]
        t0 = time.perf_counter()
        history, best_state = data.train_recall_model(
            model,
            train,
            val,
            tokenizer,
            data.RecallTrainingConfig(**TRAIN_KWARGS),
            device=device,
            verbose=True,
        )
        training_times[positional] = time.perf_counter() - t0
        histories[positional] = history

        weight_dir = data.WEIGHTS_ROOT / positional
        weight_dir.mkdir(parents=True, exist_ok=True)
        data.save_checkpoint(weight_dir / "best_model.pt", model, history, tokenizer)
        # Guarda también el historial en JSON legible.
        with (weight_dir / "training_history.json").open("w", encoding="utf-8") as handle:
            json.dump(asdict(history), handle, ensure_ascii=False, indent=2)

        val_metrics = data.evaluate_recall(model, val, tokenizer, device=device)
        length_metrics = {}
        for n_pairs in data.TEST_PAIR_LENGTHS:
            test_examples = datasets[str(n_pairs)]
            metrics = data.evaluate_recall(model, test_examples, tokenizer, device=device)
            length_metrics[str(n_pairs)] = metrics
            print(
                f"  extrapolación {n_pairs:2d} pares "
                f"({4 * n_pairs + 4:3d} tokens): "
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
        row = f"{n_pairs:>6} {4 * n_pairs + 4:>7}"
        for positional in ("learned", "rope"):
            accuracy = evaluations[positional]["by_length"][str(n_pairs)]["accuracy"]
            row += f" {accuracy:>9.2%}"
        print(row)

    return results


if __name__ == "__main__":
    run()
