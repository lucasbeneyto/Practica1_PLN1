"""FP16 CLI for full or LoRA fine-tuning of GPT-2 with RoPE."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from minigpt.checkpoints import atomic_save
from minigpt.config import GPTConfig
from minigpt.finetune_data import TokenBlockDataset
from minigpt.lora import adapter_state_dict, inject_lora, load_adapter_state_dict
from minigpt.model import GPTModel
from minigpt.weights import DEFAULT_WEIGHTS_DIR, load_gpt2_hf, load_gpt2_weights


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "lora"), required=True)
    parser.add_argument("--train", type=Path, default=Path("data/dacsa_spanish_articles_train.txt"))
    parser.add_argument("--valid", type=Path, help="Corpus aparte de validación; si se omite, se reserva una fracción del train")
    parser.add_argument("--valid-fraction", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--resume", type=Path, help="Checkpoint previo para continuar")
    parser.add_argument("--steps", type=int, default=2000, help="Microbatches de entrenamiento (no actualizaciones del optimizador)")
    parser.add_argument("--epochs", type=int, help="Entrenar una cantidad de épocas completas en vez de usar --steps")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=100)
    parser.add_argument("--early-stopping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=8, help="Evaluaciones sin mejora antes de parar")
    parser.add_argument("--min-delta", type=float, default=0.005, help="Mejora mínima de val_loss para resetear patience")
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--targets", choices=("qv", "qkv"), default="qv")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def _evaluate(model, loader, device, max_batches: int) -> float:
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for batch_index, (inputs, targets) in enumerate(loader):
            if batch_index >= max_batches:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="sum")
            total += loss.item()
            count += targets.numel()
    if was_training:
        model.train()
    if count == 0:
        raise ValueError("El conjunto de validación no contiene bloques")
    return total / count


def _make_datasets(args: argparse.Namespace):
    train_all = TokenBlockDataset(args.train, args.context_length)
    if args.valid is not None:
        if args.valid.resolve() == args.train.resolve():
            raise ValueError("--valid debe ser distinto de --train; use --valid-fraction para dividir un solo corpus")
        train_ds = train_all
        valid_ds = TokenBlockDataset(args.valid, args.context_length)
    else:
        if not 0.0 < args.valid_fraction < 0.5:
            raise ValueError("valid-fraction debe estar entre 0 y 0.5")
        boundary = int(train_all.total_blocks * (1.0 - args.valid_fraction))
        if boundary < 1 or boundary >= train_all.total_blocks:
            raise ValueError("No hay suficientes bloques para separar entrenamiento y validación")
        train_ds = TokenBlockDataset(args.train, args.context_length, block_range=(0, boundary))
        valid_ds = TokenBlockDataset(args.train, args.context_length, block_range=(boundary, train_all.total_blocks))
        del train_all
    return train_ds, valid_ds


def main() -> None:
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere CUDA compatible con FP16; no se cambia silenciosamente a FP32/BF16.")
    if min(args.steps, args.grad_accum, args.batch_size, args.eval_every, args.eval_batches, args.save_every, args.patience) < 1:
        raise ValueError("steps, grad-accum, batch-size, eval-every, eval-batches, save-every y patience deben ser positivos")
    if args.min_delta < 0:
        raise ValueError("min-delta no puede ser negativo")
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs debe ser positivo")
    if args.context_length < 2 or args.context_length > 1024:
        raise ValueError("context-length debe estar entre 2 y 1024 para este GPT-2")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)

    train_ds, valid_ds = _make_datasets(args)
    if args.epochs is not None:
        args.steps = args.epochs * math.ceil(len(train_ds) / args.batch_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Bloques: train={len(train_ds):,}, valid={len(valid_ds):,}; microbatches={args.steps:,}; contexto={args.context_length}", flush=True)

    cfg = GPTConfig.gpt2_124m(positional="rope").with_(context_length=args.context_length)
    # Load and transfer on CPU so the Hugging Face and minigpt copies do not
    # coexist in GPU memory during initialization.
    hf_model = load_gpt2_hf(args.weights_dir).cpu().eval()
    model = GPTModel(cfg)
    load_gpt2_weights(model, hf_model)
    del hf_model
    model.to(device)
    model.gradient_checkpointing = args.gradient_checkpointing

    targets = ("query", "value") if args.targets == "qv" else ("query", "key", "value")
    if args.mode == "lora":
        inject_lora(model, targets, args.rank, args.alpha, args.lora_dropout)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    no_decay = ("bias", "scale", "shift", "lora_A", "lora_B")
    groups = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(k in n for k in no_decay)], "weight_decay": 0.1},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    lr = args.lr or (2e-5 if args.mode == "full" else 1e-4)
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
    optimizer_steps = math.ceil(args.steps / args.grad_accum)
    warmup = max(1, int(optimizer_steps * 0.04))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda update: min((update + 1) / warmup, 1.0)
        * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(update, optimizer_steps) / optimizer_steps))),
    )
    scaler = torch.amp.GradScaler("cuda")
    output = args.output or Path("gpt2_weights") / args.mode / time.strftime("rope-gpt2-%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    config_payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}

    start_step, best, bad_evals = 0, float("inf"), 0
    if args.resume:
        checkpoint_data = torch.load(args.resume, map_location="cpu", weights_only=False)
        if args.mode == "lora":
            load_adapter_state_dict(model, checkpoint_data["adapter_state_dict"])
        else:
            model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint_data["scaler_state_dict"])
        start_step = int(checkpoint_data["step"])
        best = float(checkpoint_data.get("best_val_loss", checkpoint_data.get("val_loss", best)))
        bad_evals = int(checkpoint_data.get("bad_evals", 0))
        print(f"Checkpoint reanudado desde el paso {start_step}", flush=True)
    if start_step >= args.steps:
        raise ValueError(f"El checkpoint ya llegó al paso {start_step}; --steps debe ser mayor")

    step, last_loss = start_step, float("nan")
    microbatches_since_update = 0
    next_checkpoint_step = (start_step // args.save_every + 1) * args.save_every
    optimizer.zero_grad(set_to_none=True)
    iterator = iter(train_loader)
    while step < args.steps:
        model.train()
        try:
            inputs, targets_batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            inputs, targets_batch = next(iterator)
        inputs, targets_batch = inputs.to(device), targets_batch.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets_batch.reshape(-1))
            scaled_loss = loss / args.grad_accum
        scaler.scale(scaled_loss).backward()
        last_loss = loss.item()
        microbatches_since_update += 1
        optimizer_update = microbatches_since_update >= args.grad_accum or step + 1 == args.steps
        if optimizer_update:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() >= old_scale:
                scheduler.step()
            microbatches_since_update = 0
        step += 1

        save_due = optimizer_update and step >= next_checkpoint_step
        if optimizer_update and (step % args.eval_every == 0 or save_due or step == args.steps):
            val_loss = _evaluate(model, valid_loader, device, args.eval_batches)
            is_best = val_loss < best
            significant_improvement = val_loss < best - args.min_delta
            if is_best:
                best = val_loss
            if significant_improvement:
                bad_evals = 0
            else:
                bad_evals += 1
            stop_early = args.early_stopping and bad_evals >= args.patience
            print(
                f"step={step} train_loss={last_loss:.4f} val_loss={val_loss:.4f} "
                f"best={best:.4f} plateau={bad_evals}/{args.patience} "
                f"val_ppl={math.exp(min(val_loss, 80)):.3f}",
                flush=True,
            )
            weights = adapter_state_dict(model) if args.mode == "lora" else {k: v.detach().cpu() for k, v in model.state_dict().items()}
            payload = {
                "model_state_dict" if args.mode == "full" else "adapter_state_dict": weights,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "step": step,
                "val_loss": val_loss,
                "best_val_loss": best,
                "bad_evals": bad_evals,
                "config": config_payload,
                "model_config": cfg.__dict__,
            }
            if is_best:
                atomic_save(payload, output / "best.pt")
            if save_due or step == args.steps or stop_early:
                atomic_save(payload, output / f"checkpoint-step-{step:06d}.pt")
                next_checkpoint_step = (step // args.save_every + 1) * args.save_every
            if stop_early:
                print(f"Early stopping: val_loss no mejoró al menos {args.min_delta} durante {bad_evals} evaluaciones; mejor={best:.4f}, detenido en step={step}.", flush=True)
                break

    if args.mode == "lora":
        (output / "adapter_config.json").write_text(
            json.dumps({"rank": args.rank, "alpha": args.alpha, "dropout": args.lora_dropout,
                        "targets": targets, "base_weights": str(args.weights_dir),
                        "positional": "rope", "context_length": args.context_length}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
