"""CLI para fine-tuning full o LoRA de GPT-2 adaptado con RoPE (FP16)."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from minigpt.checkpoints import atomic_save
from minigpt.config import GPTConfig
from minigpt.finetune_data import TokenBlockDataset
from minigpt.lora import adapter_state_dict, inject_lora
from minigpt.model import GPTModel
from minigpt.weights import DEFAULT_WEIGHTS_DIR, load_gpt2_hf, load_gpt2_weights


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "lora"), required=True)
    parser.add_argument("--train", type=Path, default=Path("data/example_train.txt"))
    parser.add_argument("--valid", type=Path, default=Path("data/example_valid.txt"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--targets", choices=("qv", "qkv"), default="qv")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def _evaluate(model, loader, device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), reduction="sum")
            total += loss.item()
            count += targets.numel()
    model.train()
    return total / count


def main() -> None:
    args = _args()
    if not torch.cuda.is_available():
        raise RuntimeError("Este flujo requiere CUDA compatible con FP16; no se cambia silenciosamente a FP32/BF16.")
    if min(args.steps, args.grad_accum, args.batch_size, args.eval_every, args.save_every) < 1:
        raise ValueError("steps, grad-accum, batch-size, eval-every y save-every deben ser positivos")
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    cfg = GPTConfig.gpt2_124m(positional="rope").with_(context_length=args.context_length)
    train_ds = TokenBlockDataset(args.train, args.context_length)
    valid_ds = TokenBlockDataset(args.valid, args.context_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False)

    hf_model = load_gpt2_hf(args.weights_dir).to(device).eval()
    model = GPTModel(cfg)
    load_gpt2_weights(model, hf_model)
    del hf_model
    model.to(device)
    targets = ("query", "value") if args.targets == "qv" else ("query", "key", "value")
    if args.mode == "lora":
        inject_lora(model, targets, args.rank, args.alpha, args.lora_dropout)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr or (2e-5 if args.mode == "full" else 1e-4),
                                  betas=(0.9, 0.95), weight_decay=0.1)
    optimizer_steps = math.ceil(args.steps / args.grad_accum)
    warmup = max(1, int(optimizer_steps * 0.04))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, 1.0) *
        (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(step, optimizer_steps) / optimizer_steps)))
    )
    scaler = torch.cuda.amp.GradScaler()
    output = args.output or Path("gpt2_weights") / args.mode / time.strftime("rope-gpt2-%Y%m%d-%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    config_payload = vars(args).copy()
    config_payload["train"] = str(args.train)
    config_payload["valid"] = str(args.valid)
    best = float("inf")
    step = 0
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
        # Optimización y scheduler avanzan cada número de microbatches acumulados.
        if (step + 1) % args.grad_accum == 0 or step + 1 == args.steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        step += 1
        if step % args.eval_every == 0 or step == args.steps:
            val_loss = _evaluate(model, valid_loader, device)
            print(f"step={step} train_loss={loss.item():.4f} val_loss={val_loss:.4f} val_ppl={math.exp(min(val_loss, 80)):.3f}")
            is_best = val_loss < best
            best = min(best, val_loss)
            if args.mode == "lora":
                weights = adapter_state_dict(model)
            else:
                weights = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            payload = {"model_state_dict" if args.mode == "full" else "adapter_state_dict": weights,
                       "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                       "scaler_state_dict": scaler.state_dict(), "step": step, "val_loss": val_loss,
                       "config": config_payload, "model_config": cfg.__dict__}
            if is_best:
                atomic_save(payload, output / "best.pt")
            if step % args.save_every == 0 or step == args.steps:
                atomic_save(payload, output / f"checkpoint-step-{step:06d}.pt")
    if args.mode == "lora":
        import json
        (output / "adapter_config.json").write_text(json.dumps({"rank": args.rank, "alpha": args.alpha,
            "dropout": args.lora_dropout, "targets": targets, "base_weights": str(args.weights_dir),
            "positional": "rope", "context_length": args.context_length}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
