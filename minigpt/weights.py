"""Descarga y transferencia de los pesos preentrenados de GPT-2.

Los pesos se guardan una única vez en ``gpt2_weights/`` (en la raíz del
proyecto) y se reutilizan en las siguientes ejecuciones.  Ese directorio está
ignorado por Git: los binarios superan el límite de 100 MB de GitHub.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from minigpt.config import GPTConfig, PositionalEncoding
from minigpt.model import GPTModel

#: Directorio local de los pesos (raíz del proyecto, sea cual sea el cwd).
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "gpt2_weights"


def load_gpt2_hf(weights_dir: Path | str | None = None, download: bool = True) -> GPT2LMHeadModel:
    """Carga el GPT-2 (124 M) de Hugging Face, descargándolo solo si hace falta.

    Args:
        weights_dir: Carpeta con los pesos locales.  Por defecto,
            ``gpt2_weights/`` en la raíz del proyecto.
        download: Si no hay pesos locales y es ``False``, lanza error en
            lugar de descargar.

    Returns:
        Modelo Hugging Face en modo evaluación.
    """
    weights_dir = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
    local_files = (
        weights_dir / "model.safetensors",
        weights_dir / "pytorch_model.bin",
    )
    if any(path.exists() for path in local_files):
        print(f"Cargando pesos de GPT-2 desde {weights_dir} ...")
        return GPT2LMHeadModel.from_pretrained(str(weights_dir))

    if not download:
        raise FileNotFoundError(
            f"No se encontraron pesos en {weights_dir} y download=False"
        )
    weights_dir.mkdir(parents=True, exist_ok=True)
    print(f"Descargando pesos oficiales de OpenAI GPT-2 y guardando en {weights_dir} ...")
    gpt_hf = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt_hf.save_pretrained(str(weights_dir))
    print("Pesos guardados en el directorio local con éxito.")
    return gpt_hf


def load_gpt2_weights(model: GPTModel, gpt_hf: GPT2LMHeadModel) -> GPTModel:
    """Transfiere los pesos de Hugging Face a un :class:`~minigpt.model.GPTModel`.

    La incrustación posicional aprendida (``wpe``) solo se copia si el modelo
    la tiene: la variante RoPE no dispone de esa capa y descarta esa
    información posicional de forma deliberada.
    """
    with torch.no_grad():
        model.tok_emb.weight.copy_(gpt_hf.transformer.wte.weight)
        if model.pos_emb is not None:
            model.pos_emb.weight.copy_(gpt_hf.transformer.wpe.weight)

        for i, block in enumerate(gpt_hf.transformer.h):
            mine = model.trf_blocks[i]

            mine.norm1.scale.copy_(block.ln_1.weight)
            mine.norm1.shift.copy_(block.ln_1.bias)

            # c_attn almacena Q, K y V juntos: (768, 2304) en formato Conv1D.
            w_qkv = block.attn.c_attn.weight.T
            b_qkv = block.attn.c_attn.bias
            w_q, w_k, w_v = torch.chunk(w_qkv, 3, dim=0)
            b_q, b_k, b_v = torch.chunk(b_qkv, 3, dim=0)
            mine.att.W_query.weight.copy_(w_q)
            mine.att.W_query.bias.copy_(b_q)
            mine.att.W_key.weight.copy_(w_k)
            mine.att.W_key.bias.copy_(b_k)
            mine.att.W_value.weight.copy_(w_v)
            mine.att.W_value.bias.copy_(b_v)

            mine.att.out_proj.weight.copy_(block.attn.c_proj.weight.T)
            mine.att.out_proj.bias.copy_(block.attn.c_proj.bias)

            mine.norm2.scale.copy_(block.ln_2.weight)
            mine.norm2.shift.copy_(block.ln_2.bias)

            mine.ff.layers[0].weight.copy_(block.mlp.c_fc.weight.T)
            mine.ff.layers[0].bias.copy_(block.mlp.c_fc.bias)
            mine.ff.layers[2].weight.copy_(block.mlp.c_proj.weight.T)
            mine.ff.layers[2].bias.copy_(block.mlp.c_proj.bias)

        model.final_norm.scale.copy_(gpt_hf.transformer.ln_f.weight)
        model.final_norm.shift.copy_(gpt_hf.transformer.ln_f.bias)
        model.out_head.weight.copy_(gpt_hf.lm_head.weight)

    return model


def pretrained_gpt2(
    positional: PositionalEncoding = "learned",
    device: str | torch.device | None = None,
    weights_dir: Path | str | None = None,
) -> GPTModel:
    """Construye un :class:`GPTModel` GPT-2 con los pesos oficiales cargados.

    Args:
        positional: ``"learned"`` (réplica fiel de GPT-2) o ``"rope"``
            (mismos pesos con RoPE, descartando la incrustación posicional).
        device: Dispositivo destino (por defecto, CUDA si está disponible).
        weights_dir: Carpeta con los pesos locales.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    gpt_hf = load_gpt2_hf(weights_dir).to(device).eval()
    model = GPTModel(GPTConfig.gpt2_124m(positional=positional))
    load_gpt2_weights(model, gpt_hf)
    model.to(device)
    model.eval()
    return model
