"""minigpt: transformer causal de la Práctica 1 de PLN I.

El paquete implementa un modelo GPT completo con dos esquemas posicionales
intercambiables:

* ``positional="learned"``: incrustaciones posicionales absolutas entrenables
  (réplica del GPT-2 original); y
* ``positional="rope"``: Rotary Position Embeddings, sin parámetros
  posicionales y con invariancia a traslación.

Uso básico::

    from minigpt import GPTConfig, GPTModel, generate, pretrained_gpt2

    model = pretrained_gpt2(positional="rope")
    logits = model(input_ids)
"""

from minigpt.attention import MultiHeadAttention
from minigpt.blocks import TransformerBlock
from minigpt.config import GPTConfig, PositionalEncoding
from minigpt.data import shifted_copy_permutation, shifted_copy_sequences, train_val_split
from minigpt.experiments import (
    count_parameters,
    format_parameters,
    forward_throughput,
    perplexity,
    translation_invariance_error,
)
from minigpt.generation import generate
from minigpt.layers import GELU, FeedForward, LayerNorm
from minigpt.model import GPTModel
from minigpt.rope import RoPECache, apply_rotary, inverse_frequencies, rope_angles, rotate_half
from minigpt.training import (
    TrainingConfig,
    TrainingHistory,
    next_token_loss,
    train_language_model,
)
from minigpt.weights import (
    DEFAULT_WEIGHTS_DIR,
    load_gpt2_hf,
    load_gpt2_weights,
    pretrained_gpt2,
)

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_WEIGHTS_DIR",
    "FeedForward",
    "GELU",
    "GPTConfig",
    "GPTModel",
    "LayerNorm",
    "MultiHeadAttention",
    "PositionalEncoding",
    "RoPECache",
    "TrainingConfig",
    "TrainingHistory",
    "TransformerBlock",
    "__version__",
    "apply_rotary",
    "count_parameters",
    "format_parameters",
    "forward_throughput",
    "generate",
    "inverse_frequencies",
    "load_gpt2_hf",
    "load_gpt2_weights",
    "next_token_loss",
    "perplexity",
    "pretrained_gpt2",
    "rope_angles",
    "rotate_half",
    "shifted_copy_permutation",
    "shifted_copy_sequences",
    "train_language_model",
    "train_val_split",
    "translation_invariance_error",
]
