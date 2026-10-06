import torch

from src.model import GPTModel


# ============================================================
# Configuración pequeña para pruebas
# ============================================================

TEST_CONFIG = {
    "vocab_size": 50257,
    "context_length": 128,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 3,
    "drop_rate": 0.0,
    "qkv_bias": False
}


# ============================================================
# Función auxiliar para contar parámetros
# ============================================================

def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


# ============================================================
# 1. Crear modelo baseline
# ============================================================

torch.manual_seed(123)

baseline_model = GPTModel(
    TEST_CONFIG,
    use_rope=False
)


# ============================================================
# 2. Crear modelo RoPE
# ============================================================

torch.manual_seed(123)

rope_model = GPTModel(
    TEST_CONFIG,
    use_rope=True
)


# ============================================================
# 3. Crear entrada artificial
# ============================================================

torch.manual_seed(999)

input_ids = torch.randint(
    low=0,
    high=TEST_CONFIG["vocab_size"],
    size=(2, 16)
)

print("=" * 60)
print("ENTRADA")
print("=" * 60)

print("Shape input_ids:")
print(input_ids.shape)

print("\nEjemplo primera secuencia:")
print(input_ids[0])


# ============================================================
# 4. Forward baseline
# ============================================================

baseline_model.eval()

with torch.no_grad():
    baseline_logits = baseline_model(
        input_ids
    )


# ============================================================
# 5. Forward RoPE
# ============================================================

rope_model.eval()

with torch.no_grad():
    rope_logits = rope_model(
        input_ids
    )


# ============================================================
# 6. Comprobar dimensiones
# ============================================================

print("\n" + "=" * 60)
print("SHAPES DE SALIDA")
print("=" * 60)

print(
    "Baseline:",
    baseline_logits.shape
)

print(
    "RoPE:",
    rope_logits.shape
)

expected_shape = (
    2,
    16,
    TEST_CONFIG["vocab_size"]
)

assert baseline_logits.shape == expected_shape, (
    f"Shape incorrecta en baseline: "
    f"{baseline_logits.shape}"
)

assert rope_logits.shape == expected_shape, (
    f"Shape incorrecta en RoPE: "
    f"{rope_logits.shape}"
)

print("\nTEST SHAPES: OK")


# ============================================================
# 7. Comprobar que no aparecen NaN o Inf
# ============================================================

print("\n" + "=" * 60)
print("COMPROBACIÓN NUMÉRICA")
print("=" * 60)

baseline_has_nan = torch.isnan(
    baseline_logits
).any()

rope_has_nan = torch.isnan(
    rope_logits
).any()

baseline_has_inf = torch.isinf(
    baseline_logits
).any()

rope_has_inf = torch.isinf(
    rope_logits
).any()

print(
    "Baseline contiene NaN:",
    baseline_has_nan.item()
)

print(
    "RoPE contiene NaN:",
    rope_has_nan.item()
)

print(
    "Baseline contiene Inf:",
    baseline_has_inf.item()
)

print(
    "RoPE contiene Inf:",
    rope_has_inf.item()
)

assert not baseline_has_nan
assert not rope_has_nan
assert not baseline_has_inf
assert not rope_has_inf

print("\nTEST NaN / Inf: OK")


# ============================================================
# 8. Contar parámetros
# ============================================================

baseline_params = count_parameters(
    baseline_model
)

rope_params = count_parameters(
    rope_model
)

difference = (
    baseline_params
    - rope_params
)

expected_difference = (
    TEST_CONFIG["context_length"]
    *
    TEST_CONFIG["emb_dim"]
)

print("\n" + "=" * 60)
print("PARÁMETROS")
print("=" * 60)

print(
    "Baseline:",
    f"{baseline_params:,}"
)

print(
    "RoPE:",
    f"{rope_params:,}"
)

print(
    "Diferencia:",
    f"{difference:,}"
)

print(
    "Diferencia esperada:",
    f"{expected_difference:,}"
)

assert difference == expected_difference, (
    "La diferencia de parámetros no coincide "
    "con la tabla de embeddings posicionales."
)

print("\nTEST PARÁMETROS: OK")


# ============================================================
# 9. Comprobar que baseline tiene pos_emb
#    y RoPE no
# ============================================================

print("\n" + "=" * 60)
print("EMBEDDINGS POSICIONALES")
print("=" * 60)

baseline_has_pos_emb = hasattr(
    baseline_model,
    "pos_emb"
)

rope_has_pos_emb = hasattr(
    rope_model,
    "pos_emb"
)

print(
    "Baseline tiene pos_emb:",
    baseline_has_pos_emb
)

print(
    "RoPE tiene pos_emb:",
    rope_has_pos_emb
)

assert baseline_has_pos_emb is True
assert rope_has_pos_emb is False

print("\nTEST POS_EMB: OK")


# ============================================================
# 10. Comprobar buffers RoPE
# ============================================================

first_rope_attention = (
    rope_model
    .trf_blocks[0]
    .att
)

first_baseline_attention = (
    baseline_model
    .trf_blocks[0]
    .att
)

rope_has_cos = hasattr(
    first_rope_attention,
    "rope_cos"
)

rope_has_sin = hasattr(
    first_rope_attention,
    "rope_sin"
)

baseline_has_cos = hasattr(
    first_baseline_attention,
    "rope_cos"
)

baseline_has_sin = hasattr(
    first_baseline_attention,
    "rope_sin"
)

print("\n" + "=" * 60)
print("BUFFERS RoPE")
print("=" * 60)

print(
    "RoPE attention tiene cos:",
    rope_has_cos
)

print(
    "RoPE attention tiene sin:",
    rope_has_sin
)

print(
    "Baseline attention tiene cos:",
    baseline_has_cos
)

print(
    "Baseline attention tiene sin:",
    baseline_has_sin
)

assert rope_has_cos is True
assert rope_has_sin is True

assert baseline_has_cos is False
assert baseline_has_sin is False

print("\nTEST BUFFERS RoPE: OK")


# ============================================================
# 11. Comprobar que las salidas no son iguales
# ============================================================

outputs_are_equal = torch.allclose(
    baseline_logits,
    rope_logits,
    atol=1e-6
)

print("\n" + "=" * 60)
print("COMPARACIÓN SALIDAS")
print("=" * 60)

print(
    "¿Baseline y RoPE producen exactamente "
    "la misma salida?",
    outputs_are_equal
)

assert not outputs_are_equal, (
    "Las salidas son idénticas. "
    "Conviene revisar que RoPE realmente esté "
    "aplicándose y que el baseline use pos_emb."
)

print("\nTEST SALIDAS DIFERENTES: OK")


# ============================================================
# 12. Resumen final
# ============================================================

print("\n" + "=" * 60)
print("RESULTADO FINAL")
print("=" * 60)

print("✓ Forward baseline")
print("✓ Forward RoPE")
print("✓ Shapes correctas")
print("✓ Sin NaN")
print("✓ Sin Inf")
print("✓ Diferencia de parámetros correcta")
print("✓ Baseline usa embeddings posicionales")
print("✓ RoPE NO usa embeddings posicionales")
print("✓ Atención RoPE contiene cos/sin")
print("✓ Baseline no contiene buffers RoPE")
print("✓ Salidas baseline y RoPE son distintas")

print("\nTODOS LOS TESTS SUPERADOS")
print("=" * 60)