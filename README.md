# Practica1_PLN1

Ampliación de la arquitectura de un modelo de lengua básico — **Rotary
Position Embeddings (RoPE)** — para la Práctica 1 de Procesamiento del
Lenguaje Natural I.

El proyecto compara dos esquemas posicionales sobre la misma arquitectura
transformer causal (GPT-2, 124 M):

| Variante | Descripción |
|----------|-------------|
| **Base** (`positional="learned"`) | Incrustaciones posicionales absolutas aprendidas (réplica del GPT-2 original). |
| **RoPE** (`positional="rope"`) | *Rotary Position Embeddings*: sin parámetros posicionales, la posición se inyecta rotando consultas y claves dentro de la atención. |

## Estructura

```text
.
├── minigpt/                  # Paquete Python con la implementación
│   ├── config.py             #   GPTConfig: hiperparámetros y esquema posicional
│   ├── layers.py             #   LayerNorm, GELU y FeedForward
│   ├── rope.py               #   Rotaciones RoPE y caché ampliable
│   ├── attention.py          #   Atención multi-cabeza causal (con RoPE opcional)
│   ├── blocks.py             #   Bloque Transformer pre-norm
│   ├── model.py              #   GPTModel completo
│   ├── generation.py         #   Decodificación voraz / muestreada
│   ├── weights.py            #   Descarga y transferencia de pesos de GPT-2
│   ├── data.py               #   Datos sintéticos reproducibles
│   ├── training.py           #   Bucle de entrenamiento mínimo
│   └── experiments.py        #   Métricas: perplejidad, rendimiento, invariancia…
├── data/corpus.txt           # Corpus propio para la evaluación de perplejidad
├── gpt2_weights/             # Pesos GPT-2 en local (ignorado por Git)
├── memoria/                  # Memoria LaTeX de la práctica
├── PLN_practica_1_RoPE.ipynb # Cuaderno experimental principal (entrega)
├── pyproject.toml            # Empaquetado del proyecto
└── requirements.txt          # Dependencias
```

## Uso

```bash
# 1. Entorno
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Ejecutar el cuaderno experimental
jupyter notebook PLN_practica_1_RoPE.ipynb
```

Los pesos de GPT-2 (475 MB) se descargan automáticamente la primera vez que
se ejecuta el cuaderno y se guardan en `gpt2_weights/` (no se versionan:
superan el límite de 100 MB de GitHub).

Uso del paquete desde Python:

```python
from minigpt import GPTConfig, GPTModel, generate, pretrained_gpt2

model = pretrained_gpt2(positional="rope")   # carga los pesos locales
logits = model(input_ids)
```

## Experimentos del cuaderno

1. Coste de parámetros de cada esquema posicional.
2. Generación cualitativa con los pesos preentrenados de GPT-2.
3. Invariancia a traslación (propiedad matemática de RoPE).
4. Carácter visual de ambos posicionamientos (PCA de `wpe` vs. tabla de rotación).
5. Rendimiento del paso hacia adelante.
6. Perplejidad *zero-shot* sobre el corpus propio.
7. Generalización a longitudes no vistas con entrenamiento controlado desde cero.

La implementación base se valida previamente contra la referencia de
`transformers` (diferencia máxima de logits ≈ 1e-4).

## Referencias

- Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*. arXiv:2104.09864.
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners*. OpenAI.
- Raschka, Liu & Mirjalili (2024). *Build a Large Language Model (From Scratch)*. Manning.
