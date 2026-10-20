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

## Fine-tuning RoPE (FP16)

El pipeline usa por defecto `data/dacsa_spanish_articles_train.txt`. Tokeniza
los artículos por lotes, añade el token de fin de texto entre artículos y
guarda una caché compacta mapeada en memoria bajo `gpt2_weights/token_cache/`;
no carga el corpus completo en RAM. Si no se pasa `--valid`, reserva el último
1% de bloques del corpus para validación. Los corpus `data/example_*.txt` se
pueden usar solo para una prueba rápida.

El entrenamiento requiere CUDA compatible con FP16 y no cambia
automáticamente a otra precisión. Usa autocast FP16 y GradScaler, manteniendo
los parámetros maestros y estados del optimizador en FP32 para estabilidad.
`--steps` cuenta microbatches; por defecto hay 2.000 y 125 actualizaciones por
acumulación de gradiente. Usa `--epochs 1` para recorrer una época completa;
en DACSA esto puede tardar mucho más que la prueba inicial.
La parada temprana está activada por defecto: detiene si la pérdida de
validación no mejora al menos `0.005` durante ocho evaluaciones consecutivas y
conserva `best.pt` según la mejor pérdida observada.

```bash
# Fine-tuning completo; valida con el 1% reservado del propio corpus
python -m minigpt.finetune --mode full --steps 2000

# LoRA sobre Q/V
python -m minigpt.finetune --mode lora --targets qv --steps 2000
```

Los checkpoints se guardan bajo `gpt2_weights/full/` o `gpt2_weights/lora/`,
sin sobrescribir el GPT-2 base. Se puede proporcionar un split de validación
independiente con `--valid ruta/al/valid.txt`. El entrenamiento se puede
reanudar con `--resume ruta/al/checkpoint-step-XXXXXX.pt` y un `--steps` mayor.

### Preparar OPUS-100 inglés-español

El fichero OPUS-100 `en-es/train-00000-of-00001.parquet` se prepara sin
entrenar el modelo. Instala las dependencias y convierte los pares en ejemplos
con una instrucción de traducción explícita. El script crea particiones
reproducibles de entrenamiento, validación y prueba; el test se reserva para la
evaluación final.

```bash
python -m minigpt.prepare_opus100 data/opus100_en-es_train.parquet
python -m minigpt.finetune --mode full \
  --train data/opus100_en-es_train.txt \
  --valid data/opus100_en-es_valid.txt
```

La preparación solo genera datos; el segundo comando inicia entrenamiento y
debe ejecutarse por separado. La demo compara generación greedy y muestreo
(temperatura, top-k/top-p y control de repeticiones), e informa repetición de
trigramas. La validación usa pérdida/perplejidad; `data/opus100_en-es_test.txt`
no debe emplearse para seleccionar checkpoints.

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
