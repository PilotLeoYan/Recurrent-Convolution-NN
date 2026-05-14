# Recurrent Convolution Neural Network — Video Prediction

Proyecto universitario que compara tres arquitecturas de redes neuronales para la tarea de **predicción de frames en video**, usando el dataset Moving MNIST. La idea central es ver cuánto aporta el componente recurrente frente a una CNN sin memoria temporal.

Los tres modelos implementados son:

- **RCNN2d** — RNN donde los pesos escalares se reemplazan por kernels `Conv2d`, tanto en la transición input→hidden como en hidden→hidden.
- **Conv2dGRU (CGRU)** — GRU completo (update gate, reset gate, candidate) implementado con operaciones convolucionales.
- **CNN** — Baseline sin recurrencia. Cada frame se procesa de forma independiente, sin estado oculto entre timesteps.

---

## Tabla de contenidos

- [Estructura del proyecto](#estructura-del-proyecto)
- [Arquitectura](#arquitectura)
- [Dataset](#dataset)
- [Entrenamiento](#entrenamiento)
- [Instalación](#instalación)
- [Uso](#uso)
- [Configuración](#configuración)
- [Pesos preentrenados](#pesos-preentrenados)
- [Resultados](#resultados)

---

## Estructura del proyecto

```
Recurrent-Convolution-NN/
├── config.json
├── requirements.txt
├── main/
│   ├── __main__.py
│   ├── models/
│   │   ├── rcnn.py          # RCNN2d
│   │   ├── cgru.py          # Conv2dGRU
│   │   └── cnn.py           # CNN baseline
│   ├── train/
│   │   └── train.py
│   ├── evaluate/
│   │   ├── evaluate.py
│   │   └── metrics.py       # MAE y MSE
│   ├── losses/
│   │   └── __init__.py      # CombinedLoss (MSE + SSIM)
│   ├── optimizers/
│   │   └── __init__.py
│   ├── data/
│   │   ├── downloader.py    # Descarga y split del dataset
│   │   ├── dataset.py
│   │   ├── dataloader.py
│   │   └── transform.py
│   ├── visualize/
│   │   └── visualize.py
│   └── utils/
│       ├── args.py
│       ├── json_loader.py
│       ├── logger.py
│       └── csv_logger.py
└── saves/
    ├── weights/
    ├── training_logs/
    └── visualizations/
```

---

## Arquitectura

Los tres modelos comparten la misma interfaz encode-decode:

- `forward(x, h0)` — recibe una secuencia `(seq_len, batch, C, H, W)` y retorna predicciones por timestep junto con el estado oculto final.
- `decode(pred_len, last_frame, h, ...)` — genera `pred_len` frames futuros de forma autorregresiva usando el estado oculto del encoder.
- `output_proj` — capa `Conv2d 1×1 + Sigmoid` que mapea el feature map de vuelta al espacio de píxeles `[0, 1]`.

![Diagrama de arquitectura](assets/diagram.png)

**RCNN2d** apila `units` capas recurrentes. Cada capa tiene dos `Conv2d` separados: uno para el input (`conv2d_ih`) y otro para el estado oculto (`conv2d_hh`), siguiendo la fórmula `h_t = act(W_ih·x_t + W_hh·h_{t-1})` pero en espacio convolucional 2D. Se usa inicialización Kaiming normal para las convoluciones de input y ortogonal para las recurrentes.

**Conv2dGRU** reemplaza cada peso escalar de una celda GRU estándar por un kernel `Conv2d`. Cada celda calcula update gate `z`, reset gate `r` y candidate `n` mediante proyecciones convolucionales, capturando patrones espaciales y temporales al mismo tiempo.

**CNN** es un stack de capas `Conv2d + ReLU` sin estado. Sirve como cota inferior: si los modelos recurrentes no la superan, el componente temporal no está aportando nada útil.

Todos los modelos recurrentes usan `Dropout(p=0.2)` entre capas ocultas (excepto en la última) para regularizar el camino recurrente.

---

## Dataset

**Moving MNIST** — secuencias de 20 frames en escala de grises, con dos dígitos moviéndose dentro de un frame de 64×64.

- Fuente original: `https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy`
- El downloader (`data/downloader.py`) lo descarga automáticamente en el primer uso (con reintentos y barra de progreso) y genera el split dentro de la carpeta `dataset/`:

| Split | Proporción | Archivo |
|---|---|---|
| Train | 70% | `moving_mnist_train.npy` |
| Validación | 15% | `moving_mnist_valid.npy` |
| Test | 15% | `moving_mnist_test.npy` |

El split usa semilla fija (`seed=42`). Cada sample es un tensor `(seq_len, 1, 64, 64)` normalizado a `[0, 1]`. La secuencia se divide en `split_seq_len` (default `10`): los primeros 10 frames son el **input** y los restantes son el **target** a predecir.

> **Dataset pre-dividido disponible:** ver [Pesos preentrenados](#pesos-preentrenados).

---

## Entrenamiento

### Loss

Se usa una **loss combinada MSE + SSIM**:

```
L = α · MSE(pred, target) + (1 − α) · (1 − SSIM(pred, target))
```

con `α = 0.85`. El SSIM se aproxima con average pooling de ventana 11×11. La combinación busca que las predicciones sean precisas a nivel de píxel (MSE) y visualmente nítidas (SSIM).

### Optimizer y scheduler

- AdamW (`lr = 0.001`, `weight_decay = 0.01`)
- Cosine annealing con warm-up lineal (3 épocas de warm-up, `T_max = 50`, `eta_min = 1e-6`)
- Gradient clipping con `max_norm = 5.0`

### Teacher forcing

Durante el decode, la probabilidad de usar el frame real en vez de la predicción del propio modelo decae linealmente con las épocas:

```
tf_ratio = max(0.05, 1.0 − epoch / (epochs − 1))
```

Arranca en 1.0 (siempre usa ground truth) y termina en 0.05 (casi completamente autorregresivo).

### Hardware utilizado

| Componente | Especificación |
|---|---|
| GPU | NVIDIA GeForce RTX 2070 Super (8 GB VRAM) |
| RAM | 16 GB DDR4 |
| CPU | AMD Ryzen 7 — 8 cores / 16 threads |

### Tiempos de entrenamiento

Los tres modelos se entrenaron secuencialmente con la configuración exacta de `config.json` (`epochs=50`, `hidden_channels=64`, `kernel_size=3`, `units=3`, `batch_size=8`).

| Modelo | Inicio | Fin | Duración |
|---|---|---|---|
| RCNN2d | 2026-05-12 16:10:32 | 2026-05-12 17:48:11 | **1 h 37 m 39 s** |
| Conv2dGRU | 2026-05-12 17:50:37 | 2026-05-12 23:10:08 | **5 h 19 m 31 s** |
| CNN | 2026-05-12 23:10:18 | 2026-05-12 23:33:57 | **0 h 23 m 39 s** |
| **Total** | | | **≈ 7 h 20 m 49 s** |

La diferencia tan grande entre CGRU y los demás es esperable: la celda GRU requiere 6 operaciones `Conv2d` por capa por timestep (vs 2 del RCNN2d), lo que lo hace considerablemente más costoso.

Las métricas por época se guardan automáticamente en `saves/training_logs/` como archivos CSV.

### Curvas de loss

<table>
<tr>
<td><img src="assets/train_loss.png" alt="Training loss" width="420"/></td>
<td><img src="assets/val_loss.png" alt="Validation loss" width="420"/></td>
</tr>
<tr>
<td align="center">Loss de entrenamiento</td>
<td align="center">Loss de validación</td>
</tr>
</table>

El CGRU alcanza la loss más baja en ambas curvas (~0.035 en train, ~0.055 en validación), aunque el entrenamiento es considerablemente más inestable, especialmente en las primeras épocas. El punto marcado en la curva de validación indica el checkpoint guardado (mejor época). CNN y RCNN se mantienen alrededor de 0.075, lo que sugiere que sin los gates del GRU el modelo tiene dificultades para capturar la dinámica temporal del dataset.

---

## Instalación

Requiere Python ≥ 3.10 y GPU con CUDA (recomendado).

```bash
git clone https://github.com/<usuario>/Recurrent-Convolution-NN.git
cd Recurrent-Convolution-NN
pip install -r requirements.txt
```

Dependencias:

```
torch==2.11.0
torchvision==0.26.0
numpy==2.4.4
matplotlib==3.10.9
requests==2.33.1
tqdm==4.67.1
rich==15.0.0
setuptools==81.0.0
```

---

## Uso

Todos los comandos se ejecutan como módulo Python desde la raíz del proyecto:

```bash
# Entrenar todos los modelos
python -m main --fit all

# Entrenar un modelo específico
python -m main --fit rcnn
python -m main --fit cgru
python -m main --fit cnn

# Evaluar en el test set (usa la sección "eval" de config.json)
python -m main --test

# Generar visualizaciones de inferencia (usa la sección "visualize" de config.json)
python -m main --visualize

# Usar un config diferente
python -m main --fit all --config ruta/a/config.json

# Ayuda
python -m main --help
```

> En el primer uso el dataset se descarga automáticamente (~780 MB) y se divide en `dataset/`. Las siguientes ejecuciones detectan los archivos y se saltan la descarga.

Los logs se escriben en `logs/rcnn.log` (rotación automática, máx. 5 MB, 3 backups) y también se imprimen en consola.

---

## Configuración

Todo el control del experimento está en `config.json`, con tres secciones principales:

### `fit` — entrenamiento

```jsonc
{
  "fit": {
    "hidden_channels": 64,      // feature maps en capas ocultas
    "kernel_size": 3,            // tamaño del kernel conv (same padding)
    "units": 3,                  // capas recurrentes/conv apiladas
    "activation": "relu",        // "relu" o "tanh" (solo RCNN2d)
    "device": "cuda",
    "lr": 0.001,
    "weight_decay": 0.01,
    "split_seq_len": 10,         // frames de input; el resto son targets
    "epochs": 50,
    "save_best": true,
    "model_path": "saves/weights",
    "csv_log": { "output_dir": "saves/training_logs" },
    "train":  { "batch_size": 8, "data_augmentation": false, "n_workers": 4 },
    "valid":  { "batch_size": 32, "n_workers": 0 },
    "test":   { "batch_size": 32, "n_workers": 0 },
    "lr_scheduler": {
      "type": "cosine_warmup",
      "T_max": 50,
      "eta_min": 0.000001,
      "warmup_epochs": 3
    }
  }
}
```

### `eval` — evaluación

```jsonc
{
  "eval": {
    "model_type": "cnn",                         // "rcnn2d", "cgru" o "cnn"
    "model_path": "saves/weights_cnn_best.pth",
    "batch_size": 32,
    "n_workers": 0,
    "split_seq_len": 10,
    "device": "cuda",
    "hidden_channels": 64,
    "kernel_size": 3,
    "units": 3,
    "activation": "relu"
  }
}
```

### `visualize` — inferencia visual

```jsonc
{
  "visualize": {
    "model_type": "cgru",
    "model_path": "saves/weights_cgru_best.pth",
    "batch_size": 32,
    "n_workers": 0,
    "split_seq_len": 10,
    "device": "cuda",
    "hidden_channels": 64,
    "kernel_size": 3,
    "units": 3,
    "activation": "relu",
    "n_samples": 5,
    "output_dir": "saves/visualizations",
    "colormap": "gray",
    "dpi": 150
  }
}
```

---

## Pesos preentrenados

Los checkpoints guardados (`save_best=true`, mejor loss de validación) están disponibles en `weights.zip`:

| Archivo | Modelo | Tamaño |
|---|---|---|
| `weights_rcnn2d_best.pth` | RCNN2d | ~2.1 MB |
| `weights_cgru_best.pth` | Conv2dGRU | ~6.4 MB |
| `weights_cnn_best.pth` | CNN | ~863 KB |

Colocar los `.pth` en `saves/` y actualizar `model_path` en `config.json` antes de correr `--test` o `--visualize`.

El dataset pre-dividido está en `dataset.zip` (~781 MB):

| Archivo | Uso |
|---|---|
| `moving_mnist_train.npy` | Entrenamiento |
| `moving_mnist_valid.npy` | Validación |
| `moving_mnist_test.npy` | Test |

Extraer en `dataset/` en la raíz del proyecto. El downloader detecta los archivos y omite la descarga automáticamente.

---

## Resultados

Las visualizaciones muestran, de arriba a abajo: los 10 frames de input, el ground truth de los 10 frames a predecir, la predicción del modelo y el error absoluto por píxel.

![Sample 0](assets/sample_000.png)

![Sample 1](assets/sample_001.png)

![Sample 2](assets/sample_002.png)

El modelo logra predecir bien los primeros frames (~t=10 a t=12) pero la calidad se va degradando con el tiempo, lo cual es esperable dado que en inferencia el proceso es completamente autorregresivo y los errores se acumulan. El mapa de error muestra que los contornos de los dígitos son la zona de mayor pérdida, especialmente cuando ambos dígitos se superponen o cuando el movimiento cambia de dirección.

Las métricas que reporta `--test` sobre el test set son:

| Métrica | Descripción |
|---|---|
| **Combined Loss** | `α · MSE + (1−α) · (1−SSIM)` — misma loss del entrenamiento |
| **MAE** | Error absoluto medio sobre todos los píxeles predichos |
| **MSE** | Error cuadrático medio sobre todos los píxeles predichos |
