# QSim: Post-Training INT8 Quantization Engine from Scratch (NumPy)

Motor de inferencia cuantizada (INT8) implementado desde cero en **NumPy puro**, diseñado para modelar a bajo nivel cómo operan los aceleradores hardware y runtimes de producción (fusión de operadores, acumulación en enteros y aritmética de punto fijo).

Aplica técnicas de **Post-Training Quantization (PTQ)** sobre redes convolucionales (LeNetLite sobre Fashion-MNIST), logrando una **reducción del 75% del tamaño del modelo sin pérdida de precisión**.

---

## Métricas de Impacto

| Modo | Val Accuracy | Test Accuracy | Tamaño Modelo | Factor de Compresión |
| --- | --- | --- | --- | --- |
| **FP32 (Baseline)** | 86.44% | 87.19% | 80.16 KB | 1.0× |
| **INT8 (QSim Engine)** | **87.06%** | **87.87%** | **20.04 KB** | **~4.0× (-75%)** |

* **Zero Accuracy Drop:** La precisión en test no solo se mantiene, sino que experimenta una ligera regularización (+0.68 pts).
* **Aritmética entera pura:** Inferencia ejecutada íntegramente con enteros (`int8` y acumulación `int32`), simulando el comportamiento a nivel de registro hardware sin recurrir a casts flotantes intermedios.
* *Nota sobre latencia:* Al ser una implementación pura en Python/NumPy orientada a la arquitectura del dato, prioriza la transparencia funcional sobre la optimización de kernels SIMD/hardware nativo.

---

## Arquitectura y Características Técnicas

* **Matemática de Cuantización Afín:** Cálculo explícito de escala ($S$) y punto cero ($Z$), con soporte configurable para esquemas simétricos y asimétricos, tanto *per-tensor* como *per-channel*.
* **Kernel Engine (Simulación Hardware):**
* `conv2d_int` y `linear_int` con multiplicación entera y acumulación segura en 32 bits (`int32`) para evitar overflow.
* Re-cuantización (`requantize_int32`) de vuelta a `int8` mediante escalado de punto fijo y saturación (*clipping*).
* **Operator Fusion:** Fusión a bajo nivel de capas convolucionales con activaciones (Conv + ReLU) en el dominio entero.


* **Algoritmos de Calibración:**
* Búsqueda *Greedy* por capa para balancear esquemas de cuantización en pesos.
* Optimización por **Simulated Annealing (SA)** para la calibración de percentiles en tensores de activación.


* **Infraestructura y Reproducibilidad:**
* Pipeline automatizado de CI/CD vía GitHub Actions para la generación de documentación técnica.
* Suite de perfiles y generación de artefactos reproducibles (tablas CSV y matrices de confusión).



---

## Evaluación y Resultados

---

## Inicio Rápido (Quickstart)

### 1. Requisitos e Instalación

```bash
git clone https://github.com/tu-usuario/nombre-del-repo.git
cd nombre-del-repo
pip install numpy matplotlib tqdm

```

### 2. Ejecutar Inferencia y Calibración

```bash
# Entrenar/evaluar el baseline FP32
python run_baseline.py

# Ejecutar pipeline completo: Calibración PTQ + Motor QSim INT8
python run_qsim.py

```

Los resultados y figuras se exportarán automáticamente a la carpeta `results/`.

---

## Estructura del Código

```text
.
├── src/
│   ├── quantizer.py       # Cuantización afín (scale/zero-point, per-tensor/channel)
│   ├── qsim_engine.py     # Núcleo de cómputo INT8/INT32 y fusión Conv+ReLU
│   ├── calibrator.py      # Búsqueda heurística (Greedy + Simulated Annealing)
│   ├── model_fp32.py      # Definición de arquitectura LeNetLite en NumPy
│   ├── trainer.py         # Pipeline de optimización y entrenamiento FP32
│   └── profiler.py        # Métricas de memoria y medición de rendimiento
├── notebooks/             # Implementaciones interactivas paso a paso
│   ├── 01_quantizacion_y_engine.ipynb
│   └── 02_calibracion_y_busqueda.ipynb
├── run_baseline.py        # Punto de entrada para modelo base
├── run_qsim.py            # Punto de entrada para cuantización y evaluación
├── memoria/               # Documentación formal y especificación matemática
└── .github/workflows/     # Automatización CI/CD

```

---

## Documentación Profunda

Para una explicación interactiva o una derivación matemática formal:

* **Notebooks Explicativos:** Consulta `notebooks/01_quantizacion_y_engine.ipynb` para ver el paso a paso del emulado de registros enteros y `notebooks/02_calibracion_y_busqueda.ipynb` para las curvas de calibración.
* **Memoria Técnica:** Revisa `memoria/main.pdf` para la formulación completa de la propagación del error de cuantización.

---
