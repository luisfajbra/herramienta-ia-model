# Spec Técnico: Sistema de Predicción de Fallas Hidráulicas — Red Chico Sur

> **Tipo de documento:** Plan de desarrollo de código  
> **Propósito:** Definir qué se construye, cómo se organiza y en qué orden  
> **Versión:** 4.0 — versión final para implementación  
> **No incluye:** Redacción de tesis, marco teórico, justificación académica

---

## 1. Qué hace el sistema

El usuario configura los parámetros de simulación (ruta del `.inp`, rango y paso del factor multiplicador) en un único archivo `config.yaml`, ejecuta el pipeline, y obtiene:

1. Dataset de entrenamiento generado automáticamente desde SWMM.
2. Dos modelos entrenados: clasificador (¿inunda?) y regresor (¿cuánto volumen?).
3. Métricas de evaluación en tres niveles: clasificador aislado, regresor aislado (oracle), y sistema end-to-end. Reportadas para LOSO y GroupKFold(5), con estratificación por magnitud del factor.
4. Mapas visuales de la red con gradiente de inundación por nodo.
5. Módulo de inferencia para predecir nuevos escenarios sin correr SWMM.

---

## 2. Stack tecnológico

| Componente | Librería |
|---|---|
| Ejecución de SWMM | `pyswmm` |
| Parseo de `.inp` y `.rpt` | `swmm-api` |
| Grafo de red (topología) | `networkx` |
| Manipulación de datos | `pandas`, `numpy` |
| Modelos ML | `xgboost`, `scikit-learn` |
| Visualización | `matplotlib` |
| Configuración | `config.yaml` |
| Serialización de modelos | `joblib` |

**Sistema operativo:** Windows · **Python:** 3.10+

---

## 3. Arquitectura de carpetas

```
swmm_resilience/
│
├── config.yaml
├── main.py
├── requirements.txt
│
├── data/
│   ├── networks/
│   │   └── chico_sur.inp
│   └── training/
│       └── dataset_final.csv
│
├── outputs/
│   ├── models/
│   │   ├── classifier.joblib
│   │   ├── regressor.joblib
│   │   └── training_inp_hash.txt   # Hash del .inp usado en entrenamiento
│   ├── metrics/
│   │   ├── metrics_classifier.json
│   │   ├── metrics_regressor.json
│   │   ├── metrics_endtoend.json
│   │   ├── metrics_by_factor.json  # Métricas estratificadas por factor
│   │   ├── feature_importance_classifier.png
│   │   └── feature_importance_regressor.png
│   └── maps/
│       └── flood_map_factor_X.XX.png
│
└── swmm_resilience/
    ├── __init__.py
    ├── config.py
    │
    ├── simulation/
    │   ├── __init__.py
    │   ├── runner.py
    │   └── batch.py
    │
    ├── extraction/
    │   ├── __init__.py
    │   ├── static_features.py
    │   ├── topology.py
    │   ├── dynamic_features.py
    │   └── labels.py
    │
    ├── dataset/
    │   ├── __init__.py
    │   ├── assembler.py
    │   └── validator.py
    │
    ├── ml/
    │   ├── __init__.py
    │   ├── trainer.py
    │   ├── evaluator.py
    │   ├── feature_importance.py
    │   └── predict.py
    │
    └── visualization/
        ├── __init__.py
        └── flood_map.py
```

---

## 4. Archivo de configuración: `config.yaml`

```yaml
# ─── Red de alcantarillado ────────────────────────────────────────────────────
network:
  inp_path: "data/networks/chico_sur.inp"
  name: "Chico Sur"

# ─── Simulaciones ─────────────────────────────────────────────────────────────
simulation:
  factor_min: 0.2
  factor_max: 5.0
  factor_step: 0.2
  # NOTA: factor_min < 1.0 genera negativos que ayudan al modelo a aprender
  # la frontera de falla. Para estudiar solo la zona de fallo usar 1.0.
  # Con factor_min < 1.0, scale_pos_weight: "auto" compensa el desbalance.

# ─── Dataset ──────────────────────────────────────────────────────────────────
dataset:
  output_path: "data/training/dataset_final.csv"
  flood_threshold_m3: 0.0

# ─── Modelos ML ───────────────────────────────────────────────────────────────
ml:
  classifier:
    algorithm: "xgboost"      # Opciones: "xgboost", "random_forest"
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.05       # Solo xgboost
    subsample: 0.8            # Solo xgboost
    scale_pos_weight: "auto"  # "auto" = count(0)/count(1); o valor numérico

  regressor:
    algorithm: "xgboost"      # Opciones: "xgboost", "random_forest"
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8

  # XGBoost y Random Forest son invariantes a escala lineal.
  # Activar solo si se usa SVM o regresión regularizada en el futuro.
  use_scaler: false

# ─── Evaluación ───────────────────────────────────────────────────────────────
evaluation:
  methods:
    - "LOSO"        # Principal: evalúa generalización real (factor nunca visto)
    - "GroupKFold5" # Secundario: std más estables con pocos puntos de variación
  stratify_by_factor: true  # Reportar métricas por magnitud de factor además de media global

# ─── Visualización ────────────────────────────────────────────────────────────
visualization:
  factors_to_plot: [1.4, 1.6, 2.0, 3.0, 5.0]
  colormap: "RdYlBu_r"
  output_path: "outputs/maps/"
  show_labels_top_n: 5
```

---

## 5. Features del modelo

### 5.1 Features estáticas (fuente: `.inp`, una vez)

| Variable | Descripción | Fuente |
|---|---|---|
| `elev_fondo` | Cota de fondo (m s.n.m.) | `[JUNCTIONS]` |
| `prof_max` | Profundidad máxima del nodo (m) | `[JUNCTIONS]` |
| `n_tuberias_in` | N° tuberías entrantes | `[CONDUITS]` conteo |
| `n_tuberias_out` | N° tuberías salientes | `[CONDUITS]` conteo |
| `diam_max_in` | Diámetro máximo tubería entrante (m) | `[XSECTIONS]` |
| `diam_max_out` | Diámetro máximo tubería saliente (m) | `[XSECTIONS]` |
| `pendiente_max_in` | Pendiente máxima tuberías entrantes (m/m) | `[CONDUITS]` ΔZ/L |
| `pendiente_out` | Pendiente de la tubería saliente (m/m) | `[CONDUITS]` ΔZ/L — cada nodo tiene como máximo una tubería de salida |
| `base_inflow_lps` | Caudal pico base del nodo (LPS, factor=1) | max(`[TIMESERIES]`) |

**Features eliminadas respecto a versiones anteriores:**
- `area_seccion_out`: transformación monotónica de `diam_max_out` (`A = π(D/2)²`). No agrega información independiente para árboles de decisión. Eliminar evita que la importancia de esa señal hidráulica se reparta entre dos columnas en los gráficos.

**Manejo de NaN:** Nodos cabecera sin tuberías entrantes tendrán `diam_max_in = NaN` y `pendiente_max_in = NaN`. Se propagan hasta el dataset y son imputados por el pipeline de ML con `SimpleImputer(strategy="median")`. No se imputan en la extracción.

### 5.2 Features topológicas (fuente: `.inp` + grafo NetworkX, una vez)

| Variable | Descripción | Método |
|---|---|---|
| `dist_outfall_m` | Distancia acumulada al outfall (m) | Suma de longitudes en camino aguas abajo — NaN si el nodo no tiene camino al outfall (imputado por el pipeline ML) |
| `n_nodos_aguas_arriba` | N° de nodos que drenan hacia este en toda la red | BFS inverso sobre grafo dirigido |
| `q_pico_acum_base` | Suma de caudales pico base de nodos aguas arriba + propio (LPS) | Suma de `base_inflow_lps` de nodos upstream |
| `upstream_capacity_lps` | Capacidad hidráulica a sección llena de tuberías inmediatas entrantes (LPS) | Σ A×V_lleno de conduits entrantes |

**`q_pico_acum_base` como feature independiente:** Va al dataset como columna propia, no solo como intermediario. Necesario para que el modelo distinga entre un nodo con carga base alta en condición normal versus un nodo con carga base baja bajo factor elevado, cuando el producto `base × factor` es igual en ambos casos.

**`dist_outfall_m`:** Feature topológica más importante del modelo. Captura la acumulación de backpressure aguas abajo que ninguna métrica local puede expresar.

### 5.3 Features dinámicas (fuente: factor × timeseries, una por simulación)

| Variable | Descripción | Cálculo |
|---|---|---|
| `factor_mult` | Factor multiplicador de la simulación | Directo del config |
| `q_pico_nodo` | Caudal pico escalado del nodo (LPS) | `base_inflow_lps × factor` |
| `q_pico_acum_escalado` | Carga hidráulica total escalada que llega al nodo (LPS) | `q_pico_acum_base × factor` |

**Feature eliminada:** `vol_hidrograma_nodo` — transformación lineal de `q_pico_nodo` dada la forma temporal idéntica de todos los hidrogramas. No agrega varianza explicativa independiente.

### 5.4 Variables de salida (etiquetas)

| Variable | Tipo | Fuente |
|---|---|---|
| `inunda` | Binaria (0/1) | `.rpt` → vol > threshold → 1 |
| `vol_inundacion_m3` | Continua ≥ 0 | `.rpt` Node Flooding Summary |

---

## 6. Descripción de módulos

### 6.1 `config.py`

Carga `config.yaml`, valida coherencia y expone objeto `Config` tipado. Valida: `factor_min < factor_max`, `factor_step > 0`, existencia del `.inp`, rutas de salida escribibles.

---

### 6.2 `simulation/runner.py`

Ejecuta una simulación de SWMM para un factor dado.

- Lee hidrogramas base de `[TIMESERIES]` con `swmm-api`.
- Crea copia temporal del `.inp` con todos los valores de timeseries multiplicados por el factor.
- Corre SWMM sobre la copia temporal con `pyswmm`.
- Retorna ruta del `.rpt` generado.
- El `.inp` original nunca se modifica.

---

### 6.3 `simulation/batch.py`

Genera lista de factores desde config e itera sobre `runner.py`. Muestra progreso en consola. Retorna `List[(factor, ruta_rpt)]`.

---

### 6.4 `extraction/static_features.py`

Extrae features estáticas y `base_inflow_lps` por nodo desde el `.inp`. Se ejecuta una sola vez.

```
Salida: DataFrame — 1 fila por nodo junction (outfall excluido)
        node_id, elev_fondo, prof_max, 
        n_tuberias_in, n_tuberias_out,
        diam_max_in, diam_max_out,
        pendiente_max_in, pendiente_out,
        base_inflow_lps, coord_x, coord_y
```

---

### 6.5 `extraction/topology.py`

Construye grafo dirigido con NetworkX y calcula features topológicas.

- Nodos: junctions + outfall.
- Aristas: conduits dirigidos FromNode → ToNode con atributo `length`.

`dist_outfall_m`: suma de longitudes de conduits en el camino aguas abajo hasta el outfall. Si el nodo no tiene camino al outfall (nodo desconectado), se asigna `NaN`. Este NaN se propaga al dataset final y es imputado por el pipeline de ML con `SimpleImputer(strategy="median")`.

`pendiente_out`: cada nodo junction de la red Chico Sur tiene como máximo una tubería de salida. No hay bifurcaciones en la dirección aguas abajo — el cálculo es siempre uno a uno.

```
Salida: columnas agregadas al DataFrame estático:
        dist_outfall_m, n_nodos_aguas_arriba,
        q_pico_acum_base, upstream_capacity_lps
```

`q_pico_acum_base` se incluye como feature en el dataset final.

---

### 6.6 `extraction/dynamic_features.py`

Calcula features dinámicas para un (nodo, factor). No necesita el `.rpt`.

```
Entrada: DataFrame con base_inflow_lps y q_pico_acum_base, factor (float)
Salida:  columnas: factor_mult, q_pico_nodo, q_pico_acum_escalado
```

---

### 6.7 `extraction/labels.py`

Parsea Node Flooding Summary del `.rpt`.

**Verificar unidades:** Con LPS, SWMM puede reportar en 10⁶ litros → `vol_m3 = vol_reportado × 1000`. Confirmar en el primer `.rpt` real de Chico Sur. Esto determina la escala entera de la variable de salida del regresor.

```
Salida: node_id, vol_inundacion_m3, inunda
        Nodos ausentes del reporte → vol=0, inunda=0
```

---

### 6.8 `dataset/assembler.py`

Une todas las fuentes en el dataset final.

1. Features estáticas + topológicas (constantes).
2. Por cada (factor, ruta_rpt): features dinámicas + etiquetas, join por `node_id`.
3. Concatenar todos los factores → `dataset_final.csv`.

**Dimensiones esperadas para Chico Sur con 24 factores:** 2,592 filas × ~20 columnas.

---

### 6.9 `dataset/validator.py`

Valida antes de entrenar:
- Sin NaN en etiquetas (`inunda`, `vol_inundacion_m3`).
- `vol_inundacion_m3 >= 0`, `inunda ∈ {0,1}`.
- N° filas = n_nodos × n_factores.
- Al menos un nodo con `inunda = 1`.
- Distribución de clases: advertencia si ratio inunda=1 / total < 0.05.

---

### 6.10 `ml/trainer.py`

**Pipeline de ML:**
```
SimpleImputer(strategy="median")  →  XGBoost (o RF según config)
```

`StandardScaler` no se incluye por defecto: XGBoost y RF son invariantes a escala lineal. Se activa solo si `use_scaler: true` en config.

Al finalizar el entrenamiento, guarda el hash MD5 del `.inp` usado en `outputs/models/training_inp_hash.txt`. Este hash se usa en `predict.py` para validar que la inferencia se hace sobre la misma red.

**Modelo 1 — Clasificador:**
- Features: todas excepto `node_id`, `coord_x`, `coord_y`, `vol_inundacion_m3`, `inunda`.
- Target: `inunda`.
- `scale_pos_weight` calculado automáticamente si config = "auto".

**Modelo 2 — Regresor:**
- Entrenado solo sobre filas donde `inunda = 1`.
- Features: las mismas que el clasificador.
- Target: `vol_inundacion_m3`.

Salida: `classifier.joblib`, `regressor.joblib`, `training_inp_hash.txt`.

---

### 6.11 `ml/evaluator.py`

Tres niveles de evaluación para LOSO y GroupKFold(5).

#### Nivel 1 — Clasificador aislado

```
Para cada fold:
    Predecir inunda sobre test con etiquetas reales
    Métricas: Precision, Recall, F1, AUC-ROC
```

#### Nivel 2 — Regresor aislado (oracle evaluation)

```
Para cada fold:
    Predecir vol sobre test[inunda=1]  ← filtra con etiquetas REALES
    Métricas: NSE, RMSE, MAE, R²
```

**Oracle evaluation — limitación documentada:** El conjunto `test[inunda=1]` se filtra con las etiquetas verdaderas de SWMM, no con las predicciones del clasificador. Esto mide el regresor en aislamiento, sin propagación del error del clasificador. Es una evaluación optimista: los falsos negativos del clasificador no llegan al regresor y su impacto no se captura aquí. Esta limitación se declara explícitamente en la tesis y se cuantifica con la evaluación end-to-end.

#### Nivel 3 — Sistema end-to-end

```
Para cada fold:
    Clasificador predice inunda_pred sobre test completo
    Regresor predice vol sobre test[inunda_pred=1]  ← usa PREDICCIONES del clasificador
    
    Métricas del sistema completo:
    - % nodos con inundación correctamente identificados
    - RMSE del volumen sobre TODOS los nodos (FN tienen error = vol_real)
    - Volumen total predicho vs real por simulación (m³)
```

#### Estratificación por magnitud del factor

Si `stratify_by_factor: true` en config, las métricas se calculan también por factor individual (no solo media global). Esto permite visualizar cómo aumenta el error en los extremos del rango (factores 0.2 y 5.0), donde el modelo extrapola fuera del rango de entrenamiento en LOSO. Convierte una limitación esperada en un resultado de análisis.

Salida: `metrics_classifier.json`, `metrics_regressor.json`, `metrics_endtoend.json`, `metrics_by_factor.json`.

---

### 6.12 `ml/feature_importance.py`

Importancia de variables de los modelos finales:
- XGBoost: `feature_importances_` (gain).
- RF: `feature_importances_` (mean decrease impurity).
- Gráficos de barras horizontales ordenados de mayor a menor.

Salida: `feature_importance_classifier.png`, `feature_importance_regressor.png`.

---

### 6.13 `ml/predict.py`

Módulo de inferencia para usar los modelos entrenados sobre nuevos escenarios.

**Función principal: `predict_network(factor)`**

```
1. Cargar classifier.joblib y regressor.joblib
2. Leer inp_path desde config.yaml (el mismo .inp del entrenamiento)
3. Validar hash del .inp actual contra training_inp_hash.txt
   → Si no coinciden: error descriptivo indicando qué red se usó para entrenar
   → Si coinciden: continuar
4. Extraer features estáticas y topológicas del .inp
5. Calcular features dinámicas para el factor dado
6. Clasificador predice inunda_pred por nodo
7. Regresor predice vol_pred para nodos con inunda_pred=1
8. Retornar DataFrame: node_id, inunda_pred, vol_pred_m3, coord_x, coord_y
9. Opcionalmente generar mapa de inundación para ese factor
```

**Diseño de la firma:** La función usa el `inp_path` del `config.yaml`, no un parámetro externo. Esto hace explícito que el modelo es válido solo para la red con la que fue entrenado. La validación por hash garantiza que si alguien modifica el `.inp` después del entrenamiento, el sistema lo detecta y avisa.

**Caso de uso:** El usuario quiere saber qué pasa con factor = 3.5 sin correr SWMM. Ejecuta `python main.py --predict --factor 3.5` y obtiene predicciones y mapa en segundos.

**Limitación documentada:** Válido solo dentro del rango de factores simulados. Predicciones fuera del rango [factor_min, factor_max] no están validadas y el sistema emite advertencia si se solicitan.

---

### 6.14 `visualization/flood_map.py`

Genera mapa de red con gradiente de inundación. Acepta tanto datos del dataset (simulaciones reales) como predicciones de `predict.py` (inferencia).

Para cada factor:
1. Coordenadas de nodos (`[COORDINATES]`) y conexiones (`[CONDUITS]`) del `.inp`.
2. Volúmenes desde `dataset_final.csv` o desde predicciones.
3. Tuberías: líneas grises. Nodos sin inundación: círculos neutros. Nodos inundados: tamaño y color proporcional al volumen con colormap configurado.
4. Etiquetas solo en los N nodos con mayor volumen (ID + m³).
5. Nodos con coordenadas idénticas (e.g., 2C/2I): offset visual ±2m en X.
6. Colorbar en m³. Título con nombre de red y factor.

---

## 7. Punto de entrada: `main.py`

```bash
python main.py                                      # Pipeline completo
python main.py --skip-simulation                    # Ya tienes los .rpt
python main.py --skip-simulation --skip-extraction  # Ya tienes el dataset
python main.py --only-ml                            # Solo entrenar y evaluar
python main.py --only-maps                          # Solo generar mapas
python main.py --predict --factor 3.5              # Inferencia sin SWMM
```

**Orden de ejecución (pipeline completo):**

```
1.  Cargar y validar config.yaml
2.  Extraer features estáticas del .inp
3.  Construir grafo y calcular features topológicas
4.  Ejecutar simulaciones SWMM (batch)
5.  Calcular features dinámicas por factor
6.  Extraer etiquetas de cada .rpt
7.  Ensamblar dataset final
8.  Validar dataset
9.  Entrenar modelos finales + guardar hash del .inp
10. Evaluar: clasificador aislado + regresor oracle + end-to-end
    (con estratificación por factor si está activada)
11. Calcular importancia de variables
12. Generar mapas de inundación
13. Imprimir resumen de métricas en consola
```

---

## 8. Orden de desarrollo recomendado

| Prioridad | Módulo | Razón |
|---|---|---|
| 1 | `config.py` | Todo lo demás lo necesita |
| 2 | `static_features.py` | Verificar que `swmm-api` lee bien el `.inp` de Chico Sur |
| 3 | `topology.py` | Depende del `.inp` ya parseado |
| 4 | `simulation/runner.py` | Pieza más crítica: interacción con SWMM |
| 5 | `simulation/batch.py` | Usa runner.py |
| 6 | `extraction/labels.py` | Verificar unidades de volumen en el `.rpt` aquí |
| 7 | `extraction/dynamic_features.py` | Independiente del `.rpt` |
| 8 | `dataset/assembler.py` | Une todo lo anterior |
| 9 | `dataset/validator.py` | Verificación antes de entrenar |
| 10 | `ml/trainer.py` | Pipeline: imputer → modelo + hash del .inp |
| 11 | `ml/evaluator.py` | Tres niveles + estratificación por factor |
| 12 | `ml/feature_importance.py` | Requiere modelos entrenados |
| 13 | `ml/predict.py` | Validación por hash + inferencia |
| 14 | `visualization/flood_map.py` | Requiere dataset y coordenadas |
| 15 | `main.py` | Integra todo al final |

---

## 9. Notas técnicas críticas

**Unidades de volumen en el `.rpt`:** Con LPS, SWMM puede reportar en 10⁶ litros → `vol_m3 = vol_reportado × 1000`. Verificar en el primer `.rpt` real de Chico Sur. Determina la escala entera de la variable de salida del regresor.

**El `.inp` original nunca se modifica:** `runner.py` trabaja sobre copias temporales.

**Hash del `.inp`:** Se guarda el hash MD5 del `.inp` al entrenar. `predict.py` lo valida antes de inferir. Si el `.inp` fue modificado después del entrenamiento (recalibración, cambio de diámetros), el sistema lanza error descriptivo antes de producir predicciones inválidas.


**Compatibilidad `pyswmm` en Windows:** Verificar compatibilidad con la versión de SWMM usada para calibrar la red (5.1 o 5.2). La DLL puede necesitar configuración manual de ruta.

**Topología de salida única:** Cada nodo junction de Chico Sur tiene como máximo una tubería de salida. `pendiente_out` se calcula siempre sobre esa única tubería — no hay ambigüedad de agregación.

**Nodos duplicados (2C/2I, etc.):** Tratados como nodos independientes en todo el pipeline. Offset visual ±2m en el mapa.

**Outfall (109C):** Excluido del dataset de entrenamiento, incluido en el grafo como nodo terminal para `dist_outfall_m`.

**Métricas estratificadas por factor:** Los folds extremos (0.2, 5.0) van a mostrar mayor error en LOSO porque el modelo extrapola fuera del rango de entrenamiento. Reportarlo estratificado convierte esa limitación esperada en un resultado analítico de la tesis.

---

## 10. `requirements.txt`

```
pyswmm>=1.3.0
swmm-api>=0.4.0
networkx>=3.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
pyyaml>=6.0
joblib>=1.3
```