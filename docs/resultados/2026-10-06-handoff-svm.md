# Handoff: arreglar/reconfigurar SVM para que corra a esta escala

Documento de traspaso para retomar el pendiente de SVM sin tener que
re-investigar nada desde cero. No es un plan de implementación formal (no
tiene tareas TDD paso a paso) — es el contexto técnico completo: qué pasó,
por qué, qué archivos tocar, y las opciones reales con sus tradeoffs.

## 1. Resumen ejecutivo

SVM (kernel RBF) está **excluido** del benchmark de 5 familias desde la
primera corrida real (2026-10-05). No es un bug: es una familia que nunca
se configuró para el volumen de datos real (~154,000 filas). La primera
corrida completa (`--bench` con las 5 familias) acumuló **más de 1 día y
15 horas de CPU activo** sin terminar la etapa de evaluación, y se canceló.

El código de la familia (`svm_family.py`) está bien escrito y sigue el
contrato del banco correctamente — el problema es puramente de escala
algorítmica (SVC/SVR con kernel RBF es O(n²)-O(n³)), no de diseño.

**Mi recomendación corta** (desarrollada en la sección 5): no cambiar a
kernel lineal (el modelo Lineal ya en el ranking rinde mal — 201.18 RMSE,
el peor de las 4 familias evaluadas — y un SVM lineal probablemente
heredaría ese mismo problema). La opción con mejor costo/beneficio es
aproximar el kernel RBF con `Nystroem`/`RBFSampler` + `SGDClassifier`/
`SGDRegressor`, que escala ~O(n) y preserva el comportamiento no lineal
que es la razón de ser de SVM.

## 2. Por qué está excluido — números exactos

- Dataset evaluado: **154,080 filas** (160,000 totales − 5,920 excluidas
  por el filtro de continuidad de Fase C2), **46,225 inundadas**.
- Protocolo del banco: 3 protocolos configurados en
  `config.yaml` (`bench.protocols`), cada uno con su propio número de
  folds, agrupados por columnas distintas
  (`swmm_resilience/ml/bench/schemas.py:56-60`,
  `PROTOCOL_GROUP_COLUMNS`):

  | Protocolo | Agrupa por | Folds en este dataset |
  |---|---|---|
  | `LOSO` | `factor_mult` | 25 (factores 0.2 a 5.0, paso 0.2) |
  | `GroupKFold5` | `factor_mult` | 5 |
  | `LeaveOneShapeOut` | `shape_id` | 40 (39 formas + la forma base) |

  Total: **70 folds**. `evaluate_candidate()`
  (`swmm_resilience/ml/bench/evaluate.py:182-227`) hace un fit+predict de
  **clasificador y regresor por fold** (`_fit_predict_fold`, líneas
  127-179) — es decir, **140 ajustes completos de SVC/SVR** por corrida
  de evaluación completa, cada uno sobre un subconjunto de ~151,000 filas
  de entrenamiento (todas menos el fold retenido).
- `SVC`/`SVR` con kernel RBF escalan **O(n²) a O(n³)** en el número de
  filas de entrenamiento — a ~150,000 filas por fold, cada ajuste
  individual ya es costoso, y son 140 de ellos.
- Encima, `svm_family.py` fuerza `probability=True` en `SVC`
  (obligatorio porque el evaluador necesita `predict_proba` para AUC-ROC
  — ver sección 4.3), lo cual añade un **calibrado interno Platt-scaling
  con validación cruzada a 5 folds DENTRO de cada `.fit()`** — multiplica
  el costo del ajuste del clasificador varias veces más.
- Ningún documento tiene el desglose exacto fit-vs-predict ni tiempo por
  fold — la corrida se canceló a mitad de la etapa de evaluación, no se
  alcanzó a medir con esa granularidad. Si se necesita ese desglose, hay
  que volver a medirlo (ver sección 6 para cómo aislar una corrida
  pequeña y cronometrarla).

Fuente original de estos números:
`docs/resultados/2026-10-05-benchmark-fase-c2.md` (sección 3) y
`docs/resultados/2026-10-06-explicacion-completa.md` (sección 4, versión
en lenguaje llano).

## 3. Estado actual del código

### 3.1. `swmm_resilience/ml/bench/models/svm_family.py` (familia actual)

```python
FAMILY = "svm"
SCALE_FEATURES = True

_CLASSIFIER_DEFAULTS = {"C": 10.0, "kernel": "rbf", "gamma": "scale"}
_REGRESSOR_DEFAULTS = {"C": 10.0, "kernel": "rbf", "epsilon": 0.1}

def build_classifier(params: dict, scale_pos_weight: float) -> Pipeline:
    settings = {**_CLASSIFIER_DEFAULTS, **filter_estimator_params(params, exclude={"epsilon"})}
    model = SVC(
        **settings,
        probability=True,
        class_weight={0: 1.0, 1: float(scale_pos_weight)},
        random_state=ML_RANDOM_STATE,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])

def build_regressor(params: dict) -> Pipeline:
    settings = {**_REGRESSOR_DEFAULTS, **filter_estimator_params(params)}
    model = SVR(**settings)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])

def preprocessing_descriptor(params: dict) -> dict:
    return {"imputer": "median", "scaler": "standard", "pca": None}
```

Notas importantes:
- **Es la única familia con `StandardScaler`** (`SCALE_FEATURES = True`).
  XGBoost/Random Forest van sin escalador a propósito (no lo necesitan).
  Cualquier reemplazo de SVM que siga necesitando features en una escala
  comparable (SGD, Nystroem) **debe conservar el scaler**.
- El balanceo de clases usa `class_weight={0: 1.0, 1: scale_pos_weight}`
  en vez del `scale_pos_weight` nativo de XGBoost — es la forma correcta
  de pasar el mismo concepto a un estimador de sklearn que no tiene ese
  parámetro.
- `probability=True` es la causa directa de gran parte del costo extra
  del clasificador (ver 4.3) — y es la única razón por la que SVC existe
  ahí en vez de la variante más rápida `LinearSVC`/sin calibrar.
- El regresor sigue la misma convención `log1p`/`expm1` que todas las
  demás familias — eso lo maneja `evaluate.py`/`train.py`
  externamente, `build_regressor` no necesita saberlo.

### 3.2. `swmm_resilience/ml/bench/models/xgboost_family.py` (patrón a seguir)

```python
FAMILY = "xgboost"
SCALE_FEATURES = False
# ... build_classifier/build_regressor con Pipeline([imputer, model]) sin scaler
```

El contrato que cualquier familia (nueva o modificada) debe exponer, tal
como lo exige `registry.py` (docstring del módulo):

> "Una familia es un módulo que expone `FAMILY`, `SCALE_FEATURES`,
> `build_classifier(params, scale_pos_weight)`, `build_regressor(params)`
> y `preprocessing_descriptor(params)`. El banco sólo habla con esta
> interfaz."

Si cambias SVM internamente (otro kernel, otra aproximación, otro
estimador) **no necesitas tocar `registry.py`, `evaluate.py`, `train.py`
ni `main.py`** — solo el contenido de `svm_family.py` (o un archivo nuevo,
si prefieres mantener el SVM "puro" como referencia y crear una familia
separada — ver nota en sección 5.4).

### 3.3. Registro de familias — `swmm_resilience/ml/bench/registry.py`

```python
_FAMILIES = {
    module.FAMILY: module
    for module in (xgboost_family, random_forest_family, linear_family, svm_family, mlp_family)
}
```

`get_family(name)` busca por el string `FAMILY` de cada módulo. Si creas
una familia nueva en vez de modificar `svm_family.py` in-place, hay que
importarla aquí y añadirla a esta tupla.

### 3.4. `config.yaml` — bloque de la familia SVM

```yaml
  families:
    svm:
      enabled: true
      # Justificacion: C=10 y kernel RBF son los valores que usaba el stack
      # legacy (ML_MODEL_CONFIGS.svr_rbf), conservados para que los numeros
      # sean comparables con lo ya reportado.
      classifier: {C: 10.0, kernel: "rbf", gamma: "scale"}
      regressor: {C: 10.0, kernel: "rbf", epsilon: 0.1}
```

Estos valores se reenvían como `params` a `build_classifier`/
`build_regressor` — son la fuente de verdad real (los defaults dentro de
`svm_family.py` son solo un respaldo). Si cambias el enfoque (p. ej. a
Nystroem+SGD), este bloque necesita nuevas claves acordes al nuevo
estimador (`n_components`, `alpha`, `max_iter`, etc. en vez de
`kernel`/`gamma`/`epsilon`).

### 3.5. Espacio de búsqueda de Optuna — `swmm_resilience/ml/bench/tune.py:92-100`

```python
"svm": {
    "classifier": {
        "C": ("float_log", 1e-2, 100.0),
        "gamma": ("categorical", ["scale", "auto"]),
    },
    "regressor": {
        "C": ("float_log", 1e-2, 100.0),
        "epsilon": ("float", 0.01, 1.0),
    },
},
```

Ya existe un espacio de búsqueda para `--bench-tune --models svm`, pero
asume los parámetros actuales (`C`, `gamma`, `epsilon`). Si cambias el
estimador subyacente, este diccionario necesita actualizarse con los
hiperparámetros nuevos o `--bench-tune` fallará con un `KeyError`/
parámetros inválidos.

## 4. Infraestructura compartida relevante (no tocar sin necesidad)

### 4.1. `filter_estimator_params` — `swmm_resilience/ml/bench/param_filter.py`

Filtra claves reservadas (`scale_pos_weight`) y excluye claves
específicas por familia (SVM excluye `epsilon` del lado clasificador,
porque `epsilon` es solo de `SVR`). No es una allow-list: una clave
desconocida sigue generando un `TypeError` real en el `__init__` del
estimador — así que un hiperparámetro nuevo (p. ej. `n_components` de
`Nystroem`) no necesita registrarse aquí, solo pasar por `config.yaml` →
`build_classifier`/`build_regressor`.

### 4.2. El loop de folds — `swmm_resilience/ml/bench/evaluate.py:127-179, 182-227`

```python
for (protocol, fold_id), chunk in prepared.folds.groupby(["protocol", "fold_id"]):
    train_idx = chunk.loc[chunk["split"] == "train", "sample_idx"].to_numpy()
    test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"].to_numpy()
    yc_pred, yc_prob, yr_pred = _fit_predict_fold(...)
```

Esto es genérico para las 5 familias — no hace falta (ni conviene)
tocarlo para arreglar SVM.

### 4.3. Por qué `predict_proba` es obligatorio hoy — `evaluate.py:161`, `metrics.py:44`

```python
yc_prob = classifier.predict_proba(X_test)[:, 1]   # evaluate.py:161
"auc_roc": float(roc_auc_score(y_true, y_prob)) ... # metrics.py:44
```

`evaluate.py` llama `predict_proba` de forma **incondicional** — no hay
una ruta alternativa con `decision_function` hoy. Esto es importante para
la opción recomendada (sección 5.3): si usas `SGDClassifier`, elige
`loss="log_loss"` (da `predict_proba` nativo, sin calibración extra) en
vez de `loss="hinge"` (el SVM lineal "puro", que solo tiene
`decision_function` y rompería esta línea sin tocar `evaluate.py`).
`sklearn.metrics.roc_auc_score` en sí acepta perfectamente un score de
`decision_function` en vez de una probabilidad calibrada — pero
adaptarlo requeriría modificar `evaluate.py`, que es código compartido
por las 5 familias. Evitarlo (usando `log_loss`) mantiene el cambio
100% contenido a `svm_family.py`, que es preferible.

## 5. Opciones para que SVM sea evaluable, con tradeoffs

### 5.1. Kernel lineal (`kernel="linear"` en SVC/SVR, o `LinearSVC`/`LinearSVR`)

- **Velocidad**: `SVC(kernel="linear")` sigue usando libsvm con una
  complejidad parecida a la RBF en el peor caso — no resuelve el
  problema. `LinearSVC`/`LinearSVR` (liblinear) sí escalan mucho mejor,
  cerca de O(n).
- **Precisión esperada — motivo para desconfiar de esta opción**: la
  familia **Lineal** (regresión logística + ridge, ya evaluada) quedó
  **última del ranking**: RMSE 201.18, F1 0.817, NSE regresor oracle
  0.071 (`docs/resultados/2026-10-05-benchmark-fase-c2.md`, tabla
  sección 3). Si el problema no es linealmente separable para esa
  familia, es razonable esperar que un SVM con kernel lineal tenga el
  mismo techo bajo — el valor de SVM en este problema probablemente
  viene precisamente de la no linealidad del kernel RBF.
- `LinearSVC` no tiene `predict_proba` nativo (solo `decision_function`)
  — necesitaría `CalibratedClassifierCV` envolviéndolo, lo cual reintroduce
  parte del costo de calibración que se quería evitar (aunque mucho menor
  que el de SVC con RBF).
- **Veredicto**: rápido de implementar y de descartar, pero poco
  prometedor en precisión. Útil solo como punto de referencia barato, no
  como solución final.

### 5.2. Submuestreo de filas antes de `.fit()` (mantiene kernel RBF)

- Tomar una muestra (aleatoria o estratificada por `inunda`) de, p. ej.,
  10,000-20,000 filas por fold antes de ajustar `SVC`/`SVR`, conservando
  el kernel RBF y por tanto la no linealidad.
- **Pros**: cambio pequeño y localizado (unas pocas líneas en
  `build_classifier`/`build_regressor`, o más limpio, un paso
  `FunctionTransformer` adicional en el Pipeline... aunque ojo:
  submuestrear filas de `X` sin tocar `y` en el mismo paso no encaja
  naturalmente en un `Pipeline` de sklearn — un `Pipeline` transforma
  features, no reduce filas con su `y` correspondiente. Implementarlo
  limpio probablemente significa submuestrear ANTES de entrar al
  `Pipeline`, es decir dentro de `_fit_predict_fold` en `evaluate.py`, o
  envolviendo el estimador en una clase custom que submuestree dentro de
  su propio `.fit(X, y)`. Esto sí toca código compartido o añade una capa
  de estimador wrapper — más trabajo del que parece a primera vista.
- **Contras**: ya no es "SVM entrenado sobre el mismo dataset que las
  otras 4 familias" — hay que documentarlo explícitamente como
  comparación con menos datos de entrenamiento. Probablemente peor
  rendimiento que las otras familias solo por tener menos filas, aparte
  de cualquier limitación propia del kernel.
- **Veredicto**: viable, pero el submuestreo "limpio" dentro del patrón
  de Pipeline existente no es tan directo como parece — revisar bien
  antes de subestimar el esfuerzo.

### 5.3. Aproximación de kernel + solver lineal rápido (RECOMENDADA)

`sklearn.kernel_approximation.Nystroem` (o `RBFSampler`) + `SGDClassifier`/
`SGDRegressor`:

```python
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier, SGDRegressor

Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("kernel_approx", Nystroem(kernel="rbf", gamma=..., n_components=...)),
    ("model", SGDClassifier(loss="log_loss", class_weight={0: 1.0, 1: scale_pos_weight}, random_state=ML_RANDOM_STATE)),
])
```

- **Por qué es la mejor opción**: `Nystroem` aproxima el mapeo de
  features que el kernel RBF induciría, con un número fijo de
  componentes (`n_components`, p. ej. 200-500) — el costo de ese paso es
  O(n · n_components), no O(n²). `SGDClassifier`/`SGDRegressor` sobre esas
  features aproximadas es un solver lineal, O(n) por época. El resultado
  final sigue siendo (aproximadamente) un clasificador/regresor no
  lineal tipo RBF, no uno lineal puro — preserva la razón de ser de SVM.
- `SGDClassifier(loss="log_loss")` da `predict_proba` nativo sin
  calibración extra (ver 4.3) — evita el costo de `probability=True` sin
  tocar `evaluate.py`.
- `SGDRegressor` entrena directo sobre el target ya transformado con
  `log1p` (igual que todas las demás familias, sin cambios ahí).
- Nuevos hiperparámetros a exponer en `config.yaml` y en
  `tune.py:SEARCH_SPACES["svm"]`: `n_components` (tamaño de la
  aproximación — más alto = más fiel al RBF real pero más lento),
  `gamma` (igual que antes), `alpha` (regularización de SGD),
  posiblemente `max_iter`/`tol` para convergencia.
- **Contras**: es una aproximación, no un SVM "real" — hay que
  documentarlo como tal en el informe/tesis (sigue siendo honesto:
  "SVM aproximado con Nystroem" es una categoría reconocida en la
  literatura, no un atajo oculto). Requiere elegir bien `n_components`
  (trade-off directo velocidad/fidelidad) y puede necesitar algo de
  tuning para igualar el comportamiento del RBF exacto en C=10.

### 5.4. Reducir folds solo para iteración rápida durante desarrollo

No es una solución final — el ranking oficial debe seguir usando el
protocolo real (`LeaveOneShapeOut`, 40 folds completos), exactamente
igual que ya se distingue entre el `GroupKFold(5)` interno de Optuna
(exploratorio) y el ranking oficial (ver
`docs/resultados/2026-10-05-benchmark-fase-c2.md`, nota metodológica
sección 4). Pero mientras se prueba una reconfiguración nueva de SVM,
tiene sentido iterar rápido con `--models svm` (sección 6) sobre un
subconjunto de folds o de filas antes de lanzar la corrida completa de
70 folds × 2 ajustes.

### 5.5. SVM acelerado por GPU (ThunderSVM / cuML)

Investigado y descartado por ahora: **no hay ninguna librería SVM
acelerada por GPU instalada** (`requirements.txt` no tiene `thundersvm`
ni `cuml`/RAPIDS), y **no hay ninguna ruta GPU usada en este proyecto**
para SVM específicamente (`torch` sí está instalado pero es para la
familia MLP y el pipeline CNN/surrogate temporal, no relacionado). Añadir
esto implicaría una dependencia pesada, potencialmente problemática en
Windows, y sin certeza de que haya GPU disponible en el entorno real de
ejecución. Última opción, no la primera a intentar.

## 6. Cómo aislar corridas de SVM para probar una reconfiguración

Todo esto puede correrse sin tocar las otras 4 familias — `--models`
filtra el banco a una lista de familias (`main.py`, `_run_bench`,
líneas 154-158):

```bash
# Solo entrenar + evaluar SVM (reutiliza el prep_id más reciente de --bench-prepare)
C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model\venv\Scripts\python.exe main.py --bench-train --bench-evaluate --models svm

# Las 4 etapas (prepare+train+evaluate+promote) pero restringido a SVM
# (promote sigue leyendo config.bench.promote, que hoy es "xgboost" -- no promoverá SVM salvo que lo cambies ahí)
C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model\venv\Scripts\python.exe main.py --bench --models svm

# Búsqueda de hiperparámetros con Optuna, solo SVM (requiere --models con exactamente una familia)
C:\Users\lf.fajardo\Downloads\Tesis\herramienta-ia-model\venv\Scripts\python.exe main.py --bench-tune --models svm
```

Recomendación práctica: antes de lanzar `--bench-evaluate` completo (70
folds), probar primero un solo `.fit()` suelto en una sesión de Python
interactiva o un script chico (tipo
`scripts/validar_features_tributarias.py`, que ya tiene el patrón de
`LeaveOneGroupOut` manual) para medir el tiempo de un fold antes de
comprometerte a los 70.

## 7. Qué queda pendiente de actualizar una vez SVM funcione a tiempo razonable

Una vez SVM corra y dé un resultado honesto bajo `LeaveOneShapeOut`:

1. **Ranking**: re-correr `--bench` completo (o al menos
   `--bench-evaluate`) con las 5 familias para que `outputs/bench/reports/ranking.csv`
   incluya a SVM y confirme (o cambie) quién gana — no asumir que
   XGBoost sigue ganando sin volver a correr el ranking con los 5.
2. **Mapas de paridad por familia** — `scripts/generar_paridad_por_familia.py`
   hoy solo acepta `--models xgboost,random_forest,linear,mlp` (línea 8-9
   del docstring) — hay que agregar `svm` a esa lista y regenerar sus
   `parity_nodes_*`/`parity_agg_*`/`node_profile_*` para el archivo
   (`resultados_archivados/.../paridad/svm/`).
3. **`docs/resultados/2026-10-05-benchmark-fase-c2.md`**: actualizar la
   tabla de ranking (sección 3) con la fila real de SVM, y marcar
   `- [ ] Configurar SVM...` como hecho en la sección "6. Pendiente".
4. **`docs/resultados/2026-10-06-explicacion-completa.md`**: la sección 4
   ("El problema de SVM") necesita una actualización o una sección nueva
   explicando qué se hizo y el resultado final.
5. Si SVM termina ganando o quedando cerca del primer lugar, reconsiderar
   si sigue siendo XGBoost la familia promovida (`config.bench.promote` en
   `config.yaml`) — eso ya es una decisión de producto, no solo técnica.
6. Si cambiaste el contrato de hiperparámetros (opción 5.3), actualizar
   también `config.yaml` (`bench.families.svm`) y
   `tune.py:SEARCH_SPACES["svm"]` en consistencia con los nuevos nombres
   de parámetros — de lo contrario el banco falla con un `TypeError`
   silencioso al construir el estimador (ver `filter_estimator_params`,
   sección 4.1 — no es un allow-list, así que un parámetro mal escrito
   explota en el `__init__`, no se ignora).

## 8. Notas sueltas / gotchas

- El modelo final (XGBoost, `tabular_v4_21`, 21 features) ya está
  promovido a producción (`outputs/models/`) y protegido por las guardas
  de contrato añadidas en la migración reciente — nada de lo de SVM
  afecta eso. Esto es trabajo puramente dentro del banco de comparación
  (`outputs/bench/`), no toca producción salvo que explícitamente se
  decida re-promover otra familia al final (punto 5 de la sección 7).
- Todo el trabajo de SVM, igual que el resto de esta rama, debería
  evaluarse bajo el protocolo oficial `LeaveOneShapeOut` para ser
  comparable con el ranking ya reportado — no el `GroupKFold(5)` interno
  de Optuna, que da números más altos/peores por diseño (ver nota
  metodológica en `docs/resultados/2026-10-05-benchmark-fase-c2.md`
  sección 4).
- `ML_RANDOM_STATE` (de `swmm_resilience/config.py`) es la semilla que
  usan todas las familias — reutilizarla en cualquier estimador nuevo
  (`SGDClassifier(random_state=ML_RANDOM_STATE)`, etc.) para que las
  corridas sean reproducibles igual que las demás.
