# Flujo actual del proyecto (2026-09-03)

Descripción del estado real leyendo el código de la branch
`feature/hydrograph-augmentation-sql-persistence`.

## 1. Entradas

| Qué | Dónde | Formato |
|---|---|---|
| Red de drenaje | `data/networks/chico_hydro-qx1/…Qx1.00.inp` (fijado en `config.yaml`) | SWMM `.inp` |
| Parámetros del run | `config.yaml` | YAML |
| Formas de hidrograma | `data/hydrograph_shapes/*.csv` (≈2 docenas, una por archivo) | CSV `time_h,q_norm` (normalizado) |

`config.yaml` define: red, rango de factores (`0.2 … 5.0` paso `0.2` → 25
factores), umbral de inundación (`flood_threshold_m3 = 1.0`),
hiperparámetros XGBoost, métodos de evaluación y factores a graficar.

## 2. Simulación SWMM

- Forma **base**: 25 corridas (una por factor), hidrograma tomado del
  `[TIMESERIES]` del `.inp`.
- Cada forma adicional: 25 corridas más (forma × factor), reescalando el
  hidrograma normalizado por el `base_inflow` de cada nodo.
- Total ≈ `25 × (1 + nº_formas)` corridas. Cada corrida produce un `.rpt`
  temporal (no se conserva).

## 3. Extracción de features

- **Estáticas + topológicas** (una vez, desde el `.inp`): `elev_fondo`,
  `prof_max`, `n_tuberias_in/out`, `diam_max_in/out`, `pendiente_max_in`,
  `pendiente_out`, `base_inflow_lps`, `dist_outfall_m`,
  `n_nodos_aguas_arriba`, `q_pico_acum_base`, `upstream_capacity_lps`,
  `coord_x/y`.
- **Dinámicas** (por factor × forma): `q_pico_nodo`,
  `q_pico_acum_escalado`, `duracion_horas`, `tiempo_al_pico_h`.
- **Labels** (desde el `.rpt`): `vol_inundacion_m3` y `inunda` (1 si el
  volumen supera `flood_threshold_m3`).

## 4. Dataset maestro — CSV

`assemble_dataset` hace merge `static ⨝ dynamic ⨝ labels` por `node_id`,
concatena todas las corridas y escribe:

```
data/training/dataset_final.csv   (~12 MB, 24 columnas)
```

1 fila = 1 nodo × factor × forma. Columnas: las 16 estáticas + `factor_mult`
+ 4 dinámicas + `shape_id` + `vol_inundacion_m3` + `inunda`. Se valida con
`validate_dataset` (conteos nodo/corrida coherentes).

**Este CSV es el único artefacto de datos que consume el entrenamiento.**

## 5. Contrato de features (17)

`swmm_resilience/ml/contracts.py` fija `FEATURE_COLUMNS_V17`: un subconjunto
**ordenado** de 17 columnas del CSV. Quedan fuera a propósito `node_id`,
`coord_x/y`, `factor_mult`, `shape_id` (son metadata para agrupar/estratificar,
no entradas del modelo). `validate_frame` fuerza `float64`, rechaza nulos en
columnas requeridas, infinitos y columnas complejas.

## 6. Entrenamiento (CLI, activo)

`train_models` (`swmm_resilience/ml/trainer.py`):

- **Clasificador** `inunda`: XGBoost sobre las 17 features validadas de
  todas las filas, `scale_pos_weight="auto"`.
- **Regresor** `vol_inundacion_m3`: XGBoost sobre las mismas 17 features
  validadas, **solo filas `inunda == 1`**, objetivo en `log1p`.
- Salida:

```
outputs/models/classifier.joblib
outputs/models/regressor.joblib
outputs/models/training_inp_hash.txt   (hash MD5 del .inp de entrenamiento)
```

Inferencia = clasificador decide si inunda; si sí, regresor estima el volumen
(`expm1`).

## 7. Evaluación y salidas visuales

- `evaluate_models` → LOSO + GroupKFold5 estratificado por factor →
  `outputs/metrics/metrics_classifier.json`, `metrics_regressor.json`,
  `metrics_endtoend.json`, `metrics_by_factor.json`.
- Importancia de variables → `outputs/metrics/`.
- Mapas de inundación:
  - forma base → `outputs/maps/flood_map_factor_X.XX.png`
  - una subcarpeta por forma adicional → `outputs/maps/<shape_id>/…`

## 8. Persistencia SQLite v17 (opcional, ruta mínima)

`python main.py --persist-sql` (vía `swmm_resilience/database/csv_backfill.py`):

1. Lee `dataset_final.csv` (requiere columna `shape_id` y ≥1 fila inundada).
2. Aplica migraciones 001–005 sobre `outputs/training_v17.sqlite3`.
3. Vuelca `networks / nodes / scenarios / runs / node_features / node_results`.
4. Entrena **de nuevo** un par GroupKFold5-por-factor (mismos builders que el
   CLI) y persiste `training_runs → model_evaluations → oof_predictions →
   trained_models → model_metrics` (los `.joblib` van embebidos como blob).

No escribe `model_candidates / model_rankings / model_promotions /
model_selections`: los modelos quedan como **artefactos históricos válidos**,
no como “selección activa”. No toca los `.joblib` de `outputs/models/`.

## 9. Dónde se guarda cada cosa

| Formato | Contenido | Ruta | Rol |
|---|---|---|---|
| CSV | dataset maestro | `data/training/dataset_final.csv` | **entrada del entrenamiento** |
| joblib | modelos entrenados (CLI) | `outputs/models/` | inferencia |
| JSON | métricas de evaluación | `outputs/metrics/` | reporte |
| PNG | mapas, curvas, importancia | `outputs/maps/`, `outputs/metrics/`, `outputs/…` | reporte |
| SQLite v17 | evidencia de proveniencia | `outputs/training_v17.sqlite3` | solo si se corre `--persist-sql` |
| XLSX | tablas de revisión del dataset | `outputs/…/dataset_review_tables.xlsx` | opcional, `--analyze-features` / `analysis/eda.py` |
| XLSX + joblib | pipeline legacy de 7 modelos | `model_artifacts/`, `swmm_resilience.db` | solo GUI de escritorio |

Excel **no** participa del flujo productivo: solo lo generan la EDA opcional
y el pipeline legacy de la GUI.

## 10. ¿Hay pipelines para ejecutar los modelos?

No hay orquestador (Airflow/Prefect/Make). **El pipeline es `main.py`** con
flags que seleccionan el modo:

| Comando | Hace |
|---|---|
| `python main.py` | Completo: SWMM → extracción → CSV → train → eval → mapas |
| `python main.py --skip-extraction` | Reusa el CSV, corre train + eval + mapas |
| `python main.py --only-ml` | Reusa el CSV, solo train + eval |
| `python main.py --only-maps` | Solo regenera mapas desde el CSV |
| `python main.py --persist-sql` | Vuelca CSV + un entrenamiento a SQLite v17 |
| `python main.py --predict --factor X` | Inferencia sin SWMM (usa los `.joblib`) |
| `python main.py --simulate --factor X` | SWMM + ML para un factor suelto (comparación) |
| `python main.py --resilience-curve` / `--flood-volume-curve` / `--factor-comparison` | Curvas SWMM vs ML |
| `python main.py --analyze-features` | Correlación, ablación, SHAP |
| `python main.py --evaluate-shapes` / `--evaluate-generalization` | SWMM vs ML por forma / en factores no vistos |
| `python main.py --evaluate-hydrographs DIR --base-inp … --clf-path … --reg-path …` | Validación batch de un directorio de hidrogramas |
| `python -m swmm_resilience.desktop.app` | GUI legacy: comparación de 7 modelos (`ml/train.py`), pipeline separado |
| `python -m pytest -q` | Tests (517 pasan; 2 fallos ambientales: Tcl/Tk y módulo `build`) |

## 11. Orden real de los datos

```
config.yaml + .inp + data/hydrograph_shapes/*.csv
        │
        ▼  SWMM  (25 factores × ~25 formas)
     .rpt temporales
        │
        ▼  extracción  (17 features + 2 labels + metadata)
  data/training/dataset_final.csv
        │
        ├──▶ train_models → outputs/models/*.joblib → evaluación → outputs/metrics/*.json + outputs/maps/*.png
        │
        └──▶ (opcional) --persist-sql → outputs/training_v17.sqlite3
```

---

# 12. Migración a SQLite como única fuente de verdad

Decisión: **SQLite, un solo archivo, una sola máquina** (la de RAM alta). Sin
Postgres, sin DuckDB, sin servidor. El CSV pasa a ser export opcional; el
Excel sale del pipeline.

## 12.1 Estado final

| Artefacto | Antes | Después |
|---|---|---|
| Datos de features/labels | `data/training/dataset_final.csv` | tablas v17 en `outputs/training_v17.sqlite3` |
| Frame de entrenamiento | `pd.read_csv(...)` en ~8 sitios | 1 loader → `SELECT` sobre la vista `training_samples_v17` |
| CSV | fuente de verdad | export bajo demanda (`--export-csv`), no lo lee nadie del pipeline |
| Excel | EDA + GUI legacy | igual (fuera del flujo productivo), sin cambios |
| Modelos `.joblib` | `outputs/models/` | siguen en disco; la DB guarda ruta + `sha256`, **no** el blob |

Lo que **no** cambia: el entrenamiento sigue en RAM, XGBoost igual, el
contrato de 17 features (`FEATURE_COLUMNS_V17`) igual, LOSO/GroupKFold5 igual.

## 12.2 El punto crítico: contrato de la vista

**Corrección (2026-09-03):** este punto estaba mal planteado. La vista **no**
devuelve las 24 columnas del CSV, y no debe hacerlo. Lo que devuelve son
**27 columnas**:

- 8 de identidad: `run_id`, `network_id`, `scenario_id`, `scenario_key`,
  `scenario_kind`, `factor_mult`, `shape_id`, `node_id`;
- las 17 features del contrato `tabular_v3_17`;
- 2 targets: `inunda`, `vol_inundacion_m3`.

Comparado con `dataset_final.csv`: **agrega** las 5 columnas de identidad
(`run_id`/`network_id`/`scenario_id`/`scenario_key`/`scenario_kind`) y
**omite** `coord_x`/`coord_y` (las coordenadas viven en la tabla `nodes` y en
el `.inp`, no son una feature ni parte de una muestra de entrenamiento).

Se verificó consumidor por consumidor que **nadie lee `coord_x`/`coord_y` del
frame del dataset**: los mapas arman su propio dict de coordenadas desde el
`.inp` (`visualization/flood_map.py`), `ml/predict.py` construye sus features
desde el `.inp`, y `trainer`/`evaluator`/`feature_analysis` usan solo las 17
features + labels + `factor_mult`. El único consumidor de esas dos columnas es
`backfill_networks_and_runs` (para poblar `nodes`), que solo corre en la ruta
`--persist-sql` alimentada por CSV.

Garantías que sí se exigen (y que el loader ya valida):

- dtypes: 17 features → numéricas; `inunda` → int 0/1; `shape_id`/`node_id`
  → texto; `factor_mult` → float;
- nulos sólo en las columnas declaradas `nullable` en `ml/contracts.py`
  (`diam_max_*`, `pendiente_*`, `dist_outfall_m`, `upstream_capacity_lps`);
- `vol_inundacion_m3` = 0.0 cuando no hubo inundación (no NULL), y no negativo;
- solo runs `COMPLETE`, con `node_count` coherente con las filas de
  `node_features`/`node_results`.

**Ya implementado** (commits `40c2459` vista → `80d8abc`/`ec7abf2` fixes →
`2628b93`/`ad7b5d7` tests): `tests/database/test_training_view_v17.py`
(≈794 líneas) cubre orden/nombres de columnas de la vista, exclusión de
runs no-`COMPLETE`, aislamiento del snapshot de lectura vía `SAVEPOINT` bajo
WAL (una escritura concurrente a mitad de lectura no puede colar filas
inconsistentes), validación de cardinalidad `node_count` vs
features/results, y roundtrip de exportación a CSV.

**Lo que ese test NO hace** (y que la versión anterior de este documento daba
por hecho): no compara contra ningún `dataset_final.csv` de referencia.
Verifica la forma canónica propia de la vista (27 columnas), no paridad con el
CSV. Esa comparación se agrega en la Fase 2 del plan de cutover, ya con el
contrato correcto: las 22 columnas compartidas deben coincidir fila por fila
(ver `docs/superpowers/specs/2026-09-03-sqlite-v17-training-frame-cutover-design.md`).
El estado verde se infiere del historial de commits; no se re-ejecutó `pytest`
en esta revisión por falta del módulo en el entorno.

## 12.3 Loader único

**Ya implementado, pero no en la ruta que este documento proponía.** No hay
que crear `swmm_resilience/dataset/loader.py`: el equivalente ya existe en
`swmm_resilience/database/training_queries.py::load_training_samples`,
con test dedicado en `tests/database/test_training_view_v17.py` (mismo
commit range que 12.2). Hace:

- `SELECT ... FROM training_samples_v17` (columnas identidad + las 17
  features + `inunda`/`vol_inundacion_m3`, en el orden del contrato);
- valida que los `run_ids` pedidos existan y estén `COMPLETE`;
- valida cardinalidad `node_count` vs filas de `node_features`/`node_results`
  y sus claves (nunca deja pasar un run con datos huérfanos o incompletos);
- corre todo dentro de un `SAVEPOINT` para leer un snapshot estable aunque
  haya un escritor concurrente;
- valida el frame contra `TABULAR_V3_17` antes de devolverlo (nunca sale un
  frame con nulos/infinitos fuera de lo permitido);
- `export_training_samples_csv(conn, output_path, run_ids=None)` ya cubre lo
  que el paso 5 de 12.8 llama `--export-csv` (escritura atómica vía archivo
  temporal + `replace`).

Lo que **no** tiene todavía, y sí pide este punto del plan:

- no abre la DB ni corre `apply_migrations` — recibe una conexión ya
  gestionada (`connect_managed_database` + `apply_migrations` son
  responsabilidad del caller, como hace `--persist-sql` en `main.py`);
- el único filtro es `run_ids` (lista de enteros positivos); no hay filtro
  directo por `network`, `shape_id`, lista de `factor_mult`, `only_flooded`,
  `sample_frac`/`limit`, ni `chunksize` para streaming — quien quiera esos
  filtros hoy tiene que resolver primero los `run_ids` correspondientes
  (p.ej. consultando `scenarios`/`runs`) y pasarlos.

Ningún consumidor del pipeline (`main.py`, `assembler.py`, `trainer.py`,
`evaluator.py`, `analysis/*.py`) llama a `load_training_samples` todavía —
solo lo usa su propio archivo de test. Todos siguen en `pd.read_csv`. El
trabajo pendiente real de este punto es (a) decidir si se amplían los
filtros o se resuelven vía `run_ids` desde arriba, y (b) conectar los
consumidores — no reescribir el loader desde cero.

## 12.4 Cambios por archivo

| Archivo | Cambio |
|---|---|
| `config.yaml` | añadir `dataset.db_path: "outputs/training_v17.sqlite3"`; marcar `output_path` como ruta de export |
| `swmm_resilience/dataset/assembler.py` | además de (o en vez de) escribir CSV, insertar en `nodes/scenarios/runs/node_features/node_results` vía las funciones de `csv_backfill.py` (o moverlas a un módulo `dataset/persist.py`) |
| `swmm_resilience/database/training_queries.py` | **ya existe** (`load_training_samples`, `export_training_samples_csv`, 12.3); falta ampliar filtros y decidir si se mueve/renombra a `dataset/loader.py` o se deja donde está |
| `swmm_resilience/dataset/validator.py` | validar contra la DB (conteos por `network`/`run`) en vez de sobre el CSV |
| `main.py` | reemplazar cada `pd.read_csv(config.dataset.output_path)` por `load_training_frame(...)` con el filtro que ya aplica después (`base_shape_rows`, factores, etc. pasan a ser argumentos del loader); `--persist-sql` deja de existir como paso aparte (el pipeline ya escribe SQL); añadir `--export-csv` |
| `swmm_resilience/analysis/factor_comparison.py` | recibir la DB / el loader en vez de `dataset_path` |
| `swmm_resilience/analysis/resilience.py`, `analysis/flood_volume.py` | ídem; considerar hacer la agregación por factor en SQL |
| `swmm_resilience/ml/evaluator.py`, `ml/trainer.py` | sin cambios de lógica; sólo reciben el frame del loader |
| `swmm_resilience/database/csv_backfill.py` | renombrar: ya no es "backfill de CSV" sino la ruta normal de escritura; `persist_training_run` se mantiene tal cual (entrena y guarda evidencia) |

## 12.5 Modelos y blobs

- `trained_models.model_blob` → **no usar**. Guardar `.joblib` en
  `outputs/models/` y registrar en la fila `model_path` + `model_sha256`
  (ya existe la columna de hash).
- Mantiene la DB chica y el backup barato.

## 12.6 Backup y ubicación

- El `.sqlite3` vive en la máquina de RAM alta, fuera de git (ya está en
  `.gitignore`).
- Backup: `upgrade_database_with_backup` ya copia+verifica; para respaldo
  periódico basta `VACUUM INTO 'ruta/backup-YYYYMMDD.sqlite3'` (copia
  consistente en caliente) a un disco/bucket aparte.
- Un solo escritor a la vez (el pipeline). Lectores concurrentes (mapas,
  análisis) funcionan con WAL, que la conexión gestionada ya activa.

## 12.7 Escala

- Insertar siempre con `executemany` dentro de **una** transacción por run
  (ya lo hace `csv_backfill`). Nunca fila a fila en autocommit.
- Índices necesarios en `node_features`/`node_results`:
  `(run_id)`, `(network_id, node_pk)` — verificar que las migraciones ya los
  crean; si no, migración nueva.
- Correr `tests/database/test_scale_v17.py -m scale` como gate de que la
  vista + loader rinden con ~1M+ filas antes de dar por cerrada la migración.
- `node_timeseries` (datos por paso de tiempo) es lo único que llevaría a
  decenas de millones de filas; hoy no se llena. Si se activa, es tabla
  aparte y **no** entra en `training_samples_v17`.

## 12.8 Orden de ejecución

1. ~~Test de contrato de la vista (12.2) contra un `dataset_final.csv` de
   referencia. Verde.~~ **Hecho.** `training_samples_v17` (migración 001) +
   `tests/database/test_training_view_v17.py`. Ver 12.2.
2. ~~`load_training_frame` (12.3) + test: mismo frame que `read_csv` del CSV
   de referencia.~~ **Hecho, con alcance más chico que el planteado.**
   `training_queries.py::load_training_samples` + mismo archivo de test.
   Falta: filtros más allá de `run_ids`, y que algo del pipeline lo llame.
   Ver 12.3 para el detalle de la brecha.
3. Cambiar consumidores de sólo-lectura uno por uno
   (`resilience` → `flood_volume` → `factor_comparison` → `--only-maps` →
   `--only-ml`), corriendo la suite tras cada uno.
4. `assembler.py` escribe a SQL; `validate_dataset` valida contra SQL.
5. Quitar la escritura de CSV del pipeline; añadir `--export-csv`.
6. Borrar `--persist-sql` (ya redundante) o dejarlo como alias de re-entrenar
   la evidencia.
7. `-m scale` verde.
8. Actualizar este documento y `COMANDOS.md`.

## 12.9 Listo cuando

- [ ] `python main.py` completo escribe sólo en `outputs/training_v17.sqlite3` (+ `.joblib` + `.json` + `.png`).
- [ ] Ningún módulo del pipeline llama a `pd.read_csv` sobre el dataset.
- [ ] `training_samples_v17` coincide con el CSV en las 22 columnas
  compartidas, fila por fila (ver §12.2 — la vista tiene 27 columnas, no 24;
  el test de paridad es trabajo de la Fase 2, todavía no existe).
- [ ] `--export-csv` regenera el frame desde SQL — la función que lo hace
  (`export_training_samples_csv`) ya existe y tiene test, pero no está
  expuesta como flag de `main.py` todavía. Nota: el CSV exportado tiene la
  forma nueva de 27 columnas, no la histórica de 24 (decisión tomada el
  2026-09-03: se aceptan las 5 columnas de identidad y se descartan
  `coord_x`/`coord_y`, que ningún consumidor lee).
- [ ] `pytest -q` y `pytest -m scale` verdes.
- [ ] `COMANDOS.md` y este archivo actualizados.
