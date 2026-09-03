# Comandos del pipeline — estado real (2026-09-03)

Este archivo documenta únicamente lo que **funciona hoy**, verificado leyendo
el código real (no specs ni planes). Ver la sección final "Qué NO está
conectado todavía" para lo que existe como esquema pero no se usa en
producción.

> Actualizado junto con `docs/FLUJO_ACTUAL.md` (2026-09-03): se completó el
> corte de **lectura** a SQLite (Tasks 1-6 del plan de cutover) — ver más
> abajo. El corte de **escritura** (el assembler escribiendo directo a SQL,
> `--export-csv`, retirar `--persist-sql`) es una fase posterior, todavía no
> hecha.

## Arquitectura actual (resumen honesto)

- **Un solo pipeline activo por CLI** (`main.py`): SWMM → extracción → CSV
  (`data/training/dataset_final.csv`) → XGBoost (17 features) → `.joblib`
  → mapas/evaluación. Esto es lo que corren todos los comandos de abajo.
  **La escritura sigue siendo CSV** (`assembler.py` sigue llamando
  `dataset.to_csv(...)`); el corte de escritura a SQL no ha pasado todavía.
- **Corte de lectura completo (2026-09-03):** los ocho sitios de sólo-lectura
  de `main.py` — `--resilience-curve`, `--flood-volume-curve`,
  `--factor-comparison`, `--only-maps`, la rama de lectura de
  `--only-ml`/`--skip-extraction`, `--analyze-features`, `--evaluate-shapes`,
  `--evaluate-generalization` — ya no leen `dataset_final.csv`: todos llaman
  a `load_training_frame(config.dataset.db_path)`
  (`swmm_resilience/database/training_queries.py`), que hace `SELECT` sobre
  la vista `training_samples_v17` en `outputs/training_v17.sqlite3`.
  `pd.read_csv` queda en un único lugar de todo `main.py`: dentro de
  `--persist-sql` (`main.py:573`), porque esa rama necesita
  `coord_x`/`coord_y` para `backfill_networks_and_runs`, columnas que el
  frame de 27 columnas del loader no trae — es deliberado y permanece así en
  esta fase (ver `docs/FLUJO_ACTUAL.md` §12.2/§12.3).
- **Formato de almacenamiento: sigue siendo CSV + joblib para la escritura**,
  no SQL. El esquema SQLite v17 (`swmm_resilience/database/sql/001_v17_initial.sql`
  … `005_provenance_integrity.sql`) existe y está migrado y probado. Con
  `--persist-sql` **sí hay código que escribe ahí**: vuelca el CSV a
  `networks/nodes/scenarios/runs/node_features/node_results` y entrena +
  persiste `training_runs → model_evaluations → oof_predictions →
  trained_models → model_metrics` (`swmm_resilience/database/csv_backfill.py`).
  Es la única ruta que escribe SQL; el resto de `main.py` (entrenar,
  predecir, evaluar, ensamblar) sigue produciendo CSV como fuente de
  escritura — la base v17 sólo recibe datos vía `--persist-sql`, un paso
  aparte y opcional que no se retroalimenta hacia el resto del pipeline.
- **Sigue existiendo un segundo pipeline** (`swmm_resilience/ml/train.py`,
  comparación de 7 modelos) — no accesible desde `main.py`, solo desde la
  GUI de escritorio (`python -m swmm_resilience.desktop.app` o el ejecutable
  correspondiente).

## Setup

```bash
python -m pip install -r requirements.txt
```

- Python **>= 3.11** (`pyproject.toml`).
- No hace falta instalar nada para SQLite (es stdlib); no hay paquete pip
  llamado `sqlite3`.
- `config.yaml` tiene `dataset.db_path` (default `"outputs/training_v17.sqlite3"`,
  cargado en `swmm_resilience/config.py` como `config.dataset.db_path`): es
  la base que leen los ocho sitios de sólo-lectura de `main.py` vía
  `load_training_frame` (ver arriba) y donde escribe `--persist-sql`.

## Tests

```bash
# Suite completa (excluye tests de escala por defecto, ver pytest.ini)
./venv/Scripts/python.exe -m pytest -q

# Solo la base de datos v17 (migraciones, triggers, upgrade, recovery, etc.)
./venv/Scripts/python.exe -m pytest tests/database -q

# Incluir los tests de escala (1M filas OOF, ~150s, opt-in)
./venv/Scripts/python.exe -m pytest tests/database -q -m scale

# Verificar que el paquete se puede construir como wheel
./venv/Scripts/python.exe -m pytest tests/packaging -q
```

> Nota de entorno (2026-09-03): en esta máquina el `python` desnudo del PATH
> es un 3.14 sin dependencias instaladas — usar el venv de arriba. El venv
> (`venv/`) se armó con Python 3.12.10 + `requirements.txt` + el paquete
> `build` (necesario para `tests/packaging/`, ausente de
> `requirements.txt` — instalarlo aparte si el venv es nuevo).

**Medido (2026-09-03), HEAD `7a5ce36`, `./venv/Scripts/python.exe -m pytest -q`:
531 passed, 3 deselected** (105 s). Reemplaza los números previos de este
archivo (504) y de `docs/FLUJO_ACTUAL.md` (517), ninguno de los cuales había
sido confirmado por una corrida real.

La afirmación previa de este documento — que `tests/desktop/test_results_tab.py`
fallaba por falta de `tk.tcl` — **no se reprodujo** en esta corrida:
`tests/desktop/` pasa completo. Lo que sí existe, y es distinto de un fallo
duro, es un **flake intermitente** en
`tests/desktop/test_results_tab.py`: aparece solo en algunas corridas de la
suite completa y pasa en aislamiento (`pytest tests/desktop/test_results_tab.py -q`).
Es una condición de carrera preexistente de Tkinter/`PhotoImage`, no
relacionada con esta migración de lectura a SQL.

## Pipeline activo (CLI, `main.py`) — CSV + 17 features

```bash
# Pipeline completo desde cero: SWMM -> extracción -> CSV -> entrenamiento -> evaluación -> mapas
python main.py

# Ya tienes outputs/training_v17.sqlite3 poblado (--persist-sql) y solo
# quieres reentrenar + evaluar; lee vía load_training_frame, no el CSV
python main.py --only-ml

# Saltar extracción SWMM, leyendo outputs/training_v17.sqlite3 (requiere
# --persist-sql previo), pero seguir con todo lo demás
python main.py --skip-extraction

# Solo regenerar mapas leyendo outputs/training_v17.sqlite3 (sin reentrenar;
# requiere haber corrido --persist-sql al menos una vez)
python main.py --only-maps

# Inferencia para un factor arbitrario sin correr SWMM (usa los .joblib guardados)
python main.py --predict --factor 3.5

# Correr SWMM + ML para un factor arbitrario y generar ambos mapas (comparación)
python main.py --simulate --factor 3.5

# Persistir dataset_final.csv + un entrenamiento GroupKFold5 en
# outputs/training_v17.sqlite3 (esquema SQLite v17) — paso aparte y opcional;
# la escritura sigue siendo manual (ver "Qué NO está conectado todavía")
python main.py --persist-sql
```

### Análisis y visualización (requieren modelos ya generados; las seis desde
### `--resilience-curve` leen `outputs/training_v17.sqlite3` vía
### `load_training_frame` — hace falta haber corrido `--persist-sql` al menos
### una vez; `--hydrograph`/`--network-map` leen directo del `.inp`, no del
### dataset)

```bash
python main.py --hydrograph                    # Hidrograma del nodo con mayor caudal pico
python main.py --network-map                   # Mapa de topología de la red
python main.py --resilience-curve               # Curva de resiliencia SWMM vs ML
python main.py --flood-volume-curve              # Volumen total de inundación por factor
python main.py --factor-comparison               # Volumen SWMM vs XGBoost por nodo, por factor
python main.py --analyze-features                # Correlación, ablación y SHAP
python main.py --evaluate-shapes                  # SWMM vs ML por forma de hidrograma
python main.py --evaluate-generalization          # SWMM vs ML en factores no vistos en entrenamiento
```

### Validación batch de hidrogramas

```bash
python main.py --evaluate-hydrographs data/hydrograph_shapes \
    --base-inp "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp" \
    --clf-path outputs/models/classifier.joblib \
    --reg-path outputs/models/regressor.joblib \
    --out-dir ./validation_output
```

### Flags adicionales útiles

```bash
--flood-threshold 1.0      # Umbral (m3) para considerar un nodo inundado (default: config.yaml)
--allow-inp-mismatch       # Continuar si el .inp no coincide con el hash de entrenamiento (con warning)
--out-dir PATH             # Directorio de salida para validación batch (default: ./validation_output)
```

Nota: `--skip-simulation` existe pero **solo es válido combinado con**
`--skip-extraction` o `--only-ml` — el pipeline aún no indexa archivos `.rpt`
persistentes por sí solo.

## Pipeline legacy (GUI de escritorio, 7 modelos)

```bash
python -m swmm_resilience.desktop.app
```

Usa `swmm_resilience/ml/train.py` (comparación Lasso/Ridge/SVR/SVC/XGBoost…),
guarda en `model_artifacts/` con nombres tipo `regression_xgboost.joblib`
(distinto de `outputs/models/classifier.joblib` que usa el CLI). Internamente
usa una tercera base de datos SQLite (`swmm_resilience.db`, esquema en
`swmm_resilience/database/schema.py`) para almacenar simulaciones — **no** es
el esquema v17 endurecido hoy, son cosas completamente distintas.

## Base de datos SQLite v17 (esquema endurecido; escritura solo vía `--persist-sql`, lectura ya conectada a ocho comandos de `main.py`)

`python main.py --persist-sql` (ver arriba) ya escribe ahí, y los ocho
comandos de sólo-lectura listados en la sección de arriba ya leen de ahí vía
`load_training_frame`. Estos comandos son para inspeccionar/mantener la base
v17 directamente:

```python
from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations

conn = connect_managed_database("ruta/a/tu.sqlite3")
apply_migrations(conn)   # aplica migraciones 001-005 si faltan
```

```python
# Backup + upgrade seguro de una base existente
from swmm_resilience.database.upgrade import upgrade_database_with_backup
receipt = upgrade_database_with_backup("ruta/a/tu.sqlite3", "ruta/a/backups")
```

```python
# Leer de vuelta lo que --persist-sql escribió (27 columnas: identidad + las
# 17 features + labels, ver docs/FLUJO_ACTUAL.md §12.2). load_training_frame
# es lo que llaman los ocho sitios de sólo-lectura de main.py; load_training_samples
# es la función de más bajo nivel sobre la que está construido.
from swmm_resilience.database.training_queries import load_training_frame, load_training_samples
frame = load_training_frame(db_path)                 # abre la conexión, todos los runs COMPLETE
frame = load_training_samples(conn)                   # con una conexión ya abierta
frame = load_training_samples(conn, run_ids=[1, 2])    # runs específicos
```

## Qué NO está conectado todavía

- **La escritura sigue siendo CSV, no SQL.** `assembler.py` sigue llamando
  `dataset.to_csv(...)` sobre `data/training/dataset_final.csv`; nada del
  pipeline principal (`main.py` sin `--persist-sql`) escribe en
  `outputs/training_v17.sqlite3` por sí solo. **`--persist-sql` sí entrena y
  guarda resultados en las tablas SQL v17** (`training_runs`,
  `model_evaluations`, `oof_predictions`, `trained_models`, `model_metrics`)
  — ver `swmm_resilience/database/csv_backfill.py`. Lo que deliberadamente
  **no** escribe es `model_candidates` / `model_rankings` /
  `model_promotions` / `model_selections`: los modelos quedan como
  artefactos históricos válidos, no como "selección activa" (ver
  `docs/FLUJO_ACTUAL.md` §8, §12.2–12.3 para el detalle de qué tanto de la
  vista/loader de lectura ya existe). `--persist-sql` sigue leyendo su
  entrada de `dataset_final.csv` vía `pd.read_csv` (necesita
  `coord_x`/`coord_y` para poblar `nodes`), no del loader — es el único
  `pd.read_csv` que queda en `main.py`.
  **La lectura, en cambio, ya está conectada:** los ocho sitios de
  sólo-lectura de `main.py` (`--resilience-curve`, `--flood-volume-curve`,
  `--factor-comparison`, `--only-maps`, la rama de lectura de
  `--only-ml`/`--skip-extraction`, `--analyze-features`,
  `--evaluate-shapes`, `--evaluate-generalization`) llaman a
  `load_training_frame(config.dataset.db_path)` y leen de
  `outputs/training_v17.sqlite3`, no de `dataset_final.csv`. Falta la Fase 2
  del cutover: que el propio `assembler.py` escriba a SQL (para que no haga
  falta correr `--persist-sql` aparte para poblar la base que esos ocho
  comandos leen), `--export-csv` como flag de `main.py`, y la puerta de
  paridad CSV↔SQL — trabajo pendiente, no hecho hoy (ver
  `docs/FLUJO_ACTUAL.md` §12.8–12.9 y el bloqueador de Fase 2 documentado en
  el plan de cutover).
- El pipeline legacy de 7 modelos (`ml/train.py`) sigue vivo en la GUI —
  retirarlo es una tarea aparte, ya identificada pero no ejecutada.
- La búsqueda de hiperparámetros con Optuna (`docs/superpowers/specs/2026-08-04-optuna-hyperparam-search-design.md`)
  está solo diseñada, no implementada.
