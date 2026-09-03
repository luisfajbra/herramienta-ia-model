# Comandos del pipeline — estado real (2026-09-03)

Este archivo documenta únicamente lo que **funciona hoy**, verificado leyendo
el código real (no specs ni planes). Ver la sección final "Qué NO está
conectado todavía" para lo que existe como esquema pero no se usa en
producción.

> Actualizado junto con `docs/FLUJO_ACTUAL.md` (2026-09-03): faltaba
> `--persist-sql`, y la afirmación de que la base v17 no tenía ningún
> consumidor ya no era cierta — ver más abajo.

## Arquitectura actual (resumen honesto)

- **Un solo pipeline activo por CLI** (`main.py`): SWMM → extracción → CSV
  (`data/training/dataset_final.csv`) → XGBoost (17 features) → `.joblib`
  → mapas/evaluación. Esto es lo que corren todos los comandos de abajo.
- **Formato de almacenamiento del pipeline activo: CSV + joblib**, no SQL.
  El esquema SQLite v17 (`swmm_resilience/database/sql/001_v17_initial.sql`
  … `005_provenance_integrity.sql`) existe y está migrado y probado. Con
  `--persist-sql` **sí hay código que escribe ahí**: vuelca el CSV a
  `networks/nodes/scenarios/runs/node_features/node_results` y entrena +
  persiste `training_runs → model_evaluations → oof_predictions →
  trained_models → model_metrics` (`swmm_resilience/database/csv_backfill.py`).
  También existe un loader de lectura ya probado
  (`swmm_resilience/database/training_queries.py::load_training_samples`,
  con `export_training_samples_csv`), pero **nada del pipeline lo llama
  todavía** — solo lo usa su propio test. En los dos sentidos, el flujo
  real de `main.py` (entrenar, predecir, evaluar) sigue siendo siempre
  CSV → `.joblib`; la base v17 es un paso aparte y opcional que no se
  retroalimenta hacia el resto del pipeline.
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

## Tests

```bash
# Suite completa (excluye tests de escala por defecto, ver pytest.ini)
python -m pytest -q

# Solo la base de datos v17 (migraciones, triggers, upgrade, recovery, etc.)
python -m pytest tests/database -q

# Incluir los tests de escala (1M filas OOF, ~150s, opt-in)
python -m pytest tests/database -q -m scale

# Verificar que el paquete se puede construir como wheel
python -m pytest tests/packaging -q
```

Estado esperado (al 2026-08-24): **504 passed, 3 deselected**. El único
fallo conocido era `tests/desktop/test_results_tab.py` por un problema de
entorno Tcl/Tk local (falta `tk.tcl` en esta instalación de Python), no
relacionado con el código.

> Nota (2026-09-03): `docs/FLUJO_ACTUAL.md` reporta **517 pasan / 2 fallos
> ambientales (Tcl/Tk y módulo `build`)** para el mismo comando, en una
> revisión donde no había `pytest` instalado en el entorno para
> re-verificarlo. La diferencia con el número de arriba es consistente con
> los tests agregados desde el 24/08 (p.ej. `test_training_view_v17.py`,
> `test_scale_v17.py`), pero **ninguno de los dos números está confirmado
> en esta revisión** — correr `python -m pytest -q` antes de citar una
> cifra en otro documento.

## Pipeline activo (CLI, `main.py`) — CSV + 17 features

```bash
# Pipeline completo desde cero: SWMM -> extracción -> CSV -> entrenamiento -> evaluación -> mapas
python main.py

# Ya tienes dataset_final.csv y solo quieres reentrenar + evaluar
python main.py --only-ml

# Saltar extracción SWMM, usando el CSV existente, pero seguir con todo lo demás
python main.py --skip-extraction

# Solo regenerar mapas desde el CSV existente (sin reentrenar)
python main.py --only-maps

# Inferencia para un factor arbitrario sin correr SWMM (usa los .joblib guardados)
python main.py --predict --factor 3.5

# Correr SWMM + ML para un factor arbitrario y generar ambos mapas (comparación)
python main.py --simulate --factor 3.5

# Persistir dataset_final.csv + un entrenamiento GroupKFold5 en
# outputs/training_v17.sqlite3 (esquema SQLite v17) — paso aparte y opcional,
# no lo lee nadie del pipeline activo (ver "Qué NO está conectado todavía")
python main.py --persist-sql
```

### Análisis y visualización (requieren CSV/modelos ya generados)

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

## Base de datos SQLite v17 (esquema endurecido, con escritura pero sin lectura conectada al pipeline)

`python main.py --persist-sql` (ver arriba) ya escribe ahí. Estos comandos
son para inspeccionar/mantener la base v17 directamente:

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
# Leer de vuelta lo que --persist-sql escribió (mismo contrato de 17
# features que el CSV) — código probado, pero nada del pipeline lo llama
from swmm_resilience.database.training_queries import load_training_samples
frame = load_training_samples(conn)                 # todos los runs COMPLETE
frame = load_training_samples(conn, run_ids=[1, 2])  # runs específicos
```

## Qué NO está conectado todavía

- **`--persist-sql` sí entrena y guarda resultados en las tablas SQL v17**
  (`training_runs`, `model_evaluations`, `oof_predictions`, `trained_models`,
  `model_metrics`) — ver `swmm_resilience/database/csv_backfill.py`. Lo que
  deliberadamente **no** escribe es `model_candidates` / `model_rankings` /
  `model_promotions` / `model_selections`: los modelos quedan como
  artefactos históricos válidos, no como "selección activa" (ver
  `docs/FLUJO_ACTUAL.md` §8, §12.2–12.3 para el detalle de qué tanto de la
  vista/loader de lectura ya existe). El pipeline activo (`main.py` sin
  `--persist-sql`) sigue usando siempre CSV + `.joblib`: `--persist-sql` es
  un paso aparte y opcional que no se retroalimenta, y aunque ya existe un
  loader de lectura probado (`training_queries.py::load_training_samples`),
  ningún consumidor del pipeline lo llama todavía. Conectar esto como fuente
  de verdad por defecto (a veces referido como "Plan C / unified-ml" en los
  specs de `docs/superpowers/`) es trabajo pendiente, no hecho hoy.
- El pipeline legacy de 7 modelos (`ml/train.py`) sigue vivo en la GUI —
  retirarlo es una tarea aparte, ya identificada pero no ejecutada.
- La búsqueda de hiperparámetros con Optuna (`docs/superpowers/specs/2026-08-04-optuna-hyperparam-search-design.md`)
  está solo diseñada, no implementada.
