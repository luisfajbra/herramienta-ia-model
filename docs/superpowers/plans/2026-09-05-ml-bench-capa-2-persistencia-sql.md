# Banco ML — Plan 2: persistencia SQL de la cadena de proveniencia

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Que cada corrida del banco escriba la cadena de evidencia completa que el esquema v17 ya define y nadie usa — `training_runs → model_candidates → model_evaluations → oof_predictions → finalizations → trained_models → model_rankings → model_promotions → model_selections` — de modo que cualquier número que llegue a la tesis sea trazable hasta los datos y la receta que lo produjeron.

**Architecture:** Un paquete `ml/bench/persist/` con un módulo por eslabón de la cadena y un orquestador que los llama en el orden que imponen los triggers. `persist/` es el **único** lugar del banco que escribe SQL. Nada de la capa del Plan 1 cambia: recibe el `PreparedDataset`, el frame OOF y los artefactos ya producidos, y los vuelca.

**Tech Stack:** SQLite (WAL, migraciones 001–005 + la 006 nueva), Python 3.12, pandas 2.3.3, pytest 9.

**Spec:** `docs/superpowers/specs/2026-09-05-ml-bench-preproc-train-eval-design.md` §4.4, §10, §14.3

**Depende de:** Plan 1 completo (`docs/superpowers/plans/2026-09-05-ml-bench-capa-1-preproc-eval.md`). Sin `PreparedDataset`, `evaluate_candidate` y `train_candidate` no hay nada que persistir.

## Global Constraints

- **Un `training_runs` por protocolo.** `fold_count` es un entero único por fila y `grouping_strategy` acepta un solo valor (`CHECK(grouping_strategy IN ('group_kfold','loso'))`). LOSO con 25 folds y GroupKFold5 con 5 **no caben en la misma fila**. Una corrida del banco con dos protocolos escribe dos `training_runs`.
- **Clasificación y regresión son evaluaciones separadas.** `oof_predictions` tiene `PRIMARY KEY(evaluation_id, run_id, node_pk)` — **sin** `target`. No se pueden guardar los dos targets bajo la misma evaluación. Cada familia produce 2 candidatos por protocolo: uno `task='classification'`, otro `task='regression'`.
- **La cadena es inmutable.** Todas estas tablas tienen triggers `BEFORE UPDATE`/`BEFORE DELETE` que abortan. Un error de escritura no se corrige: se hace rollback de la transacción y se vuelve a intentar en un `training_run` nuevo.
- **`model_candidate_finalizations` exige tres cosas a la vez** (trigger `..._validate_insert`): (a) exactamente `training_runs.fold_count` evaluaciones `COMPLETE` enlazadas, (b) ninguna evaluación enlazada en otro estado, (c) **cobertura OOF total**: por cada run de validación (`model_evaluation_runs.role='validation'`), cada fila de `node_results` debe tener su `oof_predictions`.
- **`model_ranking_finalizations` exige** que el ganador declarado tenga el mejor `metric_ordinal = 0` válido y que **toda** entrada del ranking tenga una fila de score con `metric_ordinal = 0`. Los desempates **no** los verifica el esquema (así lo dice su propio comentario); se registran en `tie_breakers_json` para auditoría y los resuelve `ranking.py` en Python.
- **`model_promotion_rankings` exige un ranking finalizado** antes de enlazar la promoción.
- Conexión: siempre `connect_managed_database(db_path)` + `apply_migrations(conn)`. Una transacción por eslabón, `executemany` para OOF, nunca fila a fila en autocommit.
- Comando de test: `./venv/Scripts/python.exe -m pytest -q`.
- No se toca `csv_backfill.py::persist_training_run` en este plan: se retira en el Plan 3.

---

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `swmm_resilience/database/sql/006_model_artifact_paths.sql` | Migración: `model_blob` nullable + `model_path` |
| `swmm_resilience/ml/bench/persist/__init__.py` | Exporta `persist_bench_run` |
| `swmm_resilience/ml/bench/persist/identity.py` | Resolución `(network_id, node_id) → node_pk`; mapeo `sample_idx → (run_id, node_pk)` |
| `swmm_resilience/ml/bench/persist/training_run.py` | `open_training_run`, `training_run_inputs`, transiciones de estado |
| `swmm_resilience/ml/bench/persist/candidates.py` | `model_candidates` + `candidate_definition_sha256` |
| `swmm_resilience/ml/bench/persist/evaluations.py` | `model_evaluations`, `model_evaluation_runs`, `oof_predictions`, enlaces y finalización |
| `swmm_resilience/ml/bench/persist/artifacts.py` | `trained_models`, `model_artifact_candidates`, `model_metrics` |
| `swmm_resilience/ml/bench/persist/rankings.py` | `model_rankings`, entries, scores, finalización |
| `swmm_resilience/ml/bench/persist/promotions.py` | `model_promotions`, `model_promotion_rankings`, finalización, `model_selections` |
| `swmm_resilience/ml/bench/persist/runner.py` | `persist_bench_run`: orquesta todo en el orden que exigen los triggers |

Tests en `tests/ml/bench/persist/`.

---

## Task 1: Migración 006 — `model_blob` opcional, `model_path` nuevo

**Files:**
- Create: `swmm_resilience/database/sql/006_model_artifact_paths.sql`
- Modify: `swmm_resilience/database/migrations.py` (registrar la 006)
- Test: `tests/database/test_migration_006.py`

**Interfaces:**
- Consumes: `apply_migrations` de `swmm_resilience.database.migrations`.
- Produces: `trained_models` con `model_blob` nullable, `model_path TEXT`, y `CHECK(model_blob IS NOT NULL OR model_path IS NOT NULL)`.

**Por qué:** el esquema exige `model_blob BLOB NOT NULL`, pero `FLUJO_ACTUAL.md` §12.5 pide guardar el `.joblib` en disco con su hash. Con cinco familias × dos modelos × cada corrida, el blob multiplica el tamaño de la base y encarece el backup.

- [ ] **Step 1: Write the failing test**

```python
# tests/database/test_migration_006.py
import sqlite3

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations


@pytest.fixture
def migrated_db(tmp_path):
    conn = connect_managed_database(tmp_path / "v17.sqlite3")
    apply_migrations(conn)
    yield conn
    conn.close()


def _columns(conn, table: str) -> dict:
    return {row[1]: row for row in conn.execute(f"PRAGMA table_info({table})")}


def test_model_path_column_exists(migrated_db):
    assert "model_path" in _columns(migrated_db, "trained_models")


def test_model_blob_is_nullable(migrated_db):
    notnull_flag = _columns(migrated_db, "trained_models")["model_blob"][3]
    assert notnull_flag == 0, "model_blob debe ser nullable tras la 006"


def test_model_sha256_is_still_required(migrated_db):
    notnull_flag = _columns(migrated_db, "trained_models")["model_sha256"][3]
    assert notnull_flag == 1


def test_a_row_with_neither_blob_nor_path_is_rejected(migrated_db, monkeypatch):
    """El CHECK impide registrar un modelo que no existe en ningun sitio."""
    sql = (
        "INSERT INTO trained_models (training_run_id, target, algorithm, "
        "hyperparameters_json, preprocessing_json, feature_contract_id, "
        "feature_contract_sha256, ordered_features_json, target_transform_json, "
        "query_params_json, included_run_ids_json, random_seed, grouping_strategy, "
        "python_version, library_versions_json, model_sha256, model_blob, "
        "model_path, created_at_utc) "
        "VALUES (1,'inunda','xgboost','{}','{}','tabular_v3_17',?, '[]','{}','{}','[]',42,"
        "'loso','3.12','{}',?,NULL,NULL,'2026-09-05T00:00:00Z')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        migrated_db.execute(sql, ("a" * 64, "b" * 64))


def test_existing_rows_survive_the_migration(tmp_path):
    """La 006 recrea la tabla: las filas previas no se pueden perder."""
    from swmm_resilience.database.migrations import apply_migrations_up_to

    db_path = tmp_path / "v17.sqlite3"
    conn = connect_managed_database(db_path)
    apply_migrations_up_to(conn, "005")
    # ... insertar un training_run + trained_models con blob ...
    # (el implementador construye la fila minima que satisfaga las FK de la 005)
    before = conn.execute("SELECT COUNT(*) FROM trained_models").fetchone()[0]
    apply_migrations(conn)
    after = conn.execute("SELECT COUNT(*) FROM trained_models").fetchone()[0]
    assert after == before
    conn.close()


def test_migration_is_idempotent(migrated_db):
    apply_migrations(migrated_db)
    apply_migrations(migrated_db)
    assert "model_path" in _columns(migrated_db, "trained_models")
```

> **Nota:** `apply_migrations_up_to` puede no existir. Si no existe, implementa `test_existing_rows_survive_the_migration` aplicando las migraciones 001–005 a mano desde `swmm_resilience/database/sql/`, o marca ese test `xfail` y añade una función `apply_migrations(conn, up_to=None)` — pero **no lo borres**: es el único test que protege contra perder modelos ya registrados.

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/database/test_migration_006.py -q`
Expected: FAIL — `model_path` no existe

- [ ] **Step 3: Write minimal implementation**

Lee primero `swmm_resilience/database/sql/002_model_integrity.sql` para copiar el patrón de recreación de tabla (SQLite no soporta `ALTER TABLE ... DROP NOT NULL`), y `migrations.py` para ver cómo se registran.

```sql
-- swmm_resilience/database/sql/006_model_artifact_paths.sql
-- Los artefactos .joblib viven en disco (outputs/bench/candidates/<familia>/)
-- y la base guarda la ruta mas su sha256. model_blob se conserva nullable
-- para no invalidar las filas escritas antes de esta migracion.

PRAGMA foreign_keys=OFF;

CREATE TABLE trained_models_new (
    model_id INTEGER PRIMARY KEY,
    training_run_id INTEGER NOT NULL
        REFERENCES training_runs(training_run_id) ON DELETE CASCADE,
    target TEXT NOT NULL CHECK(target IN ('inunda','vol_inundacion_m3')),
    algorithm TEXT NOT NULL,
    hyperparameters_json TEXT NOT NULL,
    preprocessing_json TEXT NOT NULL,
    feature_contract_id TEXT NOT NULL CHECK(feature_contract_id='tabular_v3_17'),
    feature_contract_sha256 TEXT NOT NULL CHECK(length(feature_contract_sha256)=64),
    ordered_features_json TEXT NOT NULL,
    target_transform_json TEXT NOT NULL,
    query_params_json TEXT NOT NULL,
    included_run_ids_json TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    grouping_strategy TEXT NOT NULL,
    python_version TEXT NOT NULL,
    library_versions_json TEXT NOT NULL,
    model_sha256 TEXT NOT NULL CHECK(length(model_sha256)=64),
    model_blob BLOB,
    model_path TEXT,
    created_at_utc TEXT NOT NULL,
    UNIQUE(model_id, training_run_id, target),
    CHECK(model_blob IS NOT NULL OR model_path IS NOT NULL)
);

INSERT INTO trained_models_new (
    model_id, training_run_id, target, algorithm, hyperparameters_json,
    preprocessing_json, feature_contract_id, feature_contract_sha256,
    ordered_features_json, target_transform_json, query_params_json,
    included_run_ids_json, random_seed, grouping_strategy, python_version,
    library_versions_json, model_sha256, model_blob, model_path, created_at_utc
)
SELECT
    model_id, training_run_id, target, algorithm, hyperparameters_json,
    preprocessing_json, feature_contract_id, feature_contract_sha256,
    ordered_features_json, target_transform_json, query_params_json,
    included_run_ids_json, random_seed, grouping_strategy, python_version,
    library_versions_json, model_sha256, model_blob, NULL, created_at_utc
FROM trained_models;

DROP TABLE trained_models;
ALTER TABLE trained_models_new RENAME TO trained_models;

PRAGMA foreign_keys=ON;
```

> **Antes de escribir esto:** ejecuta `grep -n "trained_models" swmm_resilience/database/sql/00*.sql` y lista **todos** los triggers e índices que la 002/003/005 crean sobre `trained_models`. `DROP TABLE` los borra. La migración debe re-crearlos al final, o romperás garantías existentes. Ese listado es parte del trabajo de este task, no un detalle.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/database/ -q`
Expected: los tests nuevos pasan **y** toda la suite de `tests/database/` sigue verde (incluido `test_training_view_v17.py`, ~794 líneas).

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/database/sql/006_model_artifact_paths.sql swmm_resilience/database/migrations.py tests/database/test_migration_006.py
git commit -m "feat(db): migracion 006, model_blob opcional y model_path"
```

---

## Task 2: `persist/identity.py` — resolución de `node_pk`

**Files:**
- Create: `swmm_resilience/ml/bench/persist/__init__.py`
- Create: `swmm_resilience/ml/bench/persist/identity.py`
- Create: `tests/ml/bench/persist/__init__.py`
- Test: `tests/ml/bench/persist/test_identity.py`

**Interfaces:**
- Produces: `resolve_node_pks(conn, keys) -> pd.DataFrame` (añade la columna `node_pk` a `keys`); `sample_identity(prepared, conn) -> pd.DataFrame` (índice `sample_idx` → `run_id`, `node_pk`).

**Por qué existe este módulo:** la vista `training_samples_v17` expone `node_id` (texto) pero **no** `node_pk` (entero), y `oof_predictions` tiene clave `(evaluation_id, run_id, node_pk)`. Sin esta traducción no se puede escribir una sola predicción.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/persist/test_identity.py
import pandas as pd
import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.ml.bench.persist.identity import resolve_node_pks, sample_identity
from swmm_resilience.ml.bench.preprocess import prepare_dataset


@pytest.fixture
def conn(sql_training_db):
    connection = connect_managed_database(sql_training_db)
    apply_migrations(connection)
    yield connection
    connection.close()


@pytest.fixture
def prepared(sql_training_db, tmp_path):
    return prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )


def test_every_key_row_gets_a_node_pk(conn, prepared):
    resolved = resolve_node_pks(conn, prepared.keys)

    assert "node_pk" in resolved.columns
    assert resolved["node_pk"].notna().all()
    assert len(resolved) == len(prepared.keys)


def test_node_pk_is_stable_for_the_same_node(conn, prepared):
    resolved = resolve_node_pks(conn, prepared.keys)
    per_node = resolved.groupby(["network_id", "node_id"])["node_pk"].nunique()
    assert (per_node == 1).all(), "un mismo nodo no puede tener dos node_pk"


def test_row_order_is_preserved(conn, prepared):
    resolved = resolve_node_pks(conn, prepared.keys)
    pd.testing.assert_series_equal(
        resolved["node_id"].reset_index(drop=True),
        prepared.keys["node_id"].reset_index(drop=True),
    )


def test_an_unknown_node_is_reported_not_silently_dropped(conn, prepared):
    keys = prepared.keys.copy()
    keys.loc[0, "node_id"] = "NODO_QUE_NO_EXISTE"

    with pytest.raises(ValueError, match="NODO_QUE_NO_EXISTE"):
        resolve_node_pks(conn, keys)


def test_sample_identity_is_indexed_by_sample_idx(conn, prepared):
    identity = sample_identity(prepared, conn)

    assert identity.index.name == "sample_idx"
    assert list(identity.columns) == ["run_id", "node_pk"]
    assert identity.index.tolist() == list(range(len(prepared.X)))


def test_sample_identity_matches_the_prepared_key_order(conn, prepared):
    identity = sample_identity(prepared, conn)
    assert identity["run_id"].tolist() == prepared.keys["run_id"].tolist()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/persist/test_identity.py -q`
Expected: `ModuleNotFoundError: No module named 'swmm_resilience.ml.bench.persist'`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/persist/__init__.py
"""Persistencia de la cadena de proveniencia del banco.

Este paquete es el ÚNICO lugar del banco que escribe SQL. El resto de la
capa (preprocess, train, evaluate, ranking) no conoce la base de datos.
"""
```

```python
# swmm_resilience/ml/bench/persist/identity.py
"""Traducción entre la identidad del PreparedDataset y las claves de la base.

La vista training_samples_v17 expone node_id (texto) pero no node_pk
(entero), y oof_predictions tiene clave (evaluation_id, run_id, node_pk).
Sin esta traducción no se puede escribir una sola predicción.
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from ..schemas import PreparedDataset


def resolve_node_pks(conn: sqlite3.Connection, keys: pd.DataFrame) -> pd.DataFrame:
    """Añade ``node_pk`` a ``keys`` uniendo por ``(network_id, node_id)``."""
    nodes = pd.read_sql_query("SELECT node_pk, network_id, node_id FROM nodes", conn)
    merged = keys.merge(nodes, on=["network_id", "node_id"], how="left", sort=False)

    missing = merged.loc[merged["node_pk"].isna(), "node_id"].unique()
    if len(missing):
        raise ValueError(
            f"Estos nodos no existen en la tabla `nodes`: {sorted(missing)[:5]}. "
            "La base y el PreparedDataset no corresponden a la misma red."
        )
    merged["node_pk"] = merged["node_pk"].astype(int)
    return merged


def sample_identity(prepared: PreparedDataset, conn: sqlite3.Connection) -> pd.DataFrame:
    """Mapa ``sample_idx -> (run_id, node_pk)`` para escribir oof_predictions."""
    resolved = resolve_node_pks(conn, prepared.keys)
    identity = resolved.loc[:, ["run_id", "node_pk"]].copy()
    identity.index = pd.RangeIndex(len(identity), name="sample_idx")
    return identity
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/persist/test_identity.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/persist/ tests/ml/bench/persist/
git commit -m "feat(bench): resolucion node_id -> node_pk para la persistencia"
```

---

## Task 3: `persist/training_run.py` — un `training_runs` por protocolo

**Files:**
- Create: `swmm_resilience/ml/bench/persist/training_run.py`
- Test: `tests/ml/bench/persist/test_training_run.py`

**Interfaces:**
- Consumes: `PreparedDataset`; `n_folds_for` de `bench/folds.py`.
- Produces: `open_training_run(conn, prepared, protocol, criterion) -> int` (devuelve `training_run_id`, deja la fila en `RUNNING` con sus `training_run_inputs`); `mark_complete(conn, training_run_id)`; `mark_failed(conn, training_run_id, failure_type, message)`; `GROUPING_STRATEGY = {"LOSO": "loso", "GroupKFold5": "group_kfold"}`.

**Restricción central:** `fold_count` es un entero único y `grouping_strategy` acepta un solo valor. **Un `training_runs` por protocolo**, no uno por corrida del banco.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/persist/test_training_run.py
import json

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.ml.bench.persist.training_run import (
    GROUPING_STRATEGY,
    mark_complete,
    mark_failed,
    open_training_run,
)
from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.ranking import RankingCriterion

CRITERION = RankingCriterion(
    primary_metric="end_to_end.rmse_vol_todos_nodos",
    primary_direction="minimize",
    tie_breakers=(("classifier.f1", "maximize"),),
)


@pytest.fixture
def conn(sql_training_db):
    connection = connect_managed_database(sql_training_db)
    apply_migrations(connection)
    yield connection
    connection.close()


@pytest.fixture
def prepared(sql_training_db, tmp_path):
    return prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )


def _row(conn, training_run_id):
    conn.row_factory = None
    columns = [d[0] for d in conn.execute("SELECT * FROM training_runs LIMIT 0").description]
    values = conn.execute(
        "SELECT * FROM training_runs WHERE training_run_id = ?", (training_run_id,)
    ).fetchone()
    return dict(zip(columns, values))


def test_opening_a_run_leaves_it_running(conn, prepared):
    run_id = open_training_run(conn, prepared, "LOSO", CRITERION)
    assert _row(conn, run_id)["status"] == "RUNNING"


def test_grouping_strategy_maps_to_the_schema_vocabulary(conn, prepared):
    """El CHECK del esquema solo acepta 'loso' y 'group_kfold'."""
    assert GROUPING_STRATEGY == {"LOSO": "loso", "GroupKFold5": "group_kfold"}
    run_id = open_training_run(conn, prepared, "LOSO", CRITERION)
    assert _row(conn, run_id)["grouping_strategy"] == "loso"


def test_fold_count_matches_the_protocol(conn, prepared):
    run_id = open_training_run(conn, prepared, "LOSO", CRITERION)
    expected = prepared.folds[prepared.folds["protocol"] == "LOSO"]["fold_id"].nunique()
    assert _row(conn, run_id)["fold_count"] == expected


def test_the_ranking_criterion_is_recorded(conn, prepared):
    run_id = open_training_run(conn, prepared, "LOSO", CRITERION)
    row = _row(conn, run_id)

    assert row["primary_metric"] == "end_to_end.rmse_vol_todos_nodos"
    assert json.loads(row["tie_breakers_json"]) == [
        {"metric": "classifier.f1", "direction": "maximize"}
    ]


def test_the_feature_contract_is_recorded(conn, prepared):
    from swmm_resilience.ml.contracts import TABULAR_V3_17

    row = _row(conn, open_training_run(conn, prepared, "LOSO", CRITERION))
    assert row["feature_contract_id"] == "tabular_v3_17"
    assert row["feature_contract_sha256"] == TABULAR_V3_17.descriptor_sha256


def test_included_run_ids_are_recorded_and_linked(conn, prepared):
    run_id = open_training_run(conn, prepared, "LOSO", CRITERION)
    declared = json.loads(_row(conn, run_id)["included_run_ids_json"])
    linked = [
        row[0]
        for row in conn.execute(
            "SELECT run_id FROM training_run_inputs WHERE training_run_id = ? ORDER BY run_id",
            (run_id,),
        )
    ]
    assert declared == linked == sorted(prepared.keys["run_id"].unique().tolist())


def test_two_protocols_get_two_separate_training_runs(conn, sql_training_db, tmp_path):
    """fold_count es unico por fila: LOSO y GroupKFold5 no caben juntos."""
    prepared = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "p",
    )
    first = open_training_run(conn, prepared, "LOSO", CRITERION)
    second = open_training_run(conn, prepared, "LOSO", CRITERION)
    assert first != second


def test_mark_complete_sets_the_status_and_timestamp(conn, prepared):
    run_id = open_training_run(conn, prepared, "LOSO", CRITERION)
    mark_complete(conn, run_id)
    row = _row(conn, run_id)

    assert row["status"] == "COMPLETE"
    assert row["completed_at_utc"] is not None


def test_mark_failed_records_the_reason(conn, prepared):
    run_id = open_training_run(conn, prepared, "LOSO", CRITERION)
    mark_failed(conn, run_id, "EvaluationError", "el fold 3 no converge")
    row = _row(conn, run_id)

    assert row["status"] == "FAILED"
    assert row["failure_type"] == "EvaluationError"
    assert "fold 3" in row["failure_message"]


def test_an_unknown_protocol_is_rejected(conn, prepared):
    with pytest.raises(ValueError, match="Protocolo desconocido"):
        open_training_run(conn, prepared, "LeaveOneShapeOut", CRITERION)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/persist/test_training_run.py -q`
Expected: `ModuleNotFoundError: ...persist.training_run`

- [ ] **Step 3: Write minimal implementation**

Lee primero `swmm_resilience/database/csv_backfill.py:218-270` para copiar el patrón de inserción con `training_run_id` explícito (`SELECT COALESCE(MAX(...), 0) + 1`) y las transiciones de estado que los triggers permiten.

```python
# swmm_resilience/ml/bench/persist/training_run.py
"""Apertura y cierre de un training_run.

UN training_run POR PROTOCOLO. `fold_count` es un entero único por fila y
`grouping_strategy` acepta un solo valor (CHECK IN ('group_kfold','loso')),
así que LOSO con 25 folds y GroupKFold5 con 5 no caben en la misma fila.
"""

from __future__ import annotations

import json
import platform
import sqlite3
from datetime import datetime, timezone
from importlib.metadata import version as _package_version

from ...contracts import TABULAR_V3_17
from ..folds import n_folds_for
from ..ranking import RankingCriterion
from ..schemas import PreparedDataset

GROUPING_STRATEGY = {"LOSO": "loso", "GroupKFold5": "group_kfold"}

_QUERY_SQL = "SELECT ... FROM training_samples_v17 WHERE 1 = 1 ORDER BY run_id, node_id"


def _library_versions() -> dict:
    return {
        name: _package_version(name)
        for name in ("pandas", "numpy", "scikit-learn", "xgboost", "torch")
    }


def open_training_run(
    conn: sqlite3.Connection,
    prepared: PreparedDataset,
    protocol: str,
    criterion: RankingCriterion,
) -> int:
    """Crea el training_run del protocolo, lo pasa a RUNNING y enlaza sus runs."""
    if protocol not in GROUPING_STRATEGY:
        raise ValueError(
            f"Protocolo desconocido: {protocol!r}. Opciones: {sorted(GROUPING_STRATEGY)}"
        )

    run_ids = sorted(int(value) for value in prepared.keys["run_id"].unique())
    fold_count = n_folds_for(prepared.keys["factor_mult"], protocol)
    now = datetime.now(timezone.utc).isoformat()

    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id, feature_contract_sha256,
            query_sql, query_params_json, included_run_ids_json, grouping_strategy,
            fold_count, random_seed, primary_metric, tie_breakers_json,
            python_version, library_versions_json, status
        ) VALUES (
            (SELECT COALESCE(MAX(training_run_id), 0) + 1 FROM training_runs),
            'system', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING'
        )
        """,
        (
            TABULAR_V3_17.contract_id,
            TABULAR_V3_17.descriptor_sha256,
            _QUERY_SQL,
            json.dumps({"prep_id": prepared.prep_id, "protocol": protocol}),
            json.dumps(run_ids),
            GROUPING_STRATEGY[protocol],
            int(fold_count),
            int(prepared.manifest.get("random_seed", 42)),
            criterion.primary_metric,
            json.dumps(
                [{"metric": m, "direction": d} for m, d in criterion.tie_breakers]
            ),
            platform.python_version(),
            json.dumps(_library_versions()),
        ),
    )
    training_run_id = conn.execute(
        "SELECT MAX(training_run_id) FROM training_runs"
    ).fetchone()[0]

    # training_run_inputs exige que el owner siga PENDING (trigger
    # training_run_inputs_owner_pending), asi que se enlaza ANTES de RUNNING.
    conn.executemany(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (?, ?)",
        [(training_run_id, run_id) for run_id in run_ids],
    )
    conn.execute(
        "UPDATE training_runs SET status = 'RUNNING', started_at_utc = ? "
        "WHERE training_run_id = ?",
        (now, training_run_id),
    )
    return int(training_run_id)


def mark_complete(conn: sqlite3.Connection, training_run_id: int) -> None:
    conn.execute(
        "UPDATE training_runs SET status = 'COMPLETE', completed_at_utc = ? "
        "WHERE training_run_id = ?",
        (datetime.now(timezone.utc).isoformat(), training_run_id),
    )


def mark_failed(
    conn: sqlite3.Connection, training_run_id: int, failure_type: str, message: str
) -> None:
    conn.execute(
        "UPDATE training_runs SET status = 'FAILED', completed_at_utc = ?, "
        "failure_type = ?, failure_message = ? WHERE training_run_id = ?",
        (datetime.now(timezone.utc).isoformat(), failure_type, message, training_run_id),
    )
```

> **Verifica el orden PENDING → inputs → RUNNING.** El trigger `training_run_inputs_owner_pending` (migración 005, línea ~189) aborta si el `training_run` no está `PENDING` al insertar el enlace. Si tu lectura del trigger dice otra cosa, ajusta el orden y **anota cuál es la regla real** en el docstring — es exactamente el tipo de detalle que hace fallar la cadena tres tareas más adelante.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/persist/test_training_run.py -q`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/persist/training_run.py tests/ml/bench/persist/test_training_run.py
git commit -m "feat(bench): apertura de training_run por protocolo"
```

---

## Task 4: `persist/candidates.py` — recetas de candidato

**Files:**
- Create: `swmm_resilience/ml/bench/persist/candidates.py`
- Test: `tests/ml/bench/persist/test_candidates.py`

**Interfaces:**
- Produces: `candidate_definition(family, task, params, prepared) -> dict` (la receta completa, canónica); `candidate_definition_sha256(definition) -> str`; `insert_candidate(conn, training_run_id, definition) -> int`.

**Restricción:** el trigger `model_candidate_evaluations_matches_candidate` compara `(task, algorithm, hyperparameters_json)` **como texto**. El JSON de hiperparámetros debe serializarse **idénticamente** aquí y en `evaluations.py`, o el enlace se rechaza. Por eso ambos módulos usan la misma función de serialización.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/persist/test_candidates.py
import json

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.ml.bench.persist.candidates import (
    candidate_definition,
    candidate_definition_sha256,
    canonical_json,
    insert_candidate,
)
from swmm_resilience.ml.bench.persist.training_run import open_training_run
from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.ranking import RankingCriterion

CRITERION = RankingCriterion("classifier.f1", "maximize", ())
PARAMS = {"n_estimators": 200, "max_depth": 6}


@pytest.fixture
def conn(sql_training_db):
    connection = connect_managed_database(sql_training_db)
    apply_migrations(connection)
    yield connection
    connection.close()


@pytest.fixture
def prepared(sql_training_db, tmp_path):
    return prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )


def test_canonical_json_is_key_order_independent():
    """El trigger compara hyperparameters_json como TEXTO: debe ser canonico."""
    assert canonical_json({"b": 2, "a": 1}) == canonical_json({"a": 1, "b": 2})


def test_definition_records_the_full_recipe(prepared):
    definition = candidate_definition("xgboost", "classification", PARAMS, prepared)

    assert definition["algorithm"] == "xgboost"
    assert definition["task"] == "classification"
    assert definition["feature_contract_id"] == "tabular_v3_17"
    assert json.loads(definition["ordered_features_json"])[0] == "elev_fondo"
    assert definition["preprocessing_json"] == canonical_json(
        {"imputer": "median", "scaler": None, "pca": None}
    )


def test_regression_task_declares_the_log1p_transform(prepared):
    definition = candidate_definition("xgboost", "regression", PARAMS, prepared)
    transform = json.loads(definition["target_transform_json"])
    assert transform == {"regressor": "log1p", "inverse": "expm1"}


def test_classification_task_declares_no_transform(prepared):
    definition = candidate_definition("xgboost", "classification", PARAMS, prepared)
    assert json.loads(definition["target_transform_json"]) == {}


def test_the_definition_hash_is_sixty_four_hex(prepared):
    definition = candidate_definition("xgboost", "classification", PARAMS, prepared)
    digest = candidate_definition_sha256(definition)
    assert len(digest) == 64
    int(digest, 16)


def test_different_hyperparameters_give_different_hashes(prepared):
    a = candidate_definition("xgboost", "classification", {"max_depth": 3}, prepared)
    b = candidate_definition("xgboost", "classification", {"max_depth": 9}, prepared)
    assert candidate_definition_sha256(a) != candidate_definition_sha256(b)


def test_different_families_give_different_hashes(prepared):
    a = candidate_definition("xgboost", "classification", PARAMS, prepared)
    b = candidate_definition("random_forest", "classification", PARAMS, prepared)
    assert candidate_definition_sha256(a) != candidate_definition_sha256(b)


def test_insert_returns_a_candidate_id_and_persists_the_row(conn, prepared):
    training_run_id = open_training_run(conn, prepared, "LOSO", CRITERION)
    definition = candidate_definition("xgboost", "classification", PARAMS, prepared)

    candidate_id = insert_candidate(conn, training_run_id, definition)

    row = conn.execute(
        "SELECT algorithm, task, candidate_definition_sha256 FROM model_candidates "
        "WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    assert row[0] == "xgboost"
    assert row[1] == "classification"
    assert row[2] == candidate_definition_sha256(definition)


def test_five_families_times_two_tasks_gives_ten_candidates(conn, prepared):
    training_run_id = open_training_run(conn, prepared, "LOSO", CRITERION)
    for family in ("xgboost", "random_forest", "linear", "svm", "mlp"):
        for task in ("classification", "regression"):
            insert_candidate(
                conn,
                training_run_id,
                candidate_definition(family, task, PARAMS, prepared),
            )

    count = conn.execute(
        "SELECT COUNT(*) FROM model_candidates WHERE training_run_id = ?",
        (training_run_id,),
    ).fetchone()[0]
    assert count == 10


def test_candidates_are_immutable(conn, prepared):
    import sqlite3

    training_run_id = open_training_run(conn, prepared, "LOSO", CRITERION)
    candidate_id = insert_candidate(
        conn, training_run_id,
        candidate_definition("xgboost", "classification", PARAMS, prepared),
    )
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(
            "UPDATE model_candidates SET algorithm = 'otro' WHERE candidate_id = ?",
            (candidate_id,),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/persist/test_candidates.py -q`
Expected: `ModuleNotFoundError: ...persist.candidates`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/persist/candidates.py
"""Recetas de candidato: qué modelo, con qué hiperparámetros, sobre qué datos.

El trigger model_candidate_evaluations_matches_candidate compara
(task, algorithm, hyperparameters_json) COMO TEXTO entre model_candidates y
model_evaluations. Por eso todo JSON que participe de esa comparación pasa
por canonical_json(): mismo contenido, misma cadena, siempre.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

from ...contracts import FEATURE_COLUMNS_V17, TABULAR_V3_17
from ..registry import get_family
from ..schemas import PreparedDataset

PIPELINE_VERSION = "bench-1"

_TARGET_BY_TASK = {"classification": "inunda", "regression": "vol_inundacion_m3"}


def canonical_json(value) -> str:
    """Serialización estable: claves ordenadas, sin espacios."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def candidate_definition(
    family: str, task: str, params: dict, prepared: PreparedDataset
) -> dict:
    """La receta completa del candidato, lista para insertar."""
    if task not in _TARGET_BY_TASK:
        raise ValueError(f"Tarea desconocida: {task!r}. Opciones: classification, regression")

    transform = (
        {"regressor": "log1p", "inverse": "expm1"} if task == "regression" else {}
    )
    return {
        "task": task,
        "algorithm": family,
        "hyperparameters_json": canonical_json(params),
        "preprocessing_json": canonical_json(
            get_family(family).preprocessing_descriptor(params)
        ),
        "feature_contract_id": TABULAR_V3_17.contract_id,
        "feature_contract_sha256": TABULAR_V3_17.descriptor_sha256,
        "ordered_features_json": canonical_json(list(FEATURE_COLUMNS_V17)),
        "target_transform_json": canonical_json(transform),
        "pipeline_version": PIPELINE_VERSION,
        "prep_id": prepared.prep_id,
    }


def candidate_definition_sha256(definition: dict) -> str:
    return hashlib.sha256(canonical_json(definition).encode("utf-8")).hexdigest()


def insert_candidate(
    conn: sqlite3.Connection, training_run_id: int, definition: dict
) -> int:
    conn.execute(
        """
        INSERT INTO model_candidates (
            candidate_id, training_run_id, task, algorithm, hyperparameters_json,
            preprocessing_json, feature_contract_id, feature_contract_sha256,
            ordered_features_json, target_transform_json, pipeline_version,
            candidate_definition_sha256
        ) VALUES (
            (SELECT COALESCE(MAX(candidate_id), 0) + 1 FROM model_candidates),
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            training_run_id,
            definition["task"],
            definition["algorithm"],
            definition["hyperparameters_json"],
            definition["preprocessing_json"],
            definition["feature_contract_id"],
            definition["feature_contract_sha256"],
            definition["ordered_features_json"],
            definition["target_transform_json"],
            definition["pipeline_version"],
            candidate_definition_sha256(definition),
        ),
    )
    return int(
        conn.execute("SELECT MAX(candidate_id) FROM model_candidates").fetchone()[0]
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/persist/test_candidates.py -q`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/persist/candidates.py tests/ml/bench/persist/test_candidates.py
git commit -m "feat(bench): recetas de candidato con hash de definicion"
```

---

## Task 5: `persist/evaluations.py` — folds, OOF y finalización

**Files:**
- Create: `swmm_resilience/ml/bench/persist/evaluations.py`
- Test: `tests/ml/bench/persist/test_evaluations.py`

**Interfaces:**
- Consumes: `sample_identity` de `identity.py`; `canonical_json` de `candidates.py`.
- Produces: `persist_fold(conn, training_run_id, candidate_id, definition, prepared, oof, fold_id, protocol, identity, fit_seconds, predict_seconds) -> int`; `finalize_candidate(conn, candidate_id) -> None`.

**Este es el task más delicado del plan.** El trigger `model_candidate_finalizations_validate_insert` exige simultáneamente: exactamente `fold_count` evaluaciones `COMPLETE`, ninguna en otro estado, y **cobertura OOF total** — por cada run de validación, cada fila de `node_results` debe tener su `oof_predictions`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/persist/test_evaluations.py
import sqlite3

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.ml.bench.evaluate import evaluate_candidate
from swmm_resilience.ml.bench.persist.candidates import candidate_definition, insert_candidate
from swmm_resilience.ml.bench.persist.evaluations import finalize_candidate, persist_fold
from swmm_resilience.ml.bench.persist.identity import sample_identity
from swmm_resilience.ml.bench.persist.training_run import open_training_run
from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.ranking import RankingCriterion

CRITERION = RankingCriterion("classifier.f1", "maximize", ())
TINY = {"n_estimators": 5, "max_depth": 2}


@pytest.fixture
def context(sql_training_db, tmp_path):
    conn = connect_managed_database(sql_training_db)
    apply_migrations(conn)
    prepared = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )
    oof, _ = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    identity = sample_identity(prepared, conn)
    training_run_id = open_training_run(conn, prepared, "LOSO", CRITERION)
    yield conn, prepared, oof, identity, training_run_id
    conn.close()


def _persist_all_folds(conn, prepared, oof, identity, training_run_id, task):
    definition = candidate_definition("xgboost", task, TINY, prepared)
    candidate_id = insert_candidate(conn, training_run_id, definition)
    for fold_id in sorted(oof["fold_id"].unique()):
        persist_fold(
            conn, training_run_id, candidate_id, definition, prepared, oof,
            fold_id=int(fold_id), protocol="LOSO", identity=identity,
            fit_seconds=0.1, predict_seconds=0.01,
        )
    return candidate_id


def test_one_evaluation_row_per_fold(context):
    conn, prepared, oof, identity, training_run_id = context
    candidate_id = _persist_all_folds(
        conn, prepared, oof, identity, training_run_id, "classification"
    )

    count = conn.execute(
        "SELECT COUNT(*) FROM model_candidate_evaluations WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()[0]
    assert count == oof["fold_id"].nunique()


def test_evaluations_end_complete(context):
    conn, prepared, oof, identity, training_run_id = context
    _persist_all_folds(conn, prepared, oof, identity, training_run_id, "classification")

    statuses = {
        row[0] for row in conn.execute("SELECT DISTINCT status FROM model_evaluations")
    }
    assert statuses == {"COMPLETE"}


def test_evaluation_runs_record_train_and_validation_roles(context):
    conn, prepared, oof, identity, training_run_id = context
    _persist_all_folds(conn, prepared, oof, identity, training_run_id, "classification")

    roles = {
        row[0] for row in conn.execute("SELECT DISTINCT role FROM model_evaluation_runs")
    }
    assert roles == {"train", "validation"}


def test_classification_oof_stores_the_inunda_target(context):
    conn, prepared, oof, identity, training_run_id = context
    _persist_all_folds(conn, prepared, oof, identity, training_run_id, "classification")

    targets = {row[0] for row in conn.execute("SELECT DISTINCT target FROM oof_predictions")}
    assert targets == {"inunda"}


def test_classification_oof_carries_the_probability(context):
    conn, prepared, oof, identity, training_run_id = context
    _persist_all_folds(conn, prepared, oof, identity, training_run_id, "classification")

    nulls = conn.execute(
        "SELECT COUNT(*) FROM oof_predictions WHERE probability IS NULL"
    ).fetchone()[0]
    assert nulls == 0


def test_regression_oof_stores_the_volume_target(context):
    conn, prepared, oof, identity, training_run_id = context
    _persist_all_folds(conn, prepared, oof, identity, training_run_id, "regression")

    targets = {row[0] for row in conn.execute("SELECT DISTINCT target FROM oof_predictions")}
    assert targets == {"vol_inundacion_m3"}


def test_oof_covers_every_row_of_every_validation_run(context):
    """Requisito literal del trigger de finalizacion."""
    conn, prepared, oof, identity, training_run_id = context
    _persist_all_folds(conn, prepared, oof, identity, training_run_id, "classification")

    orphans = conn.execute(
        """
        SELECT COUNT(*)
        FROM model_evaluation_runs AS membership
        JOIN node_results AS result ON result.run_id = membership.run_id
        LEFT JOIN oof_predictions AS oof
            ON oof.evaluation_id = membership.evaluation_id
           AND oof.run_id = result.run_id
           AND oof.node_pk = result.node_pk
        WHERE membership.role = 'validation' AND oof.evaluation_id IS NULL
        """
    ).fetchone()[0]
    assert orphans == 0


def test_finalizing_a_complete_candidate_succeeds(context):
    conn, prepared, oof, identity, training_run_id = context
    candidate_id = _persist_all_folds(
        conn, prepared, oof, identity, training_run_id, "classification"
    )

    finalize_candidate(conn, candidate_id)

    row = conn.execute(
        "SELECT finalized_at_utc FROM model_candidate_finalizations WHERE candidate_id = ?",
        (candidate_id,),
    ).fetchone()
    assert row is not None


def test_finalizing_with_a_missing_fold_is_rejected(context):
    """El trigger exige exactamente fold_count evaluaciones COMPLETE."""
    conn, prepared, oof, identity, training_run_id = context
    definition = candidate_definition("xgboost", "classification", TINY, prepared)
    candidate_id = insert_candidate(conn, training_run_id, definition)

    folds = sorted(oof["fold_id"].unique())[:-1]      # falta el ultimo
    for fold_id in folds:
        persist_fold(
            conn, training_run_id, candidate_id, definition, prepared, oof,
            fold_id=int(fold_id), protocol="LOSO", identity=identity,
            fit_seconds=0.1, predict_seconds=0.01,
        )

    with pytest.raises(sqlite3.IntegrityError, match="finalization"):
        finalize_candidate(conn, candidate_id)


def test_adding_a_fold_after_finalization_is_rejected(context):
    conn, prepared, oof, identity, training_run_id = context
    definition = candidate_definition("xgboost", "classification", TINY, prepared)
    candidate_id = _persist_all_folds(
        conn, prepared, oof, identity, training_run_id, "classification"
    )
    finalize_candidate(conn, candidate_id)

    with pytest.raises(sqlite3.IntegrityError, match="finalization"):
        persist_fold(
            conn, training_run_id, candidate_id, definition, prepared, oof,
            fold_id=0, protocol="LOSO", identity=identity,
            fit_seconds=0.1, predict_seconds=0.01,
        )


def test_an_evaluation_that_does_not_match_its_candidate_is_rejected(context):
    """Trigger model_candidate_evaluations_matches_candidate."""
    conn, prepared, oof, identity, training_run_id = context
    definition = candidate_definition("xgboost", "classification", TINY, prepared)
    candidate_id = insert_candidate(conn, training_run_id, definition)

    mismatched = dict(definition, algorithm="svm")
    with pytest.raises(sqlite3.IntegrityError, match="does not match"):
        persist_fold(
            conn, training_run_id, candidate_id, mismatched, prepared, oof,
            fold_id=0, protocol="LOSO", identity=identity,
            fit_seconds=0.1, predict_seconds=0.01,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/persist/test_evaluations.py -q`
Expected: `ModuleNotFoundError: ...persist.evaluations`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/persist/evaluations.py
"""Evaluaciones por fold, predicciones out-of-fold y finalización del candidato.

El trigger model_candidate_finalizations_validate_insert exige TRES cosas a
la vez, y ésta es la razón de que este módulo sea el más delicado del plan:

  (a) exactamente training_runs.fold_count evaluaciones COMPLETE enlazadas,
  (b) ninguna evaluación enlazada en otro estado,
  (c) cobertura OOF TOTAL: por cada run de validación registrado en
      model_evaluation_runs, cada fila de node_results debe tener su
      oof_predictions.

Por (c) hay que escribir model_evaluation_runs con los run_id reales de cada
lado del fold, y las predicciones de TODOS los nodos de esos runs.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..schemas import PreparedDataset

_TARGET_BY_TASK = {"classification": "inunda", "regression": "vol_inundacion_m3"}


def _fold_run_ids(prepared: PreparedDataset, protocol: str, fold_id: int) -> tuple:
    chunk = prepared.folds[
        (prepared.folds["protocol"] == protocol) & (prepared.folds["fold_id"] == fold_id)
    ]
    run_ids = prepared.keys["run_id"].to_numpy()
    train = sorted(
        {int(v) for v in run_ids[chunk.loc[chunk["split"] == "train", "sample_idx"]]}
    )
    validation = sorted(
        {int(v) for v in run_ids[chunk.loc[chunk["split"] == "test", "sample_idx"]]}
    )
    return train, validation


def persist_fold(
    conn: sqlite3.Connection,
    training_run_id: int,
    candidate_id: int,
    definition: dict,
    prepared: PreparedDataset,
    oof: pd.DataFrame,
    *,
    fold_id: int,
    protocol: str,
    identity: pd.DataFrame,
    fit_seconds: float,
    predict_seconds: float,
) -> int:
    """Escribe una evaluación, sus runs y sus predicciones OOF."""
    task = definition["task"]
    target = _TARGET_BY_TASK[task]
    train_runs, validation_runs = _fold_run_ids(prepared, protocol, fold_id)

    conn.execute(
        """
        INSERT INTO model_evaluations (
            evaluation_id, training_run_id, task, algorithm, hyperparameters_json,
            fold_id, train_run_ids_json, validation_run_ids_json, status,
            fit_seconds, predict_seconds
        ) VALUES (
            (SELECT COALESCE(MAX(evaluation_id), 0) + 1 FROM model_evaluations),
            ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?
        )
        """,
        (
            training_run_id,
            task,
            definition["algorithm"],
            definition["hyperparameters_json"],
            int(fold_id),
            json.dumps(train_runs),
            json.dumps(validation_runs),
            float(fit_seconds),
            float(predict_seconds),
        ),
    )
    evaluation_id = int(
        conn.execute("SELECT MAX(evaluation_id) FROM model_evaluations").fetchone()[0]
    )

    conn.execute(
        "UPDATE model_evaluations SET status = 'RUNNING' WHERE evaluation_id = ?",
        (evaluation_id,),
    )
    conn.executemany(
        "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (?, ?, ?)",
        [(evaluation_id, "train", run_id) for run_id in train_runs]
        + [(evaluation_id, "validation", run_id) for run_id in validation_runs],
    )

    fold_oof = oof[(oof["protocol"] == protocol) & (oof["fold_id"] == fold_id)]
    indices = fold_oof["sample_idx"].to_numpy()
    observed = (
        prepared.y_clf.to_numpy()[indices]
        if task == "classification"
        else prepared.y_reg.to_numpy()[indices]
    )
    predicted = (
        fold_oof["y_pred_clf"].to_numpy()
        if task == "classification"
        else fold_oof["y_pred_reg"].to_numpy()
    )
    probability = (
        fold_oof["y_prob_clf"].to_numpy() if task == "classification" else None
    )

    rows = [
        (
            evaluation_id,
            int(identity.loc[sample_idx, "run_id"]),
            int(identity.loc[sample_idx, "node_pk"]),
            target,
            float(observed[position]),
            float(predicted[position]),
            float(probability[position]) if probability is not None else None,
            int(fold_id),
        )
        for position, sample_idx in enumerate(indices)
    ]
    conn.executemany(
        "INSERT INTO oof_predictions (evaluation_id, run_id, node_pk, target, "
        "observed, predicted, probability, fold_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )

    conn.execute(
        "INSERT INTO model_candidate_evaluations (candidate_id, evaluation_id) VALUES (?, ?)",
        (candidate_id, evaluation_id),
    )
    conn.execute(
        "UPDATE model_evaluations SET status = 'COMPLETE' WHERE evaluation_id = ?",
        (evaluation_id,),
    )
    return evaluation_id


def finalize_candidate(conn: sqlite3.Connection, candidate_id: int) -> None:
    """Cierra el candidato. El trigger verifica folds completos y cobertura OOF."""
    conn.execute(
        "INSERT INTO model_candidate_finalizations (candidate_id, finalized_at_utc) "
        "VALUES (?, ?)",
        (candidate_id, datetime.now(timezone.utc).isoformat()),
    )
```

> **Orden de escritura crítico:** el enlace `model_candidate_evaluations` se inserta **antes** de pasar la evaluación a `COMPLETE`, porque su trigger de coincidencia compara contra la fila de `model_evaluations` que ya debe existir. Si al ejecutar los tests el trigger `..._matches_candidate` aborta, revisa que `hyperparameters_json` sea **byte a byte** el mismo en `model_candidates` y `model_evaluations` — es el fallo más probable, y por eso ambos usan `canonical_json`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/persist/test_evaluations.py -q`
Expected: 12 passed

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/persist/evaluations.py tests/ml/bench/persist/test_evaluations.py
git commit -m "feat(bench): evaluaciones por fold, OOF y finalizacion de candidato"
```

---

## Task 6: `persist/artifacts.py` — modelos entrenados y métricas

**Files:**
- Create: `swmm_resilience/ml/bench/persist/artifacts.py`
- Test: `tests/ml/bench/persist/test_artifacts.py`

**Interfaces:**
- Consumes: `CandidateArtifacts` de `bench/train.py`; `canonical_json` de `candidates.py`.
- Produces: `insert_trained_model(conn, training_run_id, candidate_id, definition, artifacts, task, prepared, protocol) -> int`; `insert_metrics(conn, owner_kind, owner_id, metrics, scope) -> int` (número de filas escritas).

**Restricción:** `model_artifact_candidates_requires_finalized_matching_candidate` exige que el candidato esté **finalizado** y que su receta coincida **campo por campo** con la fila de `trained_models`: `feature_contract_id`, `feature_contract_sha256`, `ordered_features_json`, `preprocessing_json`, `target_transform_json`, `hyperparameters_json`, `algorithm`. Un solo carácter distinto y el trigger aborta.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/persist/test_artifacts.py
import sqlite3

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations
from swmm_resilience.ml.bench.evaluate import evaluate_candidate
from swmm_resilience.ml.bench.persist.artifacts import insert_metrics, insert_trained_model
from swmm_resilience.ml.bench.persist.candidates import candidate_definition, insert_candidate
from swmm_resilience.ml.bench.persist.evaluations import finalize_candidate, persist_fold
from swmm_resilience.ml.bench.persist.identity import sample_identity
from swmm_resilience.ml.bench.persist.training_run import open_training_run
from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.ranking import RankingCriterion
from swmm_resilience.ml.bench.train import train_candidate

CRITERION = RankingCriterion("classifier.f1", "maximize", ())
TINY = {"n_estimators": 5, "max_depth": 2}


@pytest.fixture
def finalized(sql_training_db, tmp_path):
    conn = connect_managed_database(sql_training_db)
    apply_migrations(conn)
    prepared = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )
    oof, metrics = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    identity = sample_identity(prepared, conn)
    training_run_id = open_training_run(conn, prepared, "LOSO", CRITERION)

    definition = candidate_definition("xgboost", "classification", TINY, prepared)
    candidate_id = insert_candidate(conn, training_run_id, definition)
    for fold_id in sorted(oof["fold_id"].unique()):
        persist_fold(
            conn, training_run_id, candidate_id, definition, prepared, oof,
            fold_id=int(fold_id), protocol="LOSO", identity=identity,
            fit_seconds=0.1, predict_seconds=0.01,
        )
    finalize_candidate(conn, candidate_id)

    artifacts = train_candidate(prepared, "xgboost", TINY, TINY, tmp_path / "candidates")
    yield conn, prepared, definition, candidate_id, training_run_id, artifacts, metrics
    conn.close()


def test_trained_model_is_inserted_with_a_path_not_a_blob(finalized):
    conn, prepared, definition, candidate_id, training_run_id, artifacts, _ = finalized

    model_id = insert_trained_model(
        conn, training_run_id, candidate_id, definition, artifacts,
        task="classification", prepared=prepared, protocol="LOSO",
    )
    row = conn.execute(
        "SELECT model_blob, model_path, model_sha256 FROM trained_models WHERE model_id = ?",
        (model_id,),
    ).fetchone()

    assert row[0] is None, "el blob no se usa: el .joblib vive en disco"
    assert row[1].endswith("classifier.joblib")
    assert len(row[2]) == 64


def test_the_artifact_is_linked_to_its_finalized_candidate(finalized):
    conn, prepared, definition, candidate_id, training_run_id, artifacts, _ = finalized

    model_id = insert_trained_model(
        conn, training_run_id, candidate_id, definition, artifacts,
        task="classification", prepared=prepared, protocol="LOSO",
    )
    linked = conn.execute(
        "SELECT candidate_id FROM model_artifact_candidates WHERE model_id = ?",
        (model_id,),
    ).fetchone()[0]
    assert linked == candidate_id


def test_the_sha256_matches_the_file_on_disk(finalized):
    conn, prepared, definition, candidate_id, training_run_id, artifacts, _ = finalized

    model_id = insert_trained_model(
        conn, training_run_id, candidate_id, definition, artifacts,
        task="classification", prepared=prepared, protocol="LOSO",
    )
    stored = conn.execute(
        "SELECT model_sha256 FROM trained_models WHERE model_id = ?", (model_id,)
    ).fetchone()[0]
    assert stored == artifacts.metadata["classifier_sha256"]


def test_linking_an_unfinalized_candidate_is_rejected(finalized, tmp_path):
    """Trigger model_artifact_candidates_requires_finalized_matching_candidate."""
    conn, prepared, _, _, training_run_id, artifacts, _ = finalized

    other = candidate_definition("random_forest", "classification", TINY, prepared)
    unfinalized_id = insert_candidate(conn, training_run_id, other)

    with pytest.raises(sqlite3.IntegrityError, match="finalized"):
        insert_trained_model(
            conn, training_run_id, unfinalized_id, other, artifacts,
            task="classification", prepared=prepared, protocol="LOSO",
        )


def test_linking_with_a_mismatched_recipe_is_rejected(finalized):
    conn, prepared, definition, candidate_id, training_run_id, artifacts, _ = finalized

    tampered = dict(definition, hyperparameters_json='{"max_depth":99}')
    with pytest.raises(sqlite3.IntegrityError, match="matches|matching"):
        insert_trained_model(
            conn, training_run_id, candidate_id, tampered, artifacts,
            task="classification", prepared=prepared, protocol="LOSO",
        )


def test_metrics_are_flattened_with_dotted_names(finalized):
    conn, _, _, _, training_run_id, _, metrics = finalized

    written = insert_metrics(
        conn, "training_run", training_run_id, metrics["LOSO"], scope="LOSO"
    )
    assert written > 0

    names = {
        row[0] for row in conn.execute("SELECT metric_name FROM model_metrics")
    }
    assert "classifier.f1" in names
    assert "end_to_end.rmse_vol_todos_nodos" in names


def test_a_nan_metric_is_stored_as_invalid_with_a_reason(finalized):
    conn, _, _, _, training_run_id, _, _ = finalized

    insert_metrics(
        conn, "training_run", training_run_id,
        {"classifier": {"auc_roc": float("nan"), "f1": 0.8}}, scope="LOSO",
    )
    row = conn.execute(
        "SELECT value, valid, reason FROM model_metrics WHERE metric_name = 'classifier.auc_roc'"
    ).fetchone()

    assert row[0] is None
    assert row[1] == 0
    assert "NaN" in row[2]


def test_valid_metrics_store_their_value(finalized):
    conn, _, _, _, training_run_id, _, _ = finalized

    insert_metrics(
        conn, "training_run", training_run_id, {"classifier": {"f1": 0.8}}, scope="LOSO"
    )
    row = conn.execute(
        "SELECT value, valid, reason FROM model_metrics WHERE metric_name = 'classifier.f1'"
    ).fetchone()

    assert row[0] == pytest.approx(0.8)
    assert row[1] == 1
    assert row[2] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/persist/test_artifacts.py -q`
Expected: `ModuleNotFoundError: ...persist.artifacts`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/persist/artifacts.py
"""Modelos entrenados y métricas.

El trigger model_artifact_candidates_requires_finalized_matching_candidate
exige que el candidato esté FINALIZADO y que su receta coincida campo por
campo con la fila de trained_models. Por eso esta función recibe la misma
``definition`` que se insertó en model_candidates, en vez de reconstruirla:
reconstruirla arriesga una diferencia de un carácter que aborta el trigger.
"""

from __future__ import annotations

import json
import math
import platform
import sqlite3
from datetime import datetime, timezone
from importlib.metadata import version as _package_version

from ...config import ML_RANDOM_STATE
from ..schemas import PreparedDataset
from .training_run import GROUPING_STRATEGY

_TARGET_BY_TASK = {"classification": "inunda", "regression": "vol_inundacion_m3"}


def insert_trained_model(
    conn: sqlite3.Connection,
    training_run_id: int,
    candidate_id: int,
    definition: dict,
    artifacts,
    *,
    task: str,
    prepared: PreparedDataset,
    protocol: str,
) -> int:
    """Registra el .joblib en disco y lo enlaza a su candidato finalizado."""
    target = _TARGET_BY_TASK[task]
    if task == "classification":
        path, digest = artifacts.classifier_path, artifacts.metadata["classifier_sha256"]
    else:
        path, digest = artifacts.regressor_path, artifacts.metadata["regressor_sha256"]

    run_ids = sorted(int(value) for value in prepared.keys["run_id"].unique())
    conn.execute(
        """
        INSERT INTO trained_models (
            model_id, training_run_id, target, algorithm, hyperparameters_json,
            preprocessing_json, feature_contract_id, feature_contract_sha256,
            ordered_features_json, target_transform_json, query_params_json,
            included_run_ids_json, random_seed, grouping_strategy, python_version,
            library_versions_json, model_sha256, model_blob, model_path, created_at_utc
        ) VALUES (
            (SELECT COALESCE(MAX(model_id), 0) + 1 FROM trained_models),
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?
        )
        """,
        (
            training_run_id,
            target,
            definition["algorithm"],
            definition["hyperparameters_json"],
            definition["preprocessing_json"],
            definition["feature_contract_id"],
            definition["feature_contract_sha256"],
            definition["ordered_features_json"],
            definition["target_transform_json"],
            json.dumps({"prep_id": prepared.prep_id, "protocol": protocol}),
            json.dumps(run_ids),
            ML_RANDOM_STATE,
            GROUPING_STRATEGY[protocol],
            platform.python_version(),
            json.dumps(artifacts.metadata["library_versions"]),
            digest,
            str(path.resolve()),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    model_id = int(conn.execute("SELECT MAX(model_id) FROM trained_models").fetchone()[0])
    conn.execute(
        "INSERT INTO model_artifact_candidates (model_id, candidate_id) VALUES (?, ?)",
        (model_id, candidate_id),
    )
    return model_id


def _flatten(metrics: dict, prefix: str = "") -> list[tuple[str, float | None]]:
    flat: list[tuple[str, float | None]] = []
    for key, value in metrics.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.extend(_flatten(value, f"{name}."))
        elif isinstance(value, (int, float)):
            flat.append((name, float(value)))
    return flat


def insert_metrics(
    conn: sqlite3.Connection,
    owner_kind: str,
    owner_id: int,
    metrics: dict,
    scope: str,
) -> int:
    """Aplana el dict de métricas a nombres con punto y lo persiste.

    Una métrica NaN no se descarta: se guarda con ``valid=0`` y su motivo.
    Esa información es evidencia, no ruido.
    """
    column = {
        "training_run": "training_run_id",
        "evaluation": "evaluation_id",
        "model": "model_id",
    }[owner_kind]

    rows = []
    for name, value in _flatten(metrics):
        if value is None or math.isnan(value):
            rows.append((owner_id, scope, name, None, 0, f"el valor es NaN"))
        else:
            rows.append((owner_id, scope, name, value, 1, None))

    conn.executemany(
        f"INSERT INTO model_metrics ({column}, scope, metric_name, value, valid, reason) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)
```

> **`model_metrics` cambió de forma entre migraciones.** La 001 la define con `owner_kind`/`owner_id` explícitos; la 002 y la 003 la recrean con columnas `training_run_id`/`evaluation_id`/`model_id` y `owner_kind`/`owner_id` como `GENERATED ALWAYS ... STORED`. El código de arriba asume la forma de la 003 (la vigente). **Verifica con `PRAGMA table_info(model_metrics)` antes de escribirlo** y ajusta si tu base dice otra cosa.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/persist/test_artifacts.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/persist/artifacts.py tests/ml/bench/persist/test_artifacts.py
git commit -m "feat(bench): registro de modelos entrenados y metricas"
```

---

## Task 7: `persist/rankings.py` y `persist/promotions.py`

**Files:**
- Create: `swmm_resilience/ml/bench/persist/rankings.py`
- Create: `swmm_resilience/ml/bench/persist/promotions.py`
- Test: `tests/ml/bench/persist/test_rankings.py`

**Interfaces:**
- Produces: `persist_ranking(conn, training_run_id, criterion, ranking, candidate_ids, metrics_by_family, protocol) -> tuple[int, int]` (devuelve `ranking_id` y `winner_entry_id`); `promote(conn, ranking_id, training_run_id, classifier_model_id, regressor_model_id, criterion, ranking) -> int` (devuelve `promotion_id`, ya finalizada y seleccionada).

**Restricción del trigger de finalización del ranking:** el ganador declarado debe tener el mejor `metric_ordinal = 0` **válido**, y **toda** entrada debe tener una fila de score con `metric_ordinal = 0`. Los desempates **no** los verifica el esquema — su propio comentario lo dice — así que `ranking.py` (Python) los resuelve y aquí sólo se registran.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/persist/test_rankings.py
import sqlite3

import pytest

from swmm_resilience.ml.bench.persist.promotions import promote
from swmm_resilience.ml.bench.persist.rankings import persist_ranking
from swmm_resilience.ml.bench.ranking import RankingCriterion, rank_candidates

CRITERION = RankingCriterion(
    primary_metric="end_to_end.rmse_vol_todos_nodos",
    primary_direction="minimize",
    tie_breakers=(("classifier.f1", "maximize"),),
)


def _metrics(rmse, f1=0.7):
    return {
        "LOSO": {
            "classifier": {"f1": f1, "precision": 0.6, "recall": 0.8, "auc_roc": 0.85},
            "regressor_oracle": {"nse": 0.6, "rmse": 2.0, "mae": 1.0, "r2": 0.6, "log_nse": 0.7},
            "end_to_end": {
                "rmse_vol_todos_nodos": rmse, "pct_nodos_correctos": 0.9,
                "vol_total_pred_m3": 10.0, "vol_total_real_m3": 11.0,
            },
            "by_factor": {},
        }
    }


# El implementador construye `bench_context`: una base migrada con dos familias
# (xgboost y random_forest) ya persistidas y finalizadas para ambas tareas,
# reutilizando las fixtures de test_artifacts.py. Devuelve
# (conn, training_run_id, candidate_ids, model_ids, metrics_by_family).


def test_ranking_row_records_the_criterion(bench_context):
    conn, training_run_id, candidate_ids, _, metrics = bench_context
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")

    ranking_id, _ = persist_ranking(
        conn, training_run_id, CRITERION, ranking, candidate_ids, metrics, "LOSO"
    )
    row = conn.execute(
        "SELECT primary_metric, primary_direction FROM model_rankings WHERE ranking_id = ?",
        (ranking_id,),
    ).fetchone()

    assert row[0] == "end_to_end.rmse_vol_todos_nodos"
    assert row[1] == "minimize"


def test_one_entry_per_family_pairing_classifier_and_regressor(bench_context):
    conn, training_run_id, candidate_ids, _, metrics = bench_context
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")

    ranking_id, _ = persist_ranking(
        conn, training_run_id, CRITERION, ranking, candidate_ids, metrics, "LOSO"
    )
    rows = conn.execute(
        "SELECT classifier_candidate_id, regressor_candidate_id FROM model_ranking_entries "
        "WHERE ranking_id = ?",
        (ranking_id,),
    ).fetchall()

    assert len(rows) == len(metrics)
    assert all(row[0] is not None and row[1] is not None for row in rows)


def test_the_primary_metric_is_stored_at_ordinal_zero(bench_context):
    """El trigger de finalizacion solo mira metric_ordinal = 0."""
    conn, training_run_id, candidate_ids, _, metrics = bench_context
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")

    ranking_id, _ = persist_ranking(
        conn, training_run_id, CRITERION, ranking, candidate_ids, metrics, "LOSO"
    )
    names = {
        row[0]
        for row in conn.execute(
            "SELECT metric_name FROM model_ranking_scores "
            "WHERE ranking_id = ? AND metric_ordinal = 0",
            (ranking_id,),
        )
    }
    assert names == {"end_to_end.rmse_vol_todos_nodos"}


def test_tie_breakers_are_stored_at_later_ordinals(bench_context):
    conn, training_run_id, candidate_ids, _, metrics = bench_context
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")

    ranking_id, _ = persist_ranking(
        conn, training_run_id, CRITERION, ranking, candidate_ids, metrics, "LOSO"
    )
    ordinals = {
        row[0]
        for row in conn.execute(
            "SELECT metric_ordinal FROM model_ranking_scores WHERE ranking_id = ?",
            (ranking_id,),
        )
    }
    assert ordinals == {0, 1}


def test_every_entry_has_an_ordinal_zero_score(bench_context):
    """Requisito literal del trigger: sin esto la finalizacion aborta."""
    conn, training_run_id, candidate_ids, _, metrics = bench_context
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")

    ranking_id, _ = persist_ranking(
        conn, training_run_id, CRITERION, ranking, candidate_ids, metrics, "LOSO"
    )
    entries = conn.execute(
        "SELECT COUNT(*) FROM model_ranking_entries WHERE ranking_id = ?", (ranking_id,)
    ).fetchone()[0]
    scored = conn.execute(
        "SELECT COUNT(*) FROM model_ranking_scores WHERE ranking_id = ? AND metric_ordinal = 0",
        (ranking_id,),
    ).fetchone()[0]
    assert entries == scored


def test_the_finalized_winner_is_the_best_valid_entry(bench_context):
    conn, training_run_id, candidate_ids, _, metrics = bench_context
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")

    ranking_id, winner_entry_id = persist_ranking(
        conn, training_run_id, CRITERION, ranking, candidate_ids, metrics, "LOSO"
    )
    stored = conn.execute(
        "SELECT winner_entry_id FROM model_ranking_finalizations WHERE ranking_id = ?",
        (ranking_id,),
    ).fetchone()[0]
    assert stored == winner_entry_id


def test_promotion_requires_a_finalized_ranking(bench_context):
    conn, training_run_id, candidate_ids, model_ids, metrics = bench_context
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")
    ranking_id, _ = persist_ranking(
        conn, training_run_id, CRITERION, ranking, candidate_ids, metrics, "LOSO"
    )

    promotion_id = promote(
        conn, ranking_id, training_run_id,
        classifier_model_id=model_ids[("xgboost", "classification")],
        regressor_model_id=model_ids[("xgboost", "regression")],
        criterion=CRITERION, ranking=ranking,
    )
    assert promotion_id > 0


def test_promotion_creates_a_selection(bench_context):
    conn, training_run_id, candidate_ids, model_ids, metrics = bench_context
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")
    ranking_id, _ = persist_ranking(
        conn, training_run_id, CRITERION, ranking, candidate_ids, metrics, "LOSO"
    )
    promote(
        conn, ranking_id, training_run_id,
        classifier_model_id=model_ids[("xgboost", "classification")],
        regressor_model_id=model_ids[("xgboost", "regression")],
        criterion=CRITERION, ranking=ranking,
    )

    count = conn.execute(
        "SELECT COUNT(*) FROM model_selections WHERE target = 'system'"
    ).fetchone()[0]
    assert count == 1


def test_promotions_are_immutable(bench_context):
    conn, training_run_id, candidate_ids, model_ids, metrics = bench_context
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")
    ranking_id, _ = persist_ranking(
        conn, training_run_id, CRITERION, ranking, candidate_ids, metrics, "LOSO"
    )
    promotion_id = promote(
        conn, ranking_id, training_run_id,
        classifier_model_id=model_ids[("xgboost", "classification")],
        regressor_model_id=model_ids[("xgboost", "regression")],
        criterion=CRITERION, ranking=ranking,
    )

    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute(
            "UPDATE model_promotions SET primary_value = 0 WHERE promotion_id = ?",
            (promotion_id,),
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/persist/test_rankings.py -q`
Expected: `ModuleNotFoundError` / `fixture 'bench_context' not found`

- [ ] **Step 3: Write minimal implementation**

Primero crea la fixture `bench_context` en `tests/ml/bench/persist/conftest.py`, factorizando lo que ya hace la fixture `finalized` de `test_artifacts.py` pero para dos familias y las dos tareas. Después:

```python
# swmm_resilience/ml/bench/persist/rankings.py
"""Ranking de candidatos en SQL.

El trigger model_ranking_finalizations_validate_insert exige que el ganador
declarado tenga el mejor metric_ordinal = 0 válido y que TODA entrada tenga
una fila de score con ese ordinal. Los desempates NO los verifica el esquema
(así lo dice su propio comentario): ranking.py los resuelve en Python y aquí
sólo quedan registrados para auditoría.
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from ...contracts import TABULAR_V3_17
from ..ranking import RankingCriterion, resolve_metric

METRIC_REGISTRY_ID = "bench-metrics-1"
INVALID_SCORE_POLICY = "rank_last"


def _score_row(value: float | None) -> tuple:
    if value is None or math.isnan(value):
        return (None, 0, "el valor es NaN o falta")
    return (float(value), 1, None)


def persist_ranking(
    conn: sqlite3.Connection,
    training_run_id: int,
    criterion: RankingCriterion,
    ranking: pd.DataFrame,
    candidate_ids: dict,
    metrics_by_family: dict,
    protocol: str,
) -> tuple[int, int]:
    """Escribe el ranking, sus entradas y scores, y lo finaliza."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO model_rankings (
            ranking_id, training_run_id, target, primary_metric, primary_direction,
            metric_registry_id, metric_registry_sha256, tie_breakers_json,
            invalid_score_policy, created_at_utc
        ) VALUES (
            (SELECT COALESCE(MAX(ranking_id), 0) + 1 FROM model_rankings),
            ?, 'system', ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            training_run_id,
            criterion.primary_metric,
            criterion.primary_direction,
            METRIC_REGISTRY_ID,
            TABULAR_V3_17.descriptor_sha256,
            json.dumps(
                [{"metric": m, "direction": d} for m, d in criterion.tie_breakers]
            ),
            INVALID_SCORE_POLICY,
            now,
        ),
    )
    ranking_id = int(
        conn.execute("SELECT MAX(ranking_id) FROM model_rankings").fetchone()[0]
    )

    winner_entry_id = None
    for entry_id, row in enumerate(ranking.itertuples(index=False), start=1):
        family = row.family
        conn.execute(
            "INSERT INTO model_ranking_entries (ranking_id, entry_id, "
            "classifier_candidate_id, regressor_candidate_id) VALUES (?, ?, ?, ?)",
            (
                ranking_id,
                entry_id,
                candidate_ids[(family, "classification")],
                candidate_ids[(family, "regression")],
            ),
        )

        protocol_metrics = metrics_by_family[family].get(protocol, {})
        paths = [criterion.primary_metric] + [m for m, _ in criterion.tie_breakers]
        for ordinal, path in enumerate(paths):
            value, valid, reason = _score_row(resolve_metric(protocol_metrics, path))
            conn.execute(
                "INSERT INTO model_ranking_scores (ranking_id, entry_id, metric_ordinal, "
                "metric_name, value, valid, invalid_reason) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ranking_id, entry_id, ordinal, path, value, valid, reason),
            )

        if winner_entry_id is None and row.valid == 1:
            winner_entry_id = entry_id

    if winner_entry_id is None:
        raise ValueError(
            "Ningún candidato tiene una métrica primaria válida; no se puede finalizar el ranking."
        )

    conn.execute(
        "INSERT INTO model_ranking_finalizations (ranking_id, winner_entry_id, "
        "finalized_at_utc) VALUES (?, ?, ?)",
        (ranking_id, winner_entry_id, now),
    )
    return ranking_id, winner_entry_id
```

```python
# swmm_resilience/ml/bench/persist/promotions.py
"""Promoción y selección del candidato ganador.

model_promotion_rankings exige un ranking ya finalizado. La selección es la
que declara qué par (clasificador, regresor) está activo; sustituye a la
anterior mediante supersedes_selection_id, sin borrar nada.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pandas as pd

from ..ranking import RankingCriterion


def promote(
    conn: sqlite3.Connection,
    ranking_id: int,
    training_run_id: int,
    *,
    classifier_model_id: int,
    regressor_model_id: int,
    criterion: RankingCriterion,
    ranking: pd.DataFrame,
) -> int:
    """Crea la promoción, la enlaza al ranking, la finaliza y la selecciona."""
    now = datetime.now(timezone.utc).isoformat()
    winner = ranking[ranking["valid"] == 1].iloc[0]

    conn.execute(
        """
        INSERT INTO model_promotions (
            promotion_id, training_run_id, target, classifier_model_id,
            regressor_model_id, primary_metric, primary_value, tie_breakers_json,
            ranking_json, promoted_at_utc
        ) VALUES (
            (SELECT COALESCE(MAX(promotion_id), 0) + 1 FROM model_promotions),
            ?, 'system', ?, ?, ?, ?, ?, ?, ?
        )
        """,
        (
            training_run_id,
            classifier_model_id,
            regressor_model_id,
            criterion.primary_metric,
            float(winner["primary_value"]),
            json.dumps([{"metric": m, "direction": d} for m, d in criterion.tie_breakers]),
            json.dumps(ranking.to_dict(orient="records"), default=str),
            now,
        ),
    )
    promotion_id = int(
        conn.execute("SELECT MAX(promotion_id) FROM model_promotions").fetchone()[0]
    )

    conn.execute(
        "INSERT INTO model_promotion_rankings (promotion_id, ranking_id) VALUES (?, ?)",
        (promotion_id, ranking_id),
    )
    conn.execute(
        "INSERT INTO model_promotion_finalizations (promotion_id, finalized_at_utc) "
        "VALUES (?, ?)",
        (promotion_id, now),
    )

    previous = conn.execute(
        "SELECT MAX(selection_id) FROM model_selections WHERE target = 'system'"
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO model_selections (selection_id, target, promotion_id, "
        "supersedes_selection_id, selected_at_utc) VALUES ("
        "(SELECT COALESCE(MAX(selection_id), 0) + 1 FROM model_selections), "
        "'system', ?, ?, ?)",
        (promotion_id, previous, now),
    )
    return promotion_id
```

> **Verifica el `CHECK` de `model_promotions`** antes de escribir: para `target='system'` exige una combinación concreta de `classifier_model_id`/`regressor_model_id` no nulos (la migración 002, líneas ~116-131, tiene el `CHECK` completo). Si tu lectura dice que `'system'` requiere ambos no nulos, el código de arriba ya cumple; si dice otra cosa, ajústalo.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/persist/test_rankings.py -q`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/persist/rankings.py swmm_resilience/ml/bench/persist/promotions.py tests/ml/bench/persist/
git commit -m "feat(bench): ranking y promocion en SQL"
```

---

## Task 8: `persist/runner.py` — orquestación y cableado al CLI

**Files:**
- Create: `swmm_resilience/ml/bench/persist/runner.py`
- Modify: `swmm_resilience/ml/bench/persist/__init__.py`
- Modify: `main.py` (`_run_bench` llama a `persist_bench_run`)
- Test: `tests/ml/bench/persist/test_runner.py`

**Interfaces:**
- Produces: `persist_bench_run(db_path, prepared, results_by_family, artifacts_by_family, criterion, protocols, promote_family=None) -> dict` (devuelve `{protocolo: {training_run_id, ranking_id, promotion_id}}`).

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/persist/test_runner.py
import sqlite3

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.ml.bench.evaluate import evaluate_candidate
from swmm_resilience.ml.bench.persist.runner import persist_bench_run
from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.ranking import RankingCriterion
from swmm_resilience.ml.bench.train import train_candidate

CRITERION = RankingCriterion(
    primary_metric="end_to_end.rmse_vol_todos_nodos",
    primary_direction="minimize",
    tie_breakers=(("classifier.f1", "maximize"),),
)
TINY = {"n_estimators": 5, "max_depth": 2}
FAMILIES = ("xgboost", "random_forest")


@pytest.fixture
def bench_run(sql_training_db, tmp_path):
    prepared = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )
    results, artifacts = {}, {}
    for family in FAMILIES:
        oof, metrics = evaluate_candidate(prepared, family, TINY, TINY)
        results[family] = (oof, metrics)
        artifacts[family] = train_candidate(
            prepared, family, TINY, TINY, tmp_path / "candidates"
        )
    written = persist_bench_run(
        sql_training_db, prepared, results, artifacts, CRITERION, ("LOSO",)
    )
    return sql_training_db, written


def _count(db_path, table):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def test_every_link_of_the_chain_has_rows(bench_run):
    db_path, _ = bench_run
    for table in (
        "training_runs", "training_run_inputs", "model_candidates",
        "model_evaluations", "model_evaluation_runs", "oof_predictions",
        "model_candidate_evaluations", "model_candidate_finalizations",
        "trained_models", "model_artifact_candidates", "model_metrics",
        "model_rankings", "model_ranking_entries", "model_ranking_scores",
        "model_ranking_finalizations", "model_promotions",
        "model_promotion_rankings", "model_promotion_finalizations",
        "model_selections",
    ):
        assert _count(db_path, table) > 0, f"{table} quedó vacía"


def test_the_training_run_ends_complete(bench_run):
    db_path, _ = bench_run
    conn = sqlite3.connect(db_path)
    try:
        statuses = {row[0] for row in conn.execute("SELECT status FROM training_runs")}
    finally:
        conn.close()
    assert statuses == {"COMPLETE"}


def test_one_training_run_per_protocol(bench_run):
    db_path, written = bench_run
    assert set(written) == {"LOSO"}
    assert _count(db_path, "training_runs") == 1


def test_two_candidates_per_family(bench_run):
    db_path, _ = bench_run
    assert _count(db_path, "model_candidates") == len(FAMILIES) * 2


def test_a_failure_mid_run_leaves_the_training_run_failed(sql_training_db, tmp_path):
    """No debe quedar un training_run RUNNING colgado."""
    prepared = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )
    oof, metrics = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    broken_artifacts = {"xgboost": None}      # revienta al registrar el modelo

    with pytest.raises(Exception):
        persist_bench_run(
            sql_training_db, prepared, {"xgboost": (oof, metrics)},
            broken_artifacts, CRITERION, ("LOSO",),
        )

    conn = sqlite3.connect(sql_training_db)
    try:
        statuses = {row[0] for row in conn.execute("SELECT status FROM training_runs")}
    finally:
        conn.close()
    assert "RUNNING" not in statuses


def test_the_promoted_family_can_be_forced(sql_training_db, tmp_path):
    prepared = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )
    results, artifacts = {}, {}
    for family in FAMILIES:
        results[family] = evaluate_candidate(prepared, family, TINY, TINY)
        artifacts[family] = train_candidate(
            prepared, family, TINY, TINY, tmp_path / "candidates"
        )

    persist_bench_run(
        sql_training_db, prepared, results, artifacts, CRITERION, ("LOSO",),
        promote_family="random_forest",
    )

    conn = sqlite3.connect(sql_training_db)
    try:
        algorithm = conn.execute(
            "SELECT tm.algorithm FROM model_promotions AS p "
            "JOIN trained_models AS tm ON tm.model_id = p.classifier_model_id"
        ).fetchone()[0]
    finally:
        conn.close()
    assert algorithm == "random_forest"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/persist/test_runner.py -q`
Expected: `ModuleNotFoundError: ...persist.runner`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/persist/runner.py
"""Orquestación de la cadena completa, en el orden que exigen los triggers.

Un training_run POR PROTOCOLO. Si algo falla a mitad, el training_run se
marca FAILED antes de propagar la excepción: nunca debe quedar una fila
RUNNING colgada, porque un lector posterior no puede distinguir "corriendo"
de "reventó hace tres semanas".
"""

from __future__ import annotations

from pathlib import Path

from ...database.connection import connect_managed_database
from ...database.migrations import apply_migrations
from ..ranking import RankingCriterion, rank_candidates
from ..schemas import PreparedDataset
from .artifacts import insert_metrics, insert_trained_model
from .candidates import candidate_definition, insert_candidate
from .evaluations import finalize_candidate, persist_fold
from .identity import sample_identity
from .promotions import promote
from .rankings import persist_ranking
from .training_run import mark_complete, mark_failed, open_training_run

_TASKS = ("classification", "regression")


def persist_bench_run(
    db_path: Path | str,
    prepared: PreparedDataset,
    results_by_family: dict,
    artifacts_by_family: dict,
    criterion: RankingCriterion,
    protocols,
    promote_family: str | None = None,
) -> dict:
    """Vuelca una corrida completa del banco a SQL, un training_run por protocolo."""
    conn = connect_managed_database(db_path)
    written: dict = {}
    try:
        apply_migrations(conn)
        identity = sample_identity(prepared, conn)

        for protocol in protocols:
            training_run_id = open_training_run(conn, prepared, protocol, criterion)
            try:
                candidate_ids, model_ids = {}, {}
                metrics_by_family = {
                    family: metrics for family, (_, metrics) in results_by_family.items()
                }

                for family, (oof, metrics) in results_by_family.items():
                    for task in _TASKS:
                        params = (
                            artifacts_by_family[family].metadata["classifier_params"]
                            if task == "classification"
                            else artifacts_by_family[family].metadata["regressor_params"]
                        )
                        definition = candidate_definition(family, task, params, prepared)
                        candidate_id = insert_candidate(conn, training_run_id, definition)
                        candidate_ids[(family, task)] = candidate_id

                        protocol_oof = oof[oof["protocol"] == protocol]
                        for fold_id in sorted(protocol_oof["fold_id"].unique()):
                            persist_fold(
                                conn, training_run_id, candidate_id, definition,
                                prepared, oof, fold_id=int(fold_id), protocol=protocol,
                                identity=identity, fit_seconds=0.0, predict_seconds=0.0,
                            )
                        finalize_candidate(conn, candidate_id)

                        model_ids[(family, task)] = insert_trained_model(
                            conn, training_run_id, candidate_id, definition,
                            artifacts_by_family[family], task=task,
                            prepared=prepared, protocol=protocol,
                        )

                    insert_metrics(
                        conn, "training_run", training_run_id,
                        metrics.get(protocol, {}), scope=f"{protocol}:{family}",
                    )

                ranking = rank_candidates(metrics_by_family, criterion, protocol)
                ranking_id, _ = persist_ranking(
                    conn, training_run_id, criterion, ranking, candidate_ids,
                    metrics_by_family, protocol,
                )

                chosen = promote_family or ranking[ranking["valid"] == 1].iloc[0]["family"]
                promotion_id = promote(
                    conn, ranking_id, training_run_id,
                    classifier_model_id=model_ids[(chosen, "classification")],
                    regressor_model_id=model_ids[(chosen, "regression")],
                    criterion=criterion, ranking=ranking,
                )
                mark_complete(conn, training_run_id)
                conn.commit()
                written[protocol] = {
                    "training_run_id": training_run_id,
                    "ranking_id": ranking_id,
                    "promotion_id": promotion_id,
                }
            except Exception as error:
                conn.rollback()
                mark_failed(conn, training_run_id, type(error).__name__, str(error))
                conn.commit()
                raise
    finally:
        conn.close()
    return written
```

> **Cuidado con el rollback:** `mark_failed` se ejecuta **después** del `rollback()`, porque el rollback descarta también la fila del `training_run`. Si al probar `test_a_failure_mid_run_leaves_the_training_run_failed` la fila desaparece del todo, es que `open_training_run` quedó dentro de la transacción abortada: haz `conn.commit()` justo después de abrirlo, para que la evidencia del intento fallido sobreviva.

En `main.py`, dentro de `_run_bench`, cambiar el bloque de evaluación para que guarde `(oof, metrics)` por familia y, tras escribir los reportes, llamar a `persist_bench_run(config.dataset.db_path, prepared, results, artifacts, criterion, config.bench.protocols, promote_family=None if config.bench.promote == "auto" else config.bench.promote)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/ -q && ./venv/Scripts/python.exe -m pytest -q`
Expected: todo verde

- [ ] **Step 5: Verificación manual**

```bash
./venv/Scripts/python.exe main.py --bench
./venv/Scripts/python.exe -c "import sqlite3; c=sqlite3.connect('outputs/training_v17.sqlite3'); print(c.execute('SELECT tm.algorithm, p.primary_metric, p.primary_value FROM model_promotions p JOIN trained_models tm ON tm.model_id=p.classifier_model_id').fetchall())"
```

Expected: la familia promovida con su valor de métrica primaria.

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/ml/bench/persist/runner.py swmm_resilience/ml/bench/persist/__init__.py main.py tests/ml/bench/persist/test_runner.py
git commit -m "feat(bench): orquestacion de la cadena de proveniencia completa"
```

---

## Verificación final del plan

- [ ] `./venv/Scripts/python.exe -m pytest -q` verde, con los ~65 tests nuevos sumados a la línea base.
- [ ] `pytest -m scale` verde: `oof_predictions` crece con familias × targets × folds; con LOSO de 25 folds sobre 175 escenarios el volumen se nota. Es el gate que la spec §16 pide antes de cerrar.
- [ ] Las 19 tablas de la cadena tienen filas tras `python main.py --bench`.
- [ ] Ningún `training_run` queda en `RUNNING` tras un fallo provocado.
- [ ] La migración 006 no perdió filas ni triggers previos de `trained_models`.

## Qué sigue

**Plan 3** — retiro del legacy: desacople de `FEATURE_COLS` hacia `ml/contracts.py`, bloque 1 (stack A completo + parte ML de la GUI), bloque 2 (stack B, sólo con el test de paridad verde), retirada de la mitad de entrenamiento de `--persist-sql`, y documentación (`docs/ML_BENCH.md`, `FLUJO_ACTUAL.md`, `COMANDOS.md`, marcado histórico de los dos documentos de la tesis).
