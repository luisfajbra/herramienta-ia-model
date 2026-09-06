# Banco ML — Plan 1: capa `bench/` (preprocesamiento, entrenamiento, evaluación)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir `swmm_resilience/ml/bench/`: un banco multi-modelo con tres etapas separadas (preprocesamiento → entrenamiento → evaluación) alimentado desde SQLite v17, que produzca un ranking comparable entre familias de modelos y reproduzca exactamente las métricas del evaluador actual.

**Architecture:** Etapa 1 (`preprocess.py` + `folds.py` + `quality.py`) lee el frame de SQL, aplica la lista blanca del contrato de 17 features, materializa folds y escribe un `PreparedDataset` con hash de proveniencia. El registro de familias (`models/*.py`) devuelve `Pipeline`s de sklearn que llevan dentro la imputación y el escalado, de modo que se ajustan solo con el train de cada fold. `evaluate.py` se parte en una capa núcleo que consume predicciones (`score_predictions`) y una de conveniencia que las produce (`evaluate_candidate`).

**Tech Stack:** Python 3.12, pandas 2.3.3, pyarrow 22, scikit-learn 1.8.0, xgboost 3.2.0, torch 2.6.0, pytest 9.

**Spec:** `docs/superpowers/specs/2026-09-05-ml-bench-preproc-train-eval-design.md`

## Global Constraints

- Contrato de features: `TABULAR_V3_17` / `FEATURE_COLUMNS_V17` de `swmm_resilience/ml/contracts.py`. **17 columnas, en ese orden exacto.** No se modifica.
- Semilla global: `ML_RANDOM_STATE = 42` de `swmm_resilience/config.py`.
- Protocolos: `"LOSO"` y `"GroupKFold5"`, ambos agrupados por `factor_mult`.
- Métrica primaria del ranking: `end_to_end.rmse_vol_todos_nodos`, dirección `minimize`.
- **Regla anti-fuga (spec §5.1 regla 1):** `preprocess.py`, `folds.py` y `quality.py` sólo pueden importar de `sklearn.model_selection`. Prohibido: cualquier otro submódulo de `sklearn`, `xgboost`, `torch`.
- El regresor se entrena sobre `log1p(vol_inundacion_m3)` y se invierte con `expm1` + `clip(min=0)`.
- XGBoost y RandomForest van **sin escalador**; linear, svm y mlp llevan `StandardScaler`.
- `scale_pos_weight` se calcula con el train de cada fold, nunca con el dataset completo.
- Agregación de métricas (asimetría heredada de `evaluator.py`, se conserva): clasificador y end-to-end **promediados entre folds**; regresor-oracle sobre el **pool concatenado**.
- Comandos de test: `./venv/Scripts/python.exe -m pytest -q` (el `python` desnudo de la máquina es un 3.14 sin dependencias).
- No se toca `swmm_resilience/ml/temporal/`, `ml/trainer.py` ni `ml/evaluator.py` en este plan. El retiro del legacy es el Plan 3.

---

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `swmm_resilience/ml/bench/__init__.py` | Exporta la API pública de la capa |
| `swmm_resilience/ml/bench/schemas.py` | `PreparedDataset`, esquema del frame OOF, constantes de columnas |
| `swmm_resilience/ml/bench/folds.py` | Construcción de folds por protocolo (parametrizada; sirve para folds internos futuros) |
| `swmm_resilience/ml/bench/quality.py` | Informe de calidad del dataset |
| `swmm_resilience/ml/bench/preprocess.py` | `prepare_dataset()`: SQL → `PreparedDataset` en disco |
| `swmm_resilience/ml/bench/registry.py` | Descubrimiento de familias por nombre |
| `swmm_resilience/ml/bench/models/xgboost_family.py` | Familia XGBoost |
| `swmm_resilience/ml/bench/models/random_forest_family.py` | Familia RandomForest |
| `swmm_resilience/ml/bench/models/linear_family.py` | Familia lineal (LogisticRegression / Ridge) |
| `swmm_resilience/ml/bench/models/svm_family.py` | Familia SVM (SVC / SVR) |
| `swmm_resilience/ml/bench/models/mlp_family.py` | Familia MLP (wrapper propio sobre torch) |
| `swmm_resilience/ml/bench/metrics.py` | Las tres capas de métricas + estratificación por factor |
| `swmm_resilience/ml/bench/evaluate.py` | `score_predictions()` (núcleo) + `evaluate_candidate()` (conveniencia) |
| `swmm_resilience/ml/bench/train.py` | `train_candidate()`: ajuste sobre todo el dataset + artefactos |
| `swmm_resilience/ml/bench/ranking.py` | Orden de candidatos por métrica primaria y desempates |
| `swmm_resilience/ml/bench/reports.py` | Export a JSON/CSV en `outputs/bench/reports/` |

Tests espejo en `tests/ml/bench/`.

---

## Task 1: Esqueleto del paquete y `schemas.py`

**Files:**
- Create: `swmm_resilience/ml/bench/__init__.py`
- Create: `swmm_resilience/ml/bench/schemas.py`
- Create: `tests/ml/bench/__init__.py`
- Test: `tests/ml/bench/test_schemas.py`

**Interfaces:**
- Consumes: `FEATURE_COLUMNS_V17` de `swmm_resilience.ml.contracts`.
- Produces: `PreparedDataset` (dataclass con `.prep_id`, `.keys`, `.X`, `.y_clf`, `.y_reg`, `.folds`, `.manifest`, `.quality`, métodos `.save(directory)` y `.load(directory)` como classmethod); constantes `KEY_COLUMNS`, `OOF_COLUMNS`, `FOLD_COLUMNS`, `PROTOCOLS`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/test_schemas.py
import pandas as pd
import pytest

from swmm_resilience.ml.bench.schemas import (
    FOLD_COLUMNS,
    KEY_COLUMNS,
    OOF_COLUMNS,
    PROTOCOLS,
    PreparedDataset,
)
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17


def _tiny_prepared() -> PreparedDataset:
    n = 6
    keys = pd.DataFrame(
        {
            "run_id": [1, 1, 1, 2, 2, 2],
            "network_id": [1] * n,
            "scenario_id": [1, 1, 1, 2, 2, 2],
            "scenario_key": ["base@1.0"] * 3 + ["base@2.0"] * 3,
            "scenario_kind": ["base"] * n,
            "node_id": ["N0", "N1", "N2"] * 2,
            "factor_mult": [1.0] * 3 + [2.0] * 3,
            "shape_id": ["base"] * n,
        }
    )
    X = pd.DataFrame(
        {col: [float(i) for i in range(n)] for col in FEATURE_COLUMNS_V17}
    )
    folds = pd.DataFrame(
        {
            "protocol": ["LOSO"] * n,
            "fold_id": [0] * n,
            "sample_idx": list(range(n)),
            "split": ["train"] * 3 + ["test"] * 3,
        }
    )
    return PreparedDataset(
        prep_id="abc123def456789a",
        keys=keys,
        X=X,
        y_clf=pd.Series([0, 0, 1, 1, 1, 0], name="inunda"),
        y_reg=pd.Series([0.0, 0.0, 5.0, 7.0, 9.0, 0.0], name="vol_inundacion_m3"),
        folds=folds,
        manifest={"prep_id": "abc123def456789a", "contract_id": "tabular_v3_17"},
        quality={"n_rows": n},
    )


def test_column_constants_match_the_contract():
    assert KEY_COLUMNS == (
        "run_id",
        "network_id",
        "scenario_id",
        "scenario_key",
        "scenario_kind",
        "node_id",
        "factor_mult",
        "shape_id",
    )
    assert FOLD_COLUMNS == ("protocol", "fold_id", "sample_idx", "split")
    assert OOF_COLUMNS == (
        "sample_idx",
        "protocol",
        "fold_id",
        "y_pred_clf",
        "y_prob_clf",
        "y_pred_reg",
    )
    assert PROTOCOLS == ("LOSO", "GroupKFold5")


def test_save_then_load_roundtrips_every_component(tmp_path):
    prepared = _tiny_prepared()
    directory = prepared.save(tmp_path)

    assert directory == tmp_path / prepared.prep_id
    reloaded = PreparedDataset.load(directory)

    assert reloaded.prep_id == prepared.prep_id
    pd.testing.assert_frame_equal(reloaded.keys, prepared.keys)
    pd.testing.assert_frame_equal(reloaded.X, prepared.X)
    pd.testing.assert_series_equal(reloaded.y_clf, prepared.y_clf)
    pd.testing.assert_series_equal(reloaded.y_reg, prepared.y_reg)
    pd.testing.assert_frame_equal(reloaded.folds, prepared.folds)
    assert reloaded.manifest == prepared.manifest
    assert reloaded.quality == prepared.quality


def test_load_rejects_a_directory_missing_a_component(tmp_path):
    prepared = _tiny_prepared()
    directory = prepared.save(tmp_path)
    (directory / "folds.parquet").unlink()

    with pytest.raises(FileNotFoundError, match="folds.parquet"):
        PreparedDataset.load(directory)


def test_feature_columns_must_match_the_contract_order():
    prepared = _tiny_prepared()
    shuffled = prepared.X.loc[:, list(reversed(FEATURE_COLUMNS_V17))]

    with pytest.raises(ValueError, match="orden del contrato"):
        PreparedDataset(
            prep_id=prepared.prep_id,
            keys=prepared.keys,
            X=shuffled,
            y_clf=prepared.y_clf,
            y_reg=prepared.y_reg,
            folds=prepared.folds,
            manifest=prepared.manifest,
            quality=prepared.quality,
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_schemas.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'swmm_resilience.ml.bench'`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/__init__.py
"""Banco ML: preprocesamiento, entrenamiento y evaluación multi-modelo.

Las tres etapas son módulos separados con una única dirección de dependencia.
Ver docs/superpowers/specs/2026-09-05-ml-bench-preproc-train-eval-design.md.
"""

from .schemas import PreparedDataset

__all__ = ["PreparedDataset"]
```

```python
# swmm_resilience/ml/bench/schemas.py
"""Estructuras de datos compartidas por las tres etapas del banco.

Este módulo no ejecuta lógica de negocio: sólo define la forma de los datos
que viajan entre etapas y cómo se persisten.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..contracts import FEATURE_COLUMNS_V17

KEY_COLUMNS = (
    "run_id",
    "network_id",
    "scenario_id",
    "scenario_key",
    "scenario_kind",
    "node_id",
    "factor_mult",
    "shape_id",
)

FOLD_COLUMNS = ("protocol", "fold_id", "sample_idx", "split")

OOF_COLUMNS = (
    "sample_idx",
    "protocol",
    "fold_id",
    "y_pred_clf",
    "y_prob_clf",
    "y_pred_reg",
)

PROTOCOLS = ("LOSO", "GroupKFold5")

_COMPONENT_FILES = {
    "keys": "keys.parquet",
    "features": "features.parquet",
    "targets": "targets.parquet",
    "folds": "folds.parquet",
    "manifest": "manifest.json",
    "quality": "quality_report.json",
}


@dataclass(frozen=True)
class PreparedDataset:
    """Salida de la etapa 1: datos listos para entrenar, sin transformar.

    Deliberadamente NO contiene nada imputado ni escalado. Esas operaciones
    pertenecen al Pipeline de cada familia, que se ajusta dentro del fold.
    """

    prep_id: str
    keys: pd.DataFrame
    X: pd.DataFrame
    y_clf: pd.Series
    y_reg: pd.Series
    folds: pd.DataFrame
    manifest: dict
    quality: dict

    def __post_init__(self) -> None:
        if tuple(self.X.columns) != FEATURE_COLUMNS_V17:
            raise ValueError(
                "Las features no están en el orden del contrato "
                f"{FEATURE_COLUMNS_V17}; recibido {tuple(self.X.columns)}"
            )
        if tuple(self.keys.columns) != KEY_COLUMNS:
            raise ValueError(
                f"keys debe tener las columnas {KEY_COLUMNS}; "
                f"recibido {tuple(self.keys.columns)}"
            )
        if tuple(self.folds.columns) != FOLD_COLUMNS:
            raise ValueError(
                f"folds debe tener las columnas {FOLD_COLUMNS}; "
                f"recibido {tuple(self.folds.columns)}"
            )

    def save(self, output_dir: Path | str) -> Path:
        """Escribe el dataset en ``output_dir/<prep_id>/`` y devuelve esa ruta."""
        directory = Path(output_dir) / self.prep_id
        directory.mkdir(parents=True, exist_ok=True)
        self.keys.to_parquet(directory / _COMPONENT_FILES["keys"], index=False)
        self.X.to_parquet(directory / _COMPONENT_FILES["features"], index=False)
        pd.DataFrame({self.y_clf.name: self.y_clf, self.y_reg.name: self.y_reg}).to_parquet(
            directory / _COMPONENT_FILES["targets"], index=False
        )
        self.folds.to_parquet(directory / _COMPONENT_FILES["folds"], index=False)
        (directory / _COMPONENT_FILES["manifest"]).write_text(
            json.dumps(self.manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (directory / _COMPONENT_FILES["quality"]).write_text(
            json.dumps(self.quality, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> PreparedDataset:
        """Lee un dataset escrito por :meth:`save`."""
        directory = Path(directory)
        for filename in _COMPONENT_FILES.values():
            if not (directory / filename).exists():
                raise FileNotFoundError(
                    f"Falta {filename} en {directory}; el PreparedDataset está incompleto"
                )
        targets = pd.read_parquet(directory / _COMPONENT_FILES["targets"])
        manifest = json.loads(
            (directory / _COMPONENT_FILES["manifest"]).read_text(encoding="utf-8")
        )
        return cls(
            prep_id=manifest["prep_id"],
            keys=pd.read_parquet(directory / _COMPONENT_FILES["keys"]),
            X=pd.read_parquet(directory / _COMPONENT_FILES["features"]),
            y_clf=targets["inunda"],
            y_reg=targets["vol_inundacion_m3"],
            folds=pd.read_parquet(directory / _COMPONENT_FILES["folds"]),
            manifest=manifest,
            quality=json.loads(
                (directory / _COMPONENT_FILES["quality"]).read_text(encoding="utf-8")
            ),
        )
```

```python
# tests/ml/bench/__init__.py
```
(archivo vacío)

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_schemas.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/__init__.py swmm_resilience/ml/bench/schemas.py tests/ml/bench/__init__.py tests/ml/bench/test_schemas.py
git commit -m "feat(bench): PreparedDataset y constantes de columnas"
```

---

## Task 2: `folds.py` — construcción de folds por protocolo

**Files:**
- Create: `swmm_resilience/ml/bench/folds.py`
- Test: `tests/ml/bench/test_folds.py`

**Interfaces:**
- Consumes: `FOLD_COLUMNS`, `PROTOCOLS` de `schemas.py`.
- Produces: `build_folds(groups: pd.Series, protocol: str) -> pd.DataFrame` (columnas `FOLD_COLUMNS`, formato largo: una fila por (fold, muestra)); `build_all_folds(groups: pd.Series, protocols: Sequence[str]) -> pd.DataFrame` (concatenación de los anteriores); `n_folds_for(groups, protocol) -> int`.

**Nota de diseño:** `build_folds` recibe `groups` en vez de un `PreparedDataset` a propósito. Es lo que permitirá construir folds internos sobre el train de un fold externo cuando llegue Optuna, sin tocar este módulo.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/test_folds.py
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from swmm_resilience.ml.bench.folds import build_all_folds, build_folds, n_folds_for
from swmm_resilience.ml.bench.schemas import FOLD_COLUMNS


@pytest.fixture
def groups_25x7() -> pd.Series:
    """25 factores x 7 formas x 4 nodos, agrupado por factor."""
    factors = [round(0.2 * i, 1) for i in range(1, 26)]
    values = []
    for factor in factors:
        values.extend([factor] * 28)
    return pd.Series(values, name="factor_mult")


def test_loso_makes_one_fold_per_group(groups_25x7):
    folds = build_folds(groups_25x7, "LOSO")

    assert tuple(folds.columns) == FOLD_COLUMNS
    assert folds["fold_id"].nunique() == groups_25x7.nunique() == 25
    assert n_folds_for(groups_25x7, "LOSO") == 25


def test_every_sample_is_tested_exactly_once_per_protocol(groups_25x7):
    for protocol in ("LOSO", "GroupKFold5"):
        folds = build_folds(groups_25x7, protocol)
        tested = folds.loc[folds["split"] == "test", "sample_idx"]
        assert sorted(tested) == list(range(len(groups_25x7)))
        assert tested.duplicated().sum() == 0


def test_a_group_never_appears_in_train_and_test_of_the_same_fold(groups_25x7):
    for protocol in ("LOSO", "GroupKFold5"):
        folds = build_folds(groups_25x7, protocol)
        for fold_id, chunk in folds.groupby("fold_id"):
            train_groups = set(groups_25x7.iloc[chunk.loc[chunk["split"] == "train", "sample_idx"]])
            test_groups = set(groups_25x7.iloc[chunk.loc[chunk["split"] == "test", "sample_idx"]])
            assert not (train_groups & test_groups), f"{protocol} fold {fold_id} filtra grupos"


def test_train_and_test_together_cover_the_dataset_in_every_fold(groups_25x7):
    folds = build_folds(groups_25x7, "GroupKFold5")
    for _, chunk in folds.groupby("fold_id"):
        assert sorted(chunk["sample_idx"]) == list(range(len(groups_25x7)))


def test_groupkfold5_matches_sklearn_split_exactly(groups_25x7):
    """La paridad con evaluator.py depende de reproducir GroupKFold literalmente."""
    folds = build_folds(groups_25x7, "GroupKFold5")
    x_dummy = np.zeros((len(groups_25x7), 1))

    expected = list(GroupKFold(n_splits=5).split(x_dummy, None, groups_25x7.values))
    for fold_id, (train_idx, test_idx) in enumerate(expected):
        chunk = folds[folds["fold_id"] == fold_id]
        got_train = chunk.loc[chunk["split"] == "train", "sample_idx"].tolist()
        got_test = chunk.loc[chunk["split"] == "test", "sample_idx"].tolist()
        assert got_train == sorted(train_idx.tolist())
        assert got_test == sorted(test_idx.tolist())


def test_loso_folds_are_ordered_by_group_value(groups_25x7):
    folds = build_folds(groups_25x7, "LOSO")
    first_test = folds[(folds["fold_id"] == 0) & (folds["split"] == "test")]["sample_idx"]
    assert groups_25x7.iloc[first_test].unique().tolist() == [0.2]


def test_build_all_folds_concatenates_both_protocols(groups_25x7):
    folds = build_all_folds(groups_25x7, ("LOSO", "GroupKFold5"))
    assert set(folds["protocol"]) == {"LOSO", "GroupKFold5"}
    assert len(folds) == len(build_folds(groups_25x7, "LOSO")) + len(
        build_folds(groups_25x7, "GroupKFold5")
    )


def test_groupkfold5_rejects_fewer_than_five_groups():
    groups = pd.Series([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], name="factor_mult")
    with pytest.raises(ValueError, match="GroupKFold5 necesita al menos 5 grupos"):
        build_folds(groups, "GroupKFold5")


def test_unknown_protocol_is_rejected(groups_25x7):
    with pytest.raises(ValueError, match="Protocolo desconocido"):
        build_folds(groups_25x7, "LeaveOneShapeOut")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_folds.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'swmm_resilience.ml.bench.folds'`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/folds.py
"""Construcción de folds de validación cruzada.

Sólo se permite importar de ``sklearn.model_selection`` (splitters, que no
ajustan nada). Importar transformadores o estimadores aquí rompería la
garantía anti-fuga de la etapa 1 — ver la spec §5.1 regla 1.

``build_folds`` recibe una serie de grupos y no un dataset: eso es lo que
permitirá construir folds internos sobre el train de un fold externo cuando
se añada la búsqueda de hiperparámetros, sin modificar este módulo.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from .schemas import FOLD_COLUMNS, PROTOCOLS

_GROUPKFOLD_SPLITS = 5


def n_folds_for(groups: pd.Series, protocol: str) -> int:
    """Número de folds que producirá ``protocol`` sobre ``groups``."""
    if protocol == "LOSO":
        return int(groups.nunique())
    if protocol == "GroupKFold5":
        return _GROUPKFOLD_SPLITS
    raise ValueError(f"Protocolo desconocido: {protocol!r}. Opciones: {PROTOCOLS}")


def _splitter(groups: pd.Series, protocol: str):
    if protocol == "LOSO":
        return LeaveOneGroupOut()
    if protocol == "GroupKFold5":
        distinct = int(groups.nunique())
        if distinct < _GROUPKFOLD_SPLITS:
            raise ValueError(
                f"GroupKFold5 necesita al menos 5 grupos; el dataset tiene {distinct}"
            )
        return GroupKFold(n_splits=_GROUPKFOLD_SPLITS)
    raise ValueError(f"Protocolo desconocido: {protocol!r}. Opciones: {PROTOCOLS}")


def build_folds(groups: pd.Series, protocol: str) -> pd.DataFrame:
    """Materializa los folds de ``protocol`` como un frame en formato largo.

    Devuelve una fila por (fold, muestra) con ``split`` en {'train','test'}.
    Persistirlo así —y no como un objeto de sklearn— es lo que permite que un
    modelo entrenado en otro script, o meses después, use exactamente los
    mismos folds.
    """
    splitter = _splitter(groups, protocol)
    x_dummy = np.zeros((len(groups), 1))
    records: list[dict] = []
    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(x_dummy, None, groups.values)
    ):
        for split_name, indices in (("train", train_idx), ("test", test_idx)):
            for sample_idx in sorted(int(value) for value in indices):
                records.append(
                    {
                        "protocol": protocol,
                        "fold_id": fold_id,
                        "sample_idx": sample_idx,
                        "split": split_name,
                    }
                )
    return pd.DataFrame.from_records(records, columns=list(FOLD_COLUMNS))


def build_all_folds(groups: pd.Series, protocols: Sequence[str]) -> pd.DataFrame:
    """Concatena los folds de varios protocolos en un solo frame."""
    frames = [build_folds(groups, protocol) for protocol in protocols]
    return pd.concat(frames, ignore_index=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_folds.py -q`
Expected: 9 passed

> **Si `test_loso_folds_are_ordered_by_group_value` falla:** `LeaveOneGroupOut` de sklearn ordena por valor de grupo ascendente. Si en tu versión no lo hace, no reordenes a mano — anota el orden real y ajusta el test, porque lo que importa para la paridad es que coincida con lo que hace `evaluator.py`, que usa el mismo `LeaveOneGroupOut`.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/folds.py tests/ml/bench/test_folds.py
git commit -m "feat(bench): construccion de folds LOSO y GroupKFold5"
```

---

## Task 3: Test estructural anti-fuga

**Files:**
- Create: `swmm_resilience/ml/bench/quality.py` (stub mínimo para que el test pueda inspeccionarlo)
- Test: `tests/ml/bench/test_stage1_imports.py`

**Interfaces:**
- Consumes: nada.
- Produces: el guardián que impide que las tareas siguientes metan un transformador en la etapa 1. `quality.py` se implementa de verdad en la Task 4.

**Por qué va antes que `preprocess.py`:** este test es la razón de ser del diseño. Escribirlo primero significa que la etapa 1 nace con la restricción puesta, en vez de auditarla después.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/test_stage1_imports.py
"""La etapa 1 no puede ajustar transformadores: garantía anti-fuga estructural.

Un splitter (sklearn.model_selection) no ajusta nada y sí está permitido —
folds.py lo necesita para reproducir exactamente los folds de evaluator.py.
Cualquier otro submódulo de sklearn, xgboost o torch queda prohibido.
"""

import ast
from pathlib import Path

import pytest

STAGE1_MODULES = ("preprocess.py", "folds.py", "quality.py")

ALLOWED_SKLEARN_PREFIXES = ("sklearn.model_selection",)
FORBIDDEN_ROOTS = ("xgboost", "torch")

BENCH_DIR = Path(__file__).resolve().parents[3] / "swmm_resilience" / "ml" / "bench"


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
    return found


@pytest.mark.parametrize("module_name", STAGE1_MODULES)
def test_stage1_module_imports_no_transformer_or_estimator(module_name):
    path = BENCH_DIR / module_name
    assert path.exists(), f"{module_name} no existe todavía"

    offenders = []
    for imported in _imported_modules(path):
        root = imported.split(".")[0]
        if root in FORBIDDEN_ROOTS:
            offenders.append(imported)
        elif root == "sklearn" and not imported.startswith(ALLOWED_SKLEARN_PREFIXES):
            offenders.append(imported)

    assert not offenders, (
        f"{module_name} importa {offenders}. La etapa 1 sólo puede importar de "
        "sklearn.model_selection; imputar o escalar aquí introduciría fuga de "
        "datos porque se ajustaría sobre el dataset completo. Ver spec §5.1."
    )


def test_the_guard_would_catch_a_real_violation(tmp_path):
    """Verifica que el detector no es un test vacío que siempre pasa."""
    offending = tmp_path / "offending.py"
    offending.write_text(
        "from sklearn.preprocessing import StandardScaler\n", encoding="utf-8"
    )
    assert "sklearn.preprocessing" in _imported_modules(offending)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_stage1_imports.py -q`
Expected: FAIL con `assert path.exists()` para `preprocess.py` y `quality.py` (`folds.py` ya pasa desde la Task 2)

- [ ] **Step 3: Write minimal implementation**

Crear `swmm_resilience/ml/bench/quality.py` con el stub, y `preprocess.py` con el stub:

```python
# swmm_resilience/ml/bench/quality.py
"""Informe de calidad del dataset preparado.

Restricción: sólo pandas/numpy. Ver tests/ml/bench/test_stage1_imports.py.
"""

from __future__ import annotations

import pandas as pd


def build_quality_report(
    keys: pd.DataFrame,
    X: pd.DataFrame,
    y_clf: pd.Series,
    y_reg: pd.Series,
    flood_threshold_m3: float,
) -> dict:
    """Implementado en la Task 4."""
    raise NotImplementedError
```

```python
# swmm_resilience/ml/bench/preprocess.py
"""Etapa 1 del banco: SQL -> PreparedDataset.

NO-OBJETIVOS de esta etapa, y son deliberados: no imputa, no escala, no
aplica PCA, no elimina filas con nulos permitidos por el contrato, y no
transforma el target. Todo eso pertenece al Pipeline de cada familia, que se
ajusta dentro de cada fold. Si algo de eso aparece en este archivo, es un bug
— y tests/ml/bench/test_stage1_imports.py lo detecta.
"""

from __future__ import annotations

from pathlib import Path

from .schemas import PreparedDataset


def prepare_dataset(db_path: Path, **kwargs) -> PreparedDataset:
    """Implementado en la Task 4."""
    raise NotImplementedError
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_stage1_imports.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/quality.py swmm_resilience/ml/bench/preprocess.py tests/ml/bench/test_stage1_imports.py
git commit -m "test(bench): guardian estructural anti-fuga de la etapa 1"
```

---

## Task 4: `quality.py` — informe de calidad

**Files:**
- Modify: `swmm_resilience/ml/bench/quality.py`
- Test: `tests/ml/bench/test_quality.py`

**Interfaces:**
- Consumes: `KEY_COLUMNS` de `schemas.py`; `FEATURE_COLUMNS_V17`, `NULLABLE_FEATURE_COLUMNS_V17` de `swmm_resilience.ml.contracts`.
- Produces: `build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3) -> dict`; `DatasetQualityError` (excepción de los dos errores duros).

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/test_quality.py
import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.bench.quality import DatasetQualityError, build_quality_report
from swmm_resilience.ml.bench.schemas import KEY_COLUMNS
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17


def _frames(n_nodes=4, factors=(1.0, 2.0, 3.0)):
    keys_rows, feature_rows, clf, reg = [], [], [], []
    for run_id, factor in enumerate(factors, start=1):
        for node_idx in range(n_nodes):
            flooded = 1 if (node_idx < 2 and factor >= 2.0) else 0
            keys_rows.append(
                {
                    "run_id": run_id,
                    "network_id": 1,
                    "scenario_id": run_id,
                    "scenario_key": f"base@{factor}",
                    "scenario_kind": "base",
                    "node_id": f"N{node_idx}",
                    "factor_mult": factor,
                    "shape_id": "base",
                }
            )
            feature_rows.append({col: float(node_idx + 1) for col in FEATURE_COLUMNS_V17})
            clf.append(flooded)
            reg.append(50.0 * factor if flooded else 0.0)
    keys = pd.DataFrame(keys_rows, columns=list(KEY_COLUMNS))
    X = pd.DataFrame(feature_rows, columns=list(FEATURE_COLUMNS_V17))
    return keys, X, pd.Series(clf, name="inunda"), pd.Series(reg, name="vol_inundacion_m3")


def test_report_counts_rows_runs_and_class_balance():
    keys, X, y_clf, y_reg = _frames()
    report = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)

    assert report["n_rows"] == 12
    assert report["n_runs"] == 3
    assert report["class_balance"]["n_flooded"] == 4
    assert report["class_balance"]["n_not_flooded"] == 8
    assert report["class_balance"]["flooded_ratio"] == pytest.approx(4 / 12)


def test_report_breaks_down_rows_by_factor_and_shape():
    keys, X, y_clf, y_reg = _frames()
    report = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)

    assert report["rows_by_factor"] == {"1.0": 4, "2.0": 4, "3.0": 4}
    assert report["rows_by_shape"] == {"base": 12}
    assert report["flooded_by_factor"] == {"1.0": 0, "2.0": 2, "3.0": 2}


def test_report_lists_nulls_against_what_the_contract_allows():
    keys, X, y_clf, y_reg = _frames()
    X = X.copy()
    X.loc[0, "diam_max_in"] = np.nan       # nullable segun el contrato
    report = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)

    assert report["nulls_by_column"]["diam_max_in"] == 1
    assert report["nulls_by_column"]["elev_fondo"] == 0
    assert "diam_max_in" in report["nullable_columns"]
    assert report["unexpected_null_columns"] == []


def test_report_flags_nulls_in_a_required_column():
    keys, X, y_clf, y_reg = _frames()
    X = X.copy()
    X.loc[0, "elev_fondo"] = np.nan        # NO es nullable
    report = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)

    assert report["unexpected_null_columns"] == ["elev_fondo"]


def test_report_lists_constant_columns():
    keys, X, y_clf, y_reg = _frames()
    X = X.copy()
    X["prof_max"] = 1.5
    report = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)

    assert "prof_max" in report["constant_columns"]
    assert "elev_fondo" not in report["constant_columns"]


def test_report_includes_percentiles_per_feature():
    keys, X, y_clf, y_reg = _frames()
    report = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)

    stats = report["feature_stats"]["elev_fondo"]
    assert set(stats) == {"min", "p1", "p25", "p50", "p75", "p99", "max"}
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(4.0)


def test_report_counts_threshold_disagreements_without_raising():
    """inunda viene persistido; si discrepa del umbral se reporta, no se corrige."""
    keys, X, y_clf, y_reg = _frames()
    y_reg = y_reg.copy()
    y_reg.iloc[0] = 99.0            # volumen alto pero inunda == 0
    report = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)

    assert report["threshold_disagreements"] == 1


def test_duplicate_run_node_pairs_are_a_hard_error():
    keys, X, y_clf, y_reg = _frames()
    keys = keys.copy()
    keys.loc[1, "node_id"] = "N0"    # duplica (run_id=1, node_id=N0)

    with pytest.raises(DatasetQualityError, match="duplicad"):
        build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)


def test_zero_flooded_rows_is_a_hard_error():
    keys, X, y_clf, y_reg = _frames()
    y_clf = pd.Series([0] * len(y_clf), name="inunda")

    with pytest.raises(DatasetQualityError, match="ninguna fila inundada"):
        build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3=1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_quality.py -q`
Expected: FAIL con `ImportError: cannot import name 'DatasetQualityError'`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/quality.py
"""Informe de calidad del dataset preparado.

Restricción: sólo pandas/numpy. Ver tests/ml/bench/test_stage1_imports.py.

El informe no bloquea la ejecución salvo en dos casos que hacen imposible
entrenar: filas duplicadas por (run_id, node_id) y cero filas inundadas.
Todo lo demás se reporta para que quede registro, no para frenar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import FEATURE_COLUMNS_V17, NULLABLE_FEATURE_COLUMNS_V17


class DatasetQualityError(ValueError):
    """El dataset tiene un defecto que hace imposible entrenar sobre él."""


def _stats(series: pd.Series) -> dict:
    clean = series.dropna()
    if clean.empty:
        return {key: None for key in ("min", "p1", "p25", "p50", "p75", "p99", "max")}
    return {
        "min": float(clean.min()),
        "p1": float(np.percentile(clean, 1)),
        "p25": float(np.percentile(clean, 25)),
        "p50": float(np.percentile(clean, 50)),
        "p75": float(np.percentile(clean, 75)),
        "p99": float(np.percentile(clean, 99)),
        "max": float(clean.max()),
    }


def build_quality_report(
    keys: pd.DataFrame,
    X: pd.DataFrame,
    y_clf: pd.Series,
    y_reg: pd.Series,
    flood_threshold_m3: float,
) -> dict:
    """Resume la calidad del dataset y aborta ante los dos defectos fatales."""
    duplicated = keys.duplicated(subset=["run_id", "node_id"]).sum()
    if duplicated:
        raise DatasetQualityError(
            f"El dataset tiene {duplicated} fila(s) duplicadas por (run_id, node_id). "
            "Suele indicar que la base guarda más de un snapshot; repuebla la base "
            "desde cero antes de entrenar."
        )

    n_flooded = int((y_clf == 1).sum())
    if n_flooded == 0:
        raise DatasetQualityError(
            "El dataset no tiene ninguna fila inundada (inunda == 1); el regresor "
            "no se puede entrenar. Revisa el umbral de inundación o el rango de factores."
        )

    nulls = {column: int(X[column].isna().sum()) for column in FEATURE_COLUMNS_V17}
    unexpected = [
        column
        for column, count in nulls.items()
        if count and column not in NULLABLE_FEATURE_COLUMNS_V17
    ]
    constant = [
        column for column in FEATURE_COLUMNS_V17 if X[column].dropna().nunique() <= 1
    ]

    factor_labels = keys["factor_mult"].astype(str)
    flooded_by_factor = (
        y_clf.groupby(factor_labels.values).sum().astype(int).to_dict()
    )

    disagreements = int(
        (((y_reg > flood_threshold_m3).astype(int)) != y_clf.astype(int)).sum()
    )

    return {
        "n_rows": int(len(keys)),
        "n_runs": int(keys["run_id"].nunique()),
        "n_nodes": int(keys["node_id"].nunique()),
        "class_balance": {
            "n_flooded": n_flooded,
            "n_not_flooded": int((y_clf == 0).sum()),
            "flooded_ratio": float(n_flooded / len(y_clf)),
        },
        "rows_by_factor": factor_labels.value_counts().sort_index().to_dict(),
        "rows_by_shape": keys["shape_id"].value_counts().sort_index().to_dict(),
        "flooded_by_factor": {str(k): int(v) for k, v in sorted(flooded_by_factor.items())},
        "nulls_by_column": nulls,
        "nullable_columns": sorted(NULLABLE_FEATURE_COLUMNS_V17),
        "unexpected_null_columns": sorted(unexpected),
        "constant_columns": constant,
        "feature_stats": {column: _stats(X[column]) for column in FEATURE_COLUMNS_V17},
        "threshold_disagreements": disagreements,
        "flood_threshold_m3": float(flood_threshold_m3),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_quality.py tests/ml/bench/test_stage1_imports.py -q`
Expected: 9 passed + 4 passed

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/quality.py tests/ml/bench/test_quality.py
git commit -m "feat(bench): informe de calidad del dataset"
```

---

## Task 5: `preprocess.py` — `prepare_dataset` y `prep_id`

**Files:**
- Modify: `swmm_resilience/ml/bench/preprocess.py`
- Test: `tests/ml/bench/test_preprocess.py`

**Interfaces:**
- Consumes: `load_training_frame` de `swmm_resilience.database.training_queries`; `build_all_folds` de `folds.py`; `build_quality_report` de `quality.py`; `PreparedDataset`, `KEY_COLUMNS`, `PROTOCOLS` de `schemas.py`; `TABULAR_V3_17`, `FEATURE_COLUMNS_V17` de `swmm_resilience.ml.contracts`.
- Produces: `prepare_dataset(db_path, *, protocols=PROTOCOLS, flood_threshold_m3, output_dir) -> PreparedDataset`; `compute_prep_id(descriptor: dict) -> str`; `resolve_prepared(output_dir, prep_id=None) -> PreparedDataset` (usado por las etapas 2 y 3 y por el CLI).

**Nota:** la fixture `sql_training_db` de `tests/conftest.py` ya monta una base v17 migrada con 2 formas × 3 factores × 4 nodos. Sirve para LOSO (3 folds) pero **no** para GroupKFold5 (necesita ≥5 grupos), así que los tests de este task usan sólo `("LOSO",)` salvo donde se indique.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/test_preprocess.py
import json

import pandas as pd
import pytest

from swmm_resilience.ml.bench.preprocess import (
    compute_prep_id,
    prepare_dataset,
    resolve_prepared,
)
from swmm_resilience.ml.bench.schemas import KEY_COLUMNS
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17


@pytest.fixture
def prepared(sql_training_db, tmp_path):
    return prepare_dataset(
        sql_training_db,
        protocols=("LOSO",),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )


def test_features_follow_the_contract_whitelist_and_order(prepared):
    assert tuple(prepared.X.columns) == FEATURE_COLUMNS_V17
    assert tuple(prepared.keys.columns) == KEY_COLUMNS
    assert "coord_x" not in prepared.X.columns


def test_targets_come_from_sql_and_are_not_re_derived(sql_training_db, tmp_path):
    """inunda ya viene persistido; recalcularlo podria discrepar de la base."""
    prepared = prepare_dataset(
        sql_training_db,
        protocols=("LOSO",),
        flood_threshold_m3=999999.0,   # umbral absurdo
        output_dir=tmp_path / "prepared",
    )
    assert prepared.y_clf.sum() > 0, "inunda debe venir de SQL, no derivarse del umbral"
    assert prepared.quality["threshold_disagreements"] > 0


def test_rows_are_sorted_canonically_by_run_id_then_node_id(prepared):
    ordered = prepared.keys.sort_values(["run_id", "node_id"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(prepared.keys, ordered)


def test_prep_id_is_stable_across_two_identical_runs(sql_training_db, tmp_path):
    first = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "a",
    )
    second = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "b",
    )
    assert first.prep_id == second.prep_id


def test_prep_id_changes_when_the_threshold_changes(sql_training_db, tmp_path):
    first = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "a",
    )
    second = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=2.5,
        output_dir=tmp_path / "b",
    )
    assert first.prep_id != second.prep_id


def test_prep_id_changes_when_the_protocol_set_changes():
    base = {
        "contract_id": "tabular_v3_17",
        "feature_contract_sha256": "a" * 64,
        "run_ids": [1, 2, 3],
        "feature_columns": list(FEATURE_COLUMNS_V17),
        "flood_threshold_m3": 1.0,
        "sort_key": ["run_id", "node_id"],
        "protocols": ["LOSO"],
    }
    other = dict(base, protocols=["LOSO", "GroupKFold5"])
    assert compute_prep_id(base) != compute_prep_id(other)


def test_prep_id_is_sixteen_hex_characters(prepared):
    assert len(prepared.prep_id) == 16
    int(prepared.prep_id, 16)


def test_manifest_records_full_provenance(prepared):
    manifest = prepared.manifest
    assert manifest["prep_id"] == prepared.prep_id
    assert manifest["contract_id"] == "tabular_v3_17"
    assert len(manifest["feature_contract_sha256"]) == 64
    assert manifest["protocols"] == ["LOSO"]
    assert manifest["feature_columns"] == list(FEATURE_COLUMNS_V17)
    assert "python_version" in manifest
    assert "library_versions" in manifest
    assert "created_at_utc" in manifest


def test_artifacts_are_written_to_disk(prepared, tmp_path):
    directory = tmp_path / "prepared" / prepared.prep_id
    for filename in (
        "keys.parquet", "features.parquet", "targets.parquet",
        "folds.parquet", "manifest.json", "quality_report.json",
    ):
        assert (directory / filename).exists(), f"falta {filename}"
    assert json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def test_folds_cover_every_sample_exactly_once_per_protocol(prepared):
    tested = prepared.folds.loc[prepared.folds["split"] == "test", "sample_idx"]
    assert sorted(tested) == list(range(len(prepared.X)))


def test_resolve_prepared_loads_the_only_dataset_when_no_id_given(prepared, tmp_path):
    loaded = resolve_prepared(tmp_path / "prepared")
    assert loaded.prep_id == prepared.prep_id


def test_resolve_prepared_errors_clearly_when_none_exists(tmp_path):
    with pytest.raises(FileNotFoundError, match="--bench-prepare"):
        resolve_prepared(tmp_path / "vacio")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_preprocess.py -q`
Expected: FAIL con `NotImplementedError` / `ImportError: cannot import name 'compute_prep_id'`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/preprocess.py
"""Etapa 1 del banco: SQL -> PreparedDataset.

NO-OBJETIVOS de esta etapa, y son deliberados: no imputa, no escala, no
aplica PCA, no elimina filas con nulos permitidos por el contrato, y no
transforma el target. Todo eso pertenece al Pipeline de cada familia, que se
ajusta dentro de cada fold. Si algo de eso aparece en este archivo, es un bug
— y tests/ml/bench/test_stage1_imports.py lo detecta.

Tampoco re-deriva ``inunda``: ese target ya viene persistido en la vista
training_samples_v17, y recalcularlo contra el umbral podría discrepar de lo
que la base afirma. La coherencia se verifica y se reporta (quality.py), no
se corrige en silencio.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import sklearn

from ...database.training_queries import load_training_frame
from ..contracts import FEATURE_COLUMNS_V17, TABULAR_V3_17
from .folds import build_all_folds
from .quality import build_quality_report
from .schemas import KEY_COLUMNS, PROTOCOLS, PreparedDataset

SORT_KEY = ("run_id", "node_id")


def _library_versions() -> dict:
    return {
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit-learn": sklearn.__version__,
    }


def compute_prep_id(descriptor: dict) -> str:
    """Hash reproducible de la definición del dataset.

    Se calcula sólo sobre la *definición* (qué runs, qué columnas, qué umbral,
    qué protocolos), no sobre los valores, para que dos corridas sobre los
    mismos datos den el mismo id.
    """
    payload = json.dumps(descriptor, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def prepare_dataset(
    db_path: Path | str,
    *,
    protocols: Sequence[str] = PROTOCOLS,
    flood_threshold_m3: float,
    output_dir: Path | str,
) -> PreparedDataset:
    """Lee el frame de SQL y materializa un PreparedDataset en ``output_dir``."""
    frame = load_training_frame(db_path)
    frame = frame.sort_values(list(SORT_KEY)).reset_index(drop=True)

    keys = frame.loc[:, list(KEY_COLUMNS)].copy()
    X = TABULAR_V3_17.validate_frame(frame.loc[:, list(FEATURE_COLUMNS_V17)])
    y_clf = frame["inunda"].astype(int).rename("inunda")
    y_reg = frame["vol_inundacion_m3"].astype(float).rename("vol_inundacion_m3")

    quality = build_quality_report(keys, X, y_clf, y_reg, flood_threshold_m3)
    folds = build_all_folds(keys["factor_mult"], tuple(protocols))

    descriptor = {
        "contract_id": TABULAR_V3_17.contract_id,
        "feature_contract_sha256": TABULAR_V3_17.descriptor_sha256,
        "run_ids": sorted(int(value) for value in keys["run_id"].unique()),
        "feature_columns": list(FEATURE_COLUMNS_V17),
        "flood_threshold_m3": float(flood_threshold_m3),
        "sort_key": list(SORT_KEY),
        "protocols": list(protocols),
    }
    prep_id = compute_prep_id(descriptor)

    manifest = dict(
        descriptor,
        prep_id=prep_id,
        db_path=str(db_path),
        n_rows=int(len(frame)),
        python_version=platform.python_version(),
        library_versions=_library_versions(),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )

    prepared = PreparedDataset(
        prep_id=prep_id,
        keys=keys,
        X=X,
        y_clf=y_clf,
        y_reg=y_reg,
        folds=folds,
        manifest=manifest,
        quality=quality,
    )
    prepared.save(output_dir)
    return prepared


def resolve_prepared(
    output_dir: Path | str, prep_id: str | None = None
) -> PreparedDataset:
    """Carga un PreparedDataset por id, o el más reciente si no se da uno."""
    root = Path(output_dir)
    if prep_id is not None:
        directory = root / prep_id
        if not directory.exists():
            raise FileNotFoundError(
                f"No existe el PreparedDataset {prep_id} en {root}. "
                "Corre `python main.py --bench-prepare` primero."
            )
        return PreparedDataset.load(directory)

    candidates = [path for path in root.glob("*") if (path / "manifest.json").exists()] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(
            f"No hay ningún PreparedDataset en {root}. "
            "Corre `python main.py --bench-prepare` primero."
        )
    newest = max(candidates, key=lambda path: (path / "manifest.json").stat().st_mtime)
    return PreparedDataset.load(newest)
```

> **Importante:** `preprocess.py` importa `sklearn` sólo para leer `sklearn.__version__`. Eso es un `import sklearn` a secas, cuyo prefijo **no** empieza por `sklearn.model_selection` y por tanto **hará fallar** `test_stage1_imports.py`. Resuélvelo leyendo la versión sin importar el paquete raíz:
>
> ```python
> from importlib.metadata import version as _package_version
>
> def _library_versions() -> dict:
>     return {
>         "pandas": pd.__version__,
>         "numpy": np.__version__,
>         "scikit-learn": _package_version("scikit-learn"),
>     }
> ```
>
> y elimina `import sklearn`. Es la solución correcta, no un rodeo: la etapa 1 no debe tener sklearn cargado en absoluto salvo el splitter de `folds.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/ -q`
Expected: todos verdes, incluido `test_stage1_imports.py`

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/preprocess.py tests/ml/bench/test_preprocess.py
git commit -m "feat(bench): etapa 1 prepare_dataset con prep_id reproducible"
```

---

## Task 6: `registry.py` y la familia `xgboost`

**Files:**
- Create: `swmm_resilience/ml/bench/registry.py`
- Create: `swmm_resilience/ml/bench/models/__init__.py`
- Create: `swmm_resilience/ml/bench/models/xgboost_family.py`
- Test: `tests/ml/bench/test_registry.py`

**Interfaces:**
- Consumes: `ML_RANDOM_STATE` de `swmm_resilience.config`.
- Produces: `get_family(name) -> ModelFamily`; `available_families() -> tuple[str, ...]`; el protocolo `ModelFamily` con atributos `FAMILY: str`, `SCALE_FEATURES: bool` y funciones `build_classifier(params, scale_pos_weight) -> Pipeline`, `build_regressor(params) -> Pipeline`, `preprocessing_descriptor(params) -> dict`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/test_registry.py
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from swmm_resilience.ml.bench.registry import available_families, get_family


def test_xgboost_family_is_registered():
    assert "xgboost" in available_families()


def test_unknown_family_lists_the_available_ones():
    with pytest.raises(ValueError, match="Familia desconocida"):
        get_family("perceptron_de_1958")


def test_xgboost_classifier_is_imputer_plus_model_without_scaler():
    family = get_family("xgboost")
    pipeline = family.build_classifier(
        {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1, "subsample": 1.0},
        scale_pos_weight=2.0,
    )

    assert isinstance(pipeline, Pipeline)
    steps = dict(pipeline.named_steps)
    assert isinstance(steps["imputer"], SimpleImputer)
    assert steps["imputer"].strategy == "median"
    assert isinstance(steps["model"], XGBClassifier)
    assert not any(isinstance(step, StandardScaler) for step in steps.values()), (
        "XGBoost debe ir sin escalador: es lo que permite la paridad con trainer.py"
    )


def test_xgboost_regressor_is_imputer_plus_model():
    family = get_family("xgboost")
    pipeline = family.build_regressor(
        {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.1, "subsample": 1.0}
    )
    assert isinstance(pipeline.named_steps["model"], XGBRegressor)


def test_scale_pos_weight_reaches_the_classifier():
    family = get_family("xgboost")
    pipeline = family.build_classifier({"n_estimators": 5}, scale_pos_weight=3.5)
    assert pipeline.named_steps["model"].scale_pos_weight == pytest.approx(3.5)


def test_family_declares_it_does_not_scale():
    assert get_family("xgboost").SCALE_FEATURES is False


def test_preprocessing_descriptor_is_json_serialisable_and_names_the_steps():
    import json

    descriptor = get_family("xgboost").preprocessing_descriptor({"n_estimators": 5})
    assert json.dumps(descriptor)
    assert descriptor["imputer"] == "median"
    assert descriptor["scaler"] is None


def test_random_state_is_fixed_for_reproducibility():
    from swmm_resilience.config import ML_RANDOM_STATE

    pipeline = get_family("xgboost").build_regressor({"n_estimators": 5})
    assert pipeline.named_steps["model"].random_state == ML_RANDOM_STATE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_registry.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'swmm_resilience.ml.bench.registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/models/__init__.py
"""Familias de modelos del banco.

Cada módulo de este paquete expone la misma interfaz (ver registry.py) y no
lee SQL ni configuración: recibe un dict de hiperparámetros ya resuelto.
Añadir una familia nueva es añadir un archivo aquí y registrarlo en
registry.py — no hay que tocar ninguna de las tres etapas.
"""
```

```python
# swmm_resilience/ml/bench/models/xgboost_family.py
"""Familia XGBoost.

Va SIN escalador a propósito: los árboles no lo necesitan, y es la condición
para que este candidato reproduzca exactamente las métricas de ml/trainer.py
en el test de paridad.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from ....config import ML_RANDOM_STATE

FAMILY = "xgboost"
SCALE_FEATURES = False

_CLASSIFIER_DEFAULTS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
}
_REGRESSOR_DEFAULTS = dict(_CLASSIFIER_DEFAULTS)


def build_classifier(params: dict, scale_pos_weight: float) -> Pipeline:
    settings = {**_CLASSIFIER_DEFAULTS, **params}
    settings.pop("scale_pos_weight", None)
    model = XGBClassifier(
        **settings,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=ML_RANDOM_STATE,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def build_regressor(params: dict) -> Pipeline:
    settings = {**_REGRESSOR_DEFAULTS, **params}
    model = XGBRegressor(**settings, random_state=ML_RANDOM_STATE)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def preprocessing_descriptor(params: dict) -> dict:
    return {"imputer": "median", "scaler": None, "pca": None}
```

```python
# swmm_resilience/ml/bench/registry.py
"""Registro de familias de modelos.

Una familia es un módulo que expone FAMILY, SCALE_FEATURES,
build_classifier(params, scale_pos_weight), build_regressor(params) y
preprocessing_descriptor(params). El banco sólo habla con esta interfaz, así
que añadir un modelo nuevo no toca ninguna de las tres etapas.
"""

from __future__ import annotations

from types import ModuleType

from .models import xgboost_family

_FAMILIES: dict[str, ModuleType] = {
    xgboost_family.FAMILY: xgboost_family,
}


def available_families() -> tuple[str, ...]:
    return tuple(sorted(_FAMILIES))


def get_family(name: str) -> ModuleType:
    try:
        return _FAMILIES[name]
    except KeyError:
        raise ValueError(
            f"Familia desconocida: {name!r}. Disponibles: {', '.join(available_families())}"
        ) from None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_registry.py -q`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/registry.py swmm_resilience/ml/bench/models/ tests/ml/bench/test_registry.py
git commit -m "feat(bench): registro de familias y familia xgboost"
```

---

## Task 7: `metrics.py` — las tres capas de métricas

**Files:**
- Create: `swmm_resilience/ml/bench/metrics.py`
- Test: `tests/ml/bench/test_metrics.py`

**Interfaces:**
- Consumes: nada del banco.
- Produces: `nse(y_true, y_pred) -> float`; `classifier_metrics(y_true, y_pred, y_prob) -> dict`; `regressor_oracle_metrics(y_true, y_pred) -> dict`; `end_to_end_metrics(y_true_vol, y_pred_vol, y_true_clf, y_pred_clf) -> dict`; `mean_metrics(list[dict]) -> dict`; `pooled_regressor_metrics(list, list) -> dict`.

**Nota:** estas funciones replican literalmente las de `ml/evaluator.py`. La asimetría (clasificador promediado, regresor sobre pool) es intencional y es lo que el test de paridad de la Task 10 va a comprobar.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/test_metrics.py
import numpy as np
import pytest

from swmm_resilience.ml.bench.metrics import (
    classifier_metrics,
    end_to_end_metrics,
    mean_metrics,
    nse,
    pooled_regressor_metrics,
    regressor_oracle_metrics,
)


def test_nse_is_one_for_a_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert nse(y, y) == pytest.approx(1.0)


def test_nse_is_zero_when_the_prediction_equals_the_mean():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert nse(y, np.full_like(y, y.mean())) == pytest.approx(0.0)


def test_nse_of_a_constant_truth_is_one_only_when_exact():
    y = np.array([5.0, 5.0, 5.0])
    assert nse(y, y) == pytest.approx(1.0)
    assert nse(y, np.array([5.0, 5.0, 6.0])) == pytest.approx(0.0)


def test_classifier_metrics_on_a_known_confusion_matrix():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.2, 0.1])

    result = classifier_metrics(y_true, y_pred, y_prob)

    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(2 / 3)
    assert result["auc_roc"] == pytest.approx(1.0)


def test_auc_is_nan_when_the_fold_has_a_single_class():
    y_true = np.array([1, 1, 1])
    result = classifier_metrics(y_true, np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))
    assert np.isnan(result["auc_roc"])


def test_regressor_oracle_metrics_on_a_perfect_prediction():
    y = np.array([10.0, 20.0, 30.0])
    result = regressor_oracle_metrics(y, y)

    assert result["nse"] == pytest.approx(1.0)
    assert result["log_nse"] == pytest.approx(1.0)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["mae"] == pytest.approx(0.0)
    assert result["r2"] == pytest.approx(1.0)


def test_end_to_end_metrics_report_totals_and_node_accuracy():
    y_true_vol = np.array([0.0, 100.0, 0.0, 50.0])
    y_pred_vol = np.array([0.0, 90.0, 0.0, 0.0])
    y_true_clf = np.array([0, 1, 0, 1])
    y_pred_clf = np.array([0, 1, 0, 0])

    result = end_to_end_metrics(y_true_vol, y_pred_vol, y_true_clf, y_pred_clf)

    assert result["pct_nodos_correctos"] == pytest.approx(0.75)
    assert result["vol_total_pred_m3"] == pytest.approx(90.0)
    assert result["vol_total_real_m3"] == pytest.approx(150.0)
    assert result["rmse_vol_todos_nodos"] == pytest.approx(np.sqrt((10**2 + 50**2) / 4))


def test_mean_metrics_averages_each_key_ignoring_nan():
    folds = [
        {"f1": 0.8, "auc_roc": float("nan")},
        {"f1": 0.6, "auc_roc": 0.9},
    ]
    result = mean_metrics(folds)
    assert result["f1"] == pytest.approx(0.7)
    assert result["auc_roc"] == pytest.approx(0.9)


def test_mean_metrics_of_an_empty_list_is_empty():
    assert mean_metrics([]) == {}


def test_pooled_regressor_metrics_concatenate_before_computing():
    """Es la asimetria heredada: el regresor NO se promedia entre folds."""
    trues = [np.array([10.0, 20.0]), np.array([30.0])]
    preds = [np.array([10.0, 20.0]), np.array([30.0])]
    result = pooled_regressor_metrics(trues, preds)
    assert result["rmse"] == pytest.approx(0.0)
    assert result["nse"] == pytest.approx(1.0)


def test_pooled_regressor_metrics_of_empty_parts_is_empty():
    assert pooled_regressor_metrics([], []) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_metrics.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'swmm_resilience.ml.bench.metrics'`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/metrics.py
"""Las tres capas de métricas del banco.

Replican literalmente las de ml/evaluator.py, incluida su asimetría de
agregación: las métricas del clasificador y las de end-to-end se PROMEDIAN
entre folds, mientras que las del regresor-oracle se calculan sobre el POOL
concatenado de todos los folds. Esa diferencia cambia los números y hoy es
invisible en el código original; aquí queda documentada y es lo que el test
de paridad comprueba.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def nse(y_true, y_pred) -> float:
    """Nash-Sutcliffe Efficiency."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def classifier_metrics(y_true, y_pred, y_prob) -> dict:
    """Nivel 1. ``auc_roc`` es NaN si el fold no tiene ambas clases."""
    y_true = np.asarray(y_true)
    has_both = y_true.sum() > 0 and (1 - y_true).sum() > 0
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
    }


def regressor_oracle_metrics(y_true, y_pred) -> dict:
    """Nivel 2. Filtrado con etiquetas REALES: es una cota superior optimista."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "nse": nse(y_true, y_pred),
        "log_nse": nse(np.log1p(y_true), np.log1p(y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def end_to_end_metrics(y_true_vol, y_pred_vol, y_true_clf, y_pred_clf) -> dict:
    """Nivel 3. Enrutado con etiquetas PREDICHAS: el error real del sistema."""
    y_true_vol = np.asarray(y_true_vol, dtype=float)
    y_pred_vol = np.asarray(y_pred_vol, dtype=float)
    return {
        "pct_nodos_correctos": float((np.asarray(y_pred_clf) == np.asarray(y_true_clf)).mean()),
        "rmse_vol_todos_nodos": float(np.sqrt(mean_squared_error(y_true_vol, y_pred_vol))),
        "vol_total_pred_m3": float(y_pred_vol.sum()),
        "vol_total_real_m3": float(y_true_vol.sum()),
    }


def _average(entries: list, key: str) -> float:
    values = [
        entry[key] for entry in entries if not np.isnan(entry.get(key, float("nan")))
    ]
    return float(np.mean(values)) if values else float("nan")


def mean_metrics(entries: list) -> dict:
    """Promedio por clave entre folds, ignorando NaN."""
    if not entries:
        return {}
    return {key: _average(entries, key) for key in entries[0]}


def pooled_regressor_metrics(true_parts: list, pred_parts: list) -> dict:
    """Métricas del regresor sobre el pool concatenado de todos los folds."""
    if not true_parts:
        return {}
    return regressor_oracle_metrics(
        np.concatenate(true_parts), np.concatenate(pred_parts)
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_metrics.py -q`
Expected: 11 passed

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/metrics.py tests/ml/bench/test_metrics.py
git commit -m "feat(bench): metricas de los tres niveles"
```

---

## Task 8: `evaluate.py::score_predictions` — la capa núcleo

**Files:**
- Create: `swmm_resilience/ml/bench/evaluate.py`
- Test: `tests/ml/bench/test_score_predictions.py`

**Interfaces:**
- Consumes: `PreparedDataset`, `OOF_COLUMNS` de `schemas.py`; todo `metrics.py`.
- Produces: `score_predictions(prepared, oof, provenance) -> dict` (devuelve `{protocolo: {classifier, regressor_oracle, end_to_end, by_factor}}`); `ProvenanceMismatchError`.

**Por qué esta capa existe:** es la puerta abierta de la spec §13.1. No sabe de modelos: recibe predicciones ya calculadas y las puntúa. Un LSTM futuro, o un `objective` de Optuna, entra por aquí sin tocar el banco.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/test_score_predictions.py
import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.bench.evaluate import ProvenanceMismatchError, score_predictions
from swmm_resilience.ml.bench.schemas import (
    FOLD_COLUMNS,
    KEY_COLUMNS,
    OOF_COLUMNS,
    PreparedDataset,
)
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17


@pytest.fixture
def prepared() -> PreparedDataset:
    """6 muestras, 2 factores, LOSO -> 2 folds."""
    n = 6
    keys = pd.DataFrame(
        {
            "run_id": [1, 1, 1, 2, 2, 2],
            "network_id": [1] * n,
            "scenario_id": [1, 1, 1, 2, 2, 2],
            "scenario_key": ["base@1.0"] * 3 + ["base@2.0"] * 3,
            "scenario_kind": ["base"] * n,
            "node_id": ["N0", "N1", "N2"] * 2,
            "factor_mult": [1.0] * 3 + [2.0] * 3,
            "shape_id": ["base"] * n,
        },
        columns=list(KEY_COLUMNS),
    )
    X = pd.DataFrame(
        {col: [float(i) for i in range(n)] for col in FEATURE_COLUMNS_V17}
    )
    folds = pd.DataFrame(
        [
            {"protocol": "LOSO", "fold_id": 0, "sample_idx": i, "split": "train" if i >= 3 else "test"}
            for i in range(n)
        ]
        + [
            {"protocol": "LOSO", "fold_id": 1, "sample_idx": i, "split": "train" if i < 3 else "test"}
            for i in range(n)
        ],
        columns=list(FOLD_COLUMNS),
    )
    return PreparedDataset(
        prep_id="0123456789abcdef",
        keys=keys,
        X=X,
        y_clf=pd.Series([0, 1, 1, 0, 1, 1], name="inunda"),
        y_reg=pd.Series([0.0, 10.0, 20.0, 0.0, 30.0, 40.0], name="vol_inundacion_m3"),
        folds=folds,
        manifest={"prep_id": "0123456789abcdef"},
        quality={},
    )


def _perfect_oof(prepared: PreparedDataset) -> pd.DataFrame:
    tests = prepared.folds[prepared.folds["split"] == "test"]
    return pd.DataFrame(
        {
            "sample_idx": tests["sample_idx"].values,
            "protocol": tests["protocol"].values,
            "fold_id": tests["fold_id"].values,
            "y_pred_clf": prepared.y_clf.iloc[tests["sample_idx"]].values,
            "y_prob_clf": prepared.y_clf.iloc[tests["sample_idx"]].values.astype(float),
            "y_pred_reg": prepared.y_reg.iloc[tests["sample_idx"]].values,
        },
        columns=list(OOF_COLUMNS),
    )


def test_perfect_predictions_score_perfectly(prepared):
    result = score_predictions(
        prepared, _perfect_oof(prepared), {"prep_id": prepared.prep_id}
    )

    assert result["LOSO"]["classifier"]["f1"] == pytest.approx(1.0)
    assert result["LOSO"]["regressor_oracle"]["nse"] == pytest.approx(1.0)
    assert result["LOSO"]["end_to_end"]["rmse_vol_todos_nodos"] == pytest.approx(0.0)


def test_result_has_the_three_levels_and_the_factor_breakdown(prepared):
    result = score_predictions(
        prepared, _perfect_oof(prepared), {"prep_id": prepared.prep_id}
    )
    assert set(result["LOSO"]) == {
        "classifier", "regressor_oracle", "end_to_end", "by_factor"
    }
    assert set(result["LOSO"]["by_factor"]) == {"1.00", "2.00"}


def test_provenance_mismatch_is_rejected(prepared):
    """La garantia de 'mismas entradas': no se puede comparar contra otro dataset."""
    with pytest.raises(ProvenanceMismatchError, match="prep_id"):
        score_predictions(
            prepared, _perfect_oof(prepared), {"prep_id": "deadbeefdeadbeef"}
        )


def test_missing_prep_id_in_provenance_is_rejected(prepared):
    with pytest.raises(ProvenanceMismatchError, match="prep_id"):
        score_predictions(prepared, _perfect_oof(prepared), {})


def test_incomplete_fold_coverage_is_rejected(prepared):
    oof = _perfect_oof(prepared).iloc[:-1]
    with pytest.raises(ValueError, match="cobertura"):
        score_predictions(prepared, oof, {"prep_id": prepared.prep_id})


def test_duplicate_predictions_for_a_sample_are_rejected(prepared):
    oof = _perfect_oof(prepared)
    oof = pd.concat([oof, oof.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicad"):
        score_predictions(prepared, oof, {"prep_id": prepared.prep_id})


def test_missing_oof_column_is_rejected(prepared):
    oof = _perfect_oof(prepared).drop(columns=["y_prob_clf"])
    with pytest.raises(ValueError, match="y_prob_clf"):
        score_predictions(prepared, oof, {"prep_id": prepared.prep_id})


def test_a_model_that_predicts_no_flooding_scores_zero_recall(prepared):
    oof = _perfect_oof(prepared)
    oof["y_pred_clf"] = 0
    oof["y_prob_clf"] = 0.0
    oof["y_pred_reg"] = 0.0

    result = score_predictions(prepared, oof, {"prep_id": prepared.prep_id})

    assert result["LOSO"]["classifier"]["recall"] == pytest.approx(0.0)
    assert result["LOSO"]["end_to_end"]["vol_total_pred_m3"] == pytest.approx(0.0)


def test_regressor_oracle_uses_true_labels_not_predicted(prepared):
    """Nivel 2 filtra con la verdad; un clasificador malo no debe afectarlo."""
    oof = _perfect_oof(prepared)
    oof["y_pred_clf"] = 0        # el clasificador falla del todo
    result = score_predictions(prepared, oof, {"prep_id": prepared.prep_id})

    assert result["LOSO"]["regressor_oracle"]["nse"] == pytest.approx(1.0)
    assert result["LOSO"]["classifier"]["recall"] == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_score_predictions.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'swmm_resilience.ml.bench.evaluate'`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/evaluate.py
"""Etapa 3 del banco, en dos capas.

score_predictions() es la capa núcleo: NO sabe de modelos. Recibe
predicciones out-of-fold ya calculadas, verifica su proveniencia contra el
PreparedDataset, y devuelve las métricas de los tres niveles. Cualquier
modelo —tabular, temporal, escrito en otro script o dentro de seis meses—
entra por aquí y su número cae en la misma tabla. Es el mecanismo que
garantiza "mismas entradas, mismos benchmarks" (spec §13.1).

evaluate_candidate() es la capa de conveniencia que produce esas
predicciones para las familias tabulares del banco (Task 9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import (
    classifier_metrics,
    end_to_end_metrics,
    mean_metrics,
    pooled_regressor_metrics,
)
from .schemas import OOF_COLUMNS, PreparedDataset


class ProvenanceMismatchError(ValueError):
    """Las predicciones no corresponden a este PreparedDataset."""


def _validate_oof(prepared: PreparedDataset, oof: pd.DataFrame) -> None:
    missing = [column for column in OOF_COLUMNS if column not in oof.columns]
    if missing:
        raise ValueError(f"Al frame OOF le faltan columnas: {missing}")

    duplicated = oof.duplicated(subset=["protocol", "fold_id", "sample_idx"]).sum()
    if duplicated:
        raise ValueError(
            f"El frame OOF tiene {duplicated} predicción(es) duplicadas para la misma "
            "(protocol, fold_id, sample_idx)."
        )

    expected = prepared.folds[prepared.folds["split"] == "test"]
    expected_keys = set(
        map(tuple, expected[["protocol", "fold_id", "sample_idx"]].itertuples(index=False))
    )
    got_keys = set(
        map(tuple, oof[["protocol", "fold_id", "sample_idx"]].itertuples(index=False))
    )
    if expected_keys != got_keys:
        raise ValueError(
            "La cobertura del frame OOF no coincide con las filas de test de los folds: "
            f"faltan {len(expected_keys - got_keys)}, sobran {len(got_keys - expected_keys)}."
        )


def score_predictions(
    prepared: PreparedDataset,
    oof: pd.DataFrame,
    provenance: dict,
) -> dict:
    """Puntúa predicciones out-of-fold contra ``prepared``.

    ``provenance`` debe traer el ``prep_id`` del dataset con el que se
    generaron. Sin esa verificación se podrían comparar números de datasets
    distintos sin notarlo, que es el modo de fallo silencioso que este
    diseño existe para cerrar.
    """
    if provenance.get("prep_id") != prepared.prep_id:
        raise ProvenanceMismatchError(
            f"prep_id de las predicciones ({provenance.get('prep_id')!r}) no coincide "
            f"con el del dataset ({prepared.prep_id!r}). No son comparables."
        )
    _validate_oof(prepared, oof)

    y_clf = prepared.y_clf.to_numpy()
    y_reg = prepared.y_reg.to_numpy()
    factors = prepared.keys["factor_mult"].to_numpy()

    results: dict = {}
    for protocol, protocol_oof in oof.groupby("protocol"):
        classifier_folds, e2e_folds = [], []
        oracle_true, oracle_pred = [], []
        by_factor: dict[str, list] = {}

        for _, fold_oof in protocol_oof.groupby("fold_id"):
            idx = fold_oof["sample_idx"].to_numpy()
            yc_true, yr_true = y_clf[idx], y_reg[idx]
            yc_pred = fold_oof["y_pred_clf"].to_numpy()
            yc_prob = fold_oof["y_prob_clf"].to_numpy()
            yr_pred = fold_oof["y_pred_reg"].to_numpy()

            classifier_folds.append(classifier_metrics(yc_true, yc_pred, yc_prob))

            flooded = yc_true == 1
            if flooded.sum():
                oracle_true.append(yr_true[flooded])
                oracle_pred.append(yr_pred[flooded])

            routed = np.where(yc_pred == 1, yr_pred, 0.0)
            e2e_folds.append(end_to_end_metrics(yr_true, routed, yc_true, yc_pred))

            for factor in np.unique(factors[idx]):
                mask = factors[idx] == factor
                key = f"{factor:.2f}"
                by_factor.setdefault(key, []).append(
                    {
                        "f1": classifier_metrics(
                            yc_true[mask], yc_pred[mask], yc_prob[mask]
                        )["f1"],
                        "rmse_vol": end_to_end_metrics(
                            yr_true[mask], routed[mask], yc_true[mask], yc_pred[mask]
                        )["rmse_vol_todos_nodos"],
                    }
                )

        results[protocol] = {
            "classifier": mean_metrics(classifier_folds),
            "regressor_oracle": pooled_regressor_metrics(oracle_true, oracle_pred),
            "end_to_end": mean_metrics(e2e_folds),
            "by_factor": {key: mean_metrics(value) for key, value in by_factor.items()},
        }
    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_score_predictions.py -q`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/evaluate.py tests/ml/bench/test_score_predictions.py
git commit -m "feat(bench): score_predictions con verificacion de proveniencia"
```

---

## Task 9: `evaluate_candidate` — la capa de conveniencia

**Files:**
- Modify: `swmm_resilience/ml/bench/evaluate.py`
- Test: `tests/ml/bench/test_evaluate_candidate.py`

**Interfaces:**
- Consumes: `get_family` de `registry.py`; `score_predictions` de este mismo módulo.
- Produces: `evaluate_candidate(prepared, family_name, classifier_params, regressor_params) -> tuple[pd.DataFrame, dict]` (el frame OOF con columnas `OOF_COLUMNS`, y las métricas de `score_predictions`).

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/test_evaluate_candidate.py
import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.bench.evaluate import evaluate_candidate, score_predictions
from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.schemas import OOF_COLUMNS

TINY = {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3, "subsample": 1.0}


@pytest.fixture
def prepared(sql_training_db, tmp_path):
    return prepare_dataset(
        sql_training_db,
        protocols=("LOSO",),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )


def test_oof_frame_has_the_declared_schema(prepared):
    oof, _ = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    assert tuple(oof.columns) == OOF_COLUMNS


def test_oof_covers_every_test_row_exactly_once(prepared):
    oof, _ = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    expected = prepared.folds[prepared.folds["split"] == "test"]
    assert len(oof) == len(expected)
    assert not oof.duplicated(subset=["protocol", "fold_id", "sample_idx"]).any()


def test_metrics_match_scoring_the_returned_oof(prepared):
    """La capa de conveniencia no debe calcular metricas por su cuenta."""
    oof, metrics = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    recomputed = score_predictions(prepared, oof, {"prep_id": prepared.prep_id})
    assert metrics == recomputed


def test_predicted_volumes_are_never_negative(prepared):
    oof, _ = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    assert (oof["y_pred_reg"] >= 0).all()


def test_probabilities_are_in_the_unit_interval(prepared):
    oof, _ = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    assert oof["y_prob_clf"].between(0.0, 1.0).all()


def test_running_it_twice_gives_identical_predictions(prepared):
    first, _ = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    second, _ = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    pd.testing.assert_frame_equal(first, second)


def test_the_scaler_is_fitted_inside_the_fold_not_on_the_whole_dataset(prepared, monkeypatch):
    """El nucleo de la garantia anti-fuga, verificado en ejecucion."""
    from sklearn.preprocessing import StandardScaler

    seen_row_counts = []
    original_fit = StandardScaler.fit

    def spy_fit(self, X, y=None, **kwargs):
        seen_row_counts.append(len(X))
        return original_fit(self, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", spy_fit)
    evaluate_candidate(prepared, "linear", {"alpha": 1.0}, {"alpha": 1.0})

    total_rows = len(prepared.X)
    assert seen_row_counts, "la familia linear deberia ajustar un StandardScaler"
    assert all(count < total_rows for count in seen_row_counts), (
        f"un scaler se ajusto con {max(seen_row_counts)} filas de {total_rows}: "
        "eso es el dataset completo, hay fuga de datos"
    )


def test_a_fold_without_flooded_training_rows_still_produces_predictions(prepared):
    """No debe reventar: predice volumen cero donde no pudo entrenar el regresor."""
    oof, metrics = evaluate_candidate(prepared, "xgboost", TINY, TINY)
    assert len(oof) > 0
    assert "LOSO" in metrics
```

> **Nota para el implementador:** `test_the_scaler_is_fitted_inside_the_fold...` usa la familia `linear`, que se crea en la Task 12. Si ejecutas este task antes, marca ese test con `@pytest.mark.xfail(reason="familia linear llega en la Task 12", strict=False)` y quita la marca al terminar la Task 12. No lo borres: es el test que demuestra la garantía anti-fuga en ejecución, no sólo por estructura.

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_evaluate_candidate.py -q`
Expected: FAIL con `ImportError: cannot import name 'evaluate_candidate'`

- [ ] **Step 3: Write minimal implementation**

Añadir a `swmm_resilience/ml/bench/evaluate.py`:

```python
def evaluate_candidate(
    prepared: PreparedDataset,
    family_name: str,
    classifier_params: dict,
    regressor_params: dict,
) -> tuple[pd.DataFrame, dict]:
    """Entrena y predice fold a fold, y puntúa el resultado.

    Es una función pura de los hiperparámetros: exactamente la forma que
    necesita un ``objective`` de Optuna (spec §13.2).

    El Pipeline de la familia lleva dentro la imputación y el escalado, así
    que ``fit`` sobre el train del fold los ajusta SOLO con esas filas. Ahí
    está la garantía anti-fuga en ejecución.
    """
    from .registry import get_family

    family = get_family(family_name)
    X = prepared.X
    y_clf = prepared.y_clf.to_numpy()
    y_reg = prepared.y_reg.to_numpy()

    records: list[pd.DataFrame] = []
    for (protocol, fold_id), chunk in prepared.folds.groupby(["protocol", "fold_id"]):
        train_idx = chunk.loc[chunk["split"] == "train", "sample_idx"].to_numpy()
        test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"].to_numpy()

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        yc_train = y_clf[train_idx]
        yr_train = y_reg[train_idx]

        n_negative, n_positive = int((yc_train == 0).sum()), int((yc_train == 1).sum())
        scale_pos_weight = n_negative / n_positive if n_positive else 1.0

        classifier = family.build_classifier(classifier_params, scale_pos_weight)
        classifier.fit(X_train, yc_train)
        yc_pred = classifier.predict(X_test)
        yc_prob = classifier.predict_proba(X_test)[:, 1]

        flooded_train = yc_train == 1
        if flooded_train.sum():
            regressor = family.build_regressor(regressor_params)
            regressor.fit(X_train.iloc[flooded_train], np.log1p(yr_train[flooded_train]))
            yr_pred = np.clip(np.expm1(regressor.predict(X_test)), a_min=0.0, a_max=None)
        else:
            yr_pred = np.zeros(len(test_idx), dtype=float)

        records.append(
            pd.DataFrame(
                {
                    "sample_idx": test_idx,
                    "protocol": protocol,
                    "fold_id": fold_id,
                    "y_pred_clf": yc_pred.astype(int),
                    "y_prob_clf": yc_prob.astype(float),
                    "y_pred_reg": yr_pred.astype(float),
                },
                columns=list(OOF_COLUMNS),
            )
        )

    oof = pd.concat(records, ignore_index=True)
    metrics = score_predictions(prepared, oof, {"prep_id": prepared.prep_id})
    return oof, metrics
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_evaluate_candidate.py -q`
Expected: 7 passed, 1 xfailed (el de `linear`, hasta la Task 12)

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/evaluate.py tests/ml/bench/test_evaluate_candidate.py
git commit -m "feat(bench): evaluate_candidate con ajuste dentro del fold"
```

---

## Task 10: Test de paridad contra `evaluator.py` — la puerta

**Files:**
- Test: `tests/ml/bench/test_parity_with_legacy_evaluator.py`

**Interfaces:**
- Consumes: `evaluate_candidate` de `evaluate.py`; `evaluate_models` de `swmm_resilience.ml.evaluator`; `prepare_dataset`.
- Produces: nada de código. **Produce la evidencia que autoriza el Plan 3 a borrar `ml/trainer.py` y `ml/evaluator.py`.**

**Este test es desechable por diseño.** Se elimina junto al stack B, en el mismo commit del Plan 3. Su trabajo es justificar el borrado, no vivir para siempre.

**Si falla:** no ajustes la tolerancia. Hay una diferencia real que no identificamos, y el Plan 3 queda bloqueado hasta entenderla. Las sospechas más probables, en orden: (a) `evaluator._run_cv` usa `df[FEATURE_COLS].values` (numpy crudo) mientras el banco usa `TABULAR_V3_17.validate_frame` (que hace `astype(float)`); (b) el orden de filas difiere porque el banco ordena canónicamente por `(run_id, node_id)` y el frame del loader ya viene con ese `ORDER BY`, pero conviene verificarlo; (c) el `scale_pos_weight` se calcula sobre distinto subconjunto.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/test_parity_with_legacy_evaluator.py
"""Paridad entre el banco nuevo y ml/evaluator.py.

DESECHABLE: se borra junto al stack B en el Plan 3. Existe para autorizar ese
borrado con evidencia, no para vivir en la suite.

Si falla, NO subas la tolerancia. Investiga.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from swmm_resilience.database.training_queries import load_training_frame
from swmm_resilience.ml.bench.evaluate import evaluate_candidate
from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.evaluator import evaluate_models

PARAMS = {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.1, "subsample": 1.0}
RTOL = 1e-6


def _legacy_config():
    return SimpleNamespace(
        ml=SimpleNamespace(
            classifier=SimpleNamespace(
                algorithm="xgboost", n_estimators=20, max_depth=3,
                learning_rate=0.1, subsample=1.0, scale_pos_weight="auto",
            ),
            regressor=SimpleNamespace(
                algorithm="xgboost", n_estimators=20, max_depth=3,
                learning_rate=0.1, subsample=1.0,
            ),
            use_scaler=False,
        ),
        evaluation=SimpleNamespace(methods=["LOSO"], stratify_by_factor=True),
    )


def _assert_metrics_match(legacy: dict, bench: dict, level: str) -> None:
    assert set(legacy) == set(bench), f"{level}: claves distintas"
    for key, legacy_value in legacy.items():
        bench_value = bench[key]
        if np.isnan(legacy_value):
            assert np.isnan(bench_value), f"{level}.{key}: legacy NaN, banco {bench_value}"
        else:
            assert bench_value == pytest.approx(legacy_value, rel=RTOL), (
                f"{level}.{key}: legacy={legacy_value} banco={bench_value}"
            )


def test_xgboost_candidate_reproduces_the_legacy_loso_metrics(sql_training_db, tmp_path):
    frame = load_training_frame(sql_training_db)
    legacy = evaluate_models(frame, _legacy_config(), tmp_path / "legacy_metrics")["LOSO"]

    prepared = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )
    _, bench = evaluate_candidate(prepared, "xgboost", PARAMS, PARAMS)

    _assert_metrics_match(legacy["classifier"], bench["LOSO"]["classifier"], "classifier")
    _assert_metrics_match(
        legacy["regressor_oracle"], bench["LOSO"]["regressor_oracle"], "regressor_oracle"
    )
    _assert_metrics_match(legacy["end_to_end"], bench["LOSO"]["end_to_end"], "end_to_end")


def test_the_factor_breakdown_also_matches(sql_training_db, tmp_path):
    frame = load_training_frame(sql_training_db)
    legacy = evaluate_models(frame, _legacy_config(), tmp_path / "legacy_metrics")["LOSO"]

    prepared = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )
    _, bench = evaluate_candidate(prepared, "xgboost", PARAMS, PARAMS)

    assert set(legacy["by_factor"]) == set(bench["LOSO"]["by_factor"])
    for factor, legacy_metrics in legacy["by_factor"].items():
        _assert_metrics_match(
            legacy_metrics, bench["LOSO"]["by_factor"][factor], f"by_factor[{factor}]"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_parity_with_legacy_evaluator.py -q`
Expected: en el primer intento puede fallar por diferencias reales. **Investiga cada una y arregla el banco, no el test.**

- [ ] **Step 3: Reconciliar las diferencias**

No hay implementación nueva que escribir: el trabajo es diagnosticar. Compara fold a fold:

```python
# script de diagnostico, no se commitea
from swmm_resilience.ml.bench.folds import build_folds
from sklearn.model_selection import LeaveOneGroupOut
import numpy as np

frame = load_training_frame(db_path)
groups = frame["factor_mult"].values
legacy_splits = list(LeaveOneGroupOut().split(np.zeros((len(frame), 1)), None, groups))
bench_folds = build_folds(frame["factor_mult"], "LOSO")
for fold_id, (train_idx, test_idx) in enumerate(legacy_splits):
    chunk = bench_folds[bench_folds["fold_id"] == fold_id]
    got = chunk.loc[chunk["split"] == "test", "sample_idx"].tolist()
    assert got == sorted(test_idx.tolist()), f"fold {fold_id} difiere"
```

Si los folds coinciden y las métricas no, el problema está en el modelo: verifica que `TABULAR_V3_17.validate_frame` no cambie los valores (sólo debe hacer `astype(float)`) y que `scale_pos_weight` se calcule sobre el mismo subconjunto.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_parity_with_legacy_evaluator.py -q`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add tests/ml/bench/test_parity_with_legacy_evaluator.py
git commit -m "test(bench): paridad con evaluator.py (puerta para el retiro del legacy)"
```

---

## Task 11: `train.py` — `train_candidate` y artefactos

**Files:**
- Create: `swmm_resilience/ml/bench/train.py`
- Test: `tests/ml/bench/test_train.py`

**Interfaces:**
- Consumes: `get_family` de `registry.py`; `PreparedDataset` de `schemas.py`.
- Produces: `train_candidate(prepared, family_name, classifier_params, regressor_params, output_dir) -> CandidateArtifacts` (dataclass con `.family`, `.classifier_path`, `.regressor_path`, `.metadata_path`, `.metadata`); `load_candidate(directory) -> CandidateArtifacts`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/test_train.py
import json

import joblib
import numpy as np
import pytest

from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.train import load_candidate, train_candidate
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17

TINY = {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3, "subsample": 1.0}


@pytest.fixture
def prepared(sql_training_db, tmp_path):
    return prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )


def test_artifacts_are_written(prepared, tmp_path):
    artifacts = train_candidate(prepared, "xgboost", TINY, TINY, tmp_path / "candidates")

    assert artifacts.classifier_path.exists()
    assert artifacts.regressor_path.exists()
    assert artifacts.metadata_path.exists()
    assert artifacts.classifier_path.parent.name == "xgboost"


def test_metadata_records_the_prep_id_and_feature_order(prepared, tmp_path):
    artifacts = train_candidate(prepared, "xgboost", TINY, TINY, tmp_path / "candidates")
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))

    assert metadata["prep_id"] == prepared.prep_id
    assert metadata["family"] == "xgboost"
    assert metadata["ordered_features"] == list(FEATURE_COLUMNS_V17)
    assert metadata["target_transform"] == {"regressor": "log1p", "inverse": "expm1"}
    assert len(metadata["classifier_sha256"]) == 64
    assert len(metadata["regressor_sha256"]) == 64
    assert "library_versions" in metadata
    assert metadata["preprocessing"]["imputer"] == "median"


def test_the_saved_classifier_predicts_on_the_contract_features(prepared, tmp_path):
    artifacts = train_candidate(prepared, "xgboost", TINY, TINY, tmp_path / "candidates")
    classifier = joblib.load(artifacts.classifier_path)

    predictions = classifier.predict(prepared.X.head(3))
    assert len(predictions) == 3
    assert set(np.unique(predictions)) <= {0, 1}


def test_the_regressor_is_trained_only_on_flooded_rows(prepared, tmp_path):
    """Igual que trainer.py: el regresor solo ve inunda == 1."""
    artifacts = train_candidate(prepared, "xgboost", TINY, TINY, tmp_path / "candidates")
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["regressor_training_rows"] == int(prepared.y_clf.sum())


def test_training_with_no_flooded_rows_raises_clearly(prepared, tmp_path):
    import pandas as pd
    from swmm_resilience.ml.bench.schemas import PreparedDataset

    dry = PreparedDataset(
        prep_id=prepared.prep_id,
        keys=prepared.keys,
        X=prepared.X,
        y_clf=pd.Series([0] * len(prepared.y_clf), name="inunda"),
        y_reg=prepared.y_reg,
        folds=prepared.folds,
        manifest=prepared.manifest,
        quality=prepared.quality,
    )
    with pytest.raises(ValueError, match="ninguna fila inundada"):
        train_candidate(dry, "xgboost", TINY, TINY, tmp_path / "candidates")


def test_load_candidate_roundtrips(prepared, tmp_path):
    written = train_candidate(prepared, "xgboost", TINY, TINY, tmp_path / "candidates")
    loaded = load_candidate(written.classifier_path.parent)

    assert loaded.family == "xgboost"
    assert loaded.metadata["prep_id"] == prepared.prep_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_train.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'swmm_resilience.ml.bench.train'`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/train.py
"""Etapa 2 del banco: ajuste sobre todo el dataset y escritura de artefactos.

Intencionalmente idéntico a lo que hace ml/trainer.py: clasificador sobre
todas las filas, regresor sobre inunda == 1 con objetivo log1p. Esa identidad
es la condición para que el test de paridad pueda pasar.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version as _package_version
from pathlib import Path

import joblib
import numpy as np

from ...config import ML_RANDOM_STATE
from ..contracts import FEATURE_COLUMNS_V17, TABULAR_V3_17
from .registry import get_family
from .schemas import PreparedDataset


@dataclass(frozen=True)
class CandidateArtifacts:
    family: str
    classifier_path: Path
    regressor_path: Path
    metadata_path: Path
    metadata: dict


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _library_versions() -> dict:
    return {
        name: _package_version(name)
        for name in ("pandas", "numpy", "scikit-learn", "xgboost")
    }


def train_candidate(
    prepared: PreparedDataset,
    family_name: str,
    classifier_params: dict,
    regressor_params: dict,
    output_dir: Path | str,
) -> CandidateArtifacts:
    """Ajusta la cascada sobre todo el dataset y persiste los artefactos."""
    family = get_family(family_name)
    directory = Path(output_dir) / family_name
    directory.mkdir(parents=True, exist_ok=True)

    X = prepared.X
    y_clf = prepared.y_clf.to_numpy()
    y_reg = prepared.y_reg.to_numpy()

    n_negative, n_positive = int((y_clf == 0).sum()), int((y_clf == 1).sum())
    if n_positive == 0:
        raise ValueError(
            "El dataset no tiene ninguna fila inundada (inunda == 1); "
            "el regresor no se puede entrenar."
        )
    scale_pos_weight = n_negative / n_positive

    classifier = family.build_classifier(classifier_params, scale_pos_weight)
    classifier.fit(X, y_clf)

    flooded = y_clf == 1
    regressor = family.build_regressor(regressor_params)
    regressor.fit(X.iloc[flooded], np.log1p(y_reg[flooded]))

    classifier_path = directory / "classifier.joblib"
    regressor_path = directory / "regressor.joblib"
    joblib.dump(classifier, classifier_path)
    joblib.dump(regressor, regressor_path)

    metadata = {
        "family": family_name,
        "prep_id": prepared.prep_id,
        "classifier_params": classifier_params,
        "regressor_params": regressor_params,
        "preprocessing": family.preprocessing_descriptor(classifier_params),
        "ordered_features": list(FEATURE_COLUMNS_V17),
        "feature_contract_id": TABULAR_V3_17.contract_id,
        "feature_contract_sha256": TABULAR_V3_17.descriptor_sha256,
        "target_transform": {"regressor": "log1p", "inverse": "expm1"},
        "scale_pos_weight": float(scale_pos_weight),
        "regressor_training_rows": n_positive,
        "random_seed": ML_RANDOM_STATE,
        "classifier_sha256": _sha256(classifier_path),
        "regressor_sha256": _sha256(regressor_path),
        "python_version": platform.python_version(),
        "library_versions": _library_versions(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path = directory / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return CandidateArtifacts(
        family=family_name,
        classifier_path=classifier_path,
        regressor_path=regressor_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def load_candidate(directory: Path | str) -> CandidateArtifacts:
    """Lee los artefactos de un candidato ya entrenado."""
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No hay metadata.json en {directory}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return CandidateArtifacts(
        family=metadata["family"],
        classifier_path=directory / "classifier.joblib",
        regressor_path=directory / "regressor.joblib",
        metadata_path=metadata_path,
        metadata=metadata,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_train.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/train.py tests/ml/bench/test_train.py
git commit -m "feat(bench): etapa 2 train_candidate con metadata de proveniencia"
```

---

## Task 12: Familias `random_forest`, `linear` y `svm`

**Files:**
- Create: `swmm_resilience/ml/bench/models/random_forest_family.py`
- Create: `swmm_resilience/ml/bench/models/linear_family.py`
- Create: `swmm_resilience/ml/bench/models/svm_family.py`
- Modify: `swmm_resilience/ml/bench/registry.py`
- Modify: `tests/ml/bench/test_evaluate_candidate.py` (quitar el `xfail` de la Task 9)
- Test: `tests/ml/bench/test_model_families.py`

**Interfaces:**
- Consumes: `ML_RANDOM_STATE` de `swmm_resilience.config`.
- Produces: tres módulos con la misma interfaz que `xgboost_family`. `available_families()` pasa a devolver `("linear", "random_forest", "svm", "xgboost")`.

**Diferencia clave respecto a `xgboost`:** `linear` y `svm` declaran `SCALE_FEATURES = True` y meten un `StandardScaler` en el `Pipeline`, entre el imputador y el modelo. `random_forest` va sin escalador, como XGBoost.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/test_model_families.py
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from swmm_resilience.ml.bench.registry import available_families, get_family
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17

SCALED_FAMILIES = ("linear", "svm")
UNSCALED_FAMILIES = ("xgboost", "random_forest")
ALL_FAMILIES = SCALED_FAMILIES + UNSCALED_FAMILIES

TINY_PARAMS = {
    "xgboost": {"n_estimators": 5, "max_depth": 2},
    "random_forest": {"n_estimators": 5, "max_depth": 2},
    "linear": {"alpha": 1.0},
    "svm": {"C": 1.0},
}


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    n = 40
    X = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMNS_V17))), columns=list(FEATURE_COLUMNS_V17)
    )
    y_clf = (X["q_pico_nodo"] > 0).astype(int).to_numpy()
    y_reg = np.abs(X["q_pico_nodo"].to_numpy()) * 10.0
    return X, y_clf, y_reg


def test_all_four_families_are_registered():
    assert set(available_families()) == set(ALL_FAMILIES)


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_every_family_exposes_the_same_interface(name):
    family = get_family(name)
    assert family.FAMILY == name
    assert isinstance(family.SCALE_FEATURES, bool)
    assert callable(family.build_classifier)
    assert callable(family.build_regressor)
    assert callable(family.preprocessing_descriptor)


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_every_pipeline_starts_with_a_median_imputer(name):
    pipeline = get_family(name).build_classifier(TINY_PARAMS[name], scale_pos_weight=1.0)
    first = pipeline.steps[0][1]
    assert isinstance(first, SimpleImputer)
    assert first.strategy == "median"


@pytest.mark.parametrize("name", SCALED_FAMILIES)
def test_scaled_families_include_a_standard_scaler(name):
    family = get_family(name)
    assert family.SCALE_FEATURES is True
    pipeline = family.build_classifier(TINY_PARAMS[name], scale_pos_weight=1.0)
    assert any(isinstance(step, StandardScaler) for _, step in pipeline.steps)


@pytest.mark.parametrize("name", UNSCALED_FAMILIES)
def test_tree_families_have_no_scaler(name):
    family = get_family(name)
    assert family.SCALE_FEATURES is False
    pipeline = family.build_classifier(TINY_PARAMS[name], scale_pos_weight=1.0)
    assert not any(isinstance(step, StandardScaler) for _, step in pipeline.steps)


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_no_family_uses_pca(name):
    """PCA era del stack A: degradaba la interpretabilidad y no vuelve."""
    pipeline = get_family(name).build_regressor(TINY_PARAMS[name])
    assert "pca" not in dict(pipeline.named_steps)


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_classifier_fits_and_predicts_binary_labels(name, toy_data):
    X, y_clf, _ = toy_data
    pipeline = get_family(name).build_classifier(TINY_PARAMS[name], scale_pos_weight=1.0)
    pipeline.fit(X, y_clf)

    predictions = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)[:, 1]

    assert set(np.unique(predictions)) <= {0, 1}
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_regressor_fits_and_predicts_floats(name, toy_data):
    X, _, y_reg = toy_data
    pipeline = get_family(name).build_regressor(TINY_PARAMS[name])
    pipeline.fit(X, np.log1p(y_reg))

    predictions = pipeline.predict(X)
    assert len(predictions) == len(X)
    assert np.isfinite(predictions).all()


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_preprocessing_descriptor_is_json_serialisable(name):
    descriptor = get_family(name).preprocessing_descriptor(TINY_PARAMS[name])
    assert json.dumps(descriptor)
    assert set(descriptor) == {"imputer", "scaler", "pca"}


@pytest.mark.parametrize("name", ALL_FAMILIES)
def test_families_handle_missing_values_via_the_imputer(name, toy_data):
    X, y_clf, _ = toy_data
    X = X.copy()
    X.loc[0, "diam_max_in"] = np.nan

    pipeline = get_family(name).build_classifier(TINY_PARAMS[name], scale_pos_weight=1.0)
    pipeline.fit(X, y_clf)
    assert len(pipeline.predict(X)) == len(X)


def test_svm_classifier_supports_probability_output():
    """SVC necesita probability=True o predict_proba revienta en el evaluador."""
    pipeline = get_family("svm").build_classifier({"C": 1.0}, scale_pos_weight=1.0)
    assert pipeline.named_steps["model"].probability is True


def test_class_weight_is_used_where_scale_pos_weight_does_not_exist():
    """linear y svm no tienen scale_pos_weight: el desbalance va por class_weight."""
    for name in SCALED_FAMILIES:
        pipeline = get_family(name).build_classifier(TINY_PARAMS[name], scale_pos_weight=4.0)
        weights = pipeline.named_steps["model"].class_weight
        assert weights == {0: 1.0, 1: pytest.approx(4.0)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_model_families.py -q`
Expected: FAIL con `ValueError: Familia desconocida: 'random_forest'`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/models/random_forest_family.py
"""Familia RandomForest. Sin escalador: los árboles no lo necesitan."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from ....config import ML_RANDOM_STATE

FAMILY = "random_forest"
SCALE_FEATURES = False

_DEFAULTS = {"n_estimators": 200, "max_depth": 6}


def build_classifier(params: dict, scale_pos_weight: float) -> Pipeline:
    settings = {**_DEFAULTS, **params}
    model = RandomForestClassifier(
        **settings,
        class_weight={0: 1.0, 1: float(scale_pos_weight)},
        random_state=ML_RANDOM_STATE,
    )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def build_regressor(params: dict) -> Pipeline:
    settings = {**_DEFAULTS, **params}
    model = RandomForestRegressor(**settings, random_state=ML_RANDOM_STATE)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def preprocessing_descriptor(params: dict) -> dict:
    return {"imputer": "median", "scaler": None, "pca": None}
```

```python
# swmm_resilience/ml/bench/models/linear_family.py
"""Familia lineal: LogisticRegression para clasificar, Ridge para regresar.

Lleva StandardScaler dentro del Pipeline, así que se ajusta con el train de
cada fold. Es el baseline honesto contra el que se mide XGBoost.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ....config import ML_RANDOM_STATE

FAMILY = "linear"
SCALE_FEATURES = True

_CLASSIFIER_DEFAULTS = {"C": 1.0, "max_iter": 5000}
_REGRESSOR_DEFAULTS = {"alpha": 1.0}


def build_classifier(params: dict, scale_pos_weight: float) -> Pipeline:
    settings = {**_CLASSIFIER_DEFAULTS, **{k: v for k, v in params.items() if k != "alpha"}}
    model = LogisticRegression(
        **settings,
        class_weight={0: 1.0, 1: float(scale_pos_weight)},
        random_state=ML_RANDOM_STATE,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_regressor(params: dict) -> Pipeline:
    settings = {**_REGRESSOR_DEFAULTS, **{k: v for k, v in params.items() if k in ("alpha",)}}
    model = Ridge(**settings, random_state=ML_RANDOM_STATE)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def preprocessing_descriptor(params: dict) -> dict:
    return {"imputer": "median", "scaler": "standard", "pca": None}
```

```python
# swmm_resilience/ml/bench/models/svm_family.py
"""Familia SVM: SVC para clasificar, SVR para regresar.

probability=True es obligatorio: el evaluador llama predict_proba para
calcular AUC-ROC. Encarece el ajuste (SVC hace Platt scaling interno) pero
sin ello esta familia no sería comparable con el resto.
"""

from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from ....config import ML_RANDOM_STATE

FAMILY = "svm"
SCALE_FEATURES = True

_CLASSIFIER_DEFAULTS = {"C": 10.0, "kernel": "rbf", "gamma": "scale"}
_REGRESSOR_DEFAULTS = {"C": 10.0, "kernel": "rbf", "epsilon": 0.1}


def build_classifier(params: dict, scale_pos_weight: float) -> Pipeline:
    settings = {**_CLASSIFIER_DEFAULTS, **{k: v for k, v in params.items() if k != "epsilon"}}
    model = SVC(
        **settings,
        probability=True,
        class_weight={0: 1.0, 1: float(scale_pos_weight)},
        random_state=ML_RANDOM_STATE,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_regressor(params: dict) -> Pipeline:
    settings = {**_REGRESSOR_DEFAULTS, **params}
    model = SVR(**settings)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def preprocessing_descriptor(params: dict) -> dict:
    return {"imputer": "median", "scaler": "standard", "pca": None}
```

Actualizar el registro:

```python
# swmm_resilience/ml/bench/registry.py — reemplazar el bloque de imports y _FAMILIES
from .models import linear_family, random_forest_family, svm_family, xgboost_family

_FAMILIES: dict[str, ModuleType] = {
    module.FAMILY: module
    for module in (xgboost_family, random_forest_family, linear_family, svm_family)
}
```

Y quitar el `xfail` de `test_the_scaler_is_fitted_inside_the_fold_not_on_the_whole_dataset` en `tests/ml/bench/test_evaluate_candidate.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/ -q`
Expected: todo verde, incluido el test anti-fuga en ejecución que ahora sí corre `linear`

> **Si `Ridge` rechaza `random_state`:** en scikit-learn 1.8 `Ridge` sólo acepta `random_state` con `solver="sag"`/`"saga"`. Quítalo del constructor de `Ridge` — el determinismo con el solver por defecto no depende de él.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/models/ swmm_resilience/ml/bench/registry.py tests/ml/bench/test_model_families.py tests/ml/bench/test_evaluate_candidate.py
git commit -m "feat(bench): familias random_forest, linear y svm"
```

---

## Task 13: Familia `mlp` — wrapper propio sobre torch

**Files:**
- Create: `swmm_resilience/ml/bench/models/mlp_family.py`
- Modify: `swmm_resilience/ml/bench/registry.py`
- Test: `tests/ml/bench/test_mlp_family.py`

**Interfaces:**
- Consumes: `torch` (ya en `requirements.txt`); `ML_RANDOM_STATE`.
- Produces: `TorchMLPClassifier` y `TorchMLPRegressor` (compatibles con la API de sklearn: `fit`, `predict`, `predict_proba`, `get_params`, `set_params`), más la interfaz de familia estándar.

**Por qué wrapper propio y no `skorch`:** no añade una dependencia para envolver un modelo de tres capas, y deja explícito el control de semilla y de épocas, que es lo que hace comparable al MLP con el resto del banco. Heredar de `BaseEstimator`/`ClassifierMixin` de sklearn da `get_params`/`set_params` gratis, que `Pipeline` necesita.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/test_mlp_family.py
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from swmm_resilience.ml.bench.registry import available_families, get_family
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17

PARAMS = {"hidden_sizes": (16, 8), "epochs": 20, "learning_rate": 0.01, "batch_size": 16}


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(7)
    n = 80
    X = pd.DataFrame(
        rng.normal(size=(n, len(FEATURE_COLUMNS_V17))), columns=list(FEATURE_COLUMNS_V17)
    )
    y_clf = (X["q_pico_nodo"] + X["prof_max"] > 0).astype(int).to_numpy()
    y_reg = np.abs(X["q_pico_nodo"].to_numpy()) * 10.0
    return X, y_clf, y_reg


def test_mlp_is_registered_as_a_family():
    assert "mlp" in available_families()
    assert get_family("mlp").SCALE_FEATURES is True


def test_pipeline_has_imputer_scaler_and_model():
    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    steps = dict(pipeline.named_steps)
    assert set(steps) == {"imputer", "scaler", "model"}
    assert isinstance(steps["scaler"], StandardScaler)


def test_classifier_fits_and_returns_binary_predictions(toy_data):
    X, y_clf, _ = toy_data
    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    pipeline.fit(X, y_clf)

    predictions = pipeline.predict(X)
    assert set(np.unique(predictions)) <= {0, 1}
    assert len(predictions) == len(X)


def test_predict_proba_returns_two_calibrated_columns(toy_data):
    X, y_clf, _ = toy_data
    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    pipeline.fit(X, y_clf)

    probabilities = pipeline.predict_proba(X)
    assert probabilities.shape == (len(X), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_regressor_fits_and_predicts_finite_values(toy_data):
    X, _, y_reg = toy_data
    pipeline = get_family("mlp").build_regressor(PARAMS)
    pipeline.fit(X, np.log1p(y_reg))

    predictions = pipeline.predict(X)
    assert predictions.shape == (len(X),)
    assert np.isfinite(predictions).all()


def test_the_same_seed_gives_the_same_predictions(toy_data):
    """Sin esto el MLP no es comparable: cada corrida daria otro numero."""
    X, y_clf, _ = toy_data
    first = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    second = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    first.fit(X, y_clf)
    second.fit(X, y_clf)

    np.testing.assert_allclose(
        first.predict_proba(X)[:, 1], second.predict_proba(X)[:, 1], rtol=1e-6
    )


def test_the_estimator_is_clonable_by_sklearn():
    """Pipeline y cross-validation dependen de get_params/set_params."""
    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    cloned = clone(pipeline)
    assert cloned.named_steps["model"].get_params()["epochs"] == PARAMS["epochs"]


def test_scale_pos_weight_reaches_the_loss(toy_data):
    """El desbalance se aplica como pos_weight de BCEWithLogitsLoss."""
    model = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=5.0).named_steps["model"]
    assert model.get_params()["scale_pos_weight"] == pytest.approx(5.0)


def test_training_runs_on_cpu_without_requiring_cuda(toy_data):
    X, y_clf, _ = toy_data
    pipeline = get_family("mlp").build_classifier(PARAMS, scale_pos_weight=1.0)
    pipeline.fit(X, y_clf)
    assert next(pipeline.named_steps["model"].network_.parameters()).device.type == "cpu"


def test_preprocessing_descriptor_declares_the_scaler():
    descriptor = get_family("mlp").preprocessing_descriptor(PARAMS)
    assert descriptor == {"imputer": "median", "scaler": "standard", "pca": None}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_mlp_family.py -q`
Expected: FAIL con `ValueError: Familia desconocida: 'mlp'`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/models/mlp_family.py
"""Familia MLP: red densa sobre las mismas 17 features que el resto.

No es un caso especial en ningún punto del banco: entra al mismo Pipeline,
a los mismos folds y a las mismas métricas que XGBoost o SVR, de modo que la
comparación es 1:1.

Wrapper propio en vez de skorch: evita una dependencia nueva para envolver
un modelo de tres capas, y deja explícito el control de semilla y de épocas,
que es lo que hace reproducible —y por tanto comparable— este candidato.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ....config import ML_RANDOM_STATE

FAMILY = "mlp"
SCALE_FEATURES = True

_DEFAULTS = {
    "hidden_sizes": (64, 32),
    "epochs": 200,
    "learning_rate": 0.001,
    "batch_size": 256,
    "dropout": 0.1,
}


def _build_network(n_features: int, hidden_sizes, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_features = n_features
    for size in hidden_sizes:
        layers.extend([nn.Linear(in_features, size), nn.ReLU(), nn.Dropout(dropout)])
        in_features = size
    layers.append(nn.Linear(in_features, 1))
    return nn.Sequential(*layers)


class _TorchMLPBase(BaseEstimator):
    """Parte común de clasificador y regresor: red, bucle de ajuste, semilla."""

    def __init__(
        self,
        hidden_sizes=(64, 32),
        epochs: int = 200,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        dropout: float = 0.1,
        random_state: int = ML_RANDOM_STATE,
    ):
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_state = random_state

    def _fit_network(self, X, y, loss_fn) -> None:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_tensor = torch.tensor(np.asarray(X, dtype=np.float32))
        y_tensor = torch.tensor(np.asarray(y, dtype=np.float32)).reshape(-1, 1)

        self.network_ = _build_network(
            X_tensor.shape[1], tuple(self.hidden_sizes), self.dropout
        )
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate)

        n_samples = len(X_tensor)
        generator = torch.Generator().manual_seed(self.random_state)
        self.network_.train()
        for _ in range(self.epochs):
            order = torch.randperm(n_samples, generator=generator)
            for start in range(0, n_samples, self.batch_size):
                batch = order[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(self.network_(X_tensor[batch]), y_tensor[batch])
                loss.backward()
                optimizer.step()
        self.network_.eval()

    def _forward(self, X) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.tensor(np.asarray(X, dtype=np.float32))
            return self.network_(tensor).numpy().ravel()


class TorchMLPClassifier(ClassifierMixin, _TorchMLPBase):
    """MLP binario. El desbalance entra como ``pos_weight`` de la pérdida."""

    def __init__(self, scale_pos_weight: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale_pos_weight = scale_pos_weight

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([float(self.scale_pos_weight)])
        )
        self._fit_network(X, y, loss_fn)
        return self

    def predict_proba(self, X) -> np.ndarray:
        positive = 1.0 / (1.0 + np.exp(-self._forward(X)))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class TorchMLPRegressor(RegressorMixin, _TorchMLPBase):
    """MLP de regresión sobre el target ya transformado con log1p."""

    def fit(self, X, y):
        self._fit_network(X, y, nn.MSELoss())
        return self

    def predict(self, X) -> np.ndarray:
        return self._forward(X)


def _settings(params: dict) -> dict:
    allowed = set(_DEFAULTS)
    return {**_DEFAULTS, **{k: v for k, v in params.items() if k in allowed}}


def build_classifier(params: dict, scale_pos_weight: float) -> Pipeline:
    model = TorchMLPClassifier(scale_pos_weight=float(scale_pos_weight), **_settings(params))
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def build_regressor(params: dict) -> Pipeline:
    model = TorchMLPRegressor(**_settings(params))
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def preprocessing_descriptor(params: dict) -> dict:
    return {"imputer": "median", "scaler": "standard", "pca": None}
```

Registrar la familia:

```python
# swmm_resilience/ml/bench/registry.py
from .models import (
    linear_family,
    mlp_family,
    random_forest_family,
    svm_family,
    xgboost_family,
)

_FAMILIES: dict[str, ModuleType] = {
    module.FAMILY: module
    for module in (
        xgboost_family,
        random_forest_family,
        linear_family,
        svm_family,
        mlp_family,
    )
}
```

Y actualizar `test_all_four_families_are_registered` en `tests/ml/bench/test_model_families.py` para incluir `"mlp"` (renómbralo a `test_all_five_families_are_registered`).

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_mlp_family.py -q`
Expected: 10 passed

> **Si `test_the_same_seed_gives_the_same_predictions` falla:** el no-determinismo suele venir del orden de los lotes o de la inicialización. Verifica que `torch.manual_seed` se llame **antes** de construir la red (la inicialización de pesos consume el generador) y que el `randperm` use su propio `Generator` sembrado, como está escrito arriba. Si persiste en tu máquina, añade `torch.use_deterministic_algorithms(True)` al inicio de `_fit_network`.

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/models/mlp_family.py swmm_resilience/ml/bench/registry.py tests/ml/bench/test_mlp_family.py tests/ml/bench/test_model_families.py
git commit -m "feat(bench): familia MLP tabular con wrapper propio sobre torch"
```

---

## Task 14: `ranking.py` y `reports.py`

**Files:**
- Create: `swmm_resilience/ml/bench/ranking.py`
- Create: `swmm_resilience/ml/bench/reports.py`
- Test: `tests/ml/bench/test_ranking.py`
- Test: `tests/ml/bench/test_reports.py`

**Interfaces:**
- Consumes: nada del banco (ambos operan sobre dicts de métricas).
- Produces: `RankingCriterion` (dataclass: `primary_metric`, `primary_direction`, `tie_breakers`); `rank_candidates(metrics_by_family, criterion, protocol) -> pd.DataFrame` (columnas `rank`, `family`, `primary_value`, `valid`, `invalid_reason`, más una por desempate); `resolve_metric(metrics, "end_to_end.rmse_vol_todos_nodos") -> float | None`; `write_reports(metrics_by_family, ranking, output_dir) -> dict[str, Path]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/test_ranking.py
import numpy as np
import pytest

from swmm_resilience.ml.bench.ranking import (
    RankingCriterion,
    rank_candidates,
    resolve_metric,
)

CRITERION = RankingCriterion(
    primary_metric="end_to_end.rmse_vol_todos_nodos",
    primary_direction="minimize",
    tie_breakers=(
        ("classifier.f1", "maximize"),
        ("regressor_oracle.nse", "maximize"),
    ),
)


def _metrics(rmse, f1=0.5, nse=0.5):
    return {
        "LOSO": {
            "classifier": {"f1": f1, "precision": 0.5, "recall": 0.5, "auc_roc": 0.7},
            "regressor_oracle": {"nse": nse, "rmse": 1.0, "mae": 1.0, "r2": 0.5, "log_nse": 0.5},
            "end_to_end": {
                "rmse_vol_todos_nodos": rmse,
                "pct_nodos_correctos": 0.9,
                "vol_total_pred_m3": 10.0,
                "vol_total_real_m3": 11.0,
            },
            "by_factor": {},
        }
    }


def test_resolve_metric_walks_the_dotted_path():
    metrics = _metrics(5.0)["LOSO"]
    assert resolve_metric(metrics, "end_to_end.rmse_vol_todos_nodos") == pytest.approx(5.0)
    assert resolve_metric(metrics, "classifier.f1") == pytest.approx(0.5)


def test_resolve_metric_returns_none_for_a_missing_path():
    assert resolve_metric(_metrics(5.0)["LOSO"], "classifier.no_existe") is None


def test_lowest_rmse_ranks_first():
    ranking = rank_candidates(
        {"xgboost": _metrics(3.0), "svm": _metrics(9.0), "linear": _metrics(6.0)},
        CRITERION,
        protocol="LOSO",
    )
    assert ranking["family"].tolist() == ["xgboost", "linear", "svm"]
    assert ranking["rank"].tolist() == [1, 2, 3]


def test_ties_are_broken_by_the_first_tie_breaker():
    ranking = rank_candidates(
        {"a": _metrics(5.0, f1=0.4), "b": _metrics(5.0, f1=0.9)},
        CRITERION,
        protocol="LOSO",
    )
    assert ranking["family"].tolist() == ["b", "a"]


def test_the_second_tie_breaker_is_used_when_the_first_also_ties():
    ranking = rank_candidates(
        {"a": _metrics(5.0, f1=0.7, nse=0.2), "b": _metrics(5.0, f1=0.7, nse=0.8)},
        CRITERION,
        protocol="LOSO",
    )
    assert ranking["family"].tolist() == ["b", "a"]


def test_a_candidate_with_a_nan_primary_metric_is_marked_invalid_and_ranked_last():
    ranking = rank_candidates(
        {"good": _metrics(5.0), "broken": _metrics(float("nan"))},
        CRITERION,
        protocol="LOSO",
    )
    assert ranking["family"].tolist() == ["good", "broken"]
    assert ranking.set_index("family").loc["broken", "valid"] == 0
    assert "NaN" in ranking.set_index("family").loc["broken", "invalid_reason"]


def test_a_candidate_missing_the_primary_metric_is_invalid():
    ranking = rank_candidates(
        {"good": _metrics(5.0), "empty": {"LOSO": {"end_to_end": {}, "classifier": {},
                                                   "regressor_oracle": {}, "by_factor": {}}}},
        CRITERION,
        protocol="LOSO",
    )
    assert ranking.set_index("family").loc["empty", "valid"] == 0


def test_maximize_direction_inverts_the_order():
    criterion = RankingCriterion(
        primary_metric="classifier.f1", primary_direction="maximize", tie_breakers=()
    )
    ranking = rank_candidates(
        {"low": _metrics(1.0, f1=0.2), "high": _metrics(9.0, f1=0.95)},
        criterion,
        protocol="LOSO",
    )
    assert ranking["family"].tolist() == ["high", "low"]


def test_unknown_direction_is_rejected():
    with pytest.raises(ValueError, match="minimize|maximize"):
        RankingCriterion(
            primary_metric="classifier.f1", primary_direction="sideways", tie_breakers=()
        )


def test_a_protocol_absent_from_a_candidate_is_reported_not_crashed():
    ranking = rank_candidates(
        {"only_kfold": {"GroupKFold5": _metrics(5.0)["LOSO"]}},
        CRITERION,
        protocol="LOSO",
    )
    assert ranking.set_index("family").loc["only_kfold", "valid"] == 0
```

```python
# tests/ml/bench/test_reports.py
import csv
import json

import pandas as pd

from swmm_resilience.ml.bench.ranking import RankingCriterion, rank_candidates
from swmm_resilience.ml.bench.reports import write_reports

CRITERION = RankingCriterion(
    primary_metric="end_to_end.rmse_vol_todos_nodos",
    primary_direction="minimize",
    tie_breakers=(("classifier.f1", "maximize"),),
)


def _metrics(rmse):
    return {
        "LOSO": {
            "classifier": {"f1": 0.8, "precision": 0.7, "recall": 0.9, "auc_roc": 0.85},
            "regressor_oracle": {"nse": 0.6, "rmse": 2.0, "mae": 1.0, "r2": 0.6, "log_nse": 0.7},
            "end_to_end": {
                "rmse_vol_todos_nodos": rmse,
                "pct_nodos_correctos": 0.9,
                "vol_total_pred_m3": 10.0,
                "vol_total_real_m3": 11.0,
            },
            "by_factor": {"1.00": {"f1": 0.8, "rmse_vol": rmse}},
        }
    }


def test_writes_one_metrics_file_per_family_plus_the_ranking(tmp_path):
    metrics = {"xgboost": _metrics(3.0), "linear": _metrics(7.0)}
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")

    paths = write_reports(metrics, ranking, tmp_path)

    assert (tmp_path / "metrics_xgboost.json").exists()
    assert (tmp_path / "metrics_linear.json").exists()
    assert (tmp_path / "ranking.json").exists()
    assert (tmp_path / "ranking.csv").exists()
    assert set(paths) >= {"ranking_json", "ranking_csv"}


def test_metrics_json_keeps_the_three_levels_and_the_factor_breakdown(tmp_path):
    metrics = {"xgboost": _metrics(3.0)}
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")
    write_reports(metrics, ranking, tmp_path)

    written = json.loads((tmp_path / "metrics_xgboost.json").read_text(encoding="utf-8"))
    assert set(written["LOSO"]) == {
        "classifier", "regressor_oracle", "end_to_end", "by_factor"
    }
    assert written["LOSO"]["by_factor"]["1.00"]["f1"] == 0.8


def test_ranking_csv_is_readable_and_ordered(tmp_path):
    metrics = {"xgboost": _metrics(3.0), "linear": _metrics(7.0)}
    ranking = rank_candidates(metrics, CRITERION, protocol="LOSO")
    write_reports(metrics, ranking, tmp_path)

    with open(tmp_path / "ranking.csv", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["family"] for row in rows] == ["xgboost", "linear"]
    assert rows[0]["rank"] == "1"


def test_report_directory_is_created_when_missing(tmp_path):
    target = tmp_path / "no" / "existe" / "todavia"
    metrics = {"xgboost": _metrics(3.0)}
    write_reports(metrics, rank_candidates(metrics, CRITERION, protocol="LOSO"), target)
    assert (target / "ranking.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_ranking.py tests/ml/bench/test_reports.py -q`
Expected: FAIL con `ModuleNotFoundError: No module named 'swmm_resilience.ml.bench.ranking'`

- [ ] **Step 3: Write minimal implementation**

```python
# swmm_resilience/ml/bench/ranking.py
"""Orden de candidatos por métrica primaria y desempates.

Un candidato cuya métrica primaria sea NaN o falte no se descarta en
silencio: queda en el ranking, marcado inválido y con su motivo, en el
último lugar. Esa información es parte de la evidencia, no ruido.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

_DIRECTIONS = ("minimize", "maximize")


@dataclass(frozen=True)
class RankingCriterion:
    primary_metric: str
    primary_direction: str
    tie_breakers: Sequence[tuple[str, str]] = ()

    def __post_init__(self) -> None:
        for direction in [self.primary_direction] + [d for _, d in self.tie_breakers]:
            if direction not in _DIRECTIONS:
                raise ValueError(
                    f"Dirección desconocida: {direction!r}. Opciones: minimize, maximize"
                )


def resolve_metric(metrics: dict, dotted_path: str) -> float | None:
    """Lee ``'end_to_end.rmse_vol_todos_nodos'`` de un dict anidado."""
    node = metrics
    for part in dotted_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, (int, float)) else None


def _sort_value(value: float | None, direction: str) -> float:
    if value is None or math.isnan(value):
        return math.inf
    return -value if direction == "maximize" else value


def rank_candidates(
    metrics_by_family: dict[str, dict],
    criterion: RankingCriterion,
    protocol: str,
) -> pd.DataFrame:
    """Ordena las familias por la métrica primaria y sus desempates."""
    rows = []
    for family, all_metrics in metrics_by_family.items():
        protocol_metrics = all_metrics.get(protocol)
        if protocol_metrics is None:
            rows.append(
                {
                    "family": family,
                    "primary_value": None,
                    "valid": 0,
                    "invalid_reason": f"el candidato no tiene resultados para {protocol}",
                }
            )
            continue

        primary = resolve_metric(protocol_metrics, criterion.primary_metric)
        if primary is None:
            reason = f"falta la métrica primaria {criterion.primary_metric}"
            valid = 0
        elif math.isnan(primary):
            reason = f"la métrica primaria {criterion.primary_metric} es NaN"
            valid = 0
        else:
            reason = None
            valid = 1

        row = {
            "family": family,
            "primary_value": primary,
            "valid": valid,
            "invalid_reason": reason,
        }
        for path, _ in criterion.tie_breakers:
            row[path] = resolve_metric(protocol_metrics, path)
        rows.append(row)

    def sort_key(row: dict) -> tuple:
        keys = [1 - row["valid"], _sort_value(row["primary_value"], criterion.primary_direction)]
        for path, direction in criterion.tie_breakers:
            keys.append(_sort_value(row.get(path), direction))
        keys.append(row["family"])
        return tuple(keys)

    ordered = sorted(rows, key=sort_key)
    frame = pd.DataFrame(ordered)
    frame.insert(0, "rank", range(1, len(frame) + 1))
    return frame
```

```python
# swmm_resilience/ml/bench/reports.py
"""Export de métricas y ranking a archivos, para lectura humana y para la tesis.

La fuente de verdad de estos números es SQL (Plan 2); estos archivos son la
copia cómoda de leer, no el registro autoritativo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_reports(
    metrics_by_family: dict[str, dict],
    ranking: pd.DataFrame,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Escribe un JSON por familia más el ranking en JSON y CSV."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for family, metrics in metrics_by_family.items():
        path = directory / f"metrics_{family}.json"
        path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written[f"metrics_{family}"] = path

    ranking_json = directory / "ranking.json"
    ranking_json.write_text(
        json.dumps(ranking.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    ranking_csv = directory / "ranking.csv"
    ranking.to_csv(ranking_csv, index=False)

    written["ranking_json"] = ranking_json
    written["ranking_csv"] = ranking_csv
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_ranking.py tests/ml/bench/test_reports.py -q`
Expected: 11 passed + 4 passed

- [ ] **Step 5: Commit**

```bash
git add swmm_resilience/ml/bench/ranking.py swmm_resilience/ml/bench/reports.py tests/ml/bench/test_ranking.py tests/ml/bench/test_reports.py
git commit -m "feat(bench): ranking por metrica primaria y export de reportes"
```

---

## Task 15: Configuración `bench:` y CLI `--bench-*`

**Files:**
- Modify: `swmm_resilience/config.py` (añadir `BenchConfig`, `BenchFamilyConfig`, `RankingConfig`; parseo en `load_config`)
- Modify: `config.yaml` (bloque `bench:`)
- Modify: `config_7shapes.yaml` (mismo bloque)
- Modify: `main.py` (cinco flags nuevos y su orquestación)
- Create: `swmm_resilience/ml/bench/promote.py`
- Test: `tests/ml/bench/test_bench_config.py`
- Test: `tests/ml/bench/test_bench_cli.py`

**Interfaces:**
- Consumes: todo lo anterior.
- Produces: `Config.bench` (opcional, `None` si el YAML no lo declara); `promote_candidate(candidate_dir, models_dir, inp_path) -> dict`; flags `--bench-prepare`, `--bench-train`, `--bench-evaluate`, `--bench-promote`, `--bench`, más `--models` y `--prep-id`.

**Compatibilidad:** `bench:` es **opcional** en esta fase. Un `config.yaml` sin ese bloque debe seguir cargando (los bloques `ml:` y `evaluation:` siguen vivos hasta el Plan 3). Eso mantiene verde toda la suite existente.

- [ ] **Step 1: Write the failing test**

```python
# tests/ml/bench/test_bench_config.py
import textwrap

import pytest

from swmm_resilience.config import load_config

_BASE_YAML = """
network:
  inp_path: "network.inp"
  name: "Test"
simulation:
  factor_min: 1.0
  factor_max: 3.0
  factor_step: 1.0
dataset:
  output_path: "data/training/dataset_final.csv"
  db_path: "outputs/training_v17.sqlite3"
  flood_threshold_m3: 1.0
ml:
  classifier: {algorithm: "xgboost", n_estimators: 10, max_depth: 3, learning_rate: 0.1, subsample: 0.8, scale_pos_weight: "auto"}
  regressor: {algorithm: "xgboost", n_estimators: 10, max_depth: 3, learning_rate: 0.1, subsample: 0.8}
  use_scaler: false
evaluation:
  methods: ["LOSO"]
  stratify_by_factor: true
visualization:
  factors_to_plot: [1.0]
  colormap: "RdBu_r"
  output_path: "outputs/maps/"
  show_labels_top_n: 5
"""

_BENCH_YAML = """
bench:
  protocols: ["LOSO", "GroupKFold5"]
  ranking:
    primary_metric: "end_to_end.rmse_vol_todos_nodos"
    primary_direction: "minimize"
    tie_breakers:
      - {metric: "classifier.f1", direction: "maximize"}
  promote: "auto"
  families:
    xgboost:
      enabled: true
      classifier: {n_estimators: 200, max_depth: 6}
      regressor: {n_estimators: 200, max_depth: 6}
    svm:
      enabled: false
      classifier: {C: 10.0}
      regressor: {C: 10.0}
"""


def _write_config(tmp_path, extra: str = "") -> str:
    (tmp_path / "network.inp").write_text("[TITLE]\n", encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(_BASE_YAML) + textwrap.dedent(extra), encoding="utf-8")
    return str(path)


def test_config_without_a_bench_block_still_loads(tmp_path):
    """Compatibilidad: el bloque bench es opcional en esta fase."""
    config = load_config(_write_config(tmp_path))
    assert config.bench is None


def test_bench_block_is_parsed(tmp_path):
    config = load_config(_write_config(tmp_path, _BENCH_YAML))

    assert config.bench is not None
    assert config.bench.protocols == ["LOSO", "GroupKFold5"]
    assert config.bench.promote == "auto"


def test_ranking_criterion_is_built_from_the_yaml(tmp_path):
    criterion = load_config(_write_config(tmp_path, _BENCH_YAML)).bench.ranking

    assert criterion.primary_metric == "end_to_end.rmse_vol_todos_nodos"
    assert criterion.primary_direction == "minimize"
    assert criterion.tie_breakers == (("classifier.f1", "maximize"),)


def test_only_enabled_families_are_returned(tmp_path):
    bench = load_config(_write_config(tmp_path, _BENCH_YAML)).bench
    assert bench.enabled_families() == ["xgboost"]


def test_family_hyperparameters_are_available_per_task(tmp_path):
    bench = load_config(_write_config(tmp_path, _BENCH_YAML)).bench
    family = bench.families["xgboost"]

    assert family.classifier["n_estimators"] == 200
    assert family.regressor["max_depth"] == 6


def test_an_unknown_family_name_is_rejected_at_load_time(tmp_path):
    bad = """
bench:
  protocols: ["LOSO"]
  ranking: {primary_metric: "classifier.f1", primary_direction: "maximize", tie_breakers: []}
  promote: "auto"
  families:
    red_neuronal_magica:
      enabled: true
      classifier: {}
      regressor: {}
"""
    with pytest.raises(ValueError, match="Familia desconocida"):
        load_config(_write_config(tmp_path, bad))


def test_an_unknown_protocol_is_rejected_at_load_time(tmp_path):
    bad = """
bench:
  protocols: ["LeaveOneShapeOut"]
  ranking: {primary_metric: "classifier.f1", primary_direction: "maximize", tie_breakers: []}
  promote: "auto"
  families:
    xgboost: {enabled: true, classifier: {}, regressor: {}}
"""
    with pytest.raises(ValueError, match="Protocolo desconocido"):
        load_config(_write_config(tmp_path, bad))


def test_promote_can_name_a_specific_family(tmp_path):
    named = _BENCH_YAML.replace('promote: "auto"', 'promote: "xgboost"')
    assert load_config(_write_config(tmp_path, named)).bench.promote == "xgboost"
```

```python
# tests/ml/bench/test_bench_cli.py
import json

import pytest

from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.promote import promote_candidate
from swmm_resilience.ml.bench.train import train_candidate

TINY = {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3, "subsample": 1.0}


@pytest.fixture
def trained(sql_training_db, tmp_path):
    prepared = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )
    artifacts = train_candidate(prepared, "xgboost", TINY, TINY, tmp_path / "candidates")
    return prepared, artifacts


def test_promote_copies_the_artifacts_to_the_classic_paths(trained, tmp_path):
    """Los siete consumidores aguas abajo leen exactamente estos dos nombres."""
    _, artifacts = trained
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")
    models_dir = tmp_path / "outputs" / "models"

    promote_candidate(artifacts.classifier_path.parent, models_dir, inp_path)

    assert (models_dir / "classifier.joblib").exists()
    assert (models_dir / "regressor.joblib").exists()
    assert (models_dir / "training_inp_hash.txt").exists()


def test_promotion_record_names_the_family_and_prep_id(trained, tmp_path):
    prepared, artifacts = trained
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")

    record = promote_candidate(
        artifacts.classifier_path.parent, tmp_path / "models", inp_path
    )

    assert record["family"] == "xgboost"
    assert record["prep_id"] == prepared.prep_id
    assert len(record["classifier_sha256"]) == 64


def test_the_promoted_classifier_loads_and_predicts(trained, tmp_path):
    import joblib

    prepared, artifacts = trained
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")
    models_dir = tmp_path / "models"

    promote_candidate(artifacts.classifier_path.parent, models_dir, inp_path)
    classifier = joblib.load(models_dir / "classifier.joblib")

    assert len(classifier.predict(prepared.X.head(2))) == 2


def test_promoting_a_directory_without_metadata_fails_clearly(tmp_path):
    empty = tmp_path / "vacio"
    empty.mkdir()
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="metadata.json"):
        promote_candidate(empty, tmp_path / "models", inp_path)


def test_promotion_record_is_written_next_to_the_models(trained, tmp_path):
    _, artifacts = trained
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\n", encoding="utf-8")
    models_dir = tmp_path / "models"

    promote_candidate(artifacts.classifier_path.parent, models_dir, inp_path)

    record = json.loads((models_dir / "promotion.json").read_text(encoding="utf-8"))
    assert record["family"] == "xgboost"


def test_bench_flags_are_registered_in_the_cli():
    """La Task 15 extrae build_parser() de main.py; sin eso este test falla."""
    import main

    assert hasattr(main, "build_parser"), (
        "main.py debe exponer build_parser() para que el parser sea testeable"
    )
    known = {action.dest for action in main.build_parser()._actions}
    assert {"bench_prepare", "bench_train", "bench_evaluate", "bench_promote", "bench"} <= known
    assert {"models", "prep_id"} <= known
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/test_bench_config.py tests/ml/bench/test_bench_cli.py -q`
Expected: FAIL con `AttributeError: 'Config' object has no attribute 'bench'`

- [ ] **Step 3: Write minimal implementation**

Añadir a `swmm_resilience/config.py`, antes de `@dataclass class Config`:

```python
BENCH_PROTOCOLS = ("LOSO", "GroupKFold5")


@dataclass
class BenchFamilyConfig:
    enabled: bool
    classifier: dict
    regressor: dict


@dataclass
class BenchRankingConfig:
    primary_metric: str
    primary_direction: str
    tie_breakers: tuple


@dataclass
class BenchConfig:
    protocols: list
    ranking: BenchRankingConfig
    promote: str
    families: dict

    def enabled_families(self) -> list:
        return [name for name, family in self.families.items() if family.enabled]
```

Añadir el campo al `Config`:

```python
    bench: Optional[BenchConfig] = None
```

Y el parseo en `load_config`, justo antes del `return Config(...)`:

```python
    def _parse_bench(raw_bench: dict | None) -> Optional[BenchConfig]:
        """El bloque bench es opcional mientras conviva con ml:/evaluation:."""
        if not raw_bench:
            return None

        from swmm_resilience.ml.bench.registry import available_families

        protocols = [str(item) for item in raw_bench["protocols"]]
        invalid = [item for item in protocols if item not in BENCH_PROTOCOLS]
        if invalid:
            raise ValueError(
                f"Protocolo desconocido: {', '.join(invalid)}. "
                f"Opciones: {', '.join(BENCH_PROTOCOLS)}"
            )

        known = set(available_families())
        families = {}
        for name, spec in (raw_bench.get("families") or {}).items():
            if name not in known:
                raise ValueError(
                    f"Familia desconocida en config.yaml: {name!r}. "
                    f"Disponibles: {', '.join(sorted(known))}"
                )
            families[name] = BenchFamilyConfig(
                enabled=bool(spec.get("enabled", True)),
                classifier=dict(spec.get("classifier") or {}),
                regressor=dict(spec.get("regressor") or {}),
            )

        ranking_raw = raw_bench["ranking"]
        return BenchConfig(
            protocols=protocols,
            ranking=BenchRankingConfig(
                primary_metric=str(ranking_raw["primary_metric"]),
                primary_direction=str(ranking_raw["primary_direction"]),
                tie_breakers=tuple(
                    (str(item["metric"]), str(item["direction"]))
                    for item in (ranking_raw.get("tie_breakers") or [])
                ),
            ),
            promote=str(raw_bench.get("promote", "auto")),
            families=families,
        )
```

y pasar `bench=_parse_bench(raw.get("bench"))` al constructor de `Config`.

Bloque a añadir a `config.yaml` y a `config_7shapes.yaml`:

```yaml
bench:
  protocols: ["LOSO", "GroupKFold5"]
  ranking:
    primary_metric: "end_to_end.rmse_vol_todos_nodos"
    primary_direction: "minimize"
    tie_breakers:
      - {metric: "classifier.f1", direction: "maximize"}
      - {metric: "regressor_oracle.nse", direction: "maximize"}
  promote: "auto"
  families:
    xgboost:
      enabled: true
      # Justificacion: valores heredados del pipeline validado en 2026-06.
      # Profundidad 6 y lr 0.05 con 200 arboles evitan sobreajuste sobre los
      # 175 escenarios sin perder capacidad en los factores extremos.
      classifier: {n_estimators: 200, max_depth: 6, learning_rate: 0.05, subsample: 0.8}
      regressor: {n_estimators: 200, max_depth: 6, learning_rate: 0.05, subsample: 0.8}
    random_forest:
      enabled: true
      # Justificacion: mismo numero de arboles y profundidad que XGBoost para
      # que la comparacion aisle el metodo (boosting vs bagging), no el tamano.
      classifier: {n_estimators: 200, max_depth: 6}
      regressor: {n_estimators: 200, max_depth: 6}
    linear:
      enabled: true
      # Justificacion: baseline. alpha=1.0 y C=1.0 son los valores por defecto
      # de sklearn; el objetivo es medir cuanto aporta la no-linealidad, no
      # exprimir el modelo lineal.
      classifier: {C: 1.0, max_iter: 5000}
      regressor: {alpha: 1.0}
    svm:
      enabled: true
      # Justificacion: C=10 y kernel RBF son los valores que usaba el stack
      # legacy (ML_MODEL_CONFIGS.svr_rbf), conservados para que los numeros
      # sean comparables con lo ya reportado.
      classifier: {C: 10.0, kernel: "rbf", gamma: "scale"}
      regressor: {C: 10.0, kernel: "rbf", epsilon: 0.1}
    mlp:
      enabled: true
      # Justificacion: dos capas ocultas (64, 32) sobre 17 features es una
      # capacidad holgada sin llegar a memorizar; 200 epocas con Adam a 1e-3
      # convergen en este tamano de dataset. Pendiente de afinar con Optuna.
      classifier: {hidden_sizes: [64, 32], epochs: 200, learning_rate: 0.001, batch_size: 256, dropout: 0.1}
      regressor: {hidden_sizes: [64, 32], epochs: 200, learning_rate: 0.001, batch_size: 256, dropout: 0.1}
```

```python
# swmm_resilience/ml/bench/promote.py
"""Promoción del candidato ganador a las rutas que lee el resto del pipeline.

outputs/models/{classifier,regressor}.joblib + training_inp_hash.txt es el
formato que consumen --predict, --simulate, los mapas, --evaluate-hydrographs,
ml/predict.py, ml/scenario_predict.py, ml/feature_analysis.py y
ml/feature_importance.py. Mantenerlo idéntico es lo que permite retirar el
legacy sin tocar ninguno de ellos.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .train import load_candidate


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote_candidate(
    candidate_dir: Path | str,
    models_dir: Path | str,
    inp_path: Path | str,
) -> dict:
    """Copia los artefactos del candidato a ``models_dir`` en el formato clásico."""
    candidate = load_candidate(candidate_dir)
    target = Path(models_dir)
    target.mkdir(parents=True, exist_ok=True)

    shutil.copy2(candidate.classifier_path, target / "classifier.joblib")
    shutil.copy2(candidate.regressor_path, target / "regressor.joblib")
    (target / "training_inp_hash.txt").write_text(_md5(Path(inp_path)))

    record = {
        "family": candidate.family,
        "prep_id": candidate.metadata["prep_id"],
        "classifier_sha256": candidate.metadata["classifier_sha256"],
        "regressor_sha256": candidate.metadata["regressor_sha256"],
        "source_dir": str(Path(candidate_dir).resolve()),
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (target / "promotion.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return record
```

En `main.py`, extraer la construcción del parser a `build_parser()` (hoy está inline) y añadir:

```python
    parser.add_argument("--bench-prepare", action="store_true",
                        help="Etapa 1 del banco: SQL -> outputs/bench/prepared/")
    parser.add_argument("--bench-train", action="store_true",
                        help="Etapa 2 del banco: entrena cada familia habilitada")
    parser.add_argument("--bench-evaluate", action="store_true",
                        help="Etapa 3 del banco: folds -> OOF -> metricas -> ranking")
    parser.add_argument("--bench-promote", action="store_true",
                        help="Copia el candidato ganador a outputs/models/")
    parser.add_argument("--bench", action="store_true",
                        help="Corre las cuatro etapas del banco en secuencia")
    parser.add_argument("--models", metavar="LISTA", default=None,
                        help="Familias separadas por coma (por defecto: las habilitadas en config.yaml)")
    parser.add_argument("--prep-id", metavar="ID", default=None,
                        help="PreparedDataset concreto (por defecto: el mas reciente)")
```

Y la orquestación, antes de las ramas existentes:

```python
BENCH_ROOT = Path("outputs/bench")


def _run_bench(args, config) -> None:
    from swmm_resilience.ml.bench.evaluate import evaluate_candidate
    from swmm_resilience.ml.bench.preprocess import prepare_dataset, resolve_prepared
    from swmm_resilience.ml.bench.promote import promote_candidate
    from swmm_resilience.ml.bench.ranking import RankingCriterion, rank_candidates
    from swmm_resilience.ml.bench.reports import write_reports
    from swmm_resilience.ml.bench.train import train_candidate

    if config.bench is None:
        raise SystemExit(
            "config.yaml no tiene bloque `bench:`. Añádelo antes de usar los comandos --bench-*."
        )

    prepared_dir = BENCH_ROOT / "prepared"
    candidates_dir = BENCH_ROOT / "candidates"
    reports_dir = BENCH_ROOT / "reports"

    run_all = args.bench
    families = (
        [name.strip() for name in args.models.split(",")]
        if args.models
        else config.bench.enabled_families()
    )

    if run_all or args.bench_prepare:
        prepared = prepare_dataset(
            config.dataset.db_path,
            protocols=tuple(config.bench.protocols),
            flood_threshold_m3=config.dataset.flood_threshold_m3,
            output_dir=prepared_dir,
        )
        print(f"PreparedDataset {prepared.prep_id}: {prepared.quality['n_rows']} filas, "
              f"{prepared.quality['class_balance']['n_flooded']} inundadas")
    else:
        prepared = resolve_prepared(prepared_dir, args.prep_id)

    if run_all or args.bench_train:
        for family in families:
            spec = config.bench.families[family]
            artifacts = train_candidate(
                prepared, family, spec.classifier, spec.regressor, candidates_dir
            )
            print(f"  {family}: {artifacts.classifier_path}")

    ranking = None
    if run_all or args.bench_evaluate:
        metrics_by_family = {}
        for family in families:
            spec = config.bench.families[family]
            _, metrics = evaluate_candidate(prepared, family, spec.classifier, spec.regressor)
            metrics_by_family[family] = metrics
            print(f"  {family} evaluado")

        criterion = RankingCriterion(
            primary_metric=config.bench.ranking.primary_metric,
            primary_direction=config.bench.ranking.primary_direction,
            tie_breakers=config.bench.ranking.tie_breakers,
        )
        ranking = rank_candidates(metrics_by_family, criterion, config.bench.protocols[0])
        write_reports(metrics_by_family, ranking, reports_dir)
        print(ranking.to_string(index=False))

    if run_all or args.bench_promote:
        chosen = config.bench.promote
        if chosen == "auto":
            if ranking is None:
                raise SystemExit(
                    "promote: 'auto' necesita un ranking. Corre --bench-evaluate antes, "
                    "o nombra la familia en config.yaml."
                )
            valid = ranking[ranking["valid"] == 1]
            if valid.empty:
                raise SystemExit("Ningún candidato tiene una métrica primaria válida.")
            chosen = valid.iloc[0]["family"]
        record = promote_candidate(
            candidates_dir / chosen, Path("outputs/models"), config.network.inp_path
        )
        print(f"Promovido: {record['family']} (prep_id={record['prep_id']})")
```

con el despacho `if args.bench or args.bench_prepare or args.bench_train or args.bench_evaluate or args.bench_promote: _run_bench(args, config); return`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/Scripts/python.exe -m pytest tests/ml/bench/ tests/test_config.py tests/test_config_defaults.py -q`
Expected: todo verde. Después, la suite completa: `./venv/Scripts/python.exe -m pytest -q` debe seguir en la línea base re-medida.

- [ ] **Step 5: Verificación manual de punta a punta**

```bash
./venv/Scripts/python.exe main.py --bench-prepare
./venv/Scripts/python.exe main.py --bench-train --models xgboost,linear
./venv/Scripts/python.exe main.py --bench-evaluate --models xgboost,linear
cat outputs/bench/reports/ranking.csv
```

Expected: una tabla ordenada por `rmse_vol_todos_nodos` ascendente, con una fila por familia.

- [ ] **Step 6: Commit**

```bash
git add swmm_resilience/config.py config.yaml config_7shapes.yaml main.py swmm_resilience/ml/bench/promote.py tests/ml/bench/test_bench_config.py tests/ml/bench/test_bench_cli.py
git commit -m "feat(bench): bloque bench en config.yaml y comandos --bench-*"
```

---

## Verificación final del plan

- [ ] **Antes de empezar la Task 1**, re-medir la línea base: `./venv/Scripts/python.exe -m pytest -q` y anotar el número. La última registrada (`531 passed, 3 deselected`) es de HEAD `7a5ce36`, tres commits atrás; sin un número actual no se puede afirmar que nada se rompió.
- [ ] `./venv/Scripts/python.exe -m pytest -q` verde al terminar, con los ~85 tests nuevos de `tests/ml/bench/` sumados a esa línea base.
- [ ] El test de paridad (Task 10) está verde: **es la condición que autoriza el Plan 3.**
- [ ] `outputs/bench/reports/ranking.csv` existe y ordena las cinco familias.
- [ ] `outputs/models/{classifier,regressor}.joblib` se regeneran vía `--bench-promote` y `python main.py --predict --factor 2.0` sigue funcionando sin cambios.
- [ ] Ningún archivo de `ml/train.py`, `ml/trainer.py`, `ml/evaluator.py` o `ml/temporal/` fue modificado en este plan.

## Qué sigue

- **Plan 2** — persistencia SQL completa: cadena `training_runs → model_candidates → model_evaluations → oof_predictions → finalizations → trained_models → rankings → promotions → selections`, más la migración 006 (`model_blob` nullable + `model_path`), y la resolución `(network_id, node_id) → node_pk` en `persist.py`.
- **Plan 3** — retiro del legacy: desacople de `FEATURE_COLS`, bloque 1 (stack A + parte ML de la GUI), bloque 2 (stack B, con la paridad verde), y documentación (`docs/ML_BENCH.md`, `FLUJO_ACTUAL.md`, `COMANDOS.md`).
