import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.ml.bench.prepared_factory import make_prepared


def _rows(shapes=("A", "B")):
    rows = []
    for shape, (dur, peak) in zip(shapes, ((1.0, 0.5), (3.0, 1.5), (6.0, 2.0))):
        for factor in (1.0, 2.0):
            for node in ("N0", "N1"):
                rows.append(dict(shape_id=shape, node_id=node, factor_mult=factor, vol=5.0 * factor,
                                 duracion_horas=dur, tiempo_al_pico_h=peak))
    return rows


def test_shape_descriptors_are_one_row_per_shape_with_the_two_descriptors():
    from scripts.ejecutar_evaluacion_v4_21 import shape_descriptors_from_frames

    prepared = make_prepared(_rows(), ["LeaveOneShapeOut"])
    table = shape_descriptors_from_frames(prepared.keys, prepared.X)

    assert list(table.columns) == ["duracion_horas", "tiempo_al_pico_h"]
    assert table.index.name == "shape_id"
    assert table.loc["A"].tolist() == [1.0, 0.5]
    assert table.loc["B"].tolist() == [3.0, 1.5]


def test_shape_descriptors_reject_a_descriptor_that_varies_within_a_shape():
    from scripts.ejecutar_evaluacion_v4_21 import shape_descriptors_from_frames

    prepared = make_prepared(_rows(), ["LeaveOneShapeOut"])
    X = prepared.X.copy()
    X.loc[0, "duracion_horas"] = 99.0

    with pytest.raises(ValueError, match="not constant"):
        shape_descriptors_from_frames(prepared.keys, X)


def test_shape_check_rejects_a_different_set_of_shapes():
    from scripts.ejecutar_evaluacion_v4_21 import check_same_shapes, shape_descriptors_from_frames

    source = make_prepared(_rows(("A", "B")), ["LeaveOneShapeOut"])
    target = make_prepared(_rows(("A", "C")), ["LeaveOneShapeOut"])
    table = shape_descriptors_from_frames(source.keys, source.X)

    check_same_shapes(table, source.keys)  # mismo conjunto: no falla
    with pytest.raises(ValueError, match="shapes"):
        check_same_shapes(table, target.keys)


@pytest.mark.parametrize(
    "relative",
    [
        "outputs/models/classifier.joblib",
        "outputs/bench/reports/metrics_xgboost.json",
        "outputs/bench/prepared",
        "outputs/bench/prepared/2db0578f875526cb",
        "outputs/training_v17.sqlite3",
        "data/hydrographs",
    ],
)
def test_protected_paths_are_refused(tmp_path, relative):
    from scripts.ejecutar_evaluacion_v4_21 import ensure_allowed_output

    with pytest.raises(ValueError, match="Protected path"):
        ensure_allowed_output(tmp_path / relative, repo_root=tmp_path)


@pytest.mark.parametrize(
    "relative",
    [
        "outputs/bench/prepared_v4_21",
        "outputs/bench/oof",
        "docs/auditoria/correcciones/datos/runs",
    ],
)
def test_run_outputs_are_allowed(tmp_path, relative):
    from scripts.ejecutar_evaluacion_v4_21 import ensure_allowed_output

    ensure_allowed_output(tmp_path / relative, repo_root=tmp_path)


def test_provenance_records_dataset_params_versions_and_timings():
    from scripts.ejecutar_evaluacion_v4_21 import build_provenance

    # Malla 2 formas x 2 factores con un hueco: falta el run (B, 2.0).
    rows = [row for row in _rows() if not (row["shape_id"] == "B" and row["factor_mult"] == 2.0)]
    prepared = make_prepared(rows, ["LeaveOneShapeOut"])
    params = {"classifier": {"n_estimators": 200}, "regressor": {"n_estimators": 200}}

    record = build_provenance(
        prepared,
        contract_id="tabular_v4_21",
        flood_threshold_m3=1.0,
        max_continuity_error_pct=5.0,
        xgboost_params=params,
        baselines=["baseline_interpolacion_factor"],
        shape_descriptors_source="outputs/bench/prepared/x",
        stage_seconds={"prepare": 1.5, "xgboost": 2.0},
        git_commit="abc123",
        git_dirty=True,
    )

    assert record["n_runs"] == 3
    assert record["n_runs_full_grid"] == 4
    assert record["n_missing_cells"] == 1
    assert record["git_dirty"] is True
    assert record["prep_id"] == prepared.prep_id
    assert record["feature_contract_id"] == "tabular_v4_21"
    assert record["n_features"] == prepared.X.shape[1]
    assert record["n_rows"] == len(prepared.keys)
    assert record["flood_threshold_m3"] == 1.0
    assert record["max_continuity_error_pct"] == 5.0
    assert record["xgboost_params"] == params
    assert record["git_commit"] == "abc123"
    assert record["stage_seconds"] == {"prepare": 1.5, "xgboost": 2.0}
    assert {"python", "xgboost", "scikit-learn", "pandas", "numpy"} <= set(record["versions"])
    assert "config.yaml" in record["xgboost_params_note"]
    assert "Optuna" in record["xgboost_params_note"]
    assert record["protocols"] == ["LeaveOneShapeOut"]
    assert record["complete"] is False
    assert record["baseline_fallback"] == {}


def test_provenance_records_baseline_fallback_counters_and_completion():
    from scripts.ejecutar_evaluacion_v4_21 import build_provenance

    prepared = make_prepared(_rows(), ["LeaveOneShapeOut"])
    counters = {"baseline_forma_mas_cercana": {"n_rows": 8, "fallback_rows": 2, "fallback_share": 0.25}}

    record = build_provenance(
        prepared,
        contract_id="tabular_v4_21",
        flood_threshold_m3=1.0,
        max_continuity_error_pct=5.0,
        xgboost_params={},
        baselines=["baseline_forma_mas_cercana"],
        shape_descriptors_source="x",
        stage_seconds={},
        git_commit="abc",
        baseline_fallback=counters,
        complete=True,
    )

    assert record["baseline_fallback"] == counters
    assert record["complete"] is True


def test_baseline_counters_read_the_oof_attrs_and_compute_the_share():
    from scripts.ejecutar_evaluacion_v4_21 import baseline_counters

    oof = pd.DataFrame({"sample_idx": range(4)})
    oof.attrs["n_rows"] = 4
    oof.attrs["fallback_rows"] = 1

    assert baseline_counters(oof) == {"n_rows": 4, "fallback_rows": 1, "fallback_share": 0.25}


def test_baseline_counters_fail_without_the_attrs():
    from scripts.ejecutar_evaluacion_v4_21 import baseline_counters

    with pytest.raises(ValueError, match="fallback_rows"):
        baseline_counters(pd.DataFrame({"sample_idx": [0]}))


def test_persist_stage_writes_the_oof_parquet_and_the_metrics_json(tmp_path):
    import json

    from scripts.ejecutar_evaluacion_v4_21 import persist_stage
    from tests.ml.bench.prepared_factory import perfect_oof

    prepared = make_prepared(_rows(), ["LeaveOneShapeOut"])
    oof = perfect_oof(prepared)

    paths = persist_stage(
        "xgboost", oof, {"LeaveOneShapeOut": {"rmse": float("nan")}}, prepared.prep_id,
        oof_root=tmp_path / "oof", runs_dir=tmp_path / "runs",
    )

    assert paths["oof"] == tmp_path / "oof" / prepared.prep_id / "xgboost.parquet"
    assert paths["metrics"] == tmp_path / "runs" / "metrics_xgboost.json"
    assert len(pd.read_parquet(paths["oof"])) == len(oof)
    assert json.loads(paths["metrics"].read_text(encoding="utf-8")) == {
        "metrics_definition": "classifier_pooled_v2",
        "prep_id": prepared.prep_id,
        "LeaveOneShapeOut": {"rmse": None},
    }
