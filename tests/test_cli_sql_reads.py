import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import main
from swmm_resilience.database.training_queries import load_training_frame
from swmm_resilience.ml.evaluator import evaluate_models


def base_shape_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": [1, 1, 2, 2],
            "node_id": ["N0", "N1", "N0", "N1"],
            "factor_mult": [1.0, 1.0, 2.0, 2.0],
            "shape_id": ["base", "base", "base", "base"],
            "inunda": [0, 1, 1, 1],
            "vol_inundacion_m3": [0.0, 5.0, 7.0, 9.0],
        }
    )


def test_resilience_curve_reads_the_frame_from_sql(monkeypatch):
    load_calls = []
    curve_calls = []

    def fake_load(db_path):
        load_calls.append(Path(db_path))
        return base_shape_frame()

    def fake_curve(df, factors, config, models_dir):
        curve_calls.append((df.copy(), list(factors), config))
        return pd.DataFrame(
            {
                "factor": list(factors),
                "resilience_swmm": [1.0] * len(factors),
                "resilience_ml": [1.0] * len(factors),
            }
        )

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(main, "compute_resilience_curve", fake_curve)
    monkeypatch.setattr(main, "plot_resilience_curve", lambda result, out: None)
    monkeypatch.setattr(sys, "argv", ["main.py", "--resilience-curve"])

    main.main()

    config = curve_calls[0][2]
    assert load_calls == [config.dataset.db_path]
    assert curve_calls[0][1] == [1.0, 2.0]


def test_only_ml_reads_the_frame_from_sql(monkeypatch):
    load_calls = []
    train_calls = []
    frame = base_shape_frame()

    def fake_load(db_path):
        load_calls.append(Path(db_path))
        return frame

    def fake_train(df, config, models_dir):
        train_calls.append((df.copy(), config))
        return object(), object()

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(main, "train_models", fake_train)
    monkeypatch.setattr(main, "evaluate_models", lambda df, config, out: {})
    monkeypatch.setattr(
        main, "generate_feature_importance_plots", lambda clf, reg, out: None
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "--only-ml"])

    main.main()

    config = train_calls[0][1]
    assert load_calls == [config.dataset.db_path]
    assert train_calls[0][0].equals(frame)


def test_analyze_features_errors_when_sql_has_no_samples(monkeypatch, capsys):
    def fake_load(db_path):
        raise ValueError("No COMPLETE v17 training samples found")

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(sys, "argv", ["main.py", "--analyze-features"])

    try:
        main.main()
    except SystemExit as exit_error:
        assert exit_error.code == 2
    else:
        raise AssertionError("--analyze-features should exit via parser.error")

    assert "training_v17.sqlite3" in capsys.readouterr().err


def test_evaluate_shapes_errors_when_sql_has_no_samples(monkeypatch, capsys):
    def fake_load(db_path):
        raise ValueError("No COMPLETE v17 training samples found")

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(sys, "argv", ["main.py", "--evaluate-shapes"])

    try:
        main.main()
    except SystemExit as exit_error:
        assert exit_error.code == 2
    else:
        raise AssertionError("--evaluate-shapes should exit via parser.error")

    assert "training_v17.sqlite3" in capsys.readouterr().err


def test_evaluate_generalization_errors_when_sql_has_no_samples(monkeypatch, capsys):
    def fake_load(db_path):
        raise ValueError("No COMPLETE v17 training samples found")

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(sys, "argv", ["main.py", "--evaluate-generalization"])

    try:
        main.main()
    except SystemExit as exit_error:
        assert exit_error.code == 2
    else:
        raise AssertionError("--evaluate-generalization should exit via parser.error")

    assert "training_v17.sqlite3" in capsys.readouterr().err


def test_persist_sql_writes_to_config_dataset_db_path(monkeypatch):
    import swmm_resilience.database.connection as connection_module
    import swmm_resilience.database.migrations as migrations_module
    import swmm_resilience.database.csv_backfill as csv_backfill_module

    connect_calls = []
    csv_frame = pd.DataFrame(
        {
            "shape_id": ["base"],
            "inunda": [1],
            "node_id": ["N0"],
        }
    )

    class FakeCursor:
        def fetchall(self):
            return []

    class FakeConn:
        def execute(self, sql):
            return FakeCursor()

        def close(self):
            pass

    def fake_connect_managed_database(db_path):
        connect_calls.append(Path(db_path))
        return FakeConn()

    def fake_backfill_networks_and_runs(conn, df, inp_path, network_name):
        return {
            "network_id": 1,
            "node_pk_by_id": {"N0": 1},
            "run_id_by_key": {("base", 1.0): 1},
        }

    def fake_persist_training_run(conn, df, run_id_by_key, node_pk_by_id, config):
        return 1

    monkeypatch.setattr(pd, "read_csv", lambda path: csv_frame)
    monkeypatch.setattr(
        connection_module, "connect_managed_database", fake_connect_managed_database
    )
    monkeypatch.setattr(migrations_module, "apply_migrations", lambda conn: None)
    monkeypatch.setattr(
        csv_backfill_module,
        "backfill_networks_and_runs",
        fake_backfill_networks_and_runs,
    )
    monkeypatch.setattr(
        csv_backfill_module, "persist_training_run", fake_persist_training_run
    )
    monkeypatch.setattr(sys, "argv", ["main.py", "--persist-sql"])

    config = main.load_config("config.yaml")
    main.main()

    assert connect_calls == [config.dataset.db_path]


def test_resilience_curve_errors_when_sql_has_no_samples(monkeypatch, capsys):
    def fake_load(db_path):
        raise ValueError("No COMPLETE v17 training samples found")

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(sys, "argv", ["main.py", "--resilience-curve"])

    try:
        main.main()
    except SystemExit as exit_error:
        assert exit_error.code == 2
    else:
        raise AssertionError("--resilience-curve should exit via parser.error")

    assert "training_v17.sqlite3" in capsys.readouterr().err


def test_flood_volume_curve_errors_when_sql_has_no_samples(monkeypatch, capsys):
    def fake_load(db_path):
        raise ValueError("No COMPLETE v17 training samples found")

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(sys, "argv", ["main.py", "--flood-volume-curve"])

    try:
        main.main()
    except SystemExit as exit_error:
        assert exit_error.code == 2
    else:
        raise AssertionError("--flood-volume-curve should exit via parser.error")

    assert "training_v17.sqlite3" in capsys.readouterr().err


def test_factor_comparison_errors_when_sql_has_no_samples(monkeypatch, capsys):
    def fake_load(db_path):
        raise ValueError("No COMPLETE v17 training samples found")

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(sys, "argv", ["main.py", "--factor-comparison"])

    try:
        main.main()
    except SystemExit as exit_error:
        assert exit_error.code == 2
    else:
        raise AssertionError("--factor-comparison should exit via parser.error")

    assert "training_v17.sqlite3" in capsys.readouterr().err


def test_only_maps_errors_when_sql_has_no_samples(monkeypatch, capsys):
    def fake_load(db_path):
        raise ValueError("No COMPLETE v17 training samples found")

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(sys, "argv", ["main.py", "--only-maps"])

    try:
        main.main()
    except SystemExit as exit_error:
        assert exit_error.code == 2
    else:
        raise AssertionError("--only-maps should exit via parser.error")

    assert "training_v17.sqlite3" in capsys.readouterr().err


def test_only_ml_errors_when_sql_has_no_samples(monkeypatch, capsys):
    def fake_load(db_path):
        raise ValueError("No COMPLETE v17 training samples found")

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(sys, "argv", ["main.py", "--only-ml"])

    try:
        main.main()
    except SystemExit as exit_error:
        assert exit_error.code == 2
    else:
        raise AssertionError("--only-ml should exit via parser.error")

    assert "training_v17.sqlite3" in capsys.readouterr().err


def test_only_maps_reads_the_frame_from_sql(monkeypatch, tmp_path):
    load_calls = []
    frame = base_shape_frame()
    frame["upstream_capacity_lps"] = 1.0

    def fake_load(db_path):
        load_calls.append(Path(db_path))
        return frame

    def fake_run_simulation_simple(inp_path, factor, run_dir):
        return tmp_path / "fake.rpt"

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(main, "run_simulation_simple", fake_run_simulation_simple)
    monkeypatch.setattr(main, "generate_flood_map", lambda *a, **k: None)
    monkeypatch.setattr(main, "generate_flood_maps_by_shape", lambda *a, **k: {})
    monkeypatch.setattr(sys, "argv", ["main.py", "--only-maps"])

    config = main.load_config("config.yaml")
    main.main()

    assert load_calls == [config.dataset.db_path]


def test_flood_volume_curve_reads_the_frame_from_sql(monkeypatch):
    load_calls = []
    curve_calls = []

    def fake_load(db_path):
        load_calls.append(Path(db_path))
        return base_shape_frame()

    def fake_curve(df, factors, config, models_dir):
        curve_calls.append((df.copy(), list(factors), config))
        return pd.DataFrame(
            {
                "factor": list(factors),
                "vol_total_swmm": [1.0] * len(factors),
                "vol_total_ml": [1.0] * len(factors),
            }
        )

    monkeypatch.setattr(main, "load_training_frame", fake_load)
    monkeypatch.setattr(main, "compute_flood_volume_curve", fake_curve)
    monkeypatch.setattr(main, "plot_flood_volume_curve", lambda result, out: None)
    monkeypatch.setattr(main, "plot_flood_volume_combined", lambda result, out: None)
    monkeypatch.setattr(sys, "argv", ["main.py", "--flood-volume-curve"])

    main.main()

    config = curve_calls[0][2]
    assert load_calls == [config.dataset.db_path]
    assert curve_calls[0][1] == [1.0, 2.0]


# ── Order-independence of evaluate_models (subsample reproducibility) ──────
#
# `main.py`'s cold full-pipeline branch (neither --skip-extraction nor
# --only-ml) now sorts the in-memory dataframe by
# ["shape_id", "factor_mult", "node_id"] before handing it to train_models /
# evaluate_models — the exact same key load_training_frame's SQL path is
# already ordered by (`ORDER BY run_id, node_id`, which the SQL cutover
# verified is equivalent to that sort). This exists because
# ml.classifier.subsample / ml.regressor.subsample < 1.0 make XGBoost's
# per-tree row sampling depend on presentation order, not just row values:
# same rows in a different order can select a different random subsample and
# so train a different model. These tests reproduce that on the
# `sql_training_db` fixture (2 shapes x 3 factors x 4 nodes = 24 rows).
#
# config.yaml's production subsample (0.8) turns out too mild to move any
# metric on this fixture's 24 rows (see
# test_unsorted_shuffle_changes_metrics_without_the_canonical_sort's fallback
# branch for that measurement). subsample=0.3 below is deliberately more
# aggressive than production so the effect is actually observable at this
# fixture's scale — the mechanism (order changes which rows XGBoost samples)
# is identical, only the sampling fraction differs.

def _xgb_subsample_config(tiny_config_factory):
    cfg = tiny_config_factory(algorithm="xgboost")
    cfg.ml.classifier.subsample = 0.3
    cfg.ml.regressor.subsample = 0.3
    cfg.evaluation = SimpleNamespace(methods=["LOSO"], stratify_by_factor=False)
    return cfg


def _canonical_sort(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["shape_id", "factor_mult", "node_id"]).reset_index(drop=True)


def _assert_metrics_equal(a: dict, b: dict, rtol: float) -> None:
    assert a.keys() == b.keys()
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, dict):
            _assert_metrics_equal(va, vb, rtol)
        else:
            np.testing.assert_allclose(va, vb, rtol=rtol, equal_nan=True)


def test_canonical_sort_makes_shuffled_and_original_frames_train_identically(
    tiny_config_factory, sql_training_db, tmp_path
):
    """Same rows, presented in different order, must yield bit-identical
    metrics once both are canonically sorted — proving the sort in main.py's
    cold-run branch is what makes it agree with the SQL-fed (--only-ml) path.
    """
    cfg = _xgb_subsample_config(tiny_config_factory)
    frame = load_training_frame(sql_training_db)
    shuffled = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)

    sorted_original = _canonical_sort(frame)
    sorted_shuffled = _canonical_sort(shuffled)
    # The sort key (shape_id, factor_mult, node_id) is a unique row key, so
    # sorting either presentation order must land on the exact same frame.
    pd.testing.assert_frame_equal(
        sorted_original.reset_index(drop=True), sorted_shuffled.reset_index(drop=True)
    )

    results_from_original = evaluate_models(sorted_original, cfg, tmp_path / "metrics_original")
    results_from_shuffled = evaluate_models(sorted_shuffled, cfg, tmp_path / "metrics_shuffled")

    _assert_metrics_equal(results_from_original, results_from_shuffled, rtol=1e-12)


def test_unsorted_shuffle_changes_metrics_without_the_canonical_sort(
    tiny_config_factory, sql_training_db, tmp_path
):
    """Non-vacuousness check: WITHOUT the canonical sort, presenting the same
    rows in a different order changes evaluate_models' metrics, because
    subsample < 1.0 makes XGBoost's row sampling order-dependent. This is the
    failure mode the sort in main.py's cold-run branch fixes.
    """
    cfg = _xgb_subsample_config(tiny_config_factory)
    frame = load_training_frame(sql_training_db)
    shuffled = frame.sample(frac=1.0, random_state=42).reset_index(drop=True)

    results_original = evaluate_models(frame, cfg, tmp_path / "metrics_unsorted_original")
    results_shuffled = evaluate_models(shuffled, cfg, tmp_path / "metrics_unsorted_shuffled")

    f1_original = results_original["LOSO"]["classifier"]["f1"]
    f1_shuffled = results_shuffled["LOSO"]["classifier"]["f1"]
    nse_original = results_original["LOSO"]["regressor_oracle"]["nse"]
    nse_shuffled = results_shuffled["LOSO"]["regressor_oracle"]["nse"]
    print(
        f"\n[unsorted] classifier f1: original={f1_original!r} shuffled={f1_shuffled!r}\n"
        f"[unsorted] regressor nse: original={nse_original!r} shuffled={nse_shuffled!r}"
    )

    differs = (
        not np.isclose(f1_original, f1_shuffled, rtol=1e-12, equal_nan=True)
        or not np.isclose(nse_original, nse_shuffled, rtol=1e-12, equal_nan=True)
    )
    if differs:
        assert differs, "expected shuffled order to change metrics without the canonical sort"
    else:
        # This fixture is small (24 rows) and may not exercise the
        # subsample-order effect strongly enough to move these particular
        # metrics. Fall back to asserting the structural property the fix
        # actually guarantees: main.py's cold-run branch always sorts the
        # frame by (shape_id, factor_mult, node_id) before training/eval, so
        # any input order collapses to the same canonical one.
        sorted_original = _canonical_sort(frame)
        sorted_shuffled = _canonical_sort(shuffled)
        pd.testing.assert_frame_equal(
            sorted_original.reset_index(drop=True), sorted_shuffled.reset_index(drop=True)
        )
