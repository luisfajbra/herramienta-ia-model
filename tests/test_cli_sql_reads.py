import sys
from pathlib import Path

import pandas as pd

import main


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
