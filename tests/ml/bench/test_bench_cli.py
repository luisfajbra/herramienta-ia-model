import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from swmm_resilience.config import (
    BenchConfig,
    BenchFamilyConfig,
    BenchRankingConfig,
    BenchTuningConfig,
    Config,
    DatasetConfig,
    NetworkConfig,
)
from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.promote import promote_candidate
from swmm_resilience.ml.bench.train import train_candidate

TINY = {"n_estimators": 5, "max_depth": 2, "learning_rate": 0.3, "subsample": 1.0}


def _make_config(
    *, inp_path, db_path, families, promote="auto", protocols=("LOSO",), tuning=None
):
    """Builds a Config with only the fields _run_bench actually reads.

    ``simulation``/``ml``/``evaluation``/``visualization`` are left as None:
    _run_bench never touches them, and Config is a plain dataclass with no
    runtime type enforcement.
    """
    family_configs = {
        name: BenchFamilyConfig(enabled=True, classifier=dict(TINY), regressor=dict(TINY))
        for name in families
    }
    bench = BenchConfig(
        protocols=list(protocols),
        ranking=BenchRankingConfig(
            primary_metric="end_to_end.rmse_vol_todos_nodos",
            primary_direction="minimize",
            tie_breakers=(("classifier.f1", "maximize"),),
        ),
        promote=promote,
        families=family_configs,
        tuning=tuning or BenchTuningConfig(n_trials=2, n_splits=5),
    )
    return Config(
        network=NetworkConfig(inp_path=inp_path, name="fixture"),
        simulation=None,
        dataset=DatasetConfig(output_path=Path("unused.csv"), flood_threshold_m3=1.0, db_path=db_path),
        ml=None,
        evaluation=None,
        visualization=None,
        bench=bench,
    )


def _make_args(**overrides):
    base = dict(
        bench=False,
        bench_prepare=False,
        bench_train=False,
        bench_evaluate=False,
        bench_promote=False,
        bench_tune=False,
        models=None,
        prep_id=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


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
    hash_path = models_dir / "training_inp_hash.txt"
    assert hash_path.exists()

    stored = hash_path.read_text(encoding="utf-8").strip()
    expected_md5 = hashlib.md5(inp_path.read_bytes()).hexdigest()
    assert len(stored) == 32, (
        "training_inp_hash.txt debe contener un digest MD5 (32 caracteres hex); "
        f"got {len(stored)} caracteres -- ml/predict.py y el resto de consumidores "
        "comparan esto contra un MD5 recien calculado del .inp."
    )
    assert stored == expected_md5


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


class _StubModelWithFeatureCount:
    """Stands in for a fitted sklearn Pipeline -- the guard only reads
    n_features_in_, so this doesn't need to be a real fitted model."""

    def __init__(self, n_features_in_):
        self.n_features_in_ = n_features_in_


def test_promote_refuses_to_overwrite_a_mismatched_contract_model(trained, tmp_path):
    """F1: promote_candidate() must refuse to silently downgrade an active
    model promoted under a different feature contract (same guard family as
    trainer.py::train_models(), reproducing the reviewer's live scenario:
    a 17-feature bench candidate promoted over a 21-feature active model)."""
    import joblib

    _, artifacts = trained
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    # Active model has 21 features; the candidate (the `trained` fixture,
    # per train.py's hardcoded v17 metadata) has 17 -- a real mismatch.
    joblib.dump(_StubModelWithFeatureCount(21), models_dir / "classifier.joblib")
    joblib.dump(_StubModelWithFeatureCount(21), models_dir / "regressor.joblib")
    clf_bytes_before = (models_dir / "classifier.joblib").read_bytes()
    reg_bytes_before = (models_dir / "regressor.joblib").read_bytes()

    with pytest.raises(RuntimeError, match=r"21.*17|17.*21"):
        promote_candidate(artifacts.classifier_path.parent, models_dir, inp_path)

    assert (models_dir / "classifier.joblib").read_bytes() == clf_bytes_before
    assert (models_dir / "regressor.joblib").read_bytes() == reg_bytes_before


def test_promote_still_works_when_models_dir_is_empty(trained, tmp_path):
    """The new guard must not break the normal --bench-promote flow when
    there is nothing promoted yet."""
    _, artifacts = trained
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")
    models_dir = tmp_path / "models"  # does not exist yet

    record = promote_candidate(artifacts.classifier_path.parent, models_dir, inp_path)

    assert (models_dir / "classifier.joblib").exists()
    assert record["family"] == "xgboost"


def test_promote_still_works_when_the_contract_matches(trained, tmp_path):
    """Re-promoting over a model from the SAME contract (e.g. re-running
    --bench-promote after a fresh bench cycle) must keep working."""
    _, artifacts = trained
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")
    models_dir = tmp_path / "models"

    # First promotion establishes the active (17-feature) model...
    promote_candidate(artifacts.classifier_path.parent, models_dir, inp_path)
    # ...a second promotion of the SAME contract must not be refused.
    record = promote_candidate(artifacts.classifier_path.parent, models_dir, inp_path)

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


# ── _run_bench: actually invoking the orchestrator ─────────────────────────


def test_config_bench_none_gives_the_clear_systemexit_not_an_attributeerror(tmp_path):
    import main

    config = _make_config(
        inp_path=tmp_path / "network.inp", db_path=tmp_path / "unused.sqlite3",
        families=["xgboost"],
    )
    config.bench = None
    args = _make_args(bench_prepare=True)

    with pytest.raises(SystemExit, match="no `bench:` block"):
        main._run_bench(args, config)


def test_unknown_family_in_models_gives_a_clear_error_naming_it(
    sql_training_db, tmp_path, monkeypatch
):
    """`--models` reaches config.bench.families[family] in the train/evaluate
    loops; an unknown name there must fail with a named SystemExit, not a
    bare KeyError. Uses --bench-train (not --bench-prepare) because the
    prepare stage never indexes config.bench.families at all -- it would
    let a typo through silently and prove nothing."""
    import main

    monkeypatch.chdir(tmp_path)
    prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=Path("outputs/bench/prepared"),
    )

    config = _make_config(
        inp_path=tmp_path / "network.inp", db_path=sql_training_db,
        families=["xgboost"],
    )
    args = _make_args(bench_train=True, models="xgbost")

    with pytest.raises(SystemExit) as excinfo:
        main._run_bench(args, config)

    message = str(excinfo.value)
    assert "xgbost" in message, "el mensaje debe nombrar la familia desconocida"
    assert "xgboost" in message, "el mensaje debe listar las familias que sí declara config.yaml"


def test_a_disabled_family_named_in_models_is_not_rejected_by_the_new_check(
    sql_training_db, tmp_path, monkeypatch
):
    """--models nombrando una familia disabled=False en config.yaml debe seguir
    funcionando (semántica de override intencional); solo un nombre ausente del
    dict de familias debe fallar."""
    import main

    config = _make_config(
        inp_path=tmp_path / "network.inp", db_path=sql_training_db,
        families=["xgboost"],
    )
    config.bench.families["xgboost"].enabled = False
    monkeypatch.chdir(tmp_path)
    args = _make_args(bench_prepare=True, models="xgboost")

    # Must not raise the unknown-family SystemExit; prepare stage runs fine.
    main._run_bench(args, config)
    assert (tmp_path / "outputs" / "bench" / "prepared").exists()


def test_bench_promote_none_requires_an_explicit_family(
    sql_training_db, tmp_path
):
    import main

    config = _make_config(
        inp_path=tmp_path / "network.inp", db_path=sql_training_db,
        families=["xgboost"], promote="none",
    )
    args = _make_args(bench_promote=True)

    with pytest.raises(SystemExit, match="explicit family"):
        main._run_bench(args, config)


def test_bench_promote_alone_promotes_without_a_prepared_directory(
    sql_training_db, tmp_path, monkeypatch
):
    """Finding 1: promote does not need outputs/bench/prepared/ at all."""
    import main

    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")

    # Train a candidate directly at the relative path _run_bench expects
    # (BENCH_ROOT = Path("outputs/bench")), from an ordinary tmp_path
    # PreparedDataset -- this stands in for a prior --bench-train run.
    prepared = prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=tmp_path / "unrelated_prepared_store",
    )
    train_candidate(
        prepared, "xgboost", TINY, TINY, tmp_path / "outputs" / "bench" / "candidates"
    )

    monkeypatch.chdir(tmp_path)
    assert not (tmp_path / "outputs" / "bench" / "prepared").exists()

    config = _make_config(
        inp_path=inp_path, db_path=sql_training_db,
        families=["xgboost"], promote="xgboost",
    )
    args = _make_args(bench_promote=True)

    main._run_bench(args, config)

    models_dir = tmp_path / "outputs" / "models"
    assert (models_dir / "classifier.joblib").exists()
    assert (models_dir / "regressor.joblib").exists()
    assert (models_dir / "training_inp_hash.txt").exists()
    assert not (tmp_path / "outputs" / "bench" / "prepared").exists(), (
        "promote must not require or create a prepared/ directory"
    )


def test_bench_evaluate_fails_clearly_when_the_prepared_dataset_lacks_the_configured_protocol(
    sql_training_db, tmp_path, monkeypatch
):
    """resolve_prepared picks the newest PreparedDataset by mtime alone, with
    no check that it actually covers what config.yaml asks for. Without the
    protocol-coverage guard, changing bench.protocols and running
    --bench-evaluate alone silently degrades every candidate to valid=0
    ("el candidato no tiene resultados para X") instead of failing clearly.
    """
    import main

    monkeypatch.chdir(tmp_path)
    # Prepared with only LOSO, but config below asks for GroupKFold5 too.
    prepare_dataset(
        sql_training_db, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=Path("outputs/bench/prepared"),
    )

    config = _make_config(
        inp_path=tmp_path / "network.inp", db_path=sql_training_db,
        families=["xgboost"], protocols=("LOSO", "GroupKFold5"),
    )
    args = _make_args(bench_evaluate=True)

    with pytest.raises(SystemExit) as excinfo:
        main._run_bench(args, config)

    message = str(excinfo.value)
    assert "GroupKFold5" in message
    assert "--bench-prepare" in message


def test_bench_train_rejects_a_prepared_dataset_with_stale_protocol_semantics(
    sql_training_db, tmp_path, monkeypatch
):
    import main

    monkeypatch.chdir(tmp_path)
    prepared = prepare_dataset(
        sql_training_db,
        protocols=("LOSO",),
        flood_threshold_m3=1.0,
        output_dir=Path("outputs/bench/prepared"),
    )
    manifest_path = (
        Path("outputs/bench/prepared") / prepared.prep_id / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol_definitions"]["LOSO"]["group_column"] = "shape_id"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    config = _make_config(
        inp_path=tmp_path / "network.inp",
        db_path=sql_training_db,
        families=["xgboost"],
        protocols=("LOSO",),
    )

    with pytest.raises(SystemExit) as excinfo:
        main._run_bench(_make_args(bench_train=True), config)

    message = str(excinfo.value)
    assert "semantics" in message
    assert "protocol" in message
    assert "--bench-prepare" in message


def test_bench_train_succeeds_when_the_prepared_dataset_covers_the_configured_protocols(
    sql_training_db_six_factors, tmp_path, monkeypatch
):
    """The guard must not be a false positive: when the PreparedDataset does
    cover every configured protocol, --bench-train proceeds normally."""
    import main

    monkeypatch.chdir(tmp_path)
    prepare_dataset(
        sql_training_db_six_factors, protocols=("LOSO", "GroupKFold5"), flood_threshold_m3=1.0,
        output_dir=Path("outputs/bench/prepared"),
    )

    config = _make_config(
        inp_path=tmp_path / "network.inp", db_path=sql_training_db_six_factors,
        families=["xgboost"], protocols=("LOSO", "GroupKFold5"),
    )
    args = _make_args(bench_train=True)

    main._run_bench(args, config)  # must not raise


def test_full_bench_run_writes_ranking_without_promoting(
    sql_training_db, tmp_path, monkeypatch
):
    import main

    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    config = _make_config(
        inp_path=inp_path, db_path=sql_training_db,
        families=["xgboost"], promote="none",
    )
    args = _make_args(bench=True, models="xgboost")

    main._run_bench(args, config)

    ranking_csv = tmp_path / "outputs" / "bench" / "reports" / "ranking.csv"
    assert ranking_csv.exists()
    ranking = pd.read_csv(ranking_csv)
    assert len(ranking) == 1
    assert set(ranking["family"]) == {"xgboost"}
    for column in ("rank", "family", "primary_value", "valid", "invalid_reason"):
        assert column in ranking.columns

    assert not (tmp_path / "outputs" / "models").exists()
    assert not (tmp_path / "outputs" / "bench" / "tuning").exists()


def test_bench_evaluate_persists_oof_baselines_and_provenance(
    sql_training_db, tmp_path, monkeypatch
):
    import main

    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config = _make_config(
        inp_path=inp_path, db_path=sql_training_db,
        families=["xgboost"], promote="none",
    )

    main._run_bench(_make_args(bench=True, models="xgboost"), config)

    bench_root = tmp_path / "outputs" / "bench"
    prep_id = next((bench_root / "prepared").iterdir()).name
    assert (bench_root / "oof" / prep_id / "xgboost.parquet").exists()
    assert (bench_root / "oof" / prep_id / "baseline_interpolacion_factor.parquet").exists()
    assert (bench_root / "reports" / "metrics_baseline_interpolacion_factor.json").exists()
    # LOSO por factor: la referencia por forma no aplica y se omite
    assert not (bench_root / "oof" / prep_id / "baseline_forma_mas_cercana.parquet").exists()

    provenance = json.loads((bench_root / "reports" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["prep_id"] == prep_id
    assert provenance["families"]["xgboost"]["classifier"] == TINY
    assert provenance["baselines"] == ["baseline_interpolacion_factor"]
    assert not (tmp_path / "outputs" / "models").exists()


# ── --bench-tune ─────────────────────────────────────────────────────────


def test_bench_tune_requires_exactly_one_family(sql_training_db, tmp_path):
    import main

    config = _make_config(
        inp_path=tmp_path / "network.inp", db_path=sql_training_db,
        families=["xgboost", "random_forest"],
    )
    args = _make_args(bench_tune=True)  # sin --models: toma las dos habilitadas

    with pytest.raises(SystemExit, match="exactly one family"):
        main._run_bench(args, config)


def test_bench_tune_rejects_a_prepared_dataset_with_stale_protocol_semantics(
    sql_training_db_six_shapes, tmp_path, monkeypatch
):
    import main

    monkeypatch.chdir(tmp_path)
    prepared = prepare_dataset(
        sql_training_db_six_shapes, protocols=("LOSO",), flood_threshold_m3=1.0,
        output_dir=Path("outputs/bench/prepared"),
    )
    manifest_path = Path("outputs/bench/prepared") / prepared.prep_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol_definitions"]["LOSO"]["group_column"] = "shape_id"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    config = _make_config(
        inp_path=tmp_path / "network.inp", db_path=sql_training_db_six_shapes,
        families=["xgboost"], protocols=("LOSO",),
    )
    args = _make_args(bench_tune=True, models="xgboost")

    with pytest.raises(SystemExit) as excinfo:
        main._run_bench(args, config)
    assert "--bench-prepare" in str(excinfo.value)


def test_bench_tune_writes_a_study_and_touches_neither_models_nor_ranking(
    sql_training_db_six_shapes, tmp_path, monkeypatch
):
    import main

    monkeypatch.chdir(tmp_path)
    config = _make_config(
        inp_path=tmp_path / "network.inp", db_path=sql_training_db_six_shapes,
        families=["xgboost"], protocols=("LeaveOneShapeOut",),
    )
    main._run_bench(_make_args(bench_prepare=True), config)
    main._run_bench(_make_args(bench_tune=True, models="xgboost"), config)

    study_files = list((tmp_path / "outputs" / "bench" / "tuning").rglob("study.json"))
    assert len(study_files) == 1
    payload = json.loads(study_files[0].read_text(encoding="utf-8"))
    assert payload["family"] == "xgboost"
    assert payload["status"] == "estimacion_exploratoria_condicionada"

    assert not (tmp_path / "outputs" / "models").exists()
    assert not (tmp_path / "outputs" / "bench" / "reports" / "ranking.csv").exists()


# ── baselines fallidas / umbral del manifiesto ───────────────────────────


def test_a_failing_baseline_does_not_lose_the_ranking_or_the_family_metrics(
    sql_training_db, tmp_path, monkeypatch, capsys
):
    import main
    from swmm_resilience.ml.bench import baselines

    def broken_baseline(prepared, name, threshold, *args, **kwargs):
        raise ValueError("la referencia se rompio")

    monkeypatch.setattr(baselines, "evaluate_baseline", broken_baseline)
    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config = _make_config(
        inp_path=inp_path, db_path=sql_training_db,
        families=["xgboost"], promote="none",
    )

    main._run_bench(_make_args(bench=True, models="xgboost"), config)  # must not raise

    reports = tmp_path / "outputs" / "bench" / "reports"
    assert (reports / "ranking.csv").exists()
    assert (reports / "ranking.json").exists()
    assert (reports / "metrics_xgboost.json").exists()
    assert "failed" in capsys.readouterr().out
    provenance = json.loads((reports / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["baselines"] == []
    assert "baseline_interpolacion_factor" in provenance["baselines_failed"]


def test_metrics_files_written_by_the_bench_are_stamped_with_the_prep_id(
    sql_training_db, tmp_path, monkeypatch
):
    import main

    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config = _make_config(
        inp_path=inp_path, db_path=sql_training_db,
        families=["xgboost"], promote="none",
    )

    main._run_bench(_make_args(bench=True, models="xgboost"), config)

    reports = tmp_path / "outputs" / "bench" / "reports"
    prep_id = next((tmp_path / "outputs" / "bench" / "prepared").iterdir()).name
    for name in ("metrics_xgboost.json", "metrics_baseline_interpolacion_factor.json"):
        written = json.loads((reports / name).read_text(encoding="utf-8"))
        assert written["metrics_definition"] == "classifier_pooled_v2"
        assert written["prep_id"] == prep_id


def test_baselines_use_the_manifest_threshold_and_warn_when_config_differs(
    sql_training_db, tmp_path, monkeypatch, capsys
):
    import main
    from swmm_resilience.ml.bench import baselines

    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config = _make_config(
        inp_path=inp_path, db_path=sql_training_db,
        families=["xgboost"], promote="none",
    )
    main._run_bench(_make_args(bench_prepare=True), config)  # manifiesto con 1.0

    config.dataset.flood_threshold_m3 = 5.0
    seen = []
    real = baselines.evaluate_baseline

    def spy(prepared, name, threshold, *args, **kwargs):
        seen.append(threshold)
        return real(prepared, name, threshold, *args, **kwargs)

    monkeypatch.setattr(baselines, "evaluate_baseline", spy)
    main._run_bench(_make_args(bench_evaluate=True, models="xgboost"), config)

    assert seen and set(seen) == {1.0}
    assert "threshold" in capsys.readouterr().out


def test_baselines_fail_clearly_when_the_manifest_has_no_threshold(
    sql_training_db, tmp_path, monkeypatch
):
    import main
    from swmm_resilience.ml.bench import preprocess

    inp_path = tmp_path / "network.inp"
    inp_path.write_text("[TITLE]\nred\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    config = _make_config(
        inp_path=inp_path, db_path=sql_training_db,
        families=["xgboost"], promote="none",
    )
    main._run_bench(_make_args(bench_prepare=True), config)

    real = preprocess.resolve_prepared

    def without_threshold(*args, **kwargs):
        prepared = real(*args, **kwargs)
        prepared.manifest.pop("flood_threshold_m3", None)
        return prepared

    monkeypatch.setattr(preprocess, "resolve_prepared", without_threshold)

    with pytest.raises(SystemExit, match="flood_threshold_m3"):
        main._run_bench(_make_args(bench_evaluate=True, models="xgboost"), config)
