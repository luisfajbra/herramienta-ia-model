"""Partes puras del script de la parte 2 (comparación justa XGBoost / MLP / SVM)."""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.ml.bench.prepared_factory import make_prepared, perfect_oof


def _prepared(n_shapes=7):
    rows = [
        dict(shape_id=f"S{index}", node_id=node, factor_mult=factor, vol=vol,
             duracion_horas=1.0 + index, tiempo_al_pico_h=0.5)
        for index in range(n_shapes)
        for factor in (1.0, 2.0)
        for node, vol in (("N0", 0.5), ("N1", 20.0))
    ]
    prepared = make_prepared(rows, ["LeaveOneShapeOut"])
    prepared.manifest["flood_threshold_m3"] = 1.0
    return prepared


GOOD_MANIFEST = {"prep_id": "ca706e27654e0cf5", "contract_id": "tabular_v4_21", "n_rows": 154080}


# --- guarda del manifiesto -------------------------------------------------------------

def test_manifest_guard_accepts_the_expected_dataset():
    from scripts.ejecutar_parte2_v4_21 import check_manifest

    check_manifest(dict(GOOD_MANIFEST))


@pytest.mark.parametrize(
    "field, value",
    [("prep_id", "0123456789abcdef"), ("contract_id", "tabular_v3_17"), ("n_rows", 154079)],
)
def test_manifest_guard_rejects_any_mismatch(field, value):
    from scripts.ejecutar_parte2_v4_21 import check_manifest

    with pytest.raises(ValueError, match=field):
        check_manifest({**GOOD_MANIFEST, field: value})


def test_manifest_guard_rejects_a_missing_field():
    from scripts.ejecutar_parte2_v4_21 import check_manifest

    manifest = dict(GOOD_MANIFEST)
    del manifest["n_rows"]
    with pytest.raises(ValueError, match="n_rows"):
        check_manifest(manifest)


# --- parámetros fijos ------------------------------------------------------------------

def test_constants_follow_the_design():
    from scripts import ejecutar_parte2_v4_21 as script
    from swmm_resilience.config import ML_RANDOM_STATE

    assert script.N_TRIALS == 20
    assert script.INNER_SPLITS == 5
    assert script.SEED == ML_RANDOM_STATE
    assert script.EVAL_PROTOCOL == "LeaveOneShapeOut"
    for task in ("classifier", "regressor"):
        assert script.FIXED_PARAMS["mlp"][task] == {"epochs": 200, "batch_size": 1024}
        assert script.FIXED_PARAMS["svm"][task] == {}
        assert script.FIXED_PARAMS["xgboost"][task]["n_jobs"] >= 1


def test_merge_fixed_overrides_the_base_without_mutating_it():
    from scripts.ejecutar_parte2_v4_21 import merge_fixed

    base = {
        "classifier": {"hidden_sizes": [64, 32], "batch_size": 256, "epochs": 50},
        "regressor": {"dropout": 0.1, "batch_size": 256},
    }
    fixed = {"classifier": {"batch_size": 1024, "epochs": 200}, "regressor": {"batch_size": 1024}}

    merged = merge_fixed(base, fixed)

    assert merged["classifier"] == {"hidden_sizes": [64, 32], "batch_size": 1024, "epochs": 200}
    assert merged["regressor"] == {"dropout": 0.1, "batch_size": 1024}
    assert base["classifier"]["batch_size"] == 256


def test_merge_fixed_tolerates_a_missing_task_in_the_fixed_params():
    from scripts.ejecutar_parte2_v4_21 import merge_fixed

    merged = merge_fixed({"classifier": {"a": 1}, "regressor": {"b": 2}}, {})
    assert merged == {"classifier": {"a": 1}, "regressor": {"b": 2}}


def test_config_params_reads_the_bench_family_and_uses_the_mlp_defaults():
    from scripts.ejecutar_parte2_v4_21 import FIXED_PARAMS, config_params, merge_fixed
    from swmm_resilience.config import load_config

    config = load_config(str(Path(__file__).resolve().parents[2] / "config.yaml"))
    mlp = merge_fixed(config_params(config, "mlp"), FIXED_PARAMS["mlp"])

    assert mlp["classifier"]["batch_size"] == 1024
    assert mlp["classifier"]["epochs"] == 200
    assert list(mlp["classifier"]["hidden_sizes"]) == [64, 32]
    assert mlp["classifier"]["learning_rate"] == 0.001
    assert mlp["classifier"]["dropout"] == 0.1
    svm = config_params(config, "svm")
    assert svm["classifier"]["n_components"] == 200


# --- nombres y etapas resumibles -------------------------------------------------------

def test_model_names_and_stage_folders(tmp_path):
    from scripts.ejecutar_parte2_v4_21 import model_name, stage_path

    assert model_name("xgboost", "default") == "xgboost_default"
    assert model_name("mlp", "tuned") == "mlp_tuned"
    with pytest.raises(ValueError):
        model_name("svm", "tune")
    assert stage_path(tmp_path, "svm", "tune") == tmp_path / "svm" / "tune"


def _header(**extra):
    return {
        "family": "svm", "stage": "default", "prep_id": "ca706e27654e0cf5",
        "fixed_params": {}, "n_jobs": 2, "git": {"git_commit": "abc", "git_dirty": False},
        "versions": {"python": "3.12"}, **extra,
    }


def test_a_stage_with_a_done_marker_is_skipped(tmp_path):
    from scripts.ejecutar_parte2_v4_21 import is_done, run_stage_once

    path = tmp_path / "svm" / "default"
    calls = []

    def runner(directory):
        calls.append(directory)
        return {"params": {"classifier": {"x": 1}, "regressor": {}}}

    assert not is_done(path)
    assert run_stage_once(path, runner, header=_header(), log=lambda *_: None) is True
    assert is_done(path)
    assert run_stage_once(path, runner, header=_header(), log=lambda *_: None) is False
    assert calls == [path]


def test_a_crashed_stage_leaves_no_done_marker_and_is_rerun(tmp_path):
    from scripts.ejecutar_parte2_v4_21 import is_done, run_stage_once

    path = tmp_path / "svm" / "default"

    def crash(directory):
        raise RuntimeError("se cayó")

    with pytest.raises(RuntimeError):
        run_stage_once(path, crash, header=_header(), log=lambda *_: None)
    assert not is_done(path)

    ran = run_stage_once(path, lambda d: {"params": {}}, header=_header(), log=lambda *_: None)
    assert ran is True and is_done(path)


def test_stage_json_has_the_provenance_fields(tmp_path):
    from scripts.ejecutar_parte2_v4_21 import run_stage_once

    path = tmp_path / "svm" / "default"
    run_stage_once(
        path,
        lambda d: {"params": {"classifier": {"gamma": 0.05}, "regressor": {}}},
        header=_header(fixed_params={"classifier": {"epochs": 200}}),
        log=lambda *_: None,
    )
    record = json.loads((path / "stage.json").read_text(encoding="utf-8"))

    for key in ("started_utc", "finished_utc", "duration_s", "params", "fixed_params", "git",
                "n_jobs", "versions", "family", "stage", "prep_id"):
        assert key in record, key
    assert record["params"]["classifier"] == {"gamma": 0.05}
    assert record["fixed_params"] == {"classifier": {"epochs": 200}}
    assert record["n_jobs"] == 2
    assert record["duration_s"] >= 0


def test_build_stage_record_computes_the_duration_from_the_timestamps():
    from scripts.ejecutar_parte2_v4_21 import build_stage_record

    start = datetime(2026, 10, 12, 10, 0, 0, tzinfo=timezone.utc)
    record = build_stage_record(_header(), {"params": {}}, start, start + timedelta(seconds=90))

    assert record["duration_s"] == 90.0
    assert record["started_utc"] == start.isoformat()
    assert record["finished_utc"] == (start + timedelta(seconds=90)).isoformat()


# --- mejores parámetros del estudio ----------------------------------------------------

def _study(path: Path, family="xgboost"):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "family": family, "n_trials": 3, "best_value": 12.5, "duration_s": 99.0,
        "best_classifier_params": {"max_depth": 5, "n_jobs": 1},
        "best_regressor_params": {"max_depth": 7, "n_jobs": 1},
        "fixed_params": {}, "fold_n_jobs": 5,
        "status": "estimacion_exploratoria_condicionada",
        "trials": [
            {"number": 0, "value": 20.0, "duration_s": 3.0},
            {"number": 1, "value": 12.5, "duration_s": 4.0},
            {"number": 2, "value": None, "duration_s": 1.0},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_best_params_come_from_the_study_and_the_fixed_ones_override(tmp_path):
    from scripts.ejecutar_parte2_v4_21 import best_params_from_study, find_study_json

    tune = tmp_path / "xgboost" / "tune"
    _study(tune / "xgboost" / "ca706e27654e0cf5_20261012T100000Z" / "study.json")

    found = find_study_json(tune, "xgboost")
    params = best_params_from_study(found, {"classifier": {"n_jobs": 4}, "regressor": {}})

    assert params["classifier"] == {"max_depth": 5, "n_jobs": 4}
    assert params["regressor"] == {"max_depth": 7, "n_jobs": 1}


def test_find_study_json_fails_when_the_tune_stage_is_missing(tmp_path):
    from scripts.ejecutar_parte2_v4_21 import find_study_json

    with pytest.raises(FileNotFoundError, match="tune"):
        find_study_json(tmp_path / "xgboost" / "tune", "xgboost")


# --- humo: subconjunto de folds --------------------------------------------------------

def test_restrict_prepared_keeps_only_the_first_folds_of_the_protocol():
    from scripts.ejecutar_parte2_v4_21 import restrict_oof, restrict_prepared

    prepared = _prepared(7)
    small = restrict_prepared(prepared, "LeaveOneShapeOut", 3)

    assert sorted(small.folds["fold_id"].unique()) == [0, 1, 2]
    assert len(small.keys) == len(prepared.keys)
    oof = restrict_oof(perfect_oof(prepared), "LeaveOneShapeOut", 3)
    assert sorted(oof["fold_id"].unique()) == [0, 1, 2]


# --- guarda de rutas -------------------------------------------------------------------

@pytest.mark.parametrize(
    "relative",
    [
        "outputs/bench/oof/ca706e27654e0cf5/x.parquet",
        "outputs/bench/prepared_v4_21/ca706e27654e0cf5/folds.parquet",
        "outputs/models/m.joblib",
        "docs/auditoria/correcciones/datos/comparacion_protocolos.csv",
        "docs/auditoria/correcciones/datos/runs",
        "config.yaml",
        "data/x.csv",
    ],
)
def test_output_guard_rejects_protected_paths(relative):
    from scripts.ejecutar_parte2_v4_21 import REPO_ROOT, ensure_parte2_output

    with pytest.raises(ValueError, match="Protected path"):
        ensure_parte2_output(REPO_ROOT / relative)


@pytest.mark.parametrize(
    "relative",
    [
        "outputs/bench/parte2/ca706e27654e0cf5/svm/default/oof.parquet",
        "docs/auditoria/correcciones/datos_parte2/comparacion_protocolos.csv",
    ],
)
def test_output_guard_accepts_the_part2_folders(relative):
    from scripts.ejecutar_parte2_v4_21 import REPO_ROOT, ensure_parte2_output

    ensure_parte2_output(REPO_ROOT / relative)


def test_output_guard_ignores_paths_outside_the_repo(tmp_path):
    from scripts.ejecutar_parte2_v4_21 import ensure_parte2_output

    ensure_parte2_output(tmp_path / "anything")


# --- armado de tablas ------------------------------------------------------------------

def _make_run(tmp_path, prepared):
    """Raíz de salida falsa: svm default+tune+tuned, xgboost default; OOF perfecto/invertido."""
    from scripts.ejecutar_parte2_v4_21 import stage_path

    root = tmp_path / "parte2"
    good = perfect_oof(prepared)
    bad = good.copy()
    bad["y_prob_clf"] = 1.0 - bad["y_prob_clf"]
    for family, stage, oof in (
        ("svm", "default", bad), ("svm", "tuned", good), ("xgboost", "default", good),
    ):
        directory = stage_path(root, family, stage)
        directory.mkdir(parents=True)
        oof.to_parquet(directory / "oof.parquet", index=False)
        (directory / "done").write_text("ok", encoding="utf-8")
    tune = stage_path(root, "svm", "tune")
    _study(tune / "svm" / "ca706e27654e0cf5_20261012T100000Z" / "study.json", family="svm")
    (tune / "done").write_text("ok", encoding="utf-8")
    baseline = tmp_path / "baseline_forma_mas_cercana.parquet"
    good.to_parquet(baseline, index=False)
    return root, baseline


def test_assemble_tables_names_models_adds_variant_and_auc(tmp_path):
    from scripts.ejecutar_parte2_v4_21 import assemble_tables

    prepared = _prepared()
    root, baseline = _make_run(tmp_path, prepared)

    tables = assemble_tables(prepared, root, baseline, flood_threshold_m3=1.0)

    assert set(tables) == {
        "comparacion_protocolos", "matriz_confusion", "banda_limitrofe", "por_forma",
        "resumen_afinacion", "convergencia_afinacion",
    }
    comparison = tables["comparacion_protocolos"]
    assert set(comparison["modelo"]) == {
        "svm_default", "svm_tuned", "xgboost_default", "baseline_forma_mas_cercana",
    }
    variant = dict(zip(comparison["modelo"], comparison["variante"]))
    assert variant == {
        "svm_default": "default", "svm_tuned": "tuned", "xgboost_default": "default",
        "baseline_forma_mas_cercana": "referencia",
    }
    auc = dict(zip(comparison["modelo"], comparison["auc_roc_agregado"]))
    assert auc["svm_default"] == 0.0 and auc["svm_tuned"] == 1.0
    assert set(comparison["protocolo"]) == {"LeaveOneShapeOut"}


def test_assemble_tables_restricts_a_wider_baseline_to_the_evaluation_protocol(tmp_path):
    from scripts.ejecutar_parte2_v4_21 import assemble_tables

    rows = [
        dict(shape_id=f"S{index}", node_id=node, factor_mult=factor, vol=vol,
             duracion_horas=1.0, tiempo_al_pico_h=0.5)
        for index in range(7)
        for factor in (1.0, 2.0)
        for node, vol in (("N0", 0.5), ("N1", 20.0))
    ]
    prepared = make_prepared(rows, ["LeaveOneShapeOut", "LOSO"])
    prepared.manifest["flood_threshold_m3"] = 1.0
    root, baseline = _make_run(tmp_path, prepared)
    full = perfect_oof(prepared)
    only_loso_shape = full[full["protocol"] == "LeaveOneShapeOut"]
    for family, stage in (("svm", "default"), ("svm", "tuned"), ("xgboost", "default")):
        only_loso_shape.to_parquet(root / family / stage / "oof.parquet", index=False)
    full.to_parquet(baseline, index=False)

    comparison = assemble_tables(prepared, root, baseline, 1.0)["comparacion_protocolos"]

    assert set(comparison["protocolo"]) == {"LeaveOneShapeOut"}


def test_tuning_tables_summarise_each_finished_study(tmp_path):
    from scripts.ejecutar_parte2_v4_21 import assemble_tables

    prepared = _prepared()
    root, baseline = _make_run(tmp_path, prepared)

    tables = assemble_tables(prepared, root, baseline, 1.0)
    summary, convergence = tables["resumen_afinacion"], tables["convergencia_afinacion"]

    assert list(summary["family"]) == ["svm"]
    row = summary.iloc[0]
    assert row["n_trials"] == 3 and row["best_value"] == 12.5 and row["duration_s"] == 99.0
    assert json.loads(row["best_classifier_params"]) == {"max_depth": 5, "n_jobs": 1}
    assert json.loads(row["best_regressor_params"]) == {"max_depth": 7, "n_jobs": 1}
    assert row["status"] == "estimacion_exploratoria_condicionada"
    assert list(convergence.columns[:4]) == ["family", "trial", "value", "duration_s"]
    assert list(convergence["trial"]) == [0, 1, 2]
    assert list(convergence["best_so_far"].iloc[:2]) == [20.0, 12.5]


def test_assemble_tables_ignores_stages_without_a_done_marker(tmp_path):
    from scripts.ejecutar_parte2_v4_21 import assemble_tables

    prepared = _prepared()
    root, baseline = _make_run(tmp_path, prepared)
    (root / "svm" / "tuned" / "done").unlink()

    comparison = assemble_tables(prepared, root, baseline, 1.0)["comparacion_protocolos"]

    assert "svm_tuned" not in set(comparison["modelo"])


def test_write_parte2_tables_writes_csvs_and_a_provenance_with_hashes(tmp_path):
    from scripts.ejecutar_parte2_v4_21 import assemble_tables, write_parte2_tables

    prepared = _prepared()
    root, baseline = _make_run(tmp_path, prepared)
    tables = assemble_tables(prepared, root, baseline, 1.0)
    out = tmp_path / "tablas"

    write_parte2_tables(
        tables, out, prepared=prepared, out_root=root, baseline_path=baseline,
        flood_threshold_m3=1.0, command="python scripts/ejecutar_parte2_v4_21.py --build-tables",
    )

    for name in tables:
        assert (out / f"{name}.csv").exists()
        assert b"\r\n" not in (out / f"{name}.csv").read_bytes()
    provenance = json.loads((out / "tables_provenance.json").read_text(encoding="utf-8"))
    assert provenance["prep_id"] == prepared.prep_id
    assert provenance["command"].endswith("--build-tables")
    assert "svm/default/oof.parquet" in provenance["inputs_sha256"]
    assert "baseline_forma_mas_cercana.parquet" in provenance["inputs_sha256"]
    assert "resumen_afinacion.csv" in provenance["outputs_sha256"]
    assert "comparacion_protocolos.csv" in provenance["outputs_sha256"]
    assert set(provenance["models"]) == {
        "svm_default", "svm_tuned", "xgboost_default", "baseline_forma_mas_cercana"
    }


def test_assemble_tables_without_any_stage_fails(tmp_path):
    from scripts.ejecutar_parte2_v4_21 import assemble_tables

    prepared = _prepared()
    _, baseline = _make_run(tmp_path, prepared)

    with pytest.raises(FileNotFoundError, match="evaluation stage"):
        assemble_tables(prepared, tmp_path / "vacio", baseline, 1.0)


# --- CLI -------------------------------------------------------------------------------

def test_cli_requires_exactly_one_of_family_or_build_tables():
    from scripts.ejecutar_parte2_v4_21 import parse_args

    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--family", "svm", "--build-tables"])
    args = parse_args(["--family", "mlp", "--eval-jobs", "8"])
    assert args.family == "mlp" and args.eval_jobs == 8
    assert parse_args(["--family", "svm"]).eval_jobs == 40
    assert parse_args(["--build-tables"]).build_tables is True


def test_smoke_forces_two_trials():
    from scripts.ejecutar_parte2_v4_21 import effective_n_trials, parse_args

    assert effective_n_trials(parse_args(["--family", "svm"])) == 20
    assert effective_n_trials(parse_args(["--family", "svm", "--n-trials", "7"])) == 7
    assert effective_n_trials(parse_args(["--family", "svm", "--smoke"])) == 2
