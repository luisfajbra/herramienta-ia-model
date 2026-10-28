import json

import numpy as np
import pandas as pd
import pytest
import optuna

from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.bench.tune import (
    build_tuning_folds,
    SEARCH_SPACES,
    _sample_params,
    run_study,
    save_tune_result,
)


def _keys_six_shapes() -> pd.DataFrame:
    """6 formas x 5 factores, para que GroupKFold(5) tenga margen."""
    shapes = ["s0", "s1", "s2", "s3", "s4", "s5"]
    factors = [1.0, 2.0, 3.0, 4.0, 5.0]
    rows = [
        {"shape_id": shape, "factor_mult": factor}
        for shape in shapes
        for factor in factors
    ]
    return pd.DataFrame(rows)


def test_build_tuning_folds_has_the_declared_columns():
    keys = _keys_six_shapes()
    folds = build_tuning_folds(keys, n_splits=5)
    assert tuple(folds.columns) == ("fold_id", "split", "sample_idx")
    assert folds["fold_id"].nunique() == 5


def test_build_tuning_folds_partitions_without_shape_leakage():
    keys = _keys_six_shapes()
    folds = build_tuning_folds(keys, n_splits=5)
    for _, chunk in folds.groupby("fold_id"):
        train_idx = chunk.loc[chunk["split"] == "train", "sample_idx"].to_numpy()
        test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"].to_numpy()
        train_shapes = set(keys["shape_id"].iloc[train_idx])
        test_shapes = set(keys["shape_id"].iloc[test_idx])
        assert train_shapes.isdisjoint(test_shapes)
        assert len(train_idx) + len(test_idx) == len(keys)


def test_build_tuning_folds_covers_every_row_across_folds_as_test_exactly_once():
    keys = _keys_six_shapes()
    folds = build_tuning_folds(keys, n_splits=5)
    test_rows = folds.loc[folds["split"] == "test", "sample_idx"]
    assert sorted(test_rows.tolist()) == list(range(len(keys)))
    assert not test_rows.duplicated().any()


def test_build_tuning_folds_is_deterministic_given_the_same_seed():
    keys = _keys_six_shapes()
    first = build_tuning_folds(keys, n_splits=5, seed=7)
    second = build_tuning_folds(keys, n_splits=5, seed=7)
    pd.testing.assert_frame_equal(first, second)


def test_build_tuning_folds_rejects_too_few_shapes_for_n_splits():
    keys = pd.DataFrame({"shape_id": ["a", "b", "c"], "factor_mult": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="5"):
        build_tuning_folds(keys, n_splits=5)


def test_build_tuning_folds_leave_one_shape_out_holds_exactly_one_shape():
    keys = _keys_six_shapes()
    folds = build_tuning_folds(keys, protocol="LeaveOneShapeOut")

    assert folds["fold_id"].nunique() == keys["shape_id"].nunique() == 6
    tested = folds.loc[folds["split"] == "test", "sample_idx"]
    assert sorted(tested.tolist()) == list(range(len(keys)))
    assert not tested.duplicated().any()
    for _, chunk in folds.groupby("fold_id"):
        test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"].to_numpy()
        train_idx = chunk.loc[chunk["split"] == "train", "sample_idx"].to_numpy()
        assert len(set(keys.iloc[test_idx]["shape_id"])) == 1
        assert set(keys.iloc[test_idx]["shape_id"]).isdisjoint(
            set(keys.iloc[train_idx]["shape_id"])
        )


@pytest.mark.parametrize("family_name", ["xgboost", "random_forest", "linear", "svm", "mlp"])
@pytest.mark.parametrize("task", ["classifier", "regressor"])
def test_sample_params_returns_only_declared_keys_within_range(family_name, task):
    space = SEARCH_SPACES[family_name][task]
    study = optuna.create_study()
    trial = study.ask()

    params = _sample_params(trial, space, prefix=f"{family_name}_{task}_")

    assert set(params) == set(space)
    for name, spec in space.items():
        kind = spec[0]
        if kind in ("int", "float", "float_log"):
            low, high = spec[1], spec[2]
            assert low <= params[name] <= high
        elif kind == "categorical":
            assert params[name] in spec[1]


def test_sample_params_does_not_collide_between_classifier_and_regressor_in_the_same_trial():
    """El prefijo evita que Optuna vea el mismo nombre de parametro dos veces
    en un trial con rangos distintos (p.ej. xgboost.max_depth en ambas tareas)."""
    study = optuna.create_study()
    trial = study.ask()

    clf_params = _sample_params(
        trial, SEARCH_SPACES["xgboost"]["classifier"], prefix="xgboost_classifier_"
    )
    reg_params = _sample_params(
        trial, SEARCH_SPACES["xgboost"]["regressor"], prefix="xgboost_regressor_"
    )

    assert set(clf_params) == set(SEARCH_SPACES["xgboost"]["classifier"])
    assert set(reg_params) == set(SEARCH_SPACES["xgboost"]["regressor"])


def test_svm_search_space_matches_the_approximate_rbf_estimators():
    assert set(SEARCH_SPACES["svm"]["classifier"]) == {
        "gamma", "n_components", "alpha"
    }
    assert set(SEARCH_SPACES["svm"]["regressor"]) == {
        "gamma", "n_components", "alpha", "epsilon"
    }


@pytest.fixture
def prepared_six_shapes(sql_training_db_six_shapes, tmp_path):
    return prepare_dataset(
        sql_training_db_six_shapes,
        protocols=("LeaveOneShapeOut",),
        flood_threshold_m3=1.0,
        output_dir=tmp_path / "prepared",
    )


def test_run_study_returns_a_finite_best_value_and_matching_param_keys(prepared_six_shapes):
    result = run_study(prepared_six_shapes, "xgboost", n_trials=2, n_splits=5)

    assert result.family == "xgboost"
    assert result.prep_id == prepared_six_shapes.prep_id
    assert result.n_trials == 2
    assert len(result.trials) == 2
    assert np.isfinite(result.best_value)
    assert set(result.best_classifier_params) == set(SEARCH_SPACES["xgboost"]["classifier"])
    assert set(result.best_regressor_params) == set(SEARCH_SPACES["xgboost"]["regressor"])
    assert result.status == "estimacion_exploratoria_condicionada"
    assert "holdout" in result.status_reason


def test_run_study_rejects_a_family_with_too_few_shapes_for_the_inner_cv(prepared_six_shapes):
    with pytest.raises(ValueError, match="99"):
        run_study(prepared_six_shapes, "xgboost", n_trials=1, n_splits=99)


def test_run_study_records_macro_network_rmse_objective(prepared_six_shapes):
    result = run_study(
        prepared_six_shapes,
        "xgboost",
        n_trials=1,
        n_splits=2,
        objective="macro_network_rmse",
    )

    assert result.objective_metric == "macro_network_rmse"
    assert result.study_id
    assert np.isfinite(result.best_value)
    assert len(result.trials) == 1
    trial = result.trials[0]
    assert trial["state"] == "COMPLETE"
    assert trial["objective_metric"] == "macro_network_rmse"
    assert len(trial["inner_fold_rmse"]) == 2
    assert trial["inner_network_rmse"]


def test_run_study_survives_a_failing_trial_and_keeps_the_completed_ones(
    prepared_six_shapes, monkeypatch
):
    """Un trial que falla (p.ej. una combinacion de hiperparametros
    degenerada que rechaza fit()) no debe abortar todo el estudio: Optuna
    debe marcarlo FAIL internamente y seguir con los trials restantes, y
    run_study() debe seguir devolviendo un TuneResult con todos los trials
    -- completos y fallidos -- y un best_value finito (spec §12: un trial
    malo debe poder descartarse sin destruir el estudio)."""
    import swmm_resilience.ml.bench.tune as tune_module

    real_fit_predict_fold = tune_module._fit_predict_fold
    calls = {"count": 0}

    def _flaky_fit_predict_fold(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("forced failure for test")
        return real_fit_predict_fold(*args, **kwargs)

    monkeypatch.setattr(tune_module, "_fit_predict_fold", _flaky_fit_predict_fold)

    result = run_study(prepared_six_shapes, "xgboost", n_trials=3, n_splits=5)

    assert len(result.trials) == 3
    assert np.isfinite(result.best_value)
    # el trial fallido no debe haber bloqueado la evaluacion de los siguientes
    assert calls["count"] > 1


def test_run_study_raises_a_clear_error_when_every_trial_fails(
    prepared_six_shapes, monkeypatch
):
    import swmm_resilience.ml.bench.tune as tune_module

    def _always_fail(*args, **kwargs):
        raise RuntimeError("forced failure for test")

    monkeypatch.setattr(tune_module, "_fit_predict_fold", _always_fail)

    with pytest.raises(ValueError, match="Optuna trials for 'xgboost' failed"):
        run_study(prepared_six_shapes, "xgboost", n_trials=3, n_splits=5)


def test_save_tune_result_writes_the_declared_schema(prepared_six_shapes, tmp_path):
    result = run_study(prepared_six_shapes, "xgboost", n_trials=2, n_splits=5)
    path = save_tune_result(result, tmp_path / "tuning")

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "family", "prep_id", "study_id", "inner_cv", "objective_metric", "direction",
        "n_trials", "seed", "optuna_version", "started_at_utc", "finished_at_utc",
        "duration_s", "trials", "best_classifier_params", "best_regressor_params",
        "best_value", "status", "status_reason",
    ):
        assert key in payload, f"falta {key} en study.json"
    assert payload["inner_cv"] == {
        "method": "GroupKFold", "n_splits": 5, "group_column": "shape_id"
    }
    assert payload["status"] == "estimacion_exploratoria_condicionada"


def test_save_tune_result_never_touches_outputs_models(prepared_six_shapes, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run_study(prepared_six_shapes, "xgboost", n_trials=2, n_splits=5)
    save_tune_result(result, tmp_path / "outputs" / "bench" / "tuning")

    assert not (tmp_path / "outputs" / "models").exists()
