from types import SimpleNamespace
from dataclasses import replace
import json

import numpy as np
import pytest

from tests.ml.bench.prepared_factory import make_prepared


def _two_network_prepared():
    rows = []
    for network_id in ("A", "B"):
        for shape_index in range(4):
            for factor in (1.0, 2.0):
                for node_index in range(2):
                    rows.append(
                        {
                            "network_id": network_id,
                            "shape_id": f"shape-{shape_index}",
                            "node_id": f"node-{node_index}",
                            "factor_mult": factor,
                            "vol": float((shape_index + node_index + factor) % 3),
                            "duracion_horas": float(shape_index + 1),
                            "tiempo_al_pico_h": float(factor),
                        }
                    )
    return make_prepared(rows, ("LeaveOneNetworkOut",))


def test_nested_evaluation_tunes_only_outer_training_networks(monkeypatch):
    import swmm_resilience.ml.bench.evaluate as evaluate_module

    prepared = _two_network_prepared()
    tuning_views = []
    tuning_kwargs = []

    def spy_run_study(view, family, **kwargs):
        tuning_views.append(view)
        tuning_kwargs.append(kwargs)
        return SimpleNamespace(
            family=family,
            prep_id=view.prep_id,
            study_id=f"study-{len(tuning_views)}",
            best_classifier_params={"stub": True},
            best_regressor_params={"stub": True},
            best_value=1.25,
            trials=[{"number": 0, "value": 1.25}],
            seed=kwargs["seed"],
            objective_metric=kwargs["objective"],
        )

    monkeypatch.setattr(evaluate_module, "run_study", spy_run_study)
    monkeypatch.setattr(
        evaluate_module,
        "_fit_predict_fold",
        lambda prepared, family, clf, reg, train_idx, test_idx: (
            np.zeros(len(test_idx), dtype=int),
            np.full(len(test_idx), 0.25),
            np.ones(len(test_idx)),
        ),
    )

    result = evaluate_module.evaluate_nested_networks(
        prepared,
        "xgboost",
        ("LeaveOneShapeOut",),
        objective="macro_network_rmse",
        n_trials=1,
        n_splits=2,
        seed=17,
    )

    assert len(result.folds) == len(tuning_views) == 2
    assert all(
        kwargs["inner_protocol"] == "LeaveOneShapeOut"
        for kwargs in tuning_kwargs
    )
    assert result.oof.groupby("sample_idx").size().eq(1).all()
    assert set(result.oof["sample_idx"]) == set(range(len(prepared.keys)))
    routed_truth = prepared.y_reg.to_numpy()
    assert result.metrics["pooled_rmse"] == pytest.approx(
        np.sqrt(np.mean(routed_truth**2))
    )
    for fold, view in zip(result.folds, tuning_views):
        outer_test_networks = set(
            prepared.keys.iloc[list(fold.outer_test_sample_idx)]["network_id"]
        )
        outer_train_networks = set(
            prepared.keys.iloc[list(fold.outer_train_sample_idx)]["network_id"]
        )
        assert len(outer_test_networks) == 1
        assert outer_train_networks.isdisjoint(outer_test_networks)
        assert set(view.keys["network_id"]) == outer_train_networks
        assert set(view.keys["run_id"]) <= set(
            prepared.keys.iloc[list(fold.outer_train_sample_idx)]["run_id"]
        )
        assert set(view.keys["run_id"]).isdisjoint(
            set(prepared.keys.iloc[list(fold.outer_test_sample_idx)]["run_id"])
        )
        assert fold.objective == "macro_network_rmse"
        assert fold.optuna_study_id


def test_outer_test_targets_do_not_change_the_inner_training_fingerprint():
    import swmm_resilience.ml.bench.evaluate as evaluate_module

    prepared = _two_network_prepared()
    outer = prepared.folds[
        (prepared.folds["protocol"] == "LeaveOneNetworkOut")
        & (prepared.folds["fold_id"] == 0)
    ]
    test_idx = outer.loc[outer["split"] == "test", "sample_idx"].to_numpy()
    train_idx = outer.loc[outer["split"] == "train", "sample_idx"].to_numpy()
    modified_targets = prepared.y_reg.copy()
    modified_targets.iloc[test_idx] += 10000.0
    modified = replace(
        prepared,
        prep_id="fedcba9876543210",
        y_reg=modified_targets,
    )

    original_view = evaluate_module._fold_training_view(prepared, train_idx)
    modified_view = evaluate_module._fold_training_view(modified, train_idx)

    assert original_view.prep_id == modified_view.prep_id
    assert original_view.run_ids == modified_view.run_ids


def test_nested_evaluation_saves_immutable_split_and_study_provenance(
    monkeypatch, tmp_path
):
    import swmm_resilience.ml.bench.evaluate as evaluate_module

    prepared = _two_network_prepared()
    monkeypatch.setattr(
        evaluate_module,
        "run_study",
        lambda view, family, **kwargs: SimpleNamespace(
            study_id=f"study-{len(view.run_ids)}",
            best_classifier_params={"stub": True},
            best_regressor_params={"stub": True},
            best_value=1.25,
            trials=[{"number": 0, "value": 1.25}],
        ),
    )
    monkeypatch.setattr(
        evaluate_module,
        "_fit_predict_fold",
        lambda prepared, family, clf, reg, train_idx, test_idx: (
            np.zeros(len(test_idx), dtype=int),
            np.full(len(test_idx), 0.25),
            np.ones(len(test_idx)),
        ),
    )
    result = evaluate_module.evaluate_nested_networks(
        prepared,
        "xgboost",
        ("LeaveOneShapeOut",),
        objective="macro_network_rmse",
        n_trials=1,
        n_splits=2,
    )

    output = evaluate_module.save_nested_network_result(result, tmp_path)

    assert (output / "oof.parquet").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["objective"] == "macro_network_rmse"
    assert manifest["network_aggregation"] == "equal_weight_per_network"
    assert len(manifest["folds"]) == 2
    assert all(fold["optuna_study_id"] for fold in manifest["folds"])
    assert all(fold["inner_prep_id"] for fold in manifest["folds"])

    with pytest.raises(FileExistsError):
        evaluate_module.save_nested_network_result(result, tmp_path)
