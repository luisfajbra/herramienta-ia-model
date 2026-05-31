# tests/ml/temporal/test_train_surrogate.py
"""TDD tests for train_surrogate() pipeline."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.temporal.schemas import TemporalWindowDataset
from swmm_resilience.ml.temporal.train_surrogate import train_surrogate


def _synthetic_surrogate_dataset(
    n_runs: int = 4, n_nodes: int = 10, T: int = 20, use_temporal: bool = True
) -> TemporalWindowDataset:
    """Minimal dataset: n_runs groups, n_nodes nodes each."""
    N = n_runs * n_nodes
    rng = np.random.RandomState(42)
    groups = np.array([f"run_{i:02d}" for i in range(n_runs) for _ in range(n_nodes)], dtype=object)
    n_static = 7 if use_temporal else 8
    return TemporalWindowDataset(
        X_seq=rng.randn(N, T, 6).astype(np.float32),
        X_static=rng.randn(N, n_static).astype(np.float32),
        y_class=rng.randint(0, 2, N).astype(np.int8),
        y_reg=(rng.rand(N) * 100).astype(np.float32),
        groups=groups,
        meta=pd.DataFrame({
            "run_id": groups,
            "node_id": [f"J-{j:03d}" for j in range(n_nodes)] * n_runs,
            "window_start_min": [0.0] * N,
        }),
    )


class TestArtifactsSaved:
    def test_all_artifacts_exist(self, tmp_path):
        dataset = _synthetic_surrogate_dataset()
        train_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=2,
            batch_size=16,
            n_cv_folds=2,
            _dataset=dataset,
        )
        artifacts_dir = tmp_path / "artifacts"
        assert (artifacts_dir / "surrogate_cnn_weights.pt").exists()
        assert (artifacts_dir / "surrogate_cnn_scaler_seq.joblib").exists()
        assert (artifacts_dir / "surrogate_cnn_scaler_static.joblib").exists()
        assert (artifacts_dir / "surrogate_cnn_metrics.csv").exists()


class TestNoDataLeakage:
    def test_train_val_groups_disjoint(self, tmp_path):
        dataset = _synthetic_surrogate_dataset()
        result = train_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1,
            batch_size=16,
            n_cv_folds=2,
            _dataset=dataset,
        )
        for fold in result["folds"]:
            train_set = set(fold["train_groups"])
            val_set = set(fold["val_groups"])
            assert not (train_set & val_set), f"Data leakage in fold {fold['fold']}"


class TestReturnedMetrics:
    def test_metrics_contain_both_tasks(self, tmp_path):
        dataset = _synthetic_surrogate_dataset()
        result = train_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1,
            batch_size=16,
            n_cv_folds=2,
            _dataset=dataset,
        )
        fold = result["folds"][0]
        assert "val_auc_roc" in fold,  "Missing classification metric"
        assert "val_rmse" in fold,     "Missing regression metric"
        assert "val_f1" in fold,       "Missing F1 metric"

    def test_result_structure(self, tmp_path):
        dataset = _synthetic_surrogate_dataset()
        result = train_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1,
            batch_size=16,
            n_cv_folds=2,
            _dataset=dataset,
        )
        assert "n_folds" in result
        assert "best_fold" in result
        assert len(result["folds"]) == 2


class TestNoTemporalMode:
    def test_no_temporal_artifacts_saved(self, tmp_path):
        dataset = _synthetic_surrogate_dataset(use_temporal=False)
        train_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=2,
            batch_size=16,
            n_cv_folds=2,
            use_temporal=False,
            _dataset=dataset,
        )
        artifacts_dir = tmp_path / "artifacts"
        assert (artifacts_dir / "surrogate_cnn_notemporal_weights.pt").exists()
        assert (artifacts_dir / "surrogate_cnn_notemporal_metrics.csv").exists()


def test_surrogate_manifest_records_training_contract(tmp_path):
    dataset = _synthetic_surrogate_dataset(n_runs=4, n_nodes=3, T=8)
    train_surrogate(
        artifacts_dir=tmp_path / "artifacts",
        n_epochs=1,
        batch_size=8,
        n_cv_folds=2,
        _dataset=dataset,
    )
    manifest = json.loads((tmp_path / "artifacts" / "surrogate_cnn_manifest.json").read_text())
    assert manifest["model_type"] == "cnn"
    assert manifest["seed"] == 42
    assert manifest["trained_run_ids"] == ["run_00", "run_01", "run_02", "run_03"]
    assert manifest["temporal_feature_names"]
    assert manifest["static_feature_names"]
    assert manifest["regression_target"] == "peak_flooding_lps"
    assert manifest["regression_target_transform"] == "log1p"


def test_train_surrogate_returns_final_training_marker(tmp_path):
    dataset = _synthetic_surrogate_dataset(n_runs=4, n_nodes=3, T=8)
    result = train_surrogate(
        artifacts_dir=tmp_path / "artifacts",
        n_epochs=1,
        batch_size=8,
        n_cv_folds=2,
        _dataset=dataset,
    )
    assert result["final_model_trained_on_all_groups"] is True
