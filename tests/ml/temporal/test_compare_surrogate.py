"""TDD tests for compare_surrogate() pipeline."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swmm_resilience.ml.temporal.schemas import TemporalWindowDataset
from swmm_resilience.ml.temporal.compare_surrogate import compare_surrogate


def _synthetic_dataset(
    n_runs: int = 4, n_nodes: int = 12, T: int = 20, n_static: int = 21
) -> TemporalWindowDataset:
    N = n_runs * n_nodes
    rng = np.random.RandomState(0)
    groups = np.array(
        [f"run_{i:02d}" for i in range(n_runs) for _ in range(n_nodes)], dtype=object
    )
    meta = pd.DataFrame({
        "run_id": groups,
        "node_id": [f"J-{j:03d}" for j in range(n_nodes)] * n_runs,
        "window_start_min": [0.0] * N,
    })
    meta.attrs["static_feature_names"] = [f"feat_{i}" for i in range(n_static)]
    return TemporalWindowDataset(
        X_seq=rng.randn(N, T, 2).astype(np.float32),
        X_static=rng.randn(N, n_static).astype(np.float32),
        y_class=rng.randint(0, 2, N).astype(np.int8),
        y_reg=(rng.rand(N) * 50).astype(np.float32),
        groups=groups,
        meta=meta,
    )


class TestReturnShape:
    def test_returns_dataframe(self, tmp_path):
        ds = _synthetic_dataset()
        result = compare_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        assert isinstance(result, pd.DataFrame)

    def test_one_row_per_fold(self, tmp_path):
        ds = _synthetic_dataset()
        result = compare_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        assert len(result) == 2


class TestMetricColumns:
    def test_xgb_metrics_present(self, tmp_path):
        ds = _synthetic_dataset()
        result = compare_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        for col in ["xgb_auc_roc", "xgb_f1", "xgb_precision", "xgb_recall", "xgb_accuracy"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_cnn_full_metrics_present(self, tmp_path):
        ds = _synthetic_dataset()
        result = compare_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        for col in ["cnn_auc_roc", "cnn_f1", "cnn_rmse"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_cnn_ablation_metrics_present(self, tmp_path):
        ds = _synthetic_dataset()
        result = compare_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        for col in ["cnn_abl_auc_roc", "cnn_abl_f1", "cnn_abl_rmse"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_lstm_metrics_present(self, tmp_path):
        ds = _synthetic_dataset()
        result = compare_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        for col in ["lstm_auc_roc", "lstm_f1", "lstm_rmse"]:
            assert col in result.columns, f"Missing column: {col}"


class TestNoDataLeakage:
    def test_train_val_groups_disjoint(self, tmp_path):
        ds = _synthetic_dataset()
        result = compare_surrogate(
            artifacts_dir=tmp_path / "artifacts",
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        for _, row in result.iterrows():
            train_set = set(row["train_groups"])
            val_set = set(row["val_groups"])
            assert not (train_set & val_set), f"Leakage in fold {row['fold']}"


class TestArtifactsSaved:
    def test_csv_saved(self, tmp_path):
        ds = _synthetic_dataset()
        artifacts_dir = tmp_path / "artifacts"
        compare_surrogate(
            artifacts_dir=artifacts_dir,
            n_epochs=1, batch_size=16, n_cv_folds=2,
            _dataset=ds,
        )
        assert (artifacts_dir / "comparison_results.csv").exists()
