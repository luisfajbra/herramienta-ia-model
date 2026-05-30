"""TDD tests for SWMMTemporalCNN and train_cnn (SP3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from swmm_resilience.ml.temporal.models.cnn import SWMMTemporalCNN
from swmm_resilience.ml.temporal.schemas import TemporalWindowDataset
from swmm_resilience.ml.temporal.train_cnn import train_cnn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_dataset(n_per_group: int = 20) -> TemporalWindowDataset:
    """Minimal dataset: 2 run_id groups, n_per_group samples each."""
    N = n_per_group * 2
    rng = np.random.RandomState(42)
    groups = np.array(["run_a"] * n_per_group + ["run_b"] * n_per_group, dtype=object)
    return TemporalWindowDataset(
        X_seq=rng.randn(N, 4, 6).astype(np.float32),
        X_static=rng.randn(N, 7).astype(np.float32),
        y_class=rng.randint(0, 2, N).astype(np.int8),
        y_reg=rng.rand(N).astype(np.float32),
        groups=groups,
        meta=pd.DataFrame(
            {
                "run_id": groups,
                "node_id": ["J-000"] * N,
                "window_start_min": np.arange(N, dtype=float),
            }
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForwardPassClassification:
    def test_output_shape_and_range(self):
        model = SWMMTemporalCNN(n_temporal_features=6, n_static_features=7, task="classification")
        x_seq = torch.randn(8, 4, 6)
        x_static = torch.randn(8, 7)
        out = model(x_seq, x_static)
        assert out.shape == (8, 1), f"Expected (8,1), got {out.shape}"
        assert (out >= 0).all() and (out <= 1).all(), "Classification output must be in [0, 1]"


class TestForwardPassRegression:
    def test_output_shape(self):
        model = SWMMTemporalCNN(n_temporal_features=6, n_static_features=7, task="regression")
        x_seq = torch.randn(8, 4, 6)
        x_static = torch.randn(8, 7)
        out = model(x_seq, x_static)
        assert out.shape == (8, 1), f"Expected (8,1), got {out.shape}"


class TestTrainingLossDecreases:
    def test_loss_decreases_over_5_epochs(self):
        torch.manual_seed(0)
        model = SWMMTemporalCNN(n_temporal_features=6, n_static_features=7, task="classification")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        criterion = nn.BCELoss()
        x_seq = torch.randn(32, 4, 6)
        x_static = torch.randn(32, 7)
        y = torch.zeros(32, 1)

        losses = []
        for _ in range(5):
            optimizer.zero_grad()
            out = model(x_seq, x_static)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], f"Loss did not decrease: {losses}"


class TestArtifactsSavedAfterTraining:
    def test_all_artifacts_exist(self, tmp_path):
        dataset = _synthetic_dataset()
        artifacts_dir = tmp_path / "artifacts"

        train_cnn(
            artifacts_dir=artifacts_dir,
            task="classification",
            n_epochs=2,
            batch_size=16,
            n_cv_folds=2,
            _dataset=dataset,
        )

        assert (artifacts_dir / "cnn_classifier_weights.pt").exists()
        assert (artifacts_dir / "cnn_classifier_scaler_seq.joblib").exists()
        assert (artifacts_dir / "cnn_classifier_scaler_static.joblib").exists()
        assert (artifacts_dir / "cnn_classifier_metrics.csv").exists()


class TestNoDataLeakageBetweenFolds:
    def test_train_val_groups_disjoint(self, tmp_path):
        dataset = _synthetic_dataset()

        result = train_cnn(
            artifacts_dir=tmp_path / "artifacts",
            task="classification",
            n_epochs=1,
            batch_size=16,
            n_cv_folds=2,
            _dataset=dataset,
        )

        for fold in result["folds"]:
            train_groups = set(fold["train_groups"])
            val_groups = set(fold["val_groups"])
            overlap = train_groups & val_groups
            assert not overlap, f"Data leakage in fold {fold['fold']}: {overlap}"
