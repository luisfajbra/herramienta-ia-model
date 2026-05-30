# tests/ml/temporal/test_surrogate_cnn.py
"""Tests for SWMMSurrogateCNN dual-head surrogate model."""
from __future__ import annotations

import pytest
import torch

from swmm_resilience.ml.temporal.models.surrogate_cnn import SWMMSurrogateCNN


class TestForwardPassFullModel:
    def test_output_shapes(self):
        model = SWMMSurrogateCNN(n_temporal_features=6, n_static_features=7)
        x_seq = torch.randn(8, 24, 6)     # batch=8, T=24, features=6
        x_static = torch.randn(8, 7)
        cls_logit, reg_out = model(x_seq, x_static)
        assert cls_logit.shape == (8, 1), f"cls_logit shape: {cls_logit.shape}"
        assert reg_out.shape == (8, 1),   f"reg_out shape: {reg_out.shape}"

    def test_classification_logit_is_unbounded(self):
        """cls_logit is raw logit — not forced to [0,1] (sigmoid applied externally)."""
        model = SWMMSurrogateCNN(n_temporal_features=6, n_static_features=7)
        x_seq = torch.randn(32, 24, 6) * 10   # large inputs to saturate sigmoid
        x_static = torch.randn(32, 7) * 10
        cls_logit, _ = model(x_seq, x_static)
        # At least some logits should fall outside [0, 1]
        assert (cls_logit.abs() > 1).any(), "Expected unbounded logits"

    def test_variable_sequence_length(self):
        """AdaptiveAvgPool must handle different T values with same output shape."""
        model = SWMMSurrogateCNN(n_temporal_features=6, n_static_features=7)
        for T in [10, 20, 40]:
            cls_logit, reg_out = model(torch.randn(4, T, 6), torch.randn(4, 7))
            assert cls_logit.shape == (4, 1)
            assert reg_out.shape == (4, 1)

    def test_regression_output_unbounded(self):
        """Regression head must be linear — no activation."""
        model = SWMMSurrogateCNN(n_temporal_features=6, n_static_features=7)
        torch.manual_seed(0)
        x_seq = torch.randn(16, 24, 6) * 100
        x_static = torch.randn(16, 7) * 100
        _, reg_out = model(x_seq, x_static)
        assert reg_out.min() < 0 or reg_out.max() > 1, "Regression head should not be bounded"


class TestNoTemporalMode:
    def test_output_shapes_no_temporal(self):
        """use_temporal=False: n_static_features=8 (7 static + 1 multiplier)."""
        model = SWMMSurrogateCNN(n_temporal_features=6, n_static_features=8, use_temporal=False)
        x_static = torch.randn(8, 8)
        cls_logit, reg_out = model(None, x_static)
        assert cls_logit.shape == (8, 1)
        assert reg_out.shape == (8, 1)

    def test_no_temporal_ignores_x_seq(self):
        """Passing x_seq=None must not raise when use_temporal=False."""
        model = SWMMSurrogateCNN(n_temporal_features=6, n_static_features=8, use_temporal=False)
        x_static = torch.randn(4, 8)
        cls_logit, reg_out = model(None, x_static)   # must not raise
        assert cls_logit.shape == (4, 1)


class TestGradientFlow:
    def test_gradients_flow_through_both_heads(self):
        model = SWMMSurrogateCNN(n_temporal_features=6, n_static_features=7)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x_seq = torch.randn(4, 24, 6, requires_grad=False)
        x_static = torch.randn(4, 7, requires_grad=False)
        cls_logit, reg_out = model(x_seq, x_static)

        y_cls = torch.zeros(4, 1)
        y_reg = torch.zeros(4, 1)
        loss = torch.nn.BCEWithLogitsLoss()(cls_logit, y_cls) + 0.01 * torch.nn.MSELoss()(reg_out, y_reg)
        optimizer.zero_grad()
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients found"
        assert all(g.abs().sum() > 0 for g in grads), "Some gradients are all-zero"
