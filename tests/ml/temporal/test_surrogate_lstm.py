"""Tests for SWMMSurrogateLSTM dual-head surrogate model."""
from __future__ import annotations

import torch
import pytest

from swmm_resilience.ml.temporal.models.surrogate_lstm import SWMMSurrogateLSTM


class TestForwardPassFullModel:
    def test_output_shapes(self):
        model = SWMMSurrogateLSTM(n_temporal_features=2, n_static_features=21)
        x_seq = torch.randn(8, 20, 2)
        x_static = torch.randn(8, 21)
        cls_logit, reg_out = model(x_seq, x_static)
        assert cls_logit.shape == (8, 1), f"cls_logit: {cls_logit.shape}"
        assert reg_out.shape == (8, 1), f"reg_out: {reg_out.shape}"

    def test_classification_logit_is_unbounded(self):
        """cls_logit is raw logit — not in [0,1]."""
        model = SWMMSurrogateLSTM(n_temporal_features=2, n_static_features=21)
        x_seq = torch.randn(32, 20, 2) * 10
        x_static = torch.randn(32, 21) * 10
        cls_logit, _ = model(x_seq, x_static)
        assert (cls_logit.abs() > 1).any(), "Expected unbounded logits"

    def test_variable_sequence_length(self):
        """LSTM handles variable T natively."""
        model = SWMMSurrogateLSTM(n_temporal_features=2, n_static_features=21)
        for T in [5, 20, 50]:
            cls_logit, reg_out = model(torch.randn(4, T, 2), torch.randn(4, 21))
            assert cls_logit.shape == (4, 1)
            assert reg_out.shape == (4, 1)

    def test_regression_output_unbounded(self):
        model = SWMMSurrogateLSTM(n_temporal_features=2, n_static_features=21)
        torch.manual_seed(0)
        x_seq = torch.randn(16, 20, 2) * 100
        x_static = torch.randn(16, 21) * 100
        _, reg_out = model(x_seq, x_static)
        assert reg_out.min() < 0 or reg_out.max() > 1


class TestNoTemporalMode:
    def test_output_shapes_no_temporal(self):
        model = SWMMSurrogateLSTM(n_temporal_features=2, n_static_features=21, use_temporal=False)
        cls_logit, reg_out = model(None, torch.randn(8, 21))
        assert cls_logit.shape == (8, 1)
        assert reg_out.shape == (8, 1)

    def test_x_seq_none_accepted(self):
        model = SWMMSurrogateLSTM(n_temporal_features=2, n_static_features=21, use_temporal=False)
        cls_logit, reg_out = model(None, torch.randn(4, 21))
        assert cls_logit.shape == (4, 1)


class TestGradientFlow:
    def test_gradients_flow_through_both_heads(self):
        model = SWMMSurrogateLSTM(n_temporal_features=2, n_static_features=21)
        x_seq = torch.randn(4, 20, 2)
        x_static = torch.randn(4, 21)
        cls_logit, reg_out = model(x_seq, x_static)

        loss = (
            torch.nn.BCEWithLogitsLoss()(cls_logit, torch.zeros(4, 1))
            + 0.01 * torch.nn.MSELoss()(reg_out, torch.zeros(4, 1))
        )
        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients"
        assert all(g.abs().sum() > 0 for g in grads), "Zero gradients found"
