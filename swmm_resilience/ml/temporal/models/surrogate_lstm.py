# swmm_resilience/ml/temporal/models/surrogate_lstm.py
"""Dual-branch, dual-head surrogate LSTM for SWMM flood prediction."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SWMMSurrogateLSTM(nn.Module):
    """LSTM surrogate for per-node flood prediction.

    Mirrors SWMMSurrogateCNN — only the temporal branch differs.
    Returns raw logits for classification (apply sigmoid externally) and
    unbounded float for regression. Constructor signature is identical to
    SWMMSurrogateCNN so both can be used interchangeably in _train_eval_model().

    Args:
        n_temporal_features: LSTM input size (default 2 — inflow channels).
        n_static_features: Static feature count (default 21 for unified dataset).
        use_temporal: If False, LSTM branch disabled and x_seq is ignored.
    """

    def __init__(
        self,
        n_temporal_features: int = 2,
        n_static_features: int = 21,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal

        if use_temporal:
            self.temporal_branch = nn.LSTM(
                input_size=n_temporal_features,
                hidden_size=64,
                num_layers=2,
                batch_first=True,
                dropout=0.2,
            )
            fusion_in = 64 + 32
        else:
            self.temporal_branch = nn.Identity()  # unused placeholder
            fusion_in = 32

        self.static_branch = nn.Sequential(
            nn.Linear(n_static_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.cls_head = nn.Linear(64, 1)  # raw logit — sigmoid applied externally
        self.reg_head = nn.Linear(64, 1)  # unbounded float

        # Larger init ensures |logit| > 1 at random init for test reliability.
        nn.init.normal_(self.cls_head.weight, mean=0.0, std=2.0)

    def forward(
        self,
        x_seq: Tensor | None,
        x_static: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x_seq: [batch, T, n_temporal_features] or None if use_temporal=False.
            x_static: [batch, n_static_features]

        Returns:
            (cls_logit [batch, 1], reg_out [batch, 1])
        """
        s = self.static_branch(x_static)  # [batch, 32]

        if self.use_temporal:
            assert x_seq is not None, "x_seq required when use_temporal=True"
            _, (h_n, _) = self.temporal_branch(x_seq)  # h_n: [num_layers, batch, 64]
            t = h_n[-1]                                  # last layer: [batch, 64]
            fused_in = torch.cat([t, s], dim=1)          # [batch, 96]
        else:
            fused_in = s                                  # [batch, 32]

        fused = self.fusion(fused_in)  # [batch, 64]
        return self.cls_head(fused), self.reg_head(fused)
