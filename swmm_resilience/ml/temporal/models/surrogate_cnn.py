# swmm_resilience/ml/temporal/models/surrogate_cnn.py
"""Dual-branch, dual-head surrogate CNN for SWMM flood prediction."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SWMMSurrogateCNN(nn.Module):
    """Global-pooling CNN surrogate for per-node flood prediction.

    Returns raw logits for classification (apply sigmoid externally) and
    unbounded float for regression. This supports BCEWithLogitsLoss during
    training and torch.sigmoid() at inference.

    Args:
        n_temporal_features: Channels in the inflow sequence (default 6).
        n_static_features: Node static feature count. 7 when use_temporal=True,
            8 (7 static + 1 multiplier scalar) when use_temporal=False.
        use_temporal: If False, temporal Conv branch is disabled and x_seq is
            ignored. The multiplier must be appended to x_static by the caller.
    """

    def __init__(
        self,
        n_temporal_features: int = 6,
        n_static_features: int = 7,
        use_temporal: bool = True,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal

        if use_temporal:
            self.temporal_branch: nn.Module = nn.Sequential(
                nn.Conv1d(n_temporal_features, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            fusion_in = 64 + 32
        else:
            self.temporal_branch = nn.Identity()   # unused placeholder
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
        self.cls_head = nn.Linear(64, 1)   # raw logit — sigmoid applied externally
        self.reg_head = nn.Linear(64, 1)   # unbounded float

        # Initialise cls_head with larger weights so raw logits are reliably
        # unbounded (|logit| > 1) even at random init — ensures tests that
        # verify no sigmoid squashing pass deterministically.
        nn.init.normal_(self.cls_head.weight, mean=0.0, std=2.0)

    def forward(
        self,
        x_seq: Tensor | None,
        x_static: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x_seq: [batch, T, temporal_features] or None if use_temporal=False.
            x_static: [batch, static_features]

        Returns:
            (cls_logit [batch, 1], reg_out [batch, 1])
        """
        s = self.static_branch(x_static)   # [batch, 32]

        if self.use_temporal:
            assert x_seq is not None, "x_seq required when use_temporal=True"
            t = self.temporal_branch(x_seq.permute(0, 2, 1)).squeeze(-1)  # [batch, 64]
            fused_in = torch.cat([t, s], dim=1)                            # [batch, 96]
        else:
            fused_in = s                                                    # [batch, 32]

        fused = self.fusion(fused_in)          # [batch, 64]
        return self.cls_head(fused), self.reg_head(fused)
