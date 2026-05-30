"""Dual-branch CNN 1D for SWMM temporal node data."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SWMMTemporalCNN(nn.Module):
    """CNN 1D + dense static branch for per-node classification or regression.

    Args:
        n_temporal_features: Number of time-series channels (default 6).
        n_static_features: Number of static node features (default 7).
        task: 'classification' (Sigmoid output in [0,1]) or 'regression' (linear output).
    """

    def __init__(
        self,
        n_temporal_features: int = 6,
        n_static_features: int = 7,
        task: str = "classification",
    ) -> None:
        super().__init__()
        if task not in ("classification", "regression"):
            raise ValueError(
                f"task must be 'classification' or 'regression', got {task!r}"
            )
        self.task = task

        # Temporal branch: Conv1d operates on [batch, channels, length]
        # Input arrives as [batch, timesteps=4, features=6] and is permuted before this branch.
        self.temporal_branch = nn.Sequential(
            nn.Conv1d(n_temporal_features, 32, kernel_size=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=2, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # → [batch, 64, 1]
        )

        # Static branch
        self.static_branch = nn.Sequential(
            nn.Linear(n_static_features, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )

        # Fusion: 64 (temporal) + 32 (static) = 96
        self.fusion = nn.Sequential(
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Task head
        if task == "classification":
            self.head: nn.Module = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        else:
            self.head = nn.Linear(64, 1)

    def forward(self, x_seq: Tensor, x_static: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x_seq: [batch, timesteps, temporal_features]
            x_static: [batch, static_features]

        Returns:
            [batch, 1] — probability for classification, raw value for regression.
        """
        # Permute for Conv1d: [batch, features, timesteps]
        t = self.temporal_branch(x_seq.permute(0, 2, 1)).squeeze(-1)  # [batch, 64]
        s = self.static_branch(x_static)                               # [batch, 32]
        fused = self.fusion(torch.cat([t, s], dim=1))                  # [batch, 64]
        return self.head(fused)                                         # [batch, 1]
