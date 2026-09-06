"""Informe de calidad del dataset preparado.

Restricción: sólo pandas/numpy. Ver tests/ml/bench/test_stage1_imports.py.
"""

from __future__ import annotations

import pandas as pd


def build_quality_report(
    keys: pd.DataFrame,
    X: pd.DataFrame,
    y_clf: pd.Series,
    y_reg: pd.Series,
    flood_threshold_m3: float,
) -> dict:
    """Implementado en la Task 4."""
    raise NotImplementedError
