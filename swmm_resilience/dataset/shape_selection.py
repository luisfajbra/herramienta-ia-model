"""Select rows of an assembled dataset by hydrograph shape.

``dataset_final.csv`` carries one row per node x factor x hydrograph shape.
Most downstream consumers (resilience/volume curves, SWMM-vs-model factor
comparisons, the flat per-factor flood maps) only ever want the canonical
``base`` shape; the extra shapes exist purely as training augmentation.
These helpers centralise that filter — including the ``shape_id``-less
fallback for datasets produced before shape tracking existed — so the
``if "shape_id" in df.columns`` dance is written once.
"""

from __future__ import annotations

import pandas as pd

BASE_SHAPE_ID = "base"


def base_shape_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for the canonical ``base`` hydrograph shape.

    A dataset with no ``shape_id`` column predates shape augmentation and is
    already base-only, so it is returned unchanged.
    """
    if "shape_id" not in df.columns:
        return df
    return df[df["shape_id"] == BASE_SHAPE_ID]


def augmented_shape_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows for every non-``base`` hydrograph shape (the augmentation set).

    A dataset with no ``shape_id`` column has no augmented shapes, so an
    empty frame with the same columns is returned.
    """
    if "shape_id" not in df.columns:
        return df.iloc[0:0]
    return df[df["shape_id"] != BASE_SHAPE_ID]
