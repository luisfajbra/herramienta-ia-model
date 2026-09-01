"""Per-factor SWMM versus XGBoost comparison generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..dataset.shape_selection import base_shape_rows
from ..ml.predict import predict_network
from ..visualization.model_comparison import (
    plot_factor_comparison,
    plot_flooded_swmm_node_profile,
)


def generate_factor_comparisons(
    dataset_path: Path,
    config,
    models_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Generate node-volume and parity plots for every dataset factor."""
    dataset = base_shape_rows(pd.read_csv(dataset_path))
    required = {"node_id", "factor_mult", "vol_inundacion_m3"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for factor in sorted(dataset["factor_mult"].astype(float).unique()):
        swmm = dataset[
            dataset["factor_mult"].astype(float).sub(factor).abs() < 1e-9
        ][["node_id", "vol_inundacion_m3"]].copy()
        swmm["node_id"] = swmm["node_id"].astype(str)
        swmm = swmm.rename(columns={"vol_inundacion_m3": "vol_swmm_m3"})
        predicted = predict_network(factor, config, Path(models_dir))[
            ["node_id", "vol_pred_m3"]
        ].copy()
        predicted["node_id"] = predicted["node_id"].astype(str)

        if swmm["node_id"].duplicated().any() or predicted["node_id"].duplicated().any():
            raise ValueError(f"Duplicate node IDs found for factor {factor:.2f}")
        if set(swmm["node_id"]) != set(predicted["node_id"]):
            raise ValueError(
                f"SWMM and XGBoost node sets differ for factor {factor:.2f}"
            )

        comparison = swmm.merge(predicted, on="node_id", how="inner")
        paths.extend(plot_factor_comparison(comparison, output_dir, factor))
        flooded_profile = plot_flooded_swmm_node_profile(
            comparison, output_dir, factor
        )
        if flooded_profile is not None:
            paths.append(flooded_profile)

    return paths
