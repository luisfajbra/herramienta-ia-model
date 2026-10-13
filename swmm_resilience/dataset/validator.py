import pandas as pd


def validate_dataset(df: pd.DataFrame, n_nodes: int, n_factors: int) -> None:
    """Validate training dataset before fitting. Raises ValueError on fatal issues."""
    for col in ("inunda", "vol_inundacion_m3"):
        if df[col].isna().any():
            raise ValueError(f"NaN in label column '{col}' — check the .rpt")

    if (df["vol_inundacion_m3"] < 0).any():
        raise ValueError("vol_inundacion_m3 contains negative values")

    invalid = ~df["inunda"].isin([0, 1])
    if invalid.any():
        raise ValueError("inunda contains values outside {0, 1}")

    expected = n_nodes * n_factors
    if len(df) != expected:
        raise ValueError(
            f"Expected rows: {n_nodes} nodes × {n_factors} factors = {expected}, "
            f"found: {len(df)}"
        )

    if df["inunda"].sum() == 0:
        raise ValueError(
            "No node floods anywhere in the dataset — verify that SWMM produces "
            "flooding within the configured factor range"
        )

    ratio = df["inunda"].mean()
    if ratio < 0.05:
        print(f"WARNING: only {ratio:.1%} of rows flood (very low ratio, check the factor range)")
