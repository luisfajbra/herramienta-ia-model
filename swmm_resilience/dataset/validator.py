import pandas as pd


def validate_dataset(df: pd.DataFrame, n_nodes: int, n_factors: int) -> None:
    """Validate training dataset before fitting. Raises ValueError on fatal issues."""
    for col in ("inunda", "vol_inundacion_m3"):
        if df[col].isna().any():
            raise ValueError(f"NaN en columna de etiqueta '{col}' — revisa el .rpt")

    if (df["vol_inundacion_m3"] < 0).any():
        raise ValueError("vol_inundacion_m3 contiene valores negativos")

    invalid = ~df["inunda"].isin([0, 1])
    if invalid.any():
        raise ValueError("inunda contiene valores fuera de {0, 1}")

    expected = n_nodes * n_factors
    if len(df) != expected:
        raise ValueError(
            f"Filas esperadas: {n_nodes} nodos × {n_factors} factores = {expected}, "
            f"encontradas: {len(df)}"
        )

    if df["inunda"].sum() == 0:
        raise ValueError(
            "Ningún nodo inunda en todo el dataset — verifica que SWMM produce "
            "flooding dentro del rango de factores configurado"
        )

    ratio = df["inunda"].mean()
    if ratio < 0.05:
        print(f"ADVERTENCIA: solo {ratio:.1%} de filas inundan (ratio muy bajo, verifica el rango de factores)")
