import pandas as pd
import pytest

from swmm_resilience.analysis import flood_volume


def swmm_df():
    rows = []
    for factor in [1.0, 2.0]:
        for node_idx in range(4):
            flooded = factor == 2.0 and node_idx >= 2
            rows.append({
                "node_id": f"J{node_idx}",
                "factor_mult": factor,
                "inunda": 1 if flooded else 0,
                "vol_inundacion_m3": 10.0 if flooded else 0.0,
            })
    return pd.DataFrame(rows)


def test_compute_flood_volume_swmm_values():
    df = swmm_df()
    factors = [1.0, 2.0]

    result = flood_volume.compute_flood_volume_curve(
        df_swmm=df,
        factors=factors,
        config=None,
        models_dir=None,
        _predict_fn=None,
    )

    assert list(result["factor"]) == [1.0, 2.0]
    assert result.loc[result["factor"] == 1.0, "vol_total_swmm"].iloc[0] == pytest.approx(0.0)
    assert result.loc[result["factor"] == 2.0, "vol_total_swmm"].iloc[0] == pytest.approx(20.0)


def test_compute_flood_volume_ml_values():
    def fake_predict(factor, config, models_dir):
        return pd.DataFrame({
            "node_id": ["J0", "J1", "J2", "J3"],
            "inunda_pred": [0, 0, 1, 1] if factor == 2.0 else [0, 0, 0, 0],
            "vol_pred_m3": [0.0, 0.0, 15.0, 5.0] if factor == 2.0 else [0.0] * 4,
        })

    df = swmm_df()
    factors = [1.0, 2.0]

    result = flood_volume.compute_flood_volume_curve(
        df_swmm=df,
        factors=factors,
        config=None,
        models_dir=None,
        _predict_fn=fake_predict,
    )

    assert result.loc[result["factor"] == 1.0, "vol_total_ml"].iloc[0] == pytest.approx(0.0)
    assert result.loc[result["factor"] == 2.0, "vol_total_ml"].iloc[0] == pytest.approx(20.0)
