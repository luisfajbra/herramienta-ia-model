import pandas as pd
import pytest

from swmm_resilience.analysis import resilience
from swmm_resilience.visualization import resilience_curve


def swmm_df():
    rows = []
    for factor in [1.0, 2.0]:
        for node_idx in range(4):
            rows.append({
                "node_id": f"J{node_idx}",
                "factor_mult": factor,
                "inunda": 1 if (factor == 2.0 and node_idx >= 2) else 0,
                "vol_inundacion_m3": 10.0 if (factor == 2.0 and node_idx >= 2) else 0.0,
            })
    return pd.DataFrame(rows)


def test_compute_resilience_swmm_values():
    df = swmm_df()
    factors = [1.0, 2.0]

    result = resilience.compute_resilience_curve(
        df_swmm=df,
        factors=factors,
        config=None,
        models_dir=None,
        _predict_fn=None,
    )

    assert list(result["factor"]) == [1.0, 2.0]
    assert result.loc[result["factor"] == 1.0, "resilience_swmm"].iloc[0] == pytest.approx(1.0)
    assert result.loc[result["factor"] == 2.0, "resilience_swmm"].iloc[0] == pytest.approx(0.5)


def test_compute_resilience_ml_values():
    import pandas as pd

    def fake_predict(factor, config, models_dir):
        return pd.DataFrame({
            "node_id": ["J0", "J1", "J2", "J3"],
            "inunda_pred": [0, 0, 1, 1] if factor == 2.0 else [0, 0, 0, 0],
            "vol_pred_m3": [0.0, 0.0, 5.0, 5.0] if factor == 2.0 else [0.0] * 4,
        })

    df = swmm_df()
    factors = [1.0, 2.0]

    result = resilience.compute_resilience_curve(
        df_swmm=df,
        factors=factors,
        config=None,
        models_dir=None,
        _predict_fn=fake_predict,
    )

    assert result.loc[result["factor"] == 1.0, "resilience_ml"].iloc[0] == pytest.approx(1.0)
    assert result.loc[result["factor"] == 2.0, "resilience_ml"].iloc[0] == pytest.approx(0.5)


def test_plot_resilience_curve_writes_png(tmp_path):
    df = pd.DataFrame({
        "factor": [1.0, 2.0],
        "resilience_swmm": [1.0, 0.5],
        "resilience_ml": [1.0, 0.6],
    })

    output = resilience_curve.plot_resilience_curve(df, tmp_path / "resilience.png")

    assert output == tmp_path / "resilience.png"
    assert (tmp_path / "resilience.png").exists()
    assert (tmp_path / "resilience.png").stat().st_size > 0
