import pandas as pd

from swmm_resilience.dataset.shape_selection import (
    BASE_SHAPE_ID,
    base_shape_rows,
    augmented_shape_rows,
)


def _df():
    return pd.DataFrame(
        {
            "node_id": ["J1", "J1", "J1"],
            "factor_mult": [1.0, 1.0, 2.0],
            "shape_id": ["base", "storm_a", "base"],
            "vol_inundacion_m3": [0.0, 1.0, 2.0],
        }
    )


def test_base_shape_rows_keeps_only_base():
    out = base_shape_rows(_df())
    assert set(out["shape_id"].unique()) == {BASE_SHAPE_ID}
    assert len(out) == 2


def test_augmented_shape_rows_drops_base():
    out = augmented_shape_rows(_df())
    assert set(out["shape_id"].unique()) == {"storm_a"}
    assert len(out) == 1


def test_base_shape_rows_passthrough_when_no_shape_id_column():
    legacy = pd.DataFrame({"node_id": ["J1", "J2"], "factor_mult": [1.0, 2.0]})
    out = base_shape_rows(legacy)
    assert out.equals(legacy)


def test_augmented_shape_rows_empty_when_no_shape_id_column():
    legacy = pd.DataFrame({"node_id": ["J1", "J2"], "factor_mult": [1.0, 2.0]})
    out = augmented_shape_rows(legacy)
    assert list(out.columns) == list(legacy.columns)
    assert out.empty


def test_helpers_do_not_mutate_or_alias_input():
    df = _df()
    base = base_shape_rows(df)
    base["vol_inundacion_m3"] = -99.0
    assert (df["vol_inundacion_m3"] == [0.0, 1.0, 2.0]).all()
