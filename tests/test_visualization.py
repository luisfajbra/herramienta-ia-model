from types import SimpleNamespace

import pandas as pd

from swmm_resilience.visualization import flood_map


def test_generate_flood_map_writes_png(monkeypatch, tmp_path):
    fake_inp = {
        "COORDINATES": {
            "J1": SimpleNamespace(x=0.0, y=0.0),
            "J2": SimpleNamespace(x=10.0, y=0.0),
        },
        "CONDUITS": {
            "C1": SimpleNamespace(from_node="J1", to_node="J2"),
        },
    }
    monkeypatch.setattr(flood_map, "load_inp", lambda path: fake_inp)

    output = flood_map.generate_flood_map(
        tmp_path / "network.inp",
        pd.DataFrame({"node_id": ["J1", "J2"], "vol_inundacion_m3": [0.0, 4.0]}),
        1.0,
        tmp_path / "map.png",
        "Test Network",
    )

    assert output == tmp_path / "map.png"
    assert (tmp_path / "map.png").exists()
    assert (tmp_path / "map.png").stat().st_size > 0


def test_generate_flood_maps_by_shape_writes_one_folder_per_shape(monkeypatch, tmp_path):
    fake_inp = {
        "COORDINATES": {
            "J1": SimpleNamespace(x=0.0, y=0.0),
            "J2": SimpleNamespace(x=10.0, y=0.0),
        },
        "CONDUITS": {
            "C1": SimpleNamespace(from_node="J1", to_node="J2"),
        },
    }
    monkeypatch.setattr(flood_map, "load_inp", lambda path: fake_inp)

    dataset = pd.DataFrame({
        "node_id": ["J1", "J2", "J1", "J2", "J1", "J2"],
        "factor_mult": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0],
        "shape_id": ["base", "base", "storm_a", "storm_a", "base", "base"],
        "vol_inundacion_m3": [0.0, 4.0, 0.0, 9.0, 0.0, 12.0],
    })

    written = flood_map.generate_flood_maps_by_shape(
        tmp_path / "network.inp", dataset, tmp_path / "maps", "Test Network",
    )

    assert set(written) == {"base", "storm_a"}
    assert (tmp_path / "maps" / "base" / "flood_map_factor_1.00.png").exists()
    assert (tmp_path / "maps" / "base" / "flood_map_factor_2.00.png").exists()
    assert (tmp_path / "maps" / "storm_a" / "flood_map_factor_1.00.png").exists()
    assert not (tmp_path / "maps" / "storm_a" / "flood_map_factor_2.00.png").exists()


def test_generate_flood_maps_by_shape_requires_shape_id_column(tmp_path):
    dataset = pd.DataFrame({
        "node_id": ["J1"], "factor_mult": [1.0], "vol_inundacion_m3": [0.0],
    })
    try:
        flood_map.generate_flood_maps_by_shape(
            tmp_path / "network.inp", dataset, tmp_path / "maps", "Test Network",
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "shape_id" in str(exc)
