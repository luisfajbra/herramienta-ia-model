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
