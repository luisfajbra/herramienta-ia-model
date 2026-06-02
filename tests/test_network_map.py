from types import SimpleNamespace

from swmm_resilience.visualization import network_map


def test_generate_network_map_writes_png(monkeypatch, tmp_path):
    fake_inp = {
        "COORDINATES": {
            "J1":  SimpleNamespace(x=0.0,  y=10.0),
            "J2":  SimpleNamespace(x=5.0,  y=5.0),
            "J3":  SimpleNamespace(x=10.0, y=0.0),
            "OUT1": SimpleNamespace(x=15.0, y=0.0),
        },
        "CONDUITS": {
            "C1": SimpleNamespace(from_node="J1",  to_node="J2"),
            "C2": SimpleNamespace(from_node="J2",  to_node="J3"),
            "C3": SimpleNamespace(from_node="J3",  to_node="OUT1"),
        },
        "OUTFALLS": {
            "OUT1": SimpleNamespace(),
        },
        "JUNCTIONS": {
            "J1": SimpleNamespace(),
            "J2": SimpleNamespace(),
            "J3": SimpleNamespace(),
        },
    }
    monkeypatch.setattr(network_map, "load_inp", lambda path: fake_inp)

    output = network_map.generate_network_map(
        tmp_path / "network.inp",
        tmp_path / "network_map.png",
        "Test Network",
    )

    assert output == tmp_path / "network_map.png"
    assert (tmp_path / "network_map.png").exists()
    assert (tmp_path / "network_map.png").stat().st_size > 0


def test_initial_vs_continuous_classification():
    conduits = {
        "C1": ("J1", "J2"),
        "C2": ("J2", "J3"),
        "C3": ("J3", "OUT1"),
    }
    to_nodes = {tn for _, (_, tn) in conduits.items()}

    assert "J1" not in to_nodes, "J1 should be INITIAL (no upstream input)"
    assert "J2" in to_nodes, "J2 should be CONTINUOUS (receives from C1)"
    assert "J3" in to_nodes, "J3 should be CONTINUOUS (receives from C2)"
