import pytest

from swmm_resilience import main


def test_delta_inflows_lps_is_rejected_before_running_simulation(monkeypatch):
    def fail_if_simulation_is_loaded():
        raise AssertionError("Simulation stack should not be loaded for rejected inputs")

    monkeypatch.setattr(main, "_load_pyswmm", fail_if_simulation_is_loaded)

    with pytest.raises(ValueError) as exc_info:
        main.run_experiment(
            delta_inflows_lps=[10.0],
        )

    message = str(exc_info.value)
    assert "delta_inflows_lps" in message
    assert "inflow_multipliers" in message
