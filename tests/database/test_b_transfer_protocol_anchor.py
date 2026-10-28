import sqlite3

import pytest


def test_freeze_api_writes_protocol_and_external_append_only_anchor(tmp_path):
    from swmm_resilience.database.b_transfer_protocol import freeze_b_transfer_protocol

    root = tmp_path / "experiment"
    root.mkdir()
    inp = root / "network.inp"
    from tests.validation.test_network_admission import _FIXTURE
    inp.write_text(_FIXTURE, encoding="utf-8")
    anchor = tmp_path / "external" / "protocol-anchor.sqlite3"
    protocol = {
        "schema_version": "b_transfer_protocol_v2",
        "status": "frozen_before_run",
        "network": {"path": "network.inp"},
        "expected_scenario_keys": ["base"],
        "scenarios": [{
            "scenario_key": "base", "shape_id": "base", "factor_mult": 1.0,
            "duracion_horas": 1.0, "tiempo_al_pico_h": 0.5,
        }],
        "hydraulic_policy": {
            "max_continuity_error_pct": 5.0, "continuity_limit_rationale": "provisional",
            "drain_down_hours": 6.0, "drain_down_extension_hours": 2.0, "max_drain_retries": 2,
            "tail_window_hours": 1.0, "min_report_steps": 2,
            "inflow_tolerance_lps": 0.01, "flooding_tolerance_lps": 0.01,
            "storage_change_tolerance_m3": 0.01, "residual_flow_tolerance_lps": 0.01,
        },
    }

    receipt = freeze_b_transfer_protocol(
        root, protocol, anchor_db_path=anchor, git_commit="a" * 40,
        execution_context={"runner": "synthetic-test", "swmm_version": "fixture"},
    )

    assert (root / "protocol.json").is_file()
    assert receipt["protocol_sha256"]
    assert receipt["anchor_sha256"]
    with sqlite3.connect(anchor) as conn:
        assert conn.execute("SELECT COUNT(*) FROM protocol_anchors").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("UPDATE protocol_anchors SET git_commit=?", ("b" * 40,))
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute("DELETE FROM protocol_anchors")


def test_protocol_cannot_be_refrozen_over_existing_file(tmp_path):
    from swmm_resilience.database.b_transfer_protocol import freeze_b_transfer_protocol

    root = tmp_path / "experiment"
    root.mkdir()
    (root / "protocol.json").write_text("already frozen", encoding="utf-8")

    with pytest.raises(FileExistsError):
        freeze_b_transfer_protocol(
            root, {}, anchor_db_path=tmp_path / "anchor.sqlite3", git_commit="a" * 40,
            execution_context={"runner": "test"},
        )


def test_v1_protocol_is_rejected_without_frozen_retry_policy(tmp_path):
    from swmm_resilience.database.b_transfer_protocol import freeze_b_transfer_protocol
    from tests.validation.test_network_admission import _FIXTURE

    root = tmp_path / "experiment"
    root.mkdir()
    (root / "network.inp").write_text(_FIXTURE, encoding="utf-8")
    protocol = {
        "schema_version": "b_transfer_protocol_v1", "status": "frozen_before_run",
        "network": {"path": "network.inp"}, "expected_scenario_keys": ["base"],
        "scenarios": [{"scenario_key": "base", "shape_id": "base", "factor_mult": 1.0,
                       "duracion_horas": 1.0, "tiempo_al_pico_h": 0.5}],
        "hydraulic_policy": {
            "max_continuity_error_pct": 5.0, "continuity_limit_rationale": "test",
            "drain_down_hours": 6.0, "drain_down_extension_hours": 2.0, "max_drain_retries": 2,
            "tail_window_hours": 1.0, "min_report_steps": 2,
            "inflow_tolerance_lps": 0.01, "flooding_tolerance_lps": 0.01,
            "storage_change_tolerance_m3": 0.01, "residual_flow_tolerance_lps": 0.01,
        },
    }
    with pytest.raises(ValueError, match="b_transfer_protocol_v2"):
        freeze_b_transfer_protocol(
            root, protocol, anchor_db_path=tmp_path / "anchor.sqlite3", git_commit="a" * 40,
            execution_context={"runner": "test"},
        )
