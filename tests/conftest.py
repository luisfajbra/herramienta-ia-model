import matplotlib
matplotlib.use('Agg')

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17

FEATURE_COLS = list(FEATURE_COLUMNS_V17)


def _qualify_synthetic_training_runs(conn):
    """Mark ordinary toy runs as clean, verified fixtures."""
    qualify_synthetic_runs(conn, max_continuity_error_pct=5.0)


def qualify_synthetic_runs(
    conn,
    *,
    max_continuity_error_pct: float = 5.0,
    default_continuity_error_pct: float | None = 0.0,
):
    """Attach internally consistent QA manifests to synthetic database rows.

    A missing continuity value is filled only when the fixture explicitly asks
    for a known-good default. Passing ``None`` keeps it unknown and ineligible.
    Existing finite values are preserved so tests can exercise threshold logic.
    """
    import hashlib
    import json

    policy = {
        "max_continuity_error_pct": float(max_continuity_error_pct),
        "continuity_limit_rationale": "synthetic fixture policy",
    }
    policy_json = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    policy_sha256 = hashlib.sha256(policy_json.encode("utf-8")).hexdigest()
    initial_condition = {
        "policy_id": "synthetic_fixture_initial_state_v1",
        "state_sha256": "a" * 64,
    }
    initial_json = json.dumps(initial_condition, sort_keys=True, separators=(",", ":"))

    rows = conn.execute(
        "SELECT run_id, continuity_error_pct FROM runs ORDER BY run_id"
    ).fetchall()
    for run_id, observed_continuity in rows:
        continuity = observed_continuity
        if continuity is None and default_continuity_error_pct is not None:
            continuity = float(default_continuity_error_pct)
            conn.execute(
                "UPDATE runs SET continuity_error_pct=? WHERE run_id=?",
                (continuity, run_id),
            )
        eligible = continuity is not None and abs(float(continuity)) <= max_continuity_error_pct
        reason_codes = [] if eligible else [
            "continuity_unavailable" if continuity is None else "continuity_exceeds_limit"
        ]
        manifest = {
            "eligible": eligible,
            "reason_codes": reason_codes,
            "continuity_error_pct": continuity,
            "finite_coverage_ok": eligible,
            "drainage_state": "drained" if eligible else "unverified",
            "flooding_report": {
                "parser": "synthetic_fixture",
                "parser_version": "1",
                "report_sha256": "b" * 64,
            },
        }
        manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        conn.execute(
            """INSERT INTO run_hydraulic_quality (
                run_id, policy_version, policy_json, policy_sha256,
                continuity_error_pct, finite_coverage_ok, drainage_state,
                initial_condition_json, eligible, reason_codes_json,
                manifest_json, created_at_utc
            ) VALUES (?, 'test-quality-v1', ?, ?, ?, ?, ?, ?, ?, ?, ?,
                      '2026-10-26T00:00:00Z')""",
            (
                run_id,
                policy_json,
                policy_sha256,
                continuity,
                int(eligible),
                "drained" if eligible else "unverified",
                initial_json,
                int(eligible),
                json.dumps(reason_codes, separators=(",", ":")),
                manifest_json,
            ),
        )


@pytest.fixture
def tiny_config_factory(tmp_path: Path):
    def _factory(algorithm: str = "random_forest", use_scaler: bool = False):
        inp = tmp_path / "network.inp"
        inp.write_text("network", encoding="utf-8")
        return SimpleNamespace(
            network=SimpleNamespace(inp_path=inp),
            ml=SimpleNamespace(
                classifier=SimpleNamespace(
                    algorithm=algorithm,
                    n_estimators=5,
                    max_depth=2,
                    learning_rate=0.1,
                    subsample=1.0,
                    scale_pos_weight="auto",
                ),
                regressor=SimpleNamespace(
                    algorithm=algorithm,
                    n_estimators=5,
                    max_depth=2,
                    learning_rate=0.1,
                    subsample=1.0,
                ),
                use_scaler=use_scaler,
            ),
        )

    return _factory


@pytest.fixture
def trainer_training_df():
    rows = []
    for i in range(8):
        row = {col: float(i + 1) for col in FEATURE_COLS}
        row["inunda"] = 1 if i >= 4 else 0
        row["vol_inundacion_m3"] = float(i * 10) if i >= 4 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def csv_shaped_dataset() -> pd.DataFrame:
    """A dataset_final.csv-shaped frame: the same 24 columns the assembler writes."""
    rows = []
    for shape_id in ("base", "storm_a"):
        for factor in (1.0, 2.0, 3.0):
            for node_idx in range(4):
                flooded = 1 if (node_idx < 2 and factor >= 2.0) else 0
                rows.append(
                    {
                        "node_id": f"N{node_idx}",
                        "elev_fondo": 100.0 + node_idx,
                        "prof_max": 1.5,
                        "n_tuberias_in": 1,
                        "n_tuberias_out": 1,
                        "diam_max_in": 0.2,
                        "diam_max_out": 0.2,
                        "pendiente_max_in": 0.01,
                        "pendiente_out": 0.01,
                        "base_inflow_lps": 5.0 + node_idx,
                        "dist_outfall_m": 100.0 + node_idx * 10,
                        "n_nodos_aguas_arriba": node_idx,
                        "q_pico_acum_base": 10.0,
                        "upstream_capacity_lps": 50.0,
                        "coord_x": float(node_idx),
                        "coord_y": float(node_idx * 2),
                        "factor_mult": factor,
                        "q_pico_nodo": 2.0 * factor,
                        "q_pico_acum_escalado": 10.0 * factor,
                        "duracion_horas": 3.0,
                        "tiempo_al_pico_h": 0.7,
                        "shape_id": shape_id,
                        "vol_inundacion_m3": 50.0 * factor if flooded else 0.0,
                        "inunda": flooded,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def sql_training_db(tmp_path: Path, csv_shaped_dataset: pd.DataFrame) -> Path:
    """Path to a migrated v17 database holding csv_shaped_dataset's rows."""
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.csv_backfill import backfill_networks_and_runs
    from swmm_resilience.database.migrations import apply_migrations

    inp_path = tmp_path / "fixture_network.inp"
    inp_path.write_text("[TITLE]\nfixture network\n", encoding="utf-8")
    db_path = tmp_path / "training_v17.sqlite3"
    conn = connect_managed_database(db_path)
    try:
        apply_migrations(conn)
        backfill_networks_and_runs(
            conn, csv_shaped_dataset, inp_path, "Fixture Network"
        )
        _qualify_synthetic_training_runs(conn)
        conn.commit()
    finally:
        conn.close()
    return db_path


@pytest.fixture
def csv_shaped_dataset_six_factors() -> pd.DataFrame:
    """Like csv_shaped_dataset but with 6 distinct factor_mult values.

    GroupKFold5 needs at least 5 groups; the 3 factors of csv_shaped_dataset
    are not enough. Same 24 columns and same flooding rule, so the two
    fixtures stay comparable. Added as a sibling on purpose: the existing
    fixtures must keep behaving exactly as before.
    """
    rows = []
    for shape_id in ("base", "storm_a"):
        for factor in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
            for node_idx in range(4):
                flooded = 1 if (node_idx < 2 and factor >= 2.0) else 0
                rows.append(
                    {
                        "node_id": f"N{node_idx}",
                        "elev_fondo": 100.0 + node_idx,
                        "prof_max": 1.5,
                        "n_tuberias_in": 1,
                        "n_tuberias_out": 1,
                        "diam_max_in": 0.2,
                        "diam_max_out": 0.2,
                        "pendiente_max_in": 0.01,
                        "pendiente_out": 0.01,
                        "base_inflow_lps": 5.0 + node_idx,
                        "dist_outfall_m": 100.0 + node_idx * 10,
                        "n_nodos_aguas_arriba": node_idx,
                        "q_pico_acum_base": 10.0,
                        "upstream_capacity_lps": 50.0,
                        "coord_x": float(node_idx),
                        "coord_y": float(node_idx * 2),
                        "factor_mult": factor,
                        "q_pico_nodo": 2.0 * factor,
                        "q_pico_acum_escalado": 10.0 * factor,
                        "duracion_horas": 3.0,
                        "tiempo_al_pico_h": 0.7,
                        "shape_id": shape_id,
                        "vol_inundacion_m3": 50.0 * factor if flooded else 0.0,
                        "inunda": flooded,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def sql_training_db_six_factors(
    tmp_path: Path, csv_shaped_dataset_six_factors: pd.DataFrame
) -> Path:
    """Path to a migrated v17 database with 6 factor groups (GroupKFold5-capable)."""
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.csv_backfill import backfill_networks_and_runs
    from swmm_resilience.database.migrations import apply_migrations

    inp_path = tmp_path / "fixture_network_six.inp"
    inp_path.write_text("[TITLE]\nfixture network six\n", encoding="utf-8")
    db_path = tmp_path / "training_v17_six.sqlite3"
    conn = connect_managed_database(db_path)
    try:
        apply_migrations(conn)
        backfill_networks_and_runs(
            conn, csv_shaped_dataset_six_factors, inp_path, "Fixture Network Six"
        )
        _qualify_synthetic_training_runs(conn)
        conn.commit()
    finally:
        conn.close()
    return db_path


@pytest.fixture
def csv_shaped_dataset_six_shapes() -> pd.DataFrame:
    """Like csv_shaped_dataset but with 6 distinct shape_id values.

    GroupKFold(5) para tuning necesita al menos 5 grupos de shape_id; las
    2 formas de csv_shaped_dataset no alcanzan. Sibling explicito, igual
    que csv_shaped_dataset_six_factors: las fixtures existentes no cambian.
    """
    rows = []
    shapes = ("base", "storm_a", "storm_b", "storm_c", "storm_d", "storm_e")
    for shape_id in shapes:
        for factor in (1.0, 2.0, 3.0):
            for node_idx in range(4):
                flooded = 1 if (node_idx < 2 and factor >= 2.0) else 0
                rows.append(
                    {
                        "node_id": f"N{node_idx}",
                        "elev_fondo": 100.0 + node_idx,
                        "prof_max": 1.5,
                        "n_tuberias_in": 1,
                        "n_tuberias_out": 1,
                        "diam_max_in": 0.2,
                        "diam_max_out": 0.2,
                        "pendiente_max_in": 0.01,
                        "pendiente_out": 0.01,
                        "base_inflow_lps": 5.0 + node_idx,
                        "dist_outfall_m": 100.0 + node_idx * 10,
                        "n_nodos_aguas_arriba": node_idx,
                        "q_pico_acum_base": 10.0,
                        "upstream_capacity_lps": 50.0,
                        "coord_x": float(node_idx),
                        "coord_y": float(node_idx * 2),
                        "factor_mult": factor,
                        "q_pico_nodo": 2.0 * factor,
                        "q_pico_acum_escalado": 10.0 * factor,
                        "duracion_horas": 3.0,
                        "tiempo_al_pico_h": 0.7,
                        "shape_id": shape_id,
                        "vol_inundacion_m3": 50.0 * factor if flooded else 0.0,
                        "inunda": flooded,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def sql_training_db_six_shapes(
    tmp_path: Path, csv_shaped_dataset_six_shapes: pd.DataFrame
) -> Path:
    """Path to a migrated v17 database with 6 shape_id groups (GroupKFold(5)-by-shape-capable)."""
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.csv_backfill import backfill_networks_and_runs
    from swmm_resilience.database.migrations import apply_migrations

    inp_path = tmp_path / "fixture_network_shapes.inp"
    inp_path.write_text("[TITLE]\nfixture network shapes\n", encoding="utf-8")
    db_path = tmp_path / "training_v17_shapes.sqlite3"
    conn = connect_managed_database(db_path)
    try:
        apply_migrations(conn)
        backfill_networks_and_runs(
            conn, csv_shaped_dataset_six_shapes, inp_path, "Fixture Network Shapes"
        )
        _qualify_synthetic_training_runs(conn)
        conn.commit()
    finally:
        conn.close()
    return db_path


_REAL_METRICS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"


def _snapshot(directory: Path) -> dict:
    return {
        str(path): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture(autouse=True)
def _tests_must_not_write_the_real_outputs_metrics():
    """Una prueba que escribe en outputs/metrics real falsea figuras y procedencia."""
    if not _REAL_METRICS_DIR.is_dir():
        before = None
    else:
        before = _snapshot(_REAL_METRICS_DIR)
    yield
    if not _REAL_METRICS_DIR.is_dir():
        return
    after = _snapshot(_REAL_METRICS_DIR)
    if before is None:
        before = {}
    changed = sorted(path for path in after if before.get(path) != after[path])
    assert not changed, (
        "La prueba escribio en el outputs/metrics real (aislala con "
        f"monkeypatch de main.METRICS_DIR / MODELS_DIR): {changed}"
    )
