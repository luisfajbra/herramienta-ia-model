import matplotlib
matplotlib.use('Agg')

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17

FEATURE_COLS = list(FEATURE_COLUMNS_V17)


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
    finally:
        conn.close()
    return db_path
