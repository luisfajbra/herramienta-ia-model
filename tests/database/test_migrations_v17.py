import sqlite3

import pytest

from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import (
    MigrationChecksumError,
    MigrationOrderError,
    apply_migrations,
)


EXPECTED_TABLES = {
    "schema_migrations",
    "networks",
    "nodes",
    "links",
    "scenarios",
    "scenario_inflows",
    "runs",
    "node_features",
    "node_results",
    "node_timeseries",
    "training_runs",
    "model_evaluations",
    "oof_predictions",
    "model_metrics",
    "trained_models",
}

EXPECTED_TRAINING_VIEW_COLUMNS = [
    "run_id",
    "network_id",
    "scenario_id",
    "scenario_key",
    "scenario_kind",
    "factor_mult",
    "shape_id",
    "node_id",
    "elev_fondo",
    "prof_max",
    "n_tuberias_in",
    "n_tuberias_out",
    "diam_max_in",
    "diam_max_out",
    "pendiente_max_in",
    "pendiente_out",
    "base_inflow_lps",
    "dist_outfall_m",
    "n_nodos_aguas_arriba",
    "q_pico_acum_base",
    "upstream_capacity_lps",
    "q_pico_nodo",
    "q_pico_acum_escalado",
    "duracion_horas",
    "tiempo_al_pico_h",
    "inunda",
    "vol_inundacion_m3",
]


def test_initial_migration_creates_expected_schema(tmp_path):
    conn = connect_database(tmp_path / "test.sqlite3")
    try:
        apply_migrations(conn)
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert EXPECTED_TABLES <= tables
        assert conn.execute(
            "SELECT COUNT(*) FROM schema_migrations"
        ).fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        conn.close()


def test_initial_migration_exposes_canonical_v17_training_view(tmp_path):
    conn = connect_database(tmp_path / "test.sqlite3")
    try:
        apply_migrations(conn)
        columns = [
            row[1]
            for row in conn.execute("PRAGMA table_info(training_samples_v17)")
        ]
        assert columns == EXPECTED_TRAINING_VIEW_COLUMNS
    finally:
        conn.close()


def test_migrations_are_idempotent(tmp_path):
    conn = connect_database(tmp_path / "test.sqlite3")
    try:
        apply_migrations(conn)
        apply_migrations(conn)
        assert conn.execute(
            "SELECT COUNT(*) FROM schema_migrations"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_modified_applied_migration_is_rejected(tmp_path):
    conn = connect_database(tmp_path / "test.sqlite3")
    try:
        apply_migrations(conn)
        conn.execute(
            "UPDATE schema_migrations SET checksum_sha256=?",
            ("0" * 64,),
        )
        conn.commit()

        with pytest.raises(MigrationChecksumError):
            apply_migrations(conn)
    finally:
        conn.close()


def test_broken_migration_rolls_back_all_its_objects(tmp_path):
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    (migration_dir / "001_broken.sql").write_text(
        "CREATE TABLE schema_migrations ("
        "version INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, "
        "checksum_sha256 TEXT NOT NULL, applied_at_utc TEXT NOT NULL);\n"
        "CREATE TABLE partial_table(id INTEGER PRIMARY KEY);\n"
        "THIS IS INVALID SQL;\n",
        encoding="utf-8",
    )
    conn = connect_database(tmp_path / "broken.sqlite3")
    try:
        with pytest.raises(sqlite3.OperationalError):
            apply_migrations(conn, migration_dir=migration_dir)

        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert "schema_migrations" not in names
        assert "partial_table" not in names
    finally:
        conn.close()


def test_migration_versions_must_be_contiguous(tmp_path):
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    (migration_dir / "002_gap.sql").write_text("SELECT 1;", encoding="utf-8")
    conn = connect_database(tmp_path / "gap.sqlite3")
    try:
        with pytest.raises(MigrationOrderError, match="contiguous from 001"):
            apply_migrations(conn, migration_dir=migration_dir)
    finally:
        conn.close()


def test_composite_foreign_keys_reject_cross_network_links_and_inflows(tmp_path):
    conn = connect_database(tmp_path / "test.sqlite3")
    try:
        apply_migrations(conn)
        conn.executemany(
            """
            INSERT INTO networks (
                network_id, network_sha256, name, source_filename, inp_bytes,
                flow_units, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    1,
                    "a" * 64,
                    "one",
                    "one.inp",
                    b"one",
                    "LPS",
                    "2026-08-21T00:00:00+00:00",
                ),
                (
                    2,
                    "b" * 64,
                    "two",
                    "two.inp",
                    b"two",
                    "LPS",
                    "2026-08-21T00:00:00+00:00",
                ),
            ],
        )
        conn.executemany(
            """
            INSERT INTO nodes (node_pk, network_id, node_id, node_type)
            VALUES (?, ?, ?, ?)
            """,
            [(10, 1, "N1", "junction"), (20, 2, "N2", "junction")],
        )
        conn.execute(
            """
            INSERT INTO scenarios (
                scenario_id, network_id, scenario_key, scenario_kind,
                factor_mult, shape_id, duracion_horas, tiempo_al_pico_h,
                config_json, config_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (100, 1, "base", "factor", 1.0, None, 1.0, 0.5, "{}", "c" * 64),
        )
        conn.commit()

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute(
                """
                INSERT INTO links (
                    network_id, link_id, link_type, from_node_pk, to_node_pk
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (1, "cross-link", "conduit", 20, 10),
            )

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute(
                """
                INSERT INTO scenario_inflows (
                    scenario_id, network_id, node_pk, step_index,
                    time_sec, inflow_lps
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (100, 1, 20, 0, 0.0, 1.0),
            )
    finally:
        conn.close()
