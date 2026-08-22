import hashlib
import sqlite3

import pytest

import swmm_resilience.database.migrations as migrations_module
from swmm_resilience.database.connection import connect_database
from swmm_resilience.database.migrations import (
    MigrationChecksumError,
    MigrationOrderError,
    apply_migrations,
)


# Keep in sync with 005_provenance_integrity.sql as it grows across increments.
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
    "model_promotions",
    "model_selections",
    "schema_migration_validators",
    "training_run_provenance_invalidations",
}

EXPECTED_INDEXES = {
    "idx_runs_status_scenario",
    "idx_scenarios_network_kind",
    "idx_timeseries_run_time",
    "idx_oof_evaluation",
    "idx_metrics_owner",
    "idx_metrics_training_run",
    "idx_metrics_evaluation",
    "idx_metrics_model",
    "idx_models_training_target",
    "idx_models_target_contract",
    "idx_promotions_training_target",
    "idx_promotions_classifier",
    "idx_promotions_regressor",
    "idx_promotions_target_metric",
    "idx_selections_target_history",
}

SCHEMA_MIGRATIONS_SQL = """
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    checksum_sha256 TEXT NOT NULL CHECK(length(checksum_sha256)=64),
    applied_at_utc TEXT NOT NULL
);
"""

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
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """
            )
        }
        indexes = {
            row[0]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='index' AND name NOT LIKE 'sqlite_%'
                """
            )
        }
        assert tables == EXPECTED_TABLES
        assert indexes == EXPECTED_INDEXES
        assert conn.execute(
            "SELECT COUNT(*) FROM schema_migrations"
        ).fetchone()[0] == 5
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


def test_training_view_returns_only_complete_joined_rows(tmp_path):
    conn = connect_database(tmp_path / "view-rows.sqlite3")
    try:
        apply_migrations(conn)
        conn.execute(
            """
            INSERT INTO networks (
                network_id, network_sha256, name, source_filename, inp_bytes,
                flow_units, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "a" * 64, "one", "one.inp", b"one", "LPS", "2026-08-21"),
        )
        conn.execute(
            """
            INSERT INTO nodes (node_pk, network_id, node_id, node_type)
            VALUES (?, ?, ?, ?)
            """,
            (10, 1, "N1", "junction"),
        )
        conn.execute(
            """
            INSERT INTO scenarios (
                scenario_id, network_id, scenario_key, scenario_kind,
                factor_mult, shape_id, duracion_horas, tiempo_al_pico_h,
                config_json, config_sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (100, 1, "base", "factor", 1.0, None, 2.0, 1.0, "{}", "b" * 64),
        )
        conn.executemany(
            """
            INSERT INTO runs (
                run_id, scenario_id, network_id, status, config_sha256
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (1000, 100, 1, "COMPLETE", "c" * 64),
                (1001, 100, 1, "FAILED", "d" * 64),
            ],
        )
        conn.executemany(
            """
            INSERT INTO node_features (
                run_id, network_id, node_pk,
                elev_fondo, prof_max, n_tuberias_in, n_tuberias_out,
                diam_max_in, diam_max_out, pendiente_max_in, pendiente_out,
                base_inflow_lps, dist_outfall_m, n_nodos_aguas_arriba,
                q_pico_acum_base, upstream_capacity_lps, q_pico_nodo,
                q_pico_acum_escalado, duracion_horas, tiempo_al_pico_h,
                feature_contract_id
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                (
                    1000,
                    1,
                    10,
                    11.0,
                    2.0,
                    1,
                    1,
                    0.5,
                    0.6,
                    0.01,
                    0.02,
                    3.0,
                    100.0,
                    2,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    2.0,
                    1.0,
                    "tabular_v3_17",
                ),
                (
                    1001,
                    1,
                    10,
                    99.0,
                    2.0,
                    1,
                    1,
                    0.5,
                    0.6,
                    0.01,
                    0.02,
                    3.0,
                    100.0,
                    2,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    2.0,
                    1.0,
                    "tabular_v3_17",
                ),
            ],
        )
        conn.executemany(
            """
            INSERT INTO node_results (
                run_id, network_id, node_pk, inunda, vol_inundacion_m3
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [(1000, 1, 10, 1, 12.5), (1001, 1, 10, 0, 0.0)],
        )

        rows = conn.execute("SELECT * FROM training_samples_v17").fetchall()

        assert [tuple(row) for row in rows] == [
            (
                1000,
                1,
                100,
                "base",
                "factor",
                1.0,
                None,
                "N1",
                11.0,
                2.0,
                1,
                1,
                0.5,
                0.6,
                0.01,
                0.02,
                3.0,
                100.0,
                2,
                4.0,
                5.0,
                6.0,
                7.0,
                2.0,
                1.0,
                1,
                12.5,
            )
        ]
    finally:
        conn.close()


def test_migrations_are_idempotent(tmp_path):
    conn = connect_database(tmp_path / "test.sqlite3")
    try:
        apply_migrations(conn)
        apply_migrations(conn)
        assert conn.execute(
            "SELECT COUNT(*) FROM schema_migrations"
        ).fetchone()[0] == 5
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


def test_explicit_empty_migration_catalog_is_rejected_without_mutation(tmp_path):
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    conn = connect_database(tmp_path / "empty.sqlite3")
    try:
        with pytest.raises(MigrationOrderError, match="empty|001"):
            apply_migrations(conn, migration_dir=migration_dir)

        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_empty_packaged_migration_catalog_is_rejected_without_mutation(
    tmp_path,
    monkeypatch,
):
    package_root = tmp_path / "package"
    (package_root / "sql").mkdir(parents=True)
    monkeypatch.setattr(migrations_module, "files", lambda _package: package_root)
    conn = connect_database(tmp_path / "packaged-empty.sqlite3")
    try:
        with pytest.raises(MigrationOrderError, match="empty|001"):
            apply_migrations(conn)

        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_invalid_sql_migration_filename_is_rejected(tmp_path):
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    (migration_dir / "manual_patch.sql").write_text(
        "SELECT 1;",
        encoding="utf-8",
    )
    conn = connect_database(tmp_path / "invalid-name.sqlite3")
    try:
        with pytest.raises(MigrationOrderError, match="Invalid migration filename"):
            apply_migrations(conn, migration_dir=migration_dir)
    finally:
        conn.close()


def test_applied_history_must_be_a_prefix_before_any_new_migration(tmp_path):
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    first_sql = "CREATE TABLE must_not_be_created(id INTEGER PRIMARY KEY);"
    second_sql = "CREATE TABLE second_table(id INTEGER PRIMARY KEY);"
    (migration_dir / "001_first.sql").write_text(first_sql, encoding="utf-8")
    (migration_dir / "002_second.sql").write_text(second_sql, encoding="utf-8")

    conn = connect_database(tmp_path / "invalid-history.sqlite3")
    try:
        conn.executescript(SCHEMA_MIGRATIONS_SQL)
        conn.execute(
            """
            INSERT INTO schema_migrations (
                version, name, checksum_sha256, applied_at_utc
            ) VALUES (?, ?, ?, ?)
            """,
            (
                2,
                "second",
                hashlib.sha256(second_sql.encode("utf-8")).hexdigest(),
                "2026-08-21T00:00:00+00:00",
            ),
        )
        conn.commit()

        with pytest.raises(MigrationOrderError, match="prefix"):
            apply_migrations(conn, migration_dir=migration_dir)

        assert conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='must_not_be_created'
            """
        ).fetchone() is None
    finally:
        conn.close()


def test_applied_migration_missing_from_catalog_is_rejected(tmp_path):
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    conn = connect_database(tmp_path / "missing-file.sqlite3")
    try:
        conn.executescript(SCHEMA_MIGRATIONS_SQL)
        conn.execute(
            """
            INSERT INTO schema_migrations (
                version, name, checksum_sha256, applied_at_utc
            ) VALUES (?, ?, ?, ?)
            """,
            (1, "removed", "a" * 64, "2026-08-21T00:00:00+00:00"),
        )
        conn.commit()

        with pytest.raises(MigrationOrderError, match="catalog"):
            apply_migrations(conn, migration_dir=migration_dir)
    finally:
        conn.close()


def test_active_transaction_is_rejected_without_implicit_commit(tmp_path):
    conn = connect_database(tmp_path / "active-transaction.sqlite3")
    try:
        conn.execute("CREATE TABLE pending_rows(value TEXT NOT NULL)")
        conn.commit()
        conn.execute("INSERT INTO pending_rows VALUES ('uncommitted')")
        assert conn.in_transaction

        with pytest.raises(RuntimeError, match="active transaction"):
            apply_migrations(conn)

        conn.rollback()
        assert conn.execute("SELECT COUNT(*) FROM pending_rows").fetchone()[0] == 0
    finally:
        conn.close()


def test_internal_commit_cannot_escape_atomic_migration(tmp_path):
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    (migration_dir / "001_internal_commit.sql").write_text(
        SCHEMA_MIGRATIONS_SQL
        + "CREATE TABLE partial_table(id INTEGER PRIMARY KEY);\n"
        + "COMMIT;\n"
        + "THIS IS INVALID SQL;\n",
        encoding="utf-8",
    )
    conn = connect_database(tmp_path / "internal-commit.sqlite3")
    try:
        with pytest.raises(RuntimeError, match="transaction control"):
            apply_migrations(conn, migration_dir=migration_dir)

        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
        assert "schema_migrations" not in names
        assert "partial_table" not in names
    finally:
        conn.close()


@pytest.mark.parametrize(
    "transaction_sql",
    [
        "BEGIN;",
        "END;",
        "ROLLBACK;",
        "SAVEPOINT nested;",
        "RELEASE nested;",
    ],
)
def test_all_internal_transaction_control_is_rejected(tmp_path, transaction_sql):
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    (migration_dir / "001_transaction.sql").write_text(
        SCHEMA_MIGRATIONS_SQL + transaction_sql,
        encoding="utf-8",
    )
    conn = connect_database(tmp_path / "transaction-control.sqlite3")
    try:
        with pytest.raises(RuntimeError, match="transaction control"):
            apply_migrations(conn, migration_dir=migration_dir)

        assert conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type='table' AND name='schema_migrations'
            """
        ).fetchone() is None
    finally:
        conn.close()


def test_transaction_words_in_literals_comments_and_trigger_are_allowed(tmp_path):
    migration_dir = tmp_path / "migrations"
    migration_dir.mkdir()
    (migration_dir / "001_valid_words.sql").write_text(
        SCHEMA_MIGRATIONS_SQL
        + """
        -- COMMIT; ROLLBACK; SAVEPOINT ignored;
        /* BEGIN; END; RELEASE ignored; */
        CREATE TABLE source_rows(value TEXT NOT NULL);
        CREATE TABLE event_log(message TEXT NOT NULL);
        CREATE TABLE "commit"(message TEXT NOT NULL);
        INSERT INTO "commit" VALUES ('BEGIN; COMMIT; ROLLBACK;');
        CREATE TRIGGER log_source_insert
        AFTER INSERT ON source_rows
        BEGIN
            INSERT INTO event_log VALUES ('END; RELEASE;');
        END;
        """,
        encoding="utf-8",
    )
    conn = connect_database(tmp_path / "valid-words.sqlite3")
    try:
        apply_migrations(conn, migration_dir=migration_dir)
        conn.execute("INSERT INTO source_rows VALUES ('row')")

        assert conn.execute("SELECT message FROM event_log").fetchone()[0] == (
            "END; RELEASE;"
        )
        assert conn.execute('SELECT message FROM "commit"').fetchone()[0] == (
            "BEGIN; COMMIT; ROLLBACK;"
        )
    finally:
        conn.close()


def test_composite_foreign_keys_reject_all_cross_network_rows(tmp_path):
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
        conn.execute(
            """
            INSERT INTO runs (
                run_id, scenario_id, network_id, status, config_sha256
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (1000, 100, 1, "COMPLETE", "d" * 64),
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

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, scenario_id, network_id, status, config_sha256
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (1001, 100, 2, "PENDING", "e" * 64),
            )

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute(
                """
                INSERT INTO node_features (
                    run_id, network_id, node_pk, elev_fondo, prof_max,
                    n_tuberias_in, n_tuberias_out, base_inflow_lps,
                    n_nodos_aguas_arriba, q_pico_acum_base, q_pico_nodo,
                    q_pico_acum_escalado, duracion_horas, tiempo_al_pico_h,
                    feature_contract_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    1000,
                    1,
                    20,
                    0.0,
                    1.0,
                    0,
                    0,
                    0.0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.5,
                    "tabular_v3_17",
                ),
            )

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute(
                """
                INSERT INTO node_results (
                    run_id, network_id, node_pk, inunda, vol_inundacion_m3
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (1000, 1, 20, 0, 0.0),
            )

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            conn.execute(
                """
                INSERT INTO node_timeseries (
                    run_id, network_id, node_pk, step_index, time_sec,
                    total_inflow_lps, lateral_inflow_lps, depth_m,
                    depth_ratio, flooding_lps, total_outflow_lps, failed_now
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (1000, 1, 20, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0),
            )
    finally:
        conn.close()
