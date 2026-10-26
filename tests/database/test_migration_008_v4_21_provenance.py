import hashlib
import sqlite3

import pytest

from swmm_resilience.database.connection import connect_managed_database
from swmm_resilience.database.migrations import apply_migrations


INP_BYTES = b"network-for-migration-008"


def _seed_run(conn):
    network_sha = hashlib.sha256(INP_BYTES).hexdigest()
    conn.execute(
        """INSERT INTO networks (network_id, network_sha256, name, source_filename,
                   inp_bytes, flow_units, created_at_utc)
           VALUES (1, ?, 'net', 'net.inp', ?, 'LPS', '2026-10-25T00:00:00+00:00')""",
        (network_sha, INP_BYTES),
    )
    conn.execute(
        """INSERT INTO scenarios (scenario_id, network_id, scenario_key, scenario_kind,
                   factor_mult, shape_id, duracion_horas, tiempo_al_pico_h,
                   config_json, config_sha256)
           VALUES (1, 1, 'baseline', 'baseline', NULL, NULL, 1.0, 0.5, '{}', ?)""",
        ("e" * 64,),
    )
    conn.execute(
        "INSERT INTO nodes (node_pk, network_id, node_id, node_type) VALUES (1, 1, 'N1', 'JUNCTION')"
    )
    conn.execute(
        "INSERT INTO nodes (node_pk, network_id, node_id, node_type) VALUES (2, 1, 'N2', 'JUNCTION')"
    )
    conn.execute(
        """INSERT INTO runs (run_id, network_id, scenario_id, status, config_sha256,
                   node_count, continuity_error_pct)
           VALUES (1, 1, 1, 'COMPLETE', ?, 1, 0.1)""",
        ("d" * 64,),
    )


def _insert_feature(conn, run_id=1, value=2.0, node_pk=1):
    conn.execute(
        """INSERT INTO node_features_v4_21 (
            run_id, network_id, node_pk, elev_fondo, prof_max, n_tuberias_in,
            n_tuberias_out, diam_max_in, diam_max_out, pendiente_max_in,
            pendiente_out, base_inflow_lps, dist_outfall_m,
            n_nodos_aguas_arriba, q_pico_acum_base, upstream_capacity_lps,
            q_pico_nodo, q_pico_acum_escalado, volumen_local_integrado_m3,
            volumen_tributario_integrado_m3, duracion_efectiva_tributario_h,
            tiempo_primer_pico_tributario_h, tiempo_sobre_capacidad_h,
            integral_exceso_capacidad_m3, feature_contract_id
        ) VALUES (?, 1, ?, 10.0, 2.0, 1, 1, 0.3, 0.3, 0.01, 0.01,
                  5.0, 100.0, 0, 1.0, 50.0, 1.0, 1.0, ?, 3.0, 1.5, 0.5,
                  NULL, NULL, 'tabular_v4_21')""",
        (run_id, node_pk, value),
    )


def _pin_run(conn):
    conn.execute(
        """INSERT INTO training_runs (
            training_run_id, target, feature_contract_id, feature_contract_sha256,
            query_sql, query_params_json, included_run_ids_json, grouping_strategy,
            fold_count, random_seed, primary_metric, tie_breakers_json,
            python_version, library_versions_json, status,
            training_query_contract_id, training_query_contract_sha256
        ) VALUES (1, 'inunda', 'tabular_v3_17', ?, 'SELECT 1', '{}', '[1]',
                  'group_kfold', 2, 42, 'roc_auc', '[]', '3.11', '{}', 'PENDING',
                  'training_samples_v17', ?)""",
        ("a" * 64, "b" * 64),
    )
    conn.execute("INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (1, 1)")
    conn.commit()


def _database(tmp_path):
    conn = connect_managed_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)
    _seed_run(conn)
    _insert_feature(conn)
    conn.commit()
    return conn


def test_pinned_v4_21_features_cannot_be_inserted_updated_or_deleted(tmp_path):
    conn = _database(tmp_path)
    _pin_run(conn)

    with pytest.raises(sqlite3.IntegrityError, match="pinned by training provenance"):
        _insert_feature(conn, run_id=1, value=9.0, node_pk=2)
    conn.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="pinned by training provenance"):
        conn.execute("UPDATE node_features_v4_21 SET q_pico_nodo=9 WHERE run_id=1")
    conn.rollback()
    with pytest.raises(sqlite3.IntegrityError, match="pinned by training provenance"):
        conn.execute("DELETE FROM node_features_v4_21 WHERE run_id=1")
    conn.rollback()
    conn.close()


def test_v4_21_feature_cannot_be_moved_into_a_pinned_run(tmp_path):
    conn = _database(tmp_path)
    conn.execute(
        """INSERT INTO runs (run_id, network_id, scenario_id, status,
                   config_sha256, node_count, continuity_error_pct)
           VALUES (2, 1, 1, 'COMPLETE', ?, 1, 0.1)""",
        ("c" * 64,),
    )
    _insert_feature(conn, run_id=2, node_pk=2)
    conn.commit()
    _pin_run(conn)

    with pytest.raises(sqlite3.IntegrityError, match="pinned by training provenance"):
        conn.execute(
            "UPDATE node_features_v4_21 SET run_id=1 WHERE run_id=2 AND node_pk=2"
        )
    conn.rollback()
    conn.close()


def test_pinned_run_hydraulic_quality_fields_are_immutable(tmp_path):
    conn = _database(tmp_path)
    _pin_run(conn)

    with pytest.raises(sqlite3.IntegrityError, match="pinned by training provenance"):
        conn.execute("UPDATE runs SET continuity_error_pct=99 WHERE run_id=1")
    conn.rollback()
    conn.close()


def test_unpinned_v4_21_features_and_run_quality_remain_editable(tmp_path):
    conn = _database(tmp_path)

    conn.execute("UPDATE node_features_v4_21 SET q_pico_nodo=9 WHERE run_id=1")
    conn.execute("DELETE FROM node_features_v4_21 WHERE run_id=1")
    _insert_feature(conn)
    conn.execute("UPDATE runs SET continuity_error_pct=99 WHERE run_id=1")
    assert conn.execute("SELECT q_pico_nodo FROM node_features_v4_21").fetchone()[0] == 1.0
    assert conn.execute("SELECT continuity_error_pct FROM runs").fetchone()[0] == 99
    conn.close()


def test_migration_008_is_additive_and_keeps_migration_006_checksum(tmp_path):
    expected_006_sha256 = "4a73a42cf5cbc443453a5d43769eb0d290a53f17212beb4bd8791a521821f51b"
    conn = connect_managed_database(tmp_path / "db.sqlite3")
    apply_migrations(conn)

    recorded = conn.execute(
        "SELECT checksum_sha256 FROM schema_migrations WHERE version=6"
    ).fetchone()[0]
    assert recorded == expected_006_sha256
    assert conn.execute(
        "SELECT name FROM schema_migrations WHERE version=8"
    ).fetchone()[0] == "v4_21_provenance_guards"
    conn.close()
