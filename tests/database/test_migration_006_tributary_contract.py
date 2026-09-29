# tests/database/test_migration_006_tributary_contract.py
import sqlite3

import pytest


def test_migration_006_creates_node_features_v4_21(tmp_path):
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.migrations import apply_migrations

    db_path = tmp_path / "test.sqlite3"
    conn = connect_managed_database(db_path)
    apply_migrations(conn)

    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(node_features_v4_21)").fetchall()
    }
    expected = {
        "run_id", "network_id", "node_pk", "feature_contract_id",
        "volumen_local_integrado_m3", "volumen_tributario_integrado_m3",
        "duracion_efectiva_tributario_h", "tiempo_primer_pico_tributario_h",
        "tiempo_sobre_capacidad_h", "integral_exceso_capacidad_m3",
    }
    assert expected.issubset(columns)
    conn.close()


def test_migration_006_does_not_alter_the_legacy_node_features_table(tmp_path):
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.migrations import apply_migrations

    db_path = tmp_path / "test.sqlite3"
    conn = connect_managed_database(db_path)
    apply_migrations(conn)

    columns = {row[1] for row in conn.execute("PRAGMA table_info(node_features)").fetchall()}
    assert "volumen_local_integrado_m3" not in columns
    conn.close()


def _seed_network_and_run(conn):
    """Mirrors tests/database/test_migration_005_candidates.py's
    `_seed_network_and_two_runs` fixture pattern -- builds the minimal
    networks/scenarios/nodes/runs rows needed to satisfy the
    (run_id, network_id) and (node_pk, network_id) foreign keys that
    node_features_v4_21 shares with node_features."""
    conn.execute(
        """
        INSERT INTO networks (network_id, network_sha256, name, source_filename,
                               inp_bytes, flow_units, created_at_utc)
        VALUES (1, ?, 'net', 'net.inp', ?, 'LPS', '2026-08-22T00:00:00+00:00')
        """,
        ("f" * 64, b"inp-bytes"),
    )
    conn.execute(
        """
        INSERT INTO scenarios (scenario_id, network_id, scenario_key, scenario_kind,
                                factor_mult, shape_id, duracion_horas,
                                tiempo_al_pico_h, config_json, config_sha256)
        VALUES (1, 1, 'baseline', 'baseline', NULL, NULL, 1.0, 0.5, '{}', ?)
        """,
        ("e" * 64,),
    )
    conn.execute(
        "INSERT INTO nodes (node_pk, network_id, node_id, node_type) "
        "VALUES (1, 1, 'N1', 'JUNCTION')"
    )
    conn.execute(
        """
        INSERT INTO runs (run_id, network_id, scenario_id, status,
                           config_sha256, node_count)
        VALUES (1, 1, 1, 'COMPLETE', ?, 1)
        """,
        ("d" * 64,),
    )
    conn.commit()


def test_migration_006_rejects_a_row_with_the_wrong_contract_id(tmp_path):
    from swmm_resilience.database.connection import connect_managed_database
    from swmm_resilience.database.migrations import apply_migrations

    db_path = tmp_path / "test.sqlite3"
    conn = connect_managed_database(db_path)
    apply_migrations(conn)

    _seed_network_and_run(conn)

    # Insertar con un feature_contract_id incorrecto debe violar el CHECK.
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            """
            INSERT INTO node_features_v4_21 (
                run_id, network_id, node_pk, elev_fondo, prof_max,
                n_tuberias_in, n_tuberias_out, diam_max_in, diam_max_out,
                pendiente_max_in, pendiente_out, base_inflow_lps,
                dist_outfall_m, n_nodos_aguas_arriba, q_pico_acum_base,
                upstream_capacity_lps, q_pico_nodo, q_pico_acum_escalado,
                volumen_local_integrado_m3, volumen_tributario_integrado_m3,
                duracion_efectiva_tributario_h, tiempo_primer_pico_tributario_h,
                tiempo_sobre_capacidad_h, integral_exceso_capacidad_m3,
                feature_contract_id
            ) VALUES (
                1, 1, 1, 10.0, 2.0,
                1, 1, 0.3, 0.3,
                0.01, 0.01, 5.0,
                100.0, 0, 1.0,
                50.0, 1.0, 1.0,
                2.0, 3.0,
                1.5, 0.5,
                NULL, NULL,
                'tabular_v3_17'
            )
            """
        )
    conn.rollback()
    conn.close()
