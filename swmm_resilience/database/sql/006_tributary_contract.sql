-- Migration 006: tributary feature contract (tabular_v4_21)
--
-- Adds a new, parallel table `node_features_v4_21` mirroring `node_features`
-- (defined in 001_v17_initial.sql) but for the 21-feature tributary
-- contract. This migration is purely additive: it does not alter, drop, or
-- redefine `node_features` or any other existing table.
--
-- Relative to `node_features`, this table:
--   - drops `duracion_horas` and `tiempo_al_pico_h` (not part of the new
--     contract's static/scaled feature set);
--   - adds six tributary features: `volumen_local_integrado_m3`,
--     `volumen_tributario_integrado_m3`, `duracion_efectiva_tributario_h`,
--     `tiempo_primer_pico_tributario_h`, `tiempo_sobre_capacidad_h`, and
--     `integral_exceso_capacidad_m3`. The latter two are capacity-dependent
--     and NULL-able (no CHECK(...>= 0), since NULL is a valid value when
--     capacity is unknown); the other four keep a CHECK(... >= 0) guard
--     like their analogues in `node_features`;
--   - constrains `feature_contract_id` to 'tabular_v4_21' instead of
--     'tabular_v3_17'.

CREATE TABLE node_features_v4_21 (
    run_id INTEGER NOT NULL,
    network_id INTEGER NOT NULL,
    node_pk INTEGER NOT NULL,
    elev_fondo REAL NOT NULL,
    prof_max REAL NOT NULL CHECK(prof_max >= 0),
    n_tuberias_in INTEGER NOT NULL CHECK(n_tuberias_in >= 0),
    n_tuberias_out INTEGER NOT NULL CHECK(n_tuberias_out >= 0),
    diam_max_in REAL CHECK(diam_max_in IS NULL OR diam_max_in >= 0),
    diam_max_out REAL CHECK(diam_max_out IS NULL OR diam_max_out >= 0),
    pendiente_max_in REAL,
    pendiente_out REAL,
    base_inflow_lps REAL NOT NULL CHECK(base_inflow_lps >= 0),
    dist_outfall_m REAL CHECK(dist_outfall_m IS NULL OR dist_outfall_m >= 0),
    n_nodos_aguas_arriba INTEGER NOT NULL CHECK(n_nodos_aguas_arriba >= 0),
    q_pico_acum_base REAL NOT NULL CHECK(q_pico_acum_base >= 0),
    upstream_capacity_lps REAL CHECK(
        upstream_capacity_lps IS NULL OR upstream_capacity_lps >= 0
    ),
    q_pico_nodo REAL NOT NULL CHECK(q_pico_nodo >= 0),
    q_pico_acum_escalado REAL NOT NULL CHECK(q_pico_acum_escalado >= 0),
    volumen_local_integrado_m3 REAL NOT NULL CHECK(volumen_local_integrado_m3 >= 0),
    volumen_tributario_integrado_m3 REAL NOT NULL CHECK(
        volumen_tributario_integrado_m3 >= 0
    ),
    duracion_efectiva_tributario_h REAL NOT NULL CHECK(
        duracion_efectiva_tributario_h >= 0
    ),
    tiempo_primer_pico_tributario_h REAL NOT NULL CHECK(
        tiempo_primer_pico_tributario_h >= 0
    ),
    tiempo_sobre_capacidad_h REAL,
    integral_exceso_capacidad_m3 REAL,
    feature_contract_id TEXT NOT NULL CHECK(feature_contract_id='tabular_v4_21'),
    PRIMARY KEY(run_id, node_pk),
    FOREIGN KEY(run_id, network_id)
        REFERENCES runs(run_id, network_id) ON DELETE CASCADE,
    FOREIGN KEY(node_pk, network_id)
        REFERENCES nodes(node_pk, network_id) ON DELETE CASCADE
);
