-- Migration 007: training_samples_v4_21 view
--
-- Espejo de training_samples_v17 (001_v17_initial.sql) pero contra
-- node_features_v4_21 (006_tributary_contract.sql) en vez de node_features.
-- Puramente aditiva: no toca ninguna tabla ni vista existente.

CREATE VIEW training_samples_v4_21 AS
SELECT
    r.run_id,
    r.network_id,
    r.scenario_id,
    s.scenario_key,
    s.scenario_kind,
    s.factor_mult,
    s.shape_id,
    n.node_id,
    f.elev_fondo,
    f.prof_max,
    f.n_tuberias_in,
    f.n_tuberias_out,
    f.diam_max_in,
    f.diam_max_out,
    f.pendiente_max_in,
    f.pendiente_out,
    f.base_inflow_lps,
    f.dist_outfall_m,
    f.n_nodos_aguas_arriba,
    f.q_pico_acum_base,
    f.upstream_capacity_lps,
    f.q_pico_nodo,
    f.q_pico_acum_escalado,
    f.volumen_local_integrado_m3,
    f.volumen_tributario_integrado_m3,
    f.duracion_efectiva_tributario_h,
    f.tiempo_primer_pico_tributario_h,
    f.tiempo_sobre_capacidad_h,
    f.integral_exceso_capacidad_m3,
    o.inunda,
    o.vol_inundacion_m3
FROM runs AS r
JOIN scenarios AS s ON s.scenario_id = r.scenario_id
JOIN node_features_v4_21 AS f
    ON f.run_id = r.run_id AND f.network_id = r.network_id
JOIN node_results AS o
    ON o.run_id = f.run_id
    AND o.node_pk = f.node_pk
    AND o.network_id = f.network_id
JOIN nodes AS n
    ON n.node_pk = f.node_pk AND n.network_id = f.network_id
WHERE r.status = 'COMPLETE';
