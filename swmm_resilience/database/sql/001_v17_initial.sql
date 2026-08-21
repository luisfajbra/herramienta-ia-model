CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    checksum_sha256 TEXT NOT NULL CHECK(length(checksum_sha256)=64),
    applied_at_utc TEXT NOT NULL
);

CREATE TABLE networks (
    network_id INTEGER PRIMARY KEY,
    network_sha256 TEXT NOT NULL UNIQUE CHECK(length(network_sha256)=64),
    name TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    inp_bytes BLOB NOT NULL,
    flow_units TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TABLE nodes (
    node_pk INTEGER PRIMARY KEY,
    network_id INTEGER NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    node_id TEXT NOT NULL,
    node_type TEXT NOT NULL,
    coord_x REAL,
    coord_y REAL,
    invert_elevation_m REAL,
    max_depth_m REAL,
    base_inflow_lps REAL,
    UNIQUE(network_id, node_id),
    UNIQUE(node_pk, network_id)
);

CREATE TABLE links (
    link_pk INTEGER PRIMARY KEY,
    network_id INTEGER NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    link_id TEXT NOT NULL,
    link_type TEXT NOT NULL,
    from_node_pk INTEGER NOT NULL,
    to_node_pk INTEGER NOT NULL,
    length_m REAL,
    diameter_m REAL,
    slope_m_per_m REAL,
    roughness REAL,
    UNIQUE(network_id, link_id),
    FOREIGN KEY(from_node_pk, network_id) REFERENCES nodes(node_pk, network_id),
    FOREIGN KEY(to_node_pk, network_id) REFERENCES nodes(node_pk, network_id)
);

CREATE TABLE scenarios (
    scenario_id INTEGER PRIMARY KEY,
    network_id INTEGER NOT NULL REFERENCES networks(network_id) ON DELETE CASCADE,
    scenario_key TEXT NOT NULL,
    scenario_kind TEXT NOT NULL,
    factor_mult REAL CHECK(factor_mult IS NULL OR factor_mult >= 0),
    shape_id TEXT,
    duracion_horas REAL NOT NULL CHECK(duracion_horas >= 0),
    tiempo_al_pico_h REAL NOT NULL CHECK(
        tiempo_al_pico_h >= 0 AND tiempo_al_pico_h <= duracion_horas
    ),
    config_json TEXT NOT NULL,
    config_sha256 TEXT NOT NULL CHECK(length(config_sha256)=64),
    UNIQUE(network_id, scenario_key, config_sha256),
    UNIQUE(scenario_id, network_id)
);

CREATE TABLE scenario_inflows (
    scenario_id INTEGER NOT NULL REFERENCES scenarios(scenario_id) ON DELETE CASCADE,
    network_id INTEGER NOT NULL,
    node_pk INTEGER NOT NULL,
    step_index INTEGER NOT NULL CHECK(step_index >= 0),
    time_sec REAL NOT NULL CHECK(time_sec >= 0),
    inflow_lps REAL NOT NULL CHECK(inflow_lps >= 0),
    PRIMARY KEY(scenario_id, node_pk, step_index),
    FOREIGN KEY(scenario_id, network_id)
        REFERENCES scenarios(scenario_id, network_id) ON DELETE CASCADE,
    FOREIGN KEY(node_pk, network_id)
        REFERENCES nodes(node_pk, network_id) ON DELETE CASCADE
);

CREATE TABLE runs (
    run_id INTEGER PRIMARY KEY,
    scenario_id INTEGER NOT NULL,
    network_id INTEGER NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('PENDING','RUNNING','COMPLETE','FAILED')),
    started_at_utc TEXT,
    completed_at_utc TEXT,
    swmm_version TEXT,
    config_sha256 TEXT NOT NULL CHECK(length(config_sha256)=64),
    continuity_error_pct REAL,
    failure_stage TEXT,
    failure_type TEXT,
    failure_message TEXT,
    node_count INTEGER,
    timestep_count INTEGER,
    UNIQUE(run_id, network_id),
    FOREIGN KEY(scenario_id, network_id)
        REFERENCES scenarios(scenario_id, network_id)
);

CREATE TABLE node_features (
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
    duracion_horas REAL NOT NULL CHECK(duracion_horas >= 0),
    tiempo_al_pico_h REAL NOT NULL CHECK(
        tiempo_al_pico_h >= 0 AND tiempo_al_pico_h <= duracion_horas
    ),
    feature_contract_id TEXT NOT NULL CHECK(feature_contract_id='tabular_v3_17'),
    PRIMARY KEY(run_id, node_pk),
    FOREIGN KEY(run_id, network_id)
        REFERENCES runs(run_id, network_id) ON DELETE CASCADE,
    FOREIGN KEY(node_pk, network_id)
        REFERENCES nodes(node_pk, network_id) ON DELETE CASCADE
);

CREATE TABLE node_results (
    run_id INTEGER NOT NULL,
    network_id INTEGER NOT NULL,
    node_pk INTEGER NOT NULL,
    inunda INTEGER NOT NULL CHECK(inunda IN (0,1)),
    vol_inundacion_m3 REAL NOT NULL CHECK(vol_inundacion_m3 >= 0),
    peak_flooding_lps REAL CHECK(
        peak_flooding_lps IS NULL OR peak_flooding_lps >= 0
    ),
    flooding_duration_min REAL CHECK(
        flooding_duration_min IS NULL OR flooding_duration_min >= 0
    ),
    max_depth_m REAL CHECK(max_depth_m IS NULL OR max_depth_m >= 0),
    max_depth_ratio REAL CHECK(
        max_depth_ratio IS NULL OR max_depth_ratio >= 0
    ),
    PRIMARY KEY(run_id, node_pk),
    FOREIGN KEY(run_id, network_id)
        REFERENCES runs(run_id, network_id) ON DELETE CASCADE,
    FOREIGN KEY(node_pk, network_id)
        REFERENCES nodes(node_pk, network_id) ON DELETE CASCADE
);

CREATE TABLE node_timeseries (
    run_id INTEGER NOT NULL,
    network_id INTEGER NOT NULL,
    node_pk INTEGER NOT NULL,
    step_index INTEGER NOT NULL CHECK(step_index >= 0),
    time_sec REAL NOT NULL CHECK(time_sec >= 0),
    total_inflow_lps REAL NOT NULL,
    lateral_inflow_lps REAL NOT NULL,
    depth_m REAL NOT NULL,
    depth_ratio REAL NOT NULL,
    flooding_lps REAL NOT NULL,
    total_outflow_lps REAL NOT NULL,
    failed_now INTEGER NOT NULL CHECK(failed_now IN (0,1)),
    PRIMARY KEY(run_id, node_pk, step_index),
    FOREIGN KEY(run_id, network_id)
        REFERENCES runs(run_id, network_id) ON DELETE CASCADE,
    FOREIGN KEY(node_pk, network_id)
        REFERENCES nodes(node_pk, network_id) ON DELETE CASCADE
);

CREATE TABLE training_runs (
    training_run_id INTEGER PRIMARY KEY,
    target TEXT NOT NULL CHECK(target IN ('inunda','vol_inundacion_m3','system')),
    feature_contract_id TEXT NOT NULL CHECK(feature_contract_id='tabular_v3_17'),
    feature_contract_sha256 TEXT NOT NULL CHECK(length(feature_contract_sha256)=64),
    query_sql TEXT NOT NULL,
    query_params_json TEXT NOT NULL,
    included_run_ids_json TEXT NOT NULL,
    grouping_strategy TEXT NOT NULL CHECK(grouping_strategy IN ('group_kfold','loso')),
    fold_count INTEGER NOT NULL CHECK(fold_count >= 2),
    random_seed INTEGER NOT NULL,
    primary_metric TEXT NOT NULL,
    tie_breakers_json TEXT NOT NULL,
    python_version TEXT NOT NULL,
    library_versions_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('PENDING','RUNNING','COMPLETE','FAILED')),
    started_at_utc TEXT,
    completed_at_utc TEXT,
    failure_type TEXT,
    failure_message TEXT
);

CREATE TABLE model_evaluations (
    evaluation_id INTEGER PRIMARY KEY,
    training_run_id INTEGER NOT NULL
        REFERENCES training_runs(training_run_id) ON DELETE CASCADE,
    task TEXT NOT NULL CHECK(task IN ('classification','regression')),
    algorithm TEXT NOT NULL,
    hyperparameters_json TEXT NOT NULL,
    fold_id INTEGER NOT NULL CHECK(fold_id >= 0),
    train_run_ids_json TEXT NOT NULL,
    validation_run_ids_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('PENDING','RUNNING','COMPLETE','FAILED')),
    fit_seconds REAL NOT NULL CHECK(fit_seconds >= 0),
    predict_seconds REAL NOT NULL CHECK(predict_seconds >= 0),
    failure_type TEXT,
    failure_message TEXT,
    UNIQUE(training_run_id, task, algorithm, fold_id)
);

CREATE TABLE oof_predictions (
    evaluation_id INTEGER NOT NULL
        REFERENCES model_evaluations(evaluation_id) ON DELETE CASCADE,
    run_id INTEGER NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    node_pk INTEGER NOT NULL REFERENCES nodes(node_pk) ON DELETE CASCADE,
    target TEXT NOT NULL CHECK(target IN ('inunda','vol_inundacion_m3')),
    observed REAL NOT NULL,
    predicted REAL NOT NULL,
    probability REAL,
    fold_id INTEGER NOT NULL CHECK(fold_id >= 0),
    PRIMARY KEY(evaluation_id, run_id, node_pk),
    FOREIGN KEY(run_id, node_pk)
        REFERENCES node_results(run_id, node_pk) ON DELETE CASCADE
);

CREATE TABLE model_metrics (
    metric_id INTEGER PRIMARY KEY,
    owner_kind TEXT NOT NULL CHECK(
        owner_kind IN ('training_run','evaluation','model')
    ),
    owner_id INTEGER NOT NULL,
    scope TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL,
    valid INTEGER NOT NULL CHECK(valid IN (0,1)),
    reason TEXT,
    UNIQUE(owner_kind, owner_id, scope, metric_name)
);

CREATE TABLE trained_models (
    model_id INTEGER PRIMARY KEY,
    training_run_id INTEGER NOT NULL
        REFERENCES training_runs(training_run_id) ON DELETE CASCADE,
    target TEXT NOT NULL CHECK(target IN ('inunda','vol_inundacion_m3')),
    algorithm TEXT NOT NULL,
    hyperparameters_json TEXT NOT NULL,
    preprocessing_json TEXT NOT NULL,
    feature_contract_id TEXT NOT NULL CHECK(feature_contract_id='tabular_v3_17'),
    feature_contract_sha256 TEXT NOT NULL CHECK(length(feature_contract_sha256)=64),
    ordered_features_json TEXT NOT NULL,
    target_transform_json TEXT NOT NULL,
    query_params_json TEXT NOT NULL,
    included_run_ids_json TEXT NOT NULL,
    random_seed INTEGER NOT NULL,
    grouping_strategy TEXT NOT NULL,
    python_version TEXT NOT NULL,
    library_versions_json TEXT NOT NULL,
    selected_metric TEXT NOT NULL,
    selected_value REAL NOT NULL,
    model_sha256 TEXT NOT NULL CHECK(length(model_sha256)=64),
    model_blob BLOB NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE(training_run_id, target)
);

CREATE INDEX idx_runs_status_scenario
    ON runs(status, scenario_id);
CREATE INDEX idx_scenarios_network_kind
    ON scenarios(network_id, scenario_kind, factor_mult, shape_id);
CREATE INDEX idx_timeseries_run_time
    ON node_timeseries(run_id, time_sec);
CREATE INDEX idx_timeseries_run_node_step
    ON node_timeseries(run_id, node_pk, step_index);
CREATE INDEX idx_oof_evaluation
    ON oof_predictions(evaluation_id, fold_id);
CREATE INDEX idx_metrics_owner
    ON model_metrics(owner_kind, owner_id, metric_name);
CREATE INDEX idx_models_target_contract_metric
    ON trained_models(
        target, feature_contract_id, selected_metric, selected_value
    );

CREATE VIEW training_samples_v17 AS
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
    f.duracion_horas,
    f.tiempo_al_pico_h,
    o.inunda,
    o.vol_inundacion_m3
FROM runs AS r
JOIN scenarios AS s ON s.scenario_id = r.scenario_id
JOIN node_features AS f
    ON f.run_id = r.run_id AND f.network_id = r.network_id
JOIN node_results AS o
    ON o.run_id = f.run_id
    AND o.node_pk = f.node_pk
    AND o.network_id = f.network_id
JOIN nodes AS n
    ON n.node_pk = f.node_pk AND n.network_id = f.network_id
WHERE r.status = 'COMPLETE';
