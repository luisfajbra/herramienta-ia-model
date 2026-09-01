ALTER TABLE trained_models RENAME TO trained_models_v1;
DROP INDEX idx_models_target_contract_metric;

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
    model_sha256 TEXT NOT NULL CHECK(length(model_sha256)=64),
    model_blob BLOB NOT NULL,
    created_at_utc TEXT NOT NULL,
    UNIQUE(model_id, training_run_id, target)
);

INSERT INTO trained_models (
    model_id, training_run_id, target, algorithm, hyperparameters_json,
    preprocessing_json, feature_contract_id, feature_contract_sha256,
    ordered_features_json, target_transform_json, query_params_json,
    included_run_ids_json, random_seed, grouping_strategy, python_version,
    library_versions_json, model_sha256, model_blob, created_at_utc
)
SELECT
    model_id, training_run_id, target, algorithm, hyperparameters_json,
    preprocessing_json, feature_contract_id, feature_contract_sha256,
    ordered_features_json, target_transform_json, query_params_json,
    included_run_ids_json, random_seed, grouping_strategy, python_version,
    library_versions_json, model_sha256, model_blob, created_at_utc
FROM trained_models_v1;

CREATE TRIGGER trained_models_immutable_update
BEFORE UPDATE ON trained_models
BEGIN
    SELECT RAISE(ABORT, 'trained model artifacts are immutable');
END;

ALTER TABLE model_metrics RENAME TO model_metrics_v1;
DROP INDEX idx_metrics_owner;

CREATE TABLE model_metrics (
    metric_id INTEGER PRIMARY KEY,
    training_run_id INTEGER
        REFERENCES training_runs(training_run_id) ON DELETE CASCADE,
    evaluation_id INTEGER
        REFERENCES model_evaluations(evaluation_id) ON DELETE CASCADE,
    model_id INTEGER
        REFERENCES trained_models(model_id) ON DELETE CASCADE,
    owner_kind TEXT GENERATED ALWAYS AS (
        CASE
            WHEN training_run_id IS NOT NULL THEN 'training_run'
            WHEN evaluation_id IS NOT NULL THEN 'evaluation'
            WHEN model_id IS NOT NULL THEN 'model'
        END
    ) STORED,
    owner_id INTEGER GENERATED ALWAYS AS (
        CASE
            WHEN training_run_id IS NOT NULL THEN training_run_id
            WHEN evaluation_id IS NOT NULL THEN evaluation_id
            WHEN model_id IS NOT NULL THEN model_id
        END
    ) STORED,
    scope TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL,
    valid INTEGER NOT NULL CHECK(valid IN (0,1)),
    reason TEXT,
    CHECK(
        (training_run_id IS NOT NULL)
        + (evaluation_id IS NOT NULL)
        + (model_id IS NOT NULL) = 1
    )
);

INSERT INTO model_metrics (
    metric_id, training_run_id, evaluation_id, model_id, scope, metric_name,
    value, valid, reason
)
SELECT
    metric_id,
    CASE WHEN owner_kind='training_run' THEN owner_id END,
    CASE WHEN owner_kind='evaluation' THEN owner_id END,
    CASE WHEN owner_kind='model' THEN owner_id END,
    scope, metric_name, value, valid, reason
FROM model_metrics_v1;

CREATE TABLE model_promotions (
    promotion_id INTEGER PRIMARY KEY,
    training_run_id INTEGER NOT NULL
        REFERENCES training_runs(training_run_id) ON DELETE RESTRICT,
    target TEXT NOT NULL CHECK(target IN ('inunda','vol_inundacion_m3','system')),
    classifier_model_id INTEGER,
    regressor_model_id INTEGER,
    classifier_target TEXT GENERATED ALWAYS AS (
        CASE WHEN classifier_model_id IS NOT NULL THEN 'inunda' END
    ) STORED,
    regressor_target TEXT GENERATED ALWAYS AS (
        CASE WHEN regressor_model_id IS NOT NULL THEN 'vol_inundacion_m3' END
    ) STORED,
    primary_metric TEXT NOT NULL,
    primary_value REAL,
    tie_breakers_json TEXT NOT NULL,
    ranking_json TEXT NOT NULL,
    promoted_at_utc TEXT NOT NULL,
    CHECK(
        (target='inunda' AND classifier_model_id IS NOT NULL
            AND regressor_model_id IS NULL)
        OR (target='vol_inundacion_m3' AND classifier_model_id IS NULL
            AND regressor_model_id IS NOT NULL)
        OR (target='system' AND classifier_model_id IS NOT NULL
            AND regressor_model_id IS NOT NULL)
    ),
    UNIQUE(promotion_id, target),
    FOREIGN KEY(classifier_model_id, training_run_id, classifier_target)
        REFERENCES trained_models(model_id, training_run_id, target)
        ON DELETE RESTRICT,
    FOREIGN KEY(regressor_model_id, training_run_id, regressor_target)
        REFERENCES trained_models(model_id, training_run_id, target)
        ON DELETE RESTRICT
);

CREATE TABLE model_selections (
    selection_id INTEGER PRIMARY KEY,
    target TEXT NOT NULL CHECK(target IN ('inunda','vol_inundacion_m3','system')),
    promotion_id INTEGER NOT NULL,
    supersedes_selection_id INTEGER UNIQUE,
    selected_at_utc TEXT NOT NULL,
    UNIQUE(selection_id, target),
    FOREIGN KEY(promotion_id, target)
        REFERENCES model_promotions(promotion_id, target) ON DELETE RESTRICT,
    FOREIGN KEY(supersedes_selection_id, target)
        REFERENCES model_selections(selection_id, target) ON DELETE RESTRICT
);

CREATE TRIGGER model_promotions_immutable_update
BEFORE UPDATE ON model_promotions
BEGIN
    SELECT RAISE(ABORT, 'model promotions are immutable');
END;

CREATE TRIGGER model_promotions_immutable_delete
BEFORE DELETE ON model_promotions
BEGIN
    SELECT RAISE(ABORT, 'model promotions are immutable');
END;

CREATE TRIGGER model_selections_validate_chain
BEFORE INSERT ON model_selections
BEGIN
    SELECT CASE
        WHEN NEW.supersedes_selection_id IS NULL
             AND EXISTS (
                 SELECT 1 FROM model_selections WHERE target=NEW.target
             )
        THEN RAISE(ABORT, 'target already has an active selection')
        WHEN NEW.supersedes_selection_id IS NOT NULL
             AND NOT EXISTS (
                 SELECT 1
                 FROM model_selections AS previous
                 WHERE previous.selection_id=NEW.supersedes_selection_id
                   AND previous.target=NEW.target
                   AND NOT EXISTS (
                       SELECT 1
                       FROM model_selections AS successor
                       WHERE successor.supersedes_selection_id=previous.selection_id
                   )
             )
        THEN RAISE(ABORT, 'selection must supersede the active selection')
    END;
END;

CREATE TRIGGER model_selections_immutable_update
BEFORE UPDATE ON model_selections
BEGIN
    SELECT RAISE(ABORT, 'model selections are immutable');
END;

CREATE TRIGGER model_selections_immutable_delete
BEFORE DELETE ON model_selections
BEGIN
    SELECT RAISE(ABORT, 'model selections are immutable');
END;

INSERT INTO model_promotions (
    training_run_id, target, classifier_model_id, regressor_model_id,
    primary_metric, primary_value, tie_breakers_json, ranking_json,
    promoted_at_utc
)
SELECT
    training_run_id,
    target,
    CASE WHEN target='inunda' THEN model_id END,
    CASE WHEN target='vol_inundacion_m3' THEN model_id END,
    selected_metric,
    selected_value,
    '[]',
    json_object('legacy_model_id', model_id, 'source', '001_v17_initial'),
    created_at_utc
FROM trained_models_v1
ORDER BY model_id;

INSERT INTO model_promotions (
    training_run_id, target, classifier_model_id, regressor_model_id,
    primary_metric, primary_value, tie_breakers_json, ranking_json,
    promoted_at_utc
)
SELECT
    classification.training_run_id,
    'system',
    classification.model_id,
    regression.model_id,
    training_runs.primary_metric,
    classification.selected_value,
    training_runs.tie_breakers_json,
    json_object(
        'legacy_classifier_metric', classification.selected_metric,
        'legacy_classifier_value', classification.selected_value,
        'legacy_regressor_metric', regression.selected_metric,
        'legacy_regressor_value', regression.selected_value,
        'source', '001_v17_initial'
    ),
    CASE
        WHEN classification.created_at_utc >= regression.created_at_utc
        THEN classification.created_at_utc
        ELSE regression.created_at_utc
    END
FROM trained_models_v1 AS classification
JOIN trained_models_v1 AS regression
    ON regression.training_run_id=classification.training_run_id
   AND regression.target='vol_inundacion_m3'
JOIN training_runs
    ON training_runs.training_run_id=classification.training_run_id
WHERE classification.target='inunda'
  AND training_runs.target='system';

INSERT INTO model_selections (
    target, promotion_id, supersedes_selection_id, selected_at_utc
)
SELECT target, promotion_id, NULL, promoted_at_utc
FROM model_promotions AS candidate
WHERE NOT EXISTS (
    SELECT 1
    FROM model_promotions AS later
    WHERE later.target=candidate.target
      AND (
          later.promoted_at_utc > candidate.promoted_at_utc
          OR (
              later.promoted_at_utc = candidate.promoted_at_utc
              AND later.promotion_id > candidate.promotion_id
          )
      )
)
ORDER BY target;

DROP TABLE model_metrics_v1;
DROP TABLE trained_models_v1;
DROP INDEX idx_timeseries_run_node_step;

CREATE UNIQUE INDEX idx_metrics_owner
    ON model_metrics(owner_kind, owner_id, scope, metric_name);
CREATE INDEX idx_metrics_training_run
    ON model_metrics(training_run_id, metric_name, scope)
    WHERE training_run_id IS NOT NULL;
CREATE INDEX idx_metrics_evaluation
    ON model_metrics(evaluation_id, metric_name, scope)
    WHERE evaluation_id IS NOT NULL;
CREATE INDEX idx_metrics_model
    ON model_metrics(model_id, metric_name, scope)
    WHERE model_id IS NOT NULL;
CREATE INDEX idx_models_training_target
    ON trained_models(training_run_id, target, model_id);
CREATE INDEX idx_models_target_contract
    ON trained_models(target, feature_contract_id, created_at_utc, model_id);
CREATE INDEX idx_promotions_training_target
    ON model_promotions(training_run_id, target, promoted_at_utc, promotion_id);
CREATE INDEX idx_promotions_classifier
    ON model_promotions(classifier_model_id, promoted_at_utc)
    WHERE classifier_model_id IS NOT NULL;
CREATE INDEX idx_promotions_regressor
    ON model_promotions(regressor_model_id, promoted_at_utc)
    WHERE regressor_model_id IS NOT NULL;
CREATE INDEX idx_promotions_target_metric
    ON model_promotions(target, primary_metric, primary_value, promotion_id);
CREATE INDEX idx_selections_target_history
    ON model_selections(target, selection_id);

CREATE VIEW active_model_selections AS
SELECT
    selection.selection_id,
    selection.target,
    selection.promotion_id,
    selection.selected_at_utc,
    promotion.training_run_id,
    promotion.classifier_model_id,
    promotion.regressor_model_id,
    promotion.primary_metric,
    promotion.primary_value,
    promotion.tie_breakers_json,
    promotion.ranking_json,
    promotion.promoted_at_utc
FROM model_selections AS selection
JOIN model_promotions AS promotion
    ON promotion.promotion_id=selection.promotion_id
   AND promotion.target=selection.target
WHERE NOT EXISTS (
    SELECT 1
    FROM model_selections AS successor
    WHERE successor.supersedes_selection_id=selection.selection_id
);
