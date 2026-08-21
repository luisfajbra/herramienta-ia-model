CREATE TABLE migration_003_integrity_guard (
    valid INTEGER NOT NULL,
    CONSTRAINT model_integrity_003 CHECK(valid=1)
);

INSERT INTO migration_003_integrity_guard(valid)
SELECT CASE WHEN EXISTS (
    SELECT 1
    FROM trained_models AS model
    LEFT JOIN training_runs AS training
        ON training.training_run_id=model.training_run_id
    WHERE training.training_run_id IS NULL
       OR training.status NOT IN ('RUNNING','COMPLETE')
       OR (model.target='inunda'
           AND training.target NOT IN ('inunda','system'))
       OR (model.target='vol_inundacion_m3'
           AND training.target NOT IN ('vol_inundacion_m3','system'))
) THEN 0 ELSE 1 END;

INSERT INTO migration_003_integrity_guard(valid)
SELECT CASE WHEN EXISTS (
    SELECT 1
    FROM model_promotions AS promotion
    LEFT JOIN training_runs AS training
        ON training.training_run_id=promotion.training_run_id
    WHERE training.training_run_id IS NULL
       OR training.status <> 'COMPLETE'
       OR (promotion.target='system' AND training.target <> 'system')
       OR (promotion.target='inunda'
           AND training.target NOT IN ('inunda','system'))
       OR (promotion.target='vol_inundacion_m3'
           AND training.target NOT IN ('vol_inundacion_m3','system'))
) THEN 0 ELSE 1 END;

WITH RECURSIVE
roots(selection_id, target) AS (
    SELECT selection_id, target
    FROM model_selections
    WHERE supersedes_selection_id IS NULL
),
reachable(selection_id, target) AS (
    SELECT selection_id, target FROM roots
    UNION ALL
    SELECT child.selection_id, child.target
    FROM model_selections AS child
    JOIN reachable AS parent
      ON child.supersedes_selection_id=parent.selection_id
     AND child.target=parent.target
)
INSERT INTO migration_003_integrity_guard(valid)
SELECT CASE WHEN
    EXISTS (
        SELECT 1 FROM model_selections
        WHERE selection_id=supersedes_selection_id
    )
    OR EXISTS (
        SELECT target
        FROM model_selections
        GROUP BY target
        HAVING SUM(supersedes_selection_id IS NULL) <> 1
    )
    OR EXISTS (
        SELECT current.target
        FROM model_selections AS current
        LEFT JOIN model_selections AS successor
          ON successor.supersedes_selection_id=current.selection_id
        GROUP BY current.target
        HAVING SUM(successor.selection_id IS NULL) <> 1
    )
    OR (SELECT COUNT(*) FROM reachable)
       <> (SELECT COUNT(*) FROM model_selections)
THEN 0 ELSE 1 END;

INSERT INTO migration_003_integrity_guard(valid)
SELECT CASE WHEN EXISTS (
    SELECT 1 FROM pragma_foreign_key_check
) THEN 0 ELSE 1 END;

DROP TABLE migration_003_integrity_guard;

ALTER TABLE model_metrics RENAME TO model_metrics_v2;
DROP INDEX idx_metrics_owner;
DROP INDEX idx_metrics_training_run;
DROP INDEX idx_metrics_evaluation;
DROP INDEX idx_metrics_model;

CREATE TABLE model_metrics (
    metric_id INTEGER PRIMARY KEY,
    training_run_id INTEGER
        REFERENCES training_runs(training_run_id) ON DELETE RESTRICT,
    evaluation_id INTEGER
        REFERENCES model_evaluations(evaluation_id) ON DELETE RESTRICT,
    model_id INTEGER
        REFERENCES trained_models(model_id) ON DELETE RESTRICT,
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
    metric_id, training_run_id, evaluation_id, model_id, scope, metric_name,
    value, valid, reason
FROM model_metrics_v2;

DROP TABLE model_metrics_v2;

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

CREATE TRIGGER model_metrics_identity_conflict
BEFORE INSERT ON model_metrics
WHEN EXISTS (
    SELECT 1
    FROM model_metrics AS existing
    WHERE existing.metric_id=NEW.metric_id
       OR (
           existing.scope=NEW.scope
           AND existing.metric_name=NEW.metric_name
           AND (
               (NEW.training_run_id IS NOT NULL
                AND existing.training_run_id=NEW.training_run_id)
               OR (NEW.evaluation_id IS NOT NULL
                   AND existing.evaluation_id=NEW.evaluation_id)
               OR (NEW.model_id IS NOT NULL
                   AND existing.model_id=NEW.model_id)
           )
       )
)
BEGIN
    SELECT RAISE(ABORT, 'model metric identity is immutable');
END;

CREATE TRIGGER model_metrics_immutable_update
BEFORE UPDATE ON model_metrics
BEGIN
    SELECT RAISE(ABORT, 'model metrics are immutable');
END;

CREATE TRIGGER model_metrics_immutable_delete
BEFORE DELETE ON model_metrics
BEGIN
    SELECT RAISE(ABORT, 'model metrics are immutable');
END;

CREATE TRIGGER trained_models_validate_insert
BEFORE INSERT ON trained_models
WHEN NOT EXISTS (
    SELECT 1
    FROM training_runs AS training
    WHERE training.training_run_id=NEW.training_run_id
      AND training.status IN ('RUNNING','COMPLETE')
      AND (
          (NEW.target='inunda'
           AND training.target IN ('inunda','system'))
          OR (NEW.target='vol_inundacion_m3'
              AND training.target IN ('vol_inundacion_m3','system'))
      )
)
BEGIN
    SELECT RAISE(
        ABORT,
        'trained model target requires a compatible RUNNING or COMPLETE training run'
    );
END;

CREATE TRIGGER trained_models_identity_conflict
BEFORE INSERT ON trained_models
WHEN EXISTS (
    SELECT 1 FROM trained_models WHERE model_id=NEW.model_id
)
BEGIN
    SELECT RAISE(ABORT, 'trained model identity is immutable');
END;

CREATE TRIGGER trained_models_immutable_delete
BEFORE DELETE ON trained_models
BEGIN
    SELECT RAISE(ABORT, 'trained model artifacts are immutable');
END;

CREATE TRIGGER model_promotions_validate_insert
BEFORE INSERT ON model_promotions
WHEN NOT EXISTS (
    SELECT 1
    FROM training_runs AS training
    WHERE training.training_run_id=NEW.training_run_id
      AND training.status='COMPLETE'
      AND (
          (NEW.target='system' AND training.target='system')
          OR (NEW.target='inunda'
              AND training.target IN ('inunda','system'))
          OR (NEW.target='vol_inundacion_m3'
              AND training.target IN ('vol_inundacion_m3','system'))
      )
)
BEGIN
    SELECT RAISE(
        ABORT,
        'model promotion target requires a compatible COMPLETE training run'
    );
END;

CREATE TRIGGER model_promotions_identity_conflict
BEFORE INSERT ON model_promotions
WHEN EXISTS (
    SELECT 1 FROM model_promotions WHERE promotion_id=NEW.promotion_id
)
BEGIN
    SELECT RAISE(ABORT, 'model promotion identity is immutable');
END;

CREATE TRIGGER model_selections_identity_conflict
BEFORE INSERT ON model_selections
WHEN EXISTS (
    SELECT 1
    FROM model_selections AS existing
    WHERE existing.selection_id=NEW.selection_id
       OR (
           NEW.supersedes_selection_id IS NOT NULL
           AND existing.supersedes_selection_id=NEW.supersedes_selection_id
       )
)
BEGIN
    SELECT RAISE(ABORT, 'model selection identity is immutable');
END;

CREATE TRIGGER model_selections_reject_self_supersession
BEFORE INSERT ON model_selections
WHEN NEW.supersedes_selection_id=NEW.selection_id
BEGIN
    SELECT RAISE(ABORT, 'model selection cannot supersede itself');
END;

CREATE TRIGGER model_selections_validate_complete_promotion
BEFORE INSERT ON model_selections
WHEN NOT EXISTS (
    SELECT 1
    FROM model_promotions AS promotion
    JOIN training_runs AS training
      ON training.training_run_id=promotion.training_run_id
    WHERE promotion.promotion_id=NEW.promotion_id
      AND promotion.target=NEW.target
      AND training.status='COMPLETE'
)
BEGIN
    SELECT RAISE(ABORT, 'selection requires a COMPLETE model promotion');
END;

CREATE TRIGGER training_runs_identity_conflict
BEFORE INSERT ON training_runs
WHEN EXISTS (
    SELECT 1 FROM training_runs WHERE training_run_id=NEW.training_run_id
)
BEGIN
    SELECT RAISE(ABORT, 'training run identity is immutable');
END;

CREATE TRIGGER training_runs_preserve_model_integrity
BEFORE UPDATE OF target, status ON training_runs
WHEN EXISTS (
    SELECT 1
    FROM trained_models AS model
    WHERE model.training_run_id=OLD.training_run_id
      AND (
          NEW.status NOT IN ('RUNNING','COMPLETE')
          OR (model.target='inunda'
              AND NEW.target NOT IN ('inunda','system'))
          OR (model.target='vol_inundacion_m3'
              AND NEW.target NOT IN ('vol_inundacion_m3','system'))
      )
)
BEGIN
    SELECT RAISE(ABORT, 'training run target/status would invalidate a model');
END;

CREATE TRIGGER training_runs_preserve_promotion_integrity
BEFORE UPDATE OF target, status ON training_runs
WHEN EXISTS (
    SELECT 1
    FROM model_promotions AS promotion
    WHERE promotion.training_run_id=OLD.training_run_id
      AND (
          NEW.status <> 'COMPLETE'
          OR (promotion.target='system' AND NEW.target <> 'system')
          OR (promotion.target='inunda'
              AND NEW.target NOT IN ('inunda','system'))
          OR (promotion.target='vol_inundacion_m3'
              AND NEW.target NOT IN ('vol_inundacion_m3','system'))
      )
)
BEGIN
    SELECT RAISE(ABORT, 'training run target/status would invalidate a promotion');
END;
