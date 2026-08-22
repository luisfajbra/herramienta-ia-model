-- swmm_resilience/database/sql/005_provenance_integrity.sql

CREATE TABLE schema_migration_validators (
    version INTEGER PRIMARY KEY
        REFERENCES schema_migrations(version) ON DELETE RESTRICT,
    validator_name TEXT UNIQUE NOT NULL,
    validator_sha256 TEXT NOT NULL CHECK(length(validator_sha256)=64)
);

CREATE TRIGGER schema_migration_validators_identity_conflict
BEFORE INSERT ON schema_migration_validators
WHEN EXISTS (
    SELECT 1 FROM schema_migration_validators WHERE version=NEW.version
)
BEGIN
    SELECT RAISE(ABORT, 'schema migration validator identity is immutable');
END;

CREATE TRIGGER schema_migration_validators_immutable_update
BEFORE UPDATE ON schema_migration_validators
BEGIN
    SELECT RAISE(ABORT, 'schema migration validators are immutable');
END;

CREATE TRIGGER schema_migration_validators_immutable_delete
BEFORE DELETE ON schema_migration_validators
BEGIN
    SELECT RAISE(ABORT, 'schema migration validators are immutable');
END;

CREATE TABLE training_run_provenance_invalidations (
    training_run_id INTEGER PRIMARY KEY
        REFERENCES training_runs(training_run_id) ON DELETE RESTRICT,
    reason TEXT NOT NULL,
    invalidated_at_utc TEXT NOT NULL
);

CREATE TRIGGER training_run_provenance_invalidations_identity_conflict
BEFORE INSERT ON training_run_provenance_invalidations
WHEN EXISTS (
    SELECT 1 FROM training_run_provenance_invalidations
    WHERE training_run_id=NEW.training_run_id
)
BEGIN
    SELECT RAISE(ABORT, 'training run provenance invalidation identity is immutable');
END;

CREATE TRIGGER training_run_provenance_invalidations_immutable_update
BEFORE UPDATE ON training_run_provenance_invalidations
BEGIN
    SELECT RAISE(ABORT, 'training run provenance invalidations are immutable');
END;

CREATE TRIGGER training_run_provenance_invalidations_immutable_delete
BEFORE DELETE ON training_run_provenance_invalidations
BEGIN
    SELECT RAISE(ABORT, 'training run provenance invalidations are immutable');
END;

CREATE VIEW valid_training_runs AS
SELECT training_runs.*
FROM training_runs
LEFT JOIN training_run_provenance_invalidations AS invalidation
    ON invalidation.training_run_id = training_runs.training_run_id
WHERE invalidation.training_run_id IS NULL;

-- Replace the migration-004 trigger: it only fires on `UPDATE OF
-- training_run_id`, which SQLite does not consider a match when the same
-- underlying column is updated through the rowid/_rowid_/oid aliases.
DROP TRIGGER training_runs_immutable_primary_key;

CREATE TRIGGER training_runs_immutable_identity
BEFORE UPDATE ON training_runs
WHEN NEW.training_run_id IS NOT OLD.training_run_id
BEGIN
    SELECT RAISE(ABORT, 'training run identity is immutable');
END;

-- Deterministically invalidate every training run that existed before 005:
-- its evidence was mutable under the old schema and cannot be proven now.
INSERT INTO training_run_provenance_invalidations (
    training_run_id, reason, invalidated_at_utc
)
SELECT training_run_id, 'pre005_mutable_provenance', datetime('now')
FROM training_runs;

-- NOTE: this fires before migration 003's training_runs_preserve_model_integrity
-- and training_runs_preserve_promotion_integrity (SQLite fires same-table BEFORE
-- triggers in reverse creation order, and this one was created later in 005).
-- Any change to `target` is now blocked here unconditionally, which is a strict
-- superset of what those older, narrower triggers checked for `target` — their
-- target-related branches are effectively superseded, not removed. Their
-- status-related branches still apply on their own terms.
CREATE TRIGGER training_runs_immutable_configuration
BEFORE UPDATE ON training_runs
WHEN NEW.target IS NOT OLD.target
   OR NEW.feature_contract_id IS NOT OLD.feature_contract_id
   OR NEW.feature_contract_sha256 IS NOT OLD.feature_contract_sha256
   OR NEW.query_sql IS NOT OLD.query_sql
   OR NEW.query_params_json IS NOT OLD.query_params_json
   OR NEW.included_run_ids_json IS NOT OLD.included_run_ids_json
   OR NEW.grouping_strategy IS NOT OLD.grouping_strategy
   OR NEW.fold_count IS NOT OLD.fold_count
   OR NEW.random_seed IS NOT OLD.random_seed
   OR NEW.primary_metric IS NOT OLD.primary_metric
   OR NEW.tie_breakers_json IS NOT OLD.tie_breakers_json
   OR NEW.python_version IS NOT OLD.python_version
   OR NEW.library_versions_json IS NOT OLD.library_versions_json
BEGIN
    SELECT RAISE(ABORT, 'training run fitting configuration is immutable');
END;

CREATE TRIGGER training_runs_valid_status_transition
BEFORE UPDATE OF status ON training_runs
WHEN NOT (
    (OLD.status='PENDING' AND NEW.status IN ('RUNNING','FAILED'))
    OR (OLD.status='RUNNING' AND NEW.status IN ('COMPLETE','FAILED'))
)
BEGIN
    SELECT RAISE(ABORT, 'invalid training run status transition');
END;

CREATE TRIGGER training_runs_immutable_delete
BEFORE DELETE ON training_runs
BEGIN
    SELECT RAISE(ABORT, 'training runs cannot be deleted');
END;

CREATE TRIGGER model_evaluations_immutable_identity
BEFORE UPDATE ON model_evaluations
WHEN NEW.evaluation_id IS NOT OLD.evaluation_id
   OR NEW.training_run_id IS NOT OLD.training_run_id
   OR NEW.task IS NOT OLD.task
   OR NEW.algorithm IS NOT OLD.algorithm
   OR NEW.hyperparameters_json IS NOT OLD.hyperparameters_json
   OR NEW.fold_id IS NOT OLD.fold_id
   OR NEW.train_run_ids_json IS NOT OLD.train_run_ids_json
   OR NEW.validation_run_ids_json IS NOT OLD.validation_run_ids_json
BEGIN
    SELECT RAISE(ABORT, 'model evaluation identity/configuration is immutable');
END;

CREATE TRIGGER model_evaluations_valid_status_transition
BEFORE UPDATE OF status ON model_evaluations
WHEN NOT (
    (OLD.status='PENDING' AND NEW.status IN ('RUNNING','FAILED'))
    OR (OLD.status='RUNNING' AND NEW.status IN ('COMPLETE','FAILED'))
)
BEGIN
    SELECT RAISE(ABORT, 'invalid model evaluation status transition');
END;

CREATE TRIGGER model_evaluations_immutable_delete
BEFORE DELETE ON model_evaluations
BEGIN
    SELECT RAISE(ABORT, 'model evaluations cannot be deleted');
END;

CREATE TABLE training_run_inputs (
    training_run_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL,
    PRIMARY KEY (training_run_id, run_id),
    FOREIGN KEY (training_run_id)
        REFERENCES training_runs(training_run_id) ON DELETE RESTRICT,
    FOREIGN KEY (run_id)
        REFERENCES runs(run_id) ON DELETE RESTRICT
);

CREATE INDEX idx_training_run_inputs_reverse
    ON training_run_inputs(run_id, training_run_id);

CREATE TABLE model_evaluation_runs (
    evaluation_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('train','validation')),
    run_id INTEGER NOT NULL,
    PRIMARY KEY (evaluation_id, role, run_id),
    FOREIGN KEY (evaluation_id)
        REFERENCES model_evaluations(evaluation_id) ON DELETE RESTRICT,
    FOREIGN KEY (run_id)
        REFERENCES runs(run_id) ON DELETE RESTRICT
);

CREATE INDEX idx_model_evaluation_runs_reverse
    ON model_evaluation_runs(run_id, evaluation_id, role);

CREATE TRIGGER training_run_inputs_immutable_update
BEFORE UPDATE ON training_run_inputs
BEGIN
    SELECT RAISE(ABORT, 'training run membership is immutable');
END;

CREATE TRIGGER training_run_inputs_immutable_delete
BEFORE DELETE ON training_run_inputs
BEGIN
    SELECT RAISE(ABORT, 'training run membership is immutable');
END;

CREATE TRIGGER training_run_inputs_owner_pending
BEFORE INSERT ON training_run_inputs
WHEN NOT EXISTS (
    SELECT 1 FROM training_runs
    WHERE training_run_id=NEW.training_run_id AND status='PENDING'
)
BEGIN
    SELECT RAISE(ABORT, 'training run membership requires a PENDING owner');
END;

CREATE TRIGGER training_run_inputs_within_canonical_json
BEFORE INSERT ON training_run_inputs
WHEN NOT EXISTS (
    SELECT 1
    FROM training_runs, json_each(training_runs.included_run_ids_json)
    WHERE training_runs.training_run_id=NEW.training_run_id
      AND json_each.value=NEW.run_id
)
BEGIN
    SELECT RAISE(ABORT, 'training run membership must be listed in included_run_ids_json');
END;

CREATE TRIGGER model_evaluation_runs_immutable_update
BEFORE UPDATE ON model_evaluation_runs
BEGIN
    SELECT RAISE(ABORT, 'evaluation membership is immutable');
END;

CREATE TRIGGER model_evaluation_runs_immutable_delete
BEFORE DELETE ON model_evaluation_runs
BEGIN
    SELECT RAISE(ABORT, 'evaluation membership is immutable');
END;

CREATE TRIGGER model_evaluation_runs_owner_pending
BEFORE INSERT ON model_evaluation_runs
WHEN NOT EXISTS (
    SELECT 1 FROM model_evaluations
    WHERE evaluation_id=NEW.evaluation_id AND status='PENDING'
)
BEGIN
    SELECT RAISE(ABORT, 'evaluation membership requires a PENDING owner');
END;

CREATE TRIGGER model_evaluation_runs_within_canonical_json
BEFORE INSERT ON model_evaluation_runs
WHEN NOT EXISTS (
    SELECT 1
    FROM model_evaluations, json_each(
        CASE NEW.role
            WHEN 'train' THEN model_evaluations.train_run_ids_json
            ELSE model_evaluations.validation_run_ids_json
        END
    )
    WHERE model_evaluations.evaluation_id=NEW.evaluation_id
      AND json_each.value=NEW.run_id
)
BEGIN
    SELECT RAISE(ABORT, 'evaluation membership must be listed in its canonical JSON array');
END;

CREATE TRIGGER model_evaluation_runs_within_training_run
BEFORE INSERT ON model_evaluation_runs
WHEN NOT EXISTS (
    SELECT 1
    FROM model_evaluations
    JOIN training_run_inputs
        ON training_run_inputs.training_run_id = model_evaluations.training_run_id
       AND training_run_inputs.run_id = NEW.run_id
    WHERE model_evaluations.evaluation_id = NEW.evaluation_id
)
BEGIN
    SELECT RAISE(ABORT, 'evaluation membership must be contained in the training run membership');
END;

-- Consistency boundary: PENDING -> RUNNING requires normalized membership
-- to be exactly equal (as a set) to the canonical JSON array.
CREATE TRIGGER training_runs_running_requires_complete_membership
BEFORE UPDATE OF status ON training_runs
WHEN NEW.status='RUNNING' AND OLD.status='PENDING'
  AND (
    (
        SELECT COUNT(*) FROM training_run_inputs
        WHERE training_run_id=OLD.training_run_id
    ) <> (SELECT COUNT(*) FROM json_each(OLD.included_run_ids_json))
    OR EXISTS (
        SELECT value FROM json_each(OLD.included_run_ids_json)
        WHERE value NOT IN (
            SELECT run_id FROM training_run_inputs
            WHERE training_run_id=OLD.training_run_id
        )
    )
  )
BEGIN
    SELECT RAISE(
        ABORT,
        'training run membership must equal included_run_ids_json before RUNNING'
    );
END;

CREATE TRIGGER model_evaluations_running_requires_complete_membership
BEFORE UPDATE OF status ON model_evaluations
WHEN NEW.status='RUNNING' AND OLD.status='PENDING'
  AND (
    EXISTS (
        SELECT run_id FROM model_evaluation_runs
        WHERE evaluation_id=OLD.evaluation_id AND role='train'
        INTERSECT
        SELECT run_id FROM model_evaluation_runs
        WHERE evaluation_id=OLD.evaluation_id AND role='validation'
    )
    OR (
        SELECT COUNT(*) FROM model_evaluation_runs
        WHERE evaluation_id=OLD.evaluation_id AND role='train'
    ) <> (SELECT COUNT(*) FROM json_each(OLD.train_run_ids_json))
    OR (
        SELECT COUNT(*) FROM model_evaluation_runs
        WHERE evaluation_id=OLD.evaluation_id AND role='validation'
    ) <> (SELECT COUNT(*) FROM json_each(OLD.validation_run_ids_json))
    OR OLD.fold_id >= (
        SELECT fold_count FROM training_runs
        WHERE training_run_id=OLD.training_run_id
    )
  )
BEGIN
    SELECT RAISE(
        ABORT,
        'evaluation membership must be complete, disjoint, and equal to its JSON arrays before RUNNING'
    );
END;
