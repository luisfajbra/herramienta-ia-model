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

CREATE TRIGGER training_runs_no_delete
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

CREATE TRIGGER model_evaluations_no_delete
BEFORE DELETE ON model_evaluations
BEGIN
    SELECT RAISE(ABORT, 'model evaluations cannot be deleted');
END;
