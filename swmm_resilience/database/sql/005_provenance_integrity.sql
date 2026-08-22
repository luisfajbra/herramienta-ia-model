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
