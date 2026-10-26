-- Migration 008: freeze v4_21 features and run-quality provenance after pinning.
-- Existing migrations remain unchanged so databases with applied history can
-- advance using the normal append-only migration mechanism.

CREATE TRIGGER node_features_v4_21_pinned_no_insert
BEFORE INSERT ON node_features_v4_21
WHEN EXISTS (
    SELECT 1 FROM training_run_inputs WHERE run_id = NEW.run_id
)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;

CREATE TRIGGER node_features_v4_21_pinned_no_write
BEFORE UPDATE ON node_features_v4_21
WHEN EXISTS (
    SELECT 1 FROM training_run_inputs
    WHERE run_id = OLD.run_id OR run_id = NEW.run_id
)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;

CREATE TRIGGER node_features_v4_21_pinned_no_delete
BEFORE DELETE ON node_features_v4_21
WHEN EXISTS (
    SELECT 1 FROM training_run_inputs WHERE run_id = OLD.run_id
)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;

-- Migration 005 already freezes network_id and status on pinned runs. Freeze
-- the remaining fields that establish scenario identity, simulation setup,
-- completion evidence, and hydraulic quality so a pinned sample cannot be
-- silently reclassified after training.
CREATE TRIGGER runs_pinned_quality_no_update
BEFORE UPDATE ON runs
WHEN EXISTS (
    SELECT 1 FROM training_run_inputs WHERE run_id = OLD.run_id
) AND (
       NEW.scenario_id IS NOT OLD.scenario_id
    OR NEW.started_at_utc IS NOT OLD.started_at_utc
    OR NEW.completed_at_utc IS NOT OLD.completed_at_utc
    OR NEW.swmm_version IS NOT OLD.swmm_version
    OR NEW.config_sha256 IS NOT OLD.config_sha256
    OR NEW.continuity_error_pct IS NOT OLD.continuity_error_pct
    OR NEW.failure_stage IS NOT OLD.failure_stage
    OR NEW.failure_type IS NOT OLD.failure_type
    OR NEW.failure_message IS NOT OLD.failure_message
    OR NEW.node_count IS NOT OLD.node_count
    OR NEW.timestep_count IS NOT OLD.timestep_count
)
BEGIN
    SELECT RAISE(ABORT, 'pinned by training provenance');
END;
