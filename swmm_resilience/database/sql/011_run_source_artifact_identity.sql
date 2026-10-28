-- Preserve prior links as a legacy, immutable record; they predate run identity
-- snapshots and canonical source-manifest digests and cannot be requalified.
ALTER TABLE run_source_artifacts RENAME TO run_source_artifacts_legacy_010;
DROP TRIGGER run_source_artifacts_valid_insert;
CREATE TRIGGER run_source_artifacts_legacy_010_no_insert
BEFORE INSERT ON run_source_artifacts_legacy_010
BEGIN
    SELECT RAISE(ABORT, 'legacy source artifact rows cannot be added after migration 011');
END;

CREATE TABLE run_source_artifacts (
    run_id INTEGER NOT NULL,
    artifact_role TEXT NOT NULL CHECK(
        artifact_role IN (
            'source_network_inp', 'effective_inp', 'event_source',
            'swmm_rpt', 'swmm_out'
        )
    ),
    registered_network_id INTEGER NOT NULL,
    registered_scenario_id INTEGER NOT NULL,
    resolved_path TEXT NOT NULL CHECK(length(trim(resolved_path)) > 0),
    sha256 TEXT NOT NULL CHECK(
        length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    byte_size INTEGER NOT NULL CHECK(typeof(byte_size) = 'integer' AND byte_size > 0),
    source_manifest_sha256 TEXT NOT NULL CHECK(
        length(source_manifest_sha256) = 64
        AND source_manifest_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    registered_at_utc TEXT NOT NULL CHECK(length(trim(registered_at_utc)) > 0),
    PRIMARY KEY(run_id, artifact_role),
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE RESTRICT,
    FOREIGN KEY(registered_network_id) REFERENCES networks(network_id) ON DELETE RESTRICT,
    FOREIGN KEY(registered_scenario_id) REFERENCES scenarios(scenario_id) ON DELETE RESTRICT
);

CREATE TRIGGER run_source_artifacts_v11_valid_insert
BEFORE INSERT ON run_source_artifacts
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM runs
        WHERE run_id = NEW.run_id
          AND status = 'COMPLETE'
          AND network_id = NEW.registered_network_id
          AND scenario_id = NEW.registered_scenario_id
    ) THEN RAISE(ABORT, 'source artifacts require a COMPLETE run with matching identity') END;

    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM run_hydraulic_quality WHERE run_id = NEW.run_id
    ) THEN RAISE(ABORT, 'source artifacts require immutable hydraulic quality') END;

    SELECT CASE WHEN NEW.artifact_role = 'source_network_inp' AND NOT EXISTS (
        SELECT 1 FROM networks
        WHERE network_id = NEW.registered_network_id
          AND network_sha256 = NEW.sha256
    ) THEN RAISE(ABORT, 'source network INP hash does not match network record') END;

    SELECT CASE WHEN NEW.artifact_role = 'swmm_rpt' AND NOT EXISTS (
        SELECT 1 FROM run_hydraulic_quality
        WHERE run_id = NEW.run_id
          AND json_extract(manifest_json, '$.flooding_report.report_sha256') = NEW.sha256
    ) THEN RAISE(ABORT, 'source RPT hash does not match hydraulic quality manifest') END;
END;

CREATE TRIGGER run_source_artifacts_v11_no_update
BEFORE UPDATE ON run_source_artifacts
BEGIN
    SELECT RAISE(ABORT, 'run source artifacts are immutable');
END;

CREATE TRIGGER run_source_artifacts_v11_no_delete
BEFORE DELETE ON run_source_artifacts
BEGIN
    SELECT RAISE(ABORT, 'run source artifacts are immutable');
END;

CREATE TRIGGER runs_source_artifacts_identity_immutable
BEFORE UPDATE OF run_id, network_id, scenario_id, config_sha256, status ON runs
WHEN EXISTS (
    SELECT 1 FROM run_source_artifacts WHERE run_id = OLD.run_id
)
AND (
    NEW.run_id IS NOT OLD.run_id
    OR NEW.network_id IS NOT OLD.network_id
    OR NEW.scenario_id IS NOT OLD.scenario_id
    OR NEW.config_sha256 IS NOT OLD.config_sha256
    OR NEW.status IS NOT OLD.status
)
BEGIN
    SELECT RAISE(ABORT, 'run identity/config/status are immutable after source artifacts are registered');
END;

CREATE TRIGGER runs_source_artifacts_no_delete
BEFORE DELETE ON runs
WHEN EXISTS (
    SELECT 1 FROM run_source_artifacts WHERE run_id = OLD.run_id
)
BEGIN
    SELECT RAISE(ABORT, 'run cannot be deleted after source artifacts are registered');
END;
