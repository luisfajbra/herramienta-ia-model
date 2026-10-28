-- Persistent references and content identities for the files behind a run.
-- Source files remain in their experiment directories; this table stores no copies.
CREATE TABLE run_source_artifacts (
    run_id INTEGER NOT NULL,
    artifact_role TEXT NOT NULL CHECK(
        artifact_role IN ('effective_inp', 'event_source', 'swmm_rpt', 'swmm_out')
    ),
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
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE RESTRICT
);

CREATE TRIGGER run_source_artifacts_valid_insert
BEFORE INSERT ON run_source_artifacts
BEGIN
    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM runs
        WHERE run_id = NEW.run_id AND status = 'COMPLETE'
    ) THEN RAISE(ABORT, 'source artifacts require a COMPLETE run') END;

    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM run_hydraulic_quality WHERE run_id = NEW.run_id
    ) THEN RAISE(ABORT, 'source artifacts require immutable hydraulic quality') END;

    SELECT CASE WHEN NEW.artifact_role = 'swmm_rpt' AND NOT EXISTS (
        SELECT 1 FROM run_hydraulic_quality
        WHERE run_id = NEW.run_id
          AND json_extract(manifest_json, '$.flooding_report.report_sha256') = NEW.sha256
    ) THEN RAISE(ABORT, 'source RPT hash does not match hydraulic quality manifest') END;
END;

CREATE TRIGGER run_source_artifacts_no_update
BEFORE UPDATE ON run_source_artifacts
BEGIN
    SELECT RAISE(ABORT, 'run source artifacts are immutable');
END;

CREATE TRIGGER run_source_artifacts_no_delete
BEFORE DELETE ON run_source_artifacts
BEGIN
    SELECT RAISE(ABORT, 'run source artifacts are immutable');
END;
