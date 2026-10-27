-- Append-only hydraulic QA manifest. Historical runs remain unqualified until
-- separately rerun and assessed; this migration deliberately performs no
-- backfill and does not alter `runs`.
CREATE TABLE run_hydraulic_quality (
    run_id INTEGER PRIMARY KEY,
    policy_version TEXT NOT NULL CHECK(length(trim(policy_version)) > 0),
    policy_json TEXT NOT NULL CHECK(json_valid(policy_json)),
    policy_sha256 TEXT NOT NULL CHECK(
        length(policy_sha256) = 64 AND policy_sha256 NOT GLOB '*[^0-9a-f]*'
    ),
    continuity_error_pct REAL CHECK(
        continuity_error_pct IS NULL OR (
            continuity_error_pct = continuity_error_pct
            AND abs(continuity_error_pct) <= 1.0e308
        )
    ),
    finite_coverage_ok INTEGER NOT NULL CHECK(finite_coverage_ok IN (0, 1)),
    drainage_state TEXT NOT NULL CHECK(
        drainage_state IN ('drained', 'not_drained', 'unverified')
    ),
    initial_condition_json TEXT NOT NULL CHECK(json_valid(initial_condition_json)),
    eligible INTEGER NOT NULL CHECK(eligible IN (0, 1)),
    reason_codes_json TEXT NOT NULL CHECK(json_valid(reason_codes_json)),
    manifest_json TEXT NOT NULL CHECK(json_valid(manifest_json)),
    created_at_utc TEXT NOT NULL CHECK(length(trim(created_at_utc)) > 0),
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE RESTRICT,
    CHECK (
        eligible = 0 OR (
            continuity_error_pct IS NOT NULL
            AND
            finite_coverage_ok = 1
            AND drainage_state = 'drained'
            AND reason_codes_json = '[]'
            AND json_type(policy_json, '$.max_continuity_error_pct')
                IN ('integer', 'real')
            AND CAST(json_extract(
                policy_json, '$.max_continuity_error_pct'
            ) AS REAL) >= 0
            AND abs(continuity_error_pct) <= CAST(json_extract(
                policy_json, '$.max_continuity_error_pct'
            ) AS REAL)
            AND length(trim(json_extract(
                policy_json, '$.continuity_limit_rationale'
            ))) > 0
            AND json_type(initial_condition_json, '$.policy_id') = 'text'
            AND length(trim(json_extract(
                initial_condition_json, '$.policy_id'
            ))) > 0
            AND json_type(initial_condition_json, '$.state_sha256') = 'text'
            AND length(json_extract(
                initial_condition_json, '$.state_sha256'
            )) = 64
            AND json_extract(
                initial_condition_json, '$.state_sha256'
            ) NOT GLOB '*[^0-9a-f]*'
            AND json_type(manifest_json, '$.flooding_report.parser') = 'text'
            AND length(trim(json_extract(
                manifest_json, '$.flooding_report.parser'
            ))) > 0
            AND json_type(manifest_json, '$.flooding_report.parser_version') = 'text'
            AND length(trim(json_extract(
                manifest_json, '$.flooding_report.parser_version'
            ))) > 0
            AND json_type(manifest_json, '$.flooding_report.report_sha256') = 'text'
            AND length(json_extract(
                manifest_json, '$.flooding_report.report_sha256'
            )) = 64
            AND json_extract(
                manifest_json, '$.flooding_report.report_sha256'
            ) NOT GLOB '*[^0-9a-f]*'
            AND json_extract(manifest_json, '$.eligible') = 1
            AND json_extract(manifest_json, '$.reason_codes') = '[]'
            AND json_extract(manifest_json, '$.continuity_error_pct')
                IS continuity_error_pct
            AND json_extract(manifest_json, '$.finite_coverage_ok') = 1
            AND json_extract(manifest_json, '$.drainage_state') = drainage_state
        )
    )
);

CREATE TRIGGER run_hydraulic_quality_valid_insert
BEFORE INSERT ON run_hydraulic_quality
BEGIN
    SELECT CASE WHEN EXISTS (
        SELECT 1 FROM training_run_inputs WHERE run_id = NEW.run_id
    ) THEN RAISE(ABORT, 'pinned run quality is immutable') END;

    SELECT CASE WHEN NOT EXISTS (
        SELECT 1 FROM runs
        WHERE run_id = NEW.run_id
          AND status = 'COMPLETE'
          AND continuity_error_pct IS NEW.continuity_error_pct
    ) THEN RAISE(ABORT, 'hydraulic quality requires matching finite continuity on a COMPLETE run') END;

END;

CREATE TRIGGER run_hydraulic_quality_no_update
BEFORE UPDATE ON run_hydraulic_quality
BEGIN
    SELECT RAISE(ABORT, 'hydraulic quality manifests are immutable');
END;

CREATE TRIGGER run_hydraulic_quality_no_delete
BEFORE DELETE ON run_hydraulic_quality
BEGIN
    SELECT RAISE(ABORT, 'hydraulic quality manifests are immutable');
END;

CREATE TRIGGER runs_quality_continuity_immutable
BEFORE UPDATE OF continuity_error_pct ON runs
WHEN EXISTS (
    SELECT 1 FROM run_hydraulic_quality
    WHERE run_id = OLD.run_id
)
AND NEW.continuity_error_pct IS NOT OLD.continuity_error_pct
BEGIN
    SELECT RAISE(ABORT, 'continuity is immutable after hydraulic quality assessment');
END;
