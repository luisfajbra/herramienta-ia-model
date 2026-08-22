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
    NOT EXISTS (
        SELECT 1 FROM training_run_inputs
        WHERE training_run_id=OLD.training_run_id
    )
    OR (
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
        'training run membership must be non-empty and equal to included_run_ids_json before RUNNING'
    );
END;

CREATE TRIGGER model_evaluations_running_requires_complete_membership
BEFORE UPDATE OF status ON model_evaluations
WHEN NEW.status='RUNNING' AND OLD.status='PENDING'
  AND (
    NOT EXISTS (
        SELECT 1 FROM model_evaluation_runs
        WHERE evaluation_id=OLD.evaluation_id AND role='train'
    )
    OR NOT EXISTS (
        SELECT 1 FROM model_evaluation_runs
        WHERE evaluation_id=OLD.evaluation_id AND role='validation'
    )
    OR EXISTS (
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

CREATE TRIGGER oof_predictions_immutable_update
BEFORE UPDATE ON oof_predictions
BEGIN
    SELECT RAISE(ABORT, 'OOF predictions are append-only');
END;

CREATE TRIGGER oof_predictions_immutable_delete
BEFORE DELETE ON oof_predictions
BEGIN
    SELECT RAISE(ABORT, 'OOF predictions are append-only');
END;

CREATE TRIGGER oof_predictions_identity_conflict
BEFORE INSERT ON oof_predictions
WHEN EXISTS (
    SELECT 1 FROM oof_predictions
    WHERE evaluation_id=NEW.evaluation_id AND run_id=NEW.run_id AND node_pk=NEW.node_pk
)
BEGIN
    SELECT RAISE(ABORT, 'OOF prediction identity is immutable');
END;

CREATE TRIGGER oof_predictions_validate_insert
BEFORE INSERT ON oof_predictions
WHEN NOT EXISTS (
    SELECT 1
    FROM model_evaluations
    JOIN training_runs
        ON training_runs.training_run_id = model_evaluations.training_run_id
    WHERE model_evaluations.evaluation_id = NEW.evaluation_id
      AND model_evaluations.status = 'RUNNING'
      AND training_runs.status = 'RUNNING'
      AND model_evaluations.fold_id = NEW.fold_id
      AND (
          (NEW.target = 'inunda' AND model_evaluations.task = 'classification')
          OR (NEW.target = 'vol_inundacion_m3' AND model_evaluations.task = 'regression')
      )
)
   OR NOT EXISTS (
       SELECT 1 FROM model_evaluation_runs
       WHERE evaluation_id = NEW.evaluation_id
         AND role = 'validation'
         AND run_id = NEW.run_id
   )
   OR NOT EXISTS (
       SELECT 1 FROM node_results
       WHERE run_id = NEW.run_id AND node_pk = NEW.node_pk
   )
   OR NEW.observed <> (
       SELECT CASE NEW.target
           WHEN 'inunda' THEN node_results.inunda
           ELSE node_results.vol_inundacion_m3
       END
       FROM node_results
       WHERE node_results.run_id = NEW.run_id AND node_results.node_pk = NEW.node_pk
   )
   OR NEW.predicted IS NULL OR NEW.predicted <> NEW.predicted  -- rejects NaN (NaN <> NaN is true)
   -- rejects +/-Infinity: any value exceeding the largest finite IEEE-754
   -- double (DBL_MAX = 1.7976931348623157e308 in magnitude) can only be
   -- +/-Infinity in SQLite's REAL storage class; empirically verified this
   -- does not false-positive on large-but-finite doubles (e.g. 1e300, 9.22e18).
   OR NEW.predicted > 1.7976931348623157e308 OR NEW.predicted < -1.7976931348623157e308
   OR (
       NEW.target = 'inunda' AND (
           NEW.observed NOT IN (0, 1)
           OR NEW.predicted NOT IN (0, 1)
           OR NEW.probability IS NULL
           OR NEW.probability < 0 OR NEW.probability > 1
       )
   )
   OR (
       NEW.target = 'vol_inundacion_m3' AND (
           NEW.observed < 0
           OR NEW.probability IS NOT NULL
       )
   )
BEGIN
    SELECT RAISE(ABORT, 'OOF prediction fails provenance/domain validation');
END;

CREATE TABLE model_candidates (
    candidate_id INTEGER PRIMARY KEY,
    training_run_id INTEGER NOT NULL
        REFERENCES training_runs(training_run_id) ON DELETE RESTRICT,
    task TEXT NOT NULL CHECK(task IN ('classification','regression')),
    algorithm TEXT NOT NULL,
    hyperparameters_json TEXT NOT NULL,
    preprocessing_json TEXT NOT NULL,
    feature_contract_id TEXT NOT NULL CHECK(feature_contract_id='tabular_v3_17'),
    feature_contract_sha256 TEXT NOT NULL CHECK(length(feature_contract_sha256)=64),
    ordered_features_json TEXT NOT NULL,
    target_transform_json TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    candidate_definition_sha256 TEXT NOT NULL CHECK(length(candidate_definition_sha256)=64)
);

CREATE INDEX idx_model_candidates_training_run
    ON model_candidates(training_run_id, task, algorithm);

CREATE TRIGGER model_candidates_immutable_update
BEFORE UPDATE ON model_candidates
BEGIN
    SELECT RAISE(ABORT, 'model candidates are immutable');
END;

CREATE TRIGGER model_candidates_immutable_delete
BEFORE DELETE ON model_candidates
BEGIN
    SELECT RAISE(ABORT, 'model candidates are immutable');
END;

CREATE TRIGGER model_candidates_owner_running
BEFORE INSERT ON model_candidates
WHEN NOT EXISTS (
    SELECT 1 FROM training_runs
    WHERE training_run_id=NEW.training_run_id AND status='RUNNING'
)
BEGIN
    SELECT RAISE(ABORT, 'candidates are inserted only while the training run is RUNNING');
END;

CREATE TABLE model_candidate_evaluations (
    candidate_id INTEGER NOT NULL
        REFERENCES model_candidates(candidate_id) ON DELETE RESTRICT,
    evaluation_id INTEGER NOT NULL UNIQUE
        REFERENCES model_evaluations(evaluation_id) ON DELETE RESTRICT,
    PRIMARY KEY (candidate_id, evaluation_id)
);

CREATE TRIGGER model_candidate_evaluations_immutable_update
BEFORE UPDATE ON model_candidate_evaluations
BEGIN
    SELECT RAISE(ABORT, 'candidate/evaluation links are immutable');
END;

CREATE TRIGGER model_candidate_evaluations_immutable_delete
BEFORE DELETE ON model_candidate_evaluations
BEGIN
    SELECT RAISE(ABORT, 'candidate/evaluation links are immutable');
END;

CREATE TRIGGER model_candidate_evaluations_matches_candidate
BEFORE INSERT ON model_candidate_evaluations
WHEN NOT EXISTS (
    SELECT 1
    FROM model_candidates
    JOIN model_evaluations
        ON model_evaluations.training_run_id = model_candidates.training_run_id
       AND model_evaluations.task = model_candidates.task
       AND model_evaluations.algorithm = model_candidates.algorithm
       AND model_evaluations.hyperparameters_json = model_candidates.hyperparameters_json
    WHERE model_candidates.candidate_id = NEW.candidate_id
      AND model_evaluations.evaluation_id = NEW.evaluation_id
)
BEGIN
    SELECT RAISE(ABORT, 'evaluation does not match its candidate task/algorithm/hyperparameters');
END;

CREATE TRIGGER model_candidate_evaluations_not_after_finalization
BEFORE INSERT ON model_candidate_evaluations
WHEN EXISTS (
    SELECT 1 FROM model_candidate_finalizations WHERE candidate_id=NEW.candidate_id
)
BEGIN
    SELECT RAISE(ABORT, 'no evaluation link is added after candidate finalization');
END;

CREATE TABLE model_candidate_finalizations (
    candidate_id INTEGER PRIMARY KEY
        REFERENCES model_candidates(candidate_id) ON DELETE RESTRICT,
    finalized_at_utc TEXT NOT NULL
);

CREATE TRIGGER model_candidate_finalizations_immutable_update
BEFORE UPDATE ON model_candidate_finalizations
BEGIN
    SELECT RAISE(ABORT, 'candidate finalizations are immutable');
END;

CREATE TRIGGER model_candidate_finalizations_immutable_delete
BEFORE DELETE ON model_candidate_finalizations
BEGIN
    SELECT RAISE(ABORT, 'candidate finalizations are immutable');
END;

CREATE TRIGGER model_candidate_finalizations_validate_insert
BEFORE INSERT ON model_candidate_finalizations
WHEN (
    SELECT COUNT(*)
    FROM model_candidate_evaluations AS link
    JOIN model_evaluations AS evaluation
        ON evaluation.evaluation_id = link.evaluation_id
    WHERE link.candidate_id = NEW.candidate_id AND evaluation.status = 'COMPLETE'
) <> (
    SELECT fold_count FROM training_runs
    WHERE training_run_id = (
        SELECT training_run_id FROM model_candidates WHERE candidate_id = NEW.candidate_id
    )
)
   OR EXISTS (
       SELECT 1
       FROM model_candidate_evaluations AS link
       JOIN model_evaluations AS evaluation
           ON evaluation.evaluation_id = link.evaluation_id
       WHERE link.candidate_id = NEW.candidate_id AND evaluation.status <> 'COMPLETE'
   )
   OR EXISTS (
       SELECT link.evaluation_id
       FROM model_candidate_evaluations AS link
       JOIN model_evaluation_runs AS membership
           ON membership.evaluation_id = link.evaluation_id AND membership.role = 'validation'
       JOIN node_results AS result
           ON result.run_id = membership.run_id
       LEFT JOIN oof_predictions AS oof
           ON oof.evaluation_id = link.evaluation_id
          AND oof.run_id = result.run_id
          AND oof.node_pk = result.node_pk
       WHERE link.candidate_id = NEW.candidate_id
         AND oof.evaluation_id IS NULL
   )
BEGIN
    SELECT RAISE(
        ABORT,
        'candidate finalization requires one COMPLETE evaluation per fold with full OOF coverage'
    );
END;

CREATE TABLE model_artifact_candidates (
    model_id INTEGER PRIMARY KEY
        REFERENCES trained_models(model_id) ON DELETE RESTRICT,
    candidate_id INTEGER NOT NULL
        REFERENCES model_candidates(candidate_id) ON DELETE RESTRICT
);

CREATE TRIGGER model_artifact_candidates_immutable_update
BEFORE UPDATE ON model_artifact_candidates
BEGIN
    SELECT RAISE(ABORT, 'artifact/candidate links are immutable');
END;

CREATE TRIGGER model_artifact_candidates_immutable_delete
BEFORE DELETE ON model_artifact_candidates
BEGIN
    SELECT RAISE(ABORT, 'artifact/candidate links are immutable');
END;

CREATE TRIGGER model_artifact_candidates_requires_finalized_matching_candidate
BEFORE INSERT ON model_artifact_candidates
WHEN NOT EXISTS (
    SELECT 1
    FROM model_candidates AS candidate
    JOIN model_candidate_finalizations AS finalization
        ON finalization.candidate_id = candidate.candidate_id
    JOIN trained_models AS artifact
        ON artifact.training_run_id = candidate.training_run_id
       AND artifact.feature_contract_id = candidate.feature_contract_id
       AND artifact.feature_contract_sha256 = candidate.feature_contract_sha256
       AND artifact.ordered_features_json = candidate.ordered_features_json
       AND artifact.preprocessing_json = candidate.preprocessing_json
       AND artifact.target_transform_json = candidate.target_transform_json
       AND artifact.hyperparameters_json = candidate.hyperparameters_json
       AND artifact.algorithm = candidate.algorithm
    WHERE candidate.candidate_id = NEW.candidate_id
      AND artifact.model_id = NEW.model_id
)
BEGIN
    SELECT RAISE(
        ABORT,
        'artifact link requires a finalized candidate whose recipe matches the artifact'
    );
END;

CREATE TABLE model_rankings (
    ranking_id INTEGER PRIMARY KEY,
    training_run_id INTEGER NOT NULL
        REFERENCES training_runs(training_run_id) ON DELETE RESTRICT,
    target TEXT NOT NULL CHECK(target IN ('inunda','vol_inundacion_m3','system')),
    primary_metric TEXT NOT NULL,
    primary_direction TEXT NOT NULL CHECK(primary_direction IN ('maximize','minimize')),
    metric_registry_id TEXT NOT NULL,
    metric_registry_sha256 TEXT NOT NULL CHECK(length(metric_registry_sha256)=64),
    tie_breakers_json TEXT NOT NULL,
    invalid_score_policy TEXT NOT NULL,
    created_at_utc TEXT NOT NULL
);

CREATE TRIGGER model_rankings_immutable_update
BEFORE UPDATE ON model_rankings
BEGIN
    SELECT RAISE(ABORT, 'model rankings are immutable');
END;

CREATE TRIGGER model_rankings_immutable_delete
BEFORE DELETE ON model_rankings
BEGIN
    SELECT RAISE(ABORT, 'model rankings are immutable');
END;

CREATE TABLE model_ranking_entries (
    ranking_id INTEGER NOT NULL
        REFERENCES model_rankings(ranking_id) ON DELETE RESTRICT,
    entry_id INTEGER NOT NULL,
    classifier_candidate_id INTEGER
        REFERENCES model_candidates(candidate_id) ON DELETE RESTRICT,
    regressor_candidate_id INTEGER
        REFERENCES model_candidates(candidate_id) ON DELETE RESTRICT,
    PRIMARY KEY (ranking_id, entry_id),
    CHECK (classifier_candidate_id IS NOT NULL OR regressor_candidate_id IS NOT NULL)
);

CREATE TRIGGER model_ranking_entries_immutable_update
BEFORE UPDATE ON model_ranking_entries
BEGIN
    SELECT RAISE(ABORT, 'ranking entries are immutable');
END;

CREATE TRIGGER model_ranking_entries_immutable_delete
BEFORE DELETE ON model_ranking_entries
BEGIN
    SELECT RAISE(ABORT, 'ranking entries are immutable');
END;

CREATE TRIGGER model_ranking_entries_not_after_finalization
BEFORE INSERT ON model_ranking_entries
WHEN EXISTS (
    SELECT 1 FROM model_ranking_finalizations WHERE ranking_id=NEW.ranking_id
)
BEGIN
    SELECT RAISE(ABORT, 'no entry is added after ranking finalization');
END;

CREATE TABLE model_ranking_scores (
    ranking_id INTEGER NOT NULL,
    entry_id INTEGER NOT NULL,
    metric_ordinal INTEGER NOT NULL,
    metric_name TEXT NOT NULL,
    value REAL,
    valid INTEGER NOT NULL CHECK(valid IN (0,1)),
    invalid_reason TEXT,
    PRIMARY KEY (ranking_id, entry_id, metric_ordinal),
    FOREIGN KEY (ranking_id, entry_id)
        REFERENCES model_ranking_entries(ranking_id, entry_id) ON DELETE RESTRICT,
    CHECK ((valid=1 AND value IS NOT NULL) OR (valid=0 AND invalid_reason IS NOT NULL))
);

CREATE TRIGGER model_ranking_scores_immutable_update
BEFORE UPDATE ON model_ranking_scores
BEGIN
    SELECT RAISE(ABORT, 'ranking scores are immutable');
END;

CREATE TRIGGER model_ranking_scores_immutable_delete
BEFORE DELETE ON model_ranking_scores
BEGIN
    SELECT RAISE(ABORT, 'ranking scores are immutable');
END;

CREATE TRIGGER model_ranking_scores_not_after_finalization
BEFORE INSERT ON model_ranking_scores
WHEN EXISTS (
    SELECT 1 FROM model_ranking_finalizations WHERE ranking_id=NEW.ranking_id
)
BEGIN
    SELECT RAISE(ABORT, 'no score is added after ranking finalization');
END;

CREATE TABLE model_ranking_finalizations (
    ranking_id INTEGER PRIMARY KEY
        REFERENCES model_rankings(ranking_id) ON DELETE RESTRICT,
    winner_entry_id INTEGER NOT NULL,
    finalized_at_utc TEXT NOT NULL,
    FOREIGN KEY (ranking_id, winner_entry_id)
        REFERENCES model_ranking_entries(ranking_id, entry_id) ON DELETE RESTRICT
);

CREATE TRIGGER model_ranking_finalizations_immutable_update
BEFORE UPDATE ON model_ranking_finalizations
BEGIN
    SELECT RAISE(ABORT, 'ranking finalizations are immutable');
END;

CREATE TRIGGER model_ranking_finalizations_immutable_delete
BEFORE DELETE ON model_ranking_finalizations
BEGIN
    SELECT RAISE(ABORT, 'ranking finalizations are immutable');
END;

CREATE TABLE model_promotion_rankings (
    promotion_id INTEGER PRIMARY KEY
        REFERENCES model_promotions(promotion_id) ON DELETE RESTRICT,
    ranking_id INTEGER NOT NULL
        REFERENCES model_rankings(ranking_id) ON DELETE RESTRICT
);

CREATE TRIGGER model_promotion_rankings_immutable_update
BEFORE UPDATE ON model_promotion_rankings
BEGIN
    SELECT RAISE(ABORT, 'promotion/ranking links are immutable');
END;

CREATE TRIGGER model_promotion_rankings_immutable_delete
BEFORE DELETE ON model_promotion_rankings
BEGIN
    SELECT RAISE(ABORT, 'promotion/ranking links are immutable');
END;

CREATE TRIGGER model_promotion_rankings_requires_finalized_ranking
BEFORE INSERT ON model_promotion_rankings
WHEN NOT EXISTS (
    SELECT 1 FROM model_ranking_finalizations WHERE ranking_id=NEW.ranking_id
)
BEGIN
    SELECT RAISE(ABORT, 'promotion must link a finalized ranking');
END;

CREATE TABLE model_promotion_finalizations (
    promotion_id INTEGER PRIMARY KEY
        REFERENCES model_promotions(promotion_id) ON DELETE RESTRICT,
    finalized_at_utc TEXT NOT NULL
);

CREATE TRIGGER model_promotion_finalizations_immutable_update
BEFORE UPDATE ON model_promotion_finalizations
BEGIN
    SELECT RAISE(ABORT, 'promotion finalizations are immutable');
END;

CREATE TRIGGER model_promotion_finalizations_immutable_delete
BEFORE DELETE ON model_promotion_finalizations
BEGIN
    SELECT RAISE(ABORT, 'promotion finalizations are immutable');
END;

CREATE TRIGGER model_promotion_finalizations_validate_insert
BEFORE INSERT ON model_promotion_finalizations
WHEN NOT EXISTS (
    SELECT 1
    FROM model_promotion_rankings AS link
    JOIN model_ranking_finalizations AS finalization
        ON finalization.ranking_id = link.ranking_id
    JOIN model_ranking_scores AS score
        ON score.ranking_id = finalization.ranking_id
       AND score.entry_id = finalization.winner_entry_id
       AND score.metric_ordinal = 0
       AND score.valid = 1
    JOIN model_promotions AS promotion
        ON promotion.promotion_id = link.promotion_id
       AND promotion.primary_metric = (
           SELECT primary_metric FROM model_rankings WHERE ranking_id = finalization.ranking_id
       )
       AND promotion.primary_value = score.value
    JOIN model_ranking_entries AS entry
        ON entry.ranking_id = finalization.ranking_id
       AND entry.entry_id = finalization.winner_entry_id
    WHERE link.promotion_id = NEW.promotion_id
      AND (
          entry.classifier_candidate_id IS NULL
          OR EXISTS (
              SELECT 1 FROM model_artifact_candidates
              WHERE candidate_id = entry.classifier_candidate_id
                AND model_id = promotion.classifier_model_id
          )
      )
      AND (
          entry.regressor_candidate_id IS NULL
          OR EXISTS (
              SELECT 1 FROM model_artifact_candidates
              WHERE candidate_id = entry.regressor_candidate_id
                AND model_id = promotion.regressor_model_id
          )
      )
)
BEGIN
    SELECT RAISE(
        ABORT,
        'promotion finalization requires its artifacts and metric to match the finalized ranking winner'
    );
END;
