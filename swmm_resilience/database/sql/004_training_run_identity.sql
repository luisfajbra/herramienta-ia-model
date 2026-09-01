CREATE TRIGGER training_runs_immutable_primary_key
BEFORE UPDATE OF training_run_id ON training_runs
BEGIN
    SELECT RAISE(ABORT, 'training run identity is immutable');
END;
