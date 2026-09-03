# SQLite V17 Provenance Hardening Design

> **PARTIALLY SUPERSEDED (2026-09-03):** migration 005 and its append-only
> lifecycle (§3.1-3.4) shipped and are in production, referenced directly by
> `csv_backfill.py`. §3.4.1's full `model_candidates`/`model_rankings`/
> `model_promotions`/`model_selections` writer was deliberately *not* built —
> `csv_backfill.py` calls this the "minimal path, not the full evidence
> chain." The active plan going forward is `docs/FLUJO_ACTUAL.md` §12; treat
> the unimplemented parts of this spec as historical, not a live target.

**Date:** 2026-08-22  
**Status:** Draft for user review  
**Branch:** `cleanup/sqlite-v17-pipeline-consolidation`

## 1. Purpose

This addendum closes integrity gaps discovered during the final adversarial
review of the SQLite V17 foundation. It is authoritative where it refines the
original consolidation spec and Plans A/C. It does not change the canonical
17-feature set, model-selection policy, or the decision to use one local
SQLite database as the source of truth.

The required outcome is stronger than passing foreign-key checks: a model,
metric, promotion, or prediction must remain traceable to the exact immutable
training inputs and OOF evidence that produced it.

## 2. Confirmed Problems

The final Plan A review reproduced these failures on a fully migrated database:

1. `training_run_id` can be changed through SQLite's `rowid`, `_rowid_`, and
   `oid` aliases without firing the migration 004 trigger.
2. Fitting provenance in `training_runs` can be edited after artifacts and
   selections exist.
3. `model_evaluations` and `oof_predictions` can be updated or deleted;
   deleting an evaluation cascades into its OOF evidence.
4. A model artifact can be promoted even when its contract hash or included
   run IDs disagree with its training run.
5. Migration 002 inferred a legacy `system` promotion by pairing two artifacts
   but copied the classifier's selected value into the system primary metric.
   The resulting metric/value pair is not trustworthy.
6. The feature contract accepts NumPy complex dtypes and silently discards the
   imaginary component when converting to `float`.
7. `.sqlite3`, WAL, and SHM files are not ignored by Git.

All of these failures are closed by default. No compatibility path may
silently weaken the checks.

## 3. Decisions

### 3.1 Append-only migration 005

Create `005_provenance_integrity.sql`. Migrations 001 through 004 remain
byte-identical. Migration 005 runs through the existing atomic migration
executor and either commits all guards/invalidations or leaves the database at
version 004 unchanged.

The migration performs read-only preflight checks before changing the schema.
Structural corruption or unverifiable migration history aborts with a clear
integrity error. Structurally readable pre-005 ML evidence is preserved but
explicitly invalidated; it is never blessed as operational provenance.

### 3.2 Normalized membership instead of per-row JSON parsing

Add two normalized membership tables:

```sql
CREATE TABLE training_run_inputs (
    training_run_id INTEGER NOT NULL,
    run_id INTEGER NOT NULL,
    PRIMARY KEY (training_run_id, run_id),
    FOREIGN KEY (training_run_id)
        REFERENCES training_runs(training_run_id) ON DELETE RESTRICT,
    FOREIGN KEY (run_id)
        REFERENCES runs(run_id) ON DELETE RESTRICT
);

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
```

Migration 005 does not turn mutable pre-005 JSON into trusted normalized
evidence. It records every pre-005 `training_run_id` in append-only
`training_run_provenance_invalidations` with reason
`pre005_mutable_provenance`; their historical JSON/evaluations/OOF remain
queryable but receive no normalized memberships, candidates, or finalizations.
All operational training after 005 uses newly inserted IDs and these normalized
tables. This deterministic invalidation lets the upgrade succeed without
pretending that previously mutable evidence was verified.

Every persisted run-ID array is canonical: JSON array, positive integer IDs,
strictly ascending order, and no duplicates. Therefore `[1,2]` and `[2,1]` do
not represent two accepted serializations of the same membership; writers
store only `[1,2]`. Object-valued provenance JSON is compared structurally,
while membership arrays are compared against their normalized rows and their
canonical serialization.

Future writers use this lifecycle:

1. insert a `training_runs` row as `PENDING` with its immutable configuration;
2. insert all `training_run_inputs` rows;
3. transition the run to `RUNNING`;
4. insert each evaluation as `PENDING`;
5. insert its train/validation membership rows;
6. transition the evaluation to `RUNNING`;
7. append OOF rows and metrics;
8. transition the evaluation to `COMPLETE` or `FAILED`;
9. finalize candidates and the complete initial ranking;
10. while the training run is still `RUNNING`, insert/link winner artifacts;
11. transition the training run to `COMPLETE` and append finalized promotion
    and selection events in the same transaction.

Membership rows are insertable only while their owner is `PENDING`. They are
never updated or deleted. OOF inserts therefore use indexed membership checks
and never call `json_each` for every prediction.

Each membership insert must already appear in the owner's canonical JSON array;
an accidental extra row cannot pin unrelated simulation data even if the owner
later fails before reaching `RUNNING`.

The transition from `PENDING` to `RUNNING` is the consistency boundary. For a
training run, the normalized membership must be non-empty and equal to the
integer set in `included_run_ids_json`. For an evaluation, train and validation
memberships must be non-empty, disjoint, contained in the owning training-run
membership, and equal to their JSON arrays. Its fold must be within the
training run's declared fold count. Every evaluation in the same training run
and fold reuses the same train/validation partition. A mismatch aborts the
transition before any fitting or OOF insert can occur.

The same complete checks run again before `RUNNING -> COMPLETE`. Pre-005
PENDING/RUNNING rows are invalidated rather than resumed; recovery may mark
them failed, but no post-005 operational candidate can reference them.

### 3.2.1 Pin the exact source rows used for tabular training

Normalized IDs alone do not make their underlying data immutable. Inserting a
run into `training_run_inputs` pins the complete tabular sample source used by
`training_samples_v17`:

- the selected `runs` row;
- its `node_features` and `node_results` rows;
- the referenced scenario and its grouping fields;
- the participating node identities;
- the referenced network identity and source bytes.

Membership insertion requires the simulation run to be `COMPLETE` and to pass
the same positive `node_count`, feature/result cardinality, and exact-key
symmetry checks used by the canonical loader. Inside the same snapshot it also
runs the complete `TABULAR_V3_17` type/nullability/finiteness validation and
target-domain validation. It verifies every pinned
`network_sha256 == sha256(inp_bytes)` and any authoritative scenario/run config
hash defined by Plan B. SQLite triggers repeat the enforceable storage checks;
managed connections provide the versioned SHA-256 function and fail closed if
it is unavailable.

Operational training does not accept arbitrary query text. Migration 005 adds
an immutable `training_query_contract_id` and SHA-256 to `training_runs`; V17
training requires the allowlisted canonical `training_samples_v17` descriptor.
`query_sql` remains historical diagnostics, not authority. The validation gate
runs again before the training run becomes `COMPLETE`. Once pinned:

- the selected `runs`, `scenarios`, `nodes`, and `networks` rows cannot be
  updated or deleted;
- matching `node_features` and `node_results` cannot be inserted, updated, or
  deleted;
- cascaded cleanup that would reach pinned provenance fails closed.

`node_timeseries` is not part of the tabular V17 training contract and is not
pinned by this migration. The later CNN-LSTM design must define its own temporal
snapshot/freeze boundary.

This intentionally refines the earlier cascade policy. Cleanup repositories
must report `pinned by training provenance` instead of deleting referenced
simulation data. Add reverse indexes on `training_run_inputs(run_id,
training_run_id)` and `model_evaluation_runs(run_id, evaluation_id, role)` so
retention checks do not scan entire membership tables.

### 3.3 Immutable training identity and fitting configuration

Migration 005 replaces the migration 004 trigger with a general trigger:

```sql
BEFORE UPDATE ON training_runs
WHEN NEW.training_run_id IS NOT OLD.training_run_id
```

Because it observes every update, it also catches changes made through all
rowid aliases.

The following fitting fields are immutable after insertion:

- `target`;
- feature contract ID and SHA-256;
- query SQL and query parameters;
- included run IDs;
- grouping strategy and fold count;
- random seed;
- primary metric and tie breakers;
- Python and library versions.

Direct deletion of a training run is forbidden. Status transitions are
explicit and one-way:

```text
PENDING -> RUNNING | FAILED
RUNNING -> COMPLETE | FAILED
COMPLETE -> no transition
FAILED -> no transition
```

Lifecycle timestamps and failure details may change only as part of the
corresponding non-terminal transition. Once a run is terminal, its complete
row is immutable.

`training_run_provenance_invalidations(training_run_id PRIMARY KEY REFERENCES
training_runs ON DELETE RESTRICT, reason, invalidated_at_utc)` is append-only
and protected from UPDATE/DELETE/conflicting REPLACE. `valid_training_runs`
anti-joins it, and every new candidate, ranking, artifact link, promotion, or
operational metric must join `valid_training_runs`. Thus a pre-005 status such
as `COMPLETE` never makes its mutable evidence operational.

### 3.4 Immutable evaluations and OOF evidence

Evaluation identity, training-run ownership, task, algorithm,
hyperparameters, fold, and train/validation membership are immutable.
Evaluation status follows the same `PENDING -> RUNNING -> terminal` state
machine. Evaluations cannot be deleted, including failed evaluations.
`fit_seconds`, `predict_seconds`, failure details, and lifecycle status may be
updated only by a valid non-terminal transition. A terminal evaluation is
fully immutable.

OOF rows are append-only: `UPDATE` and `DELETE` always fail, and no conflicting
`REPLACE`/UPSERT may overwrite an existing identity. SQLite triggers cannot
distinguish a conflict-free `INSERT OR REPLACE` from a plain insert, so Python
repositories additionally forbid REPLACE syntax. A new OOF row is accepted
only when:

- its evaluation and owning training run are both `RUNNING`;
- classification maps to target `inunda` and regression maps to
  `vol_inundacion_m3`;
- its `fold_id` matches the evaluation fold;
- its `run_id` is an indexed validation member of that evaluation;
- its `(run_id, node_pk)` exists in persisted `node_results`;
- `observed` equals the corresponding persisted target value;
- numeric prediction fields are finite;
- classification `observed` and `predicted` are in `{0,1}`, and probability is
  non-null and within `[0,1]`;
- regression observation is non-negative and probability is null.

Partial OOF belonging to a failed evaluation remains stored as failed-run
evidence but is never eligible for ranking, metrics used for selection, or
promotion. Operational OOF queries always join an evaluation whose status is
`COMPLETE`.

### 3.4.1 Normalize candidates and the promotion evidence chain

OOF rows are evidence for a candidate, not directly for a serialized final
artifact. Add these append-only entities:

- `model_candidates`: one immutable full fitting recipe;
- `model_candidate_evaluations`: assigns every evaluation to exactly one
  candidate;
- `model_candidate_finalizations`: marks a candidate complete only after all
  fold and OOF coverage checks pass;
- `model_artifact_candidates`: links each final BLOB to the exact completed
  candidate that justified its fit;
- `model_rankings`: freezes one ranking definition and eligible universe;
- `model_ranking_entries`: enumerates every eligible candidate or system pair;
- `model_ranking_scores`: stores every required metric for every entry;
- `model_ranking_finalizations`: freezes the deterministic winner;
- `model_promotion_rankings`: links a promotion to exactly one finalized
  ranking and its winner;
- `model_promotion_finalizations`: makes a promotion operational only after its
  models, candidates, ranking, and metric/value fields agree.

The immutable candidate recipe includes:

- training run, target and task;
- feature contract ID/hash and exact ordered features;
- preprocessing/imputation/scaling definition;
- target transform;
- estimator algorithm and canonical hyperparameters;
- pipeline-builder/registry version;
- `candidate_definition_sha256` over that canonical descriptor.

An evaluation is linked before it can enter `RUNNING`, and its task, algorithm,
hyperparameters and pipeline definition must equal its candidate. The link has
`evaluation_id UNIQUE`, so one evaluation cannot justify two candidates.

Candidate finalization requires exactly one `COMPLETE` evaluation for every
fold `0..fold_count-1`. For every fold, train and validation memberships are
disjoint and their union is the complete training population. Across folds,
validation memberships are disjoint and their union is exactly that
population. Every expected validation `(run_id,node_pk)` key has exactly one
OOF row for the candidate, with no extra keys. Candidates in the same training
run and fold reuse the identical partition.

The artifact link requires the same training run, compatible target/task,
algorithm, contract, ordered features, preprocessing, target transform,
hyperparameters, and candidate-definition hash. `model_id` is the primary key
of the link, so one artifact has exactly one candidate. An artifact without a
completed candidate link is historical storage only and cannot enter a valid
promotion.

Each `model_rankings` row freezes the training run, target, primary metric and
direction, metric-registry definition/hash, canonical ranking parameters,
ordered tie breakers with directions, invalid-score policy, and creation time.
Target-specific rankings enumerate every finalized candidate eligible for that
target. System rankings enumerate the full Cartesian product of every eligible
finalized classifier and regressor candidate from the same run. The universe
is immutable after ranking finalization.

Every ranking entry must have exactly one primary score and every ordered tie
score required by the definition. Scores contain finite value or an explicit
invalid reason and belong to one ranking only; scores from different rerankings
cannot be mixed. Finalization rejects an omitted candidate/pair, missing score,
duplicate metric ordinal, definition mismatch, or policy mismatch. It applies
the frozen invalid-score policy and deterministic directions/tie order, then
records exactly the entry at rank 1 as winner.

The managed SQLite connection registers the versioned ranking validator used
by the finalization trigger. It recomputes the required scores from the linked
complete OOF evidence and validates the winner. A raw connection that lacks the
function fails closed at finalization. The ranking definition, normalized score
set, and winner proof each have a persisted SHA-256 covered by regression
fixtures.

Promotion finalization requires exactly one finalized ranking, requires the
promotion artifacts to link to that ranking's winner candidate(s), and requires
`primary_metric`/`primary_value` to equal the winner's valid primary score.
Descriptive `ranking_json` remains useful for reports but is never the
integrity authority.

Cardinality/freeze rules are exact:

- candidates are inserted only while the training run is `RUNNING`;
- `evaluation_id` is unique across candidate links;
- one finalization row exists per candidate, ranking, and promotion;
- no evaluation link is added after candidate finalization;
- no entry or score is added after ranking finalization;
- one candidate link exists per artifact;
- a selection can reference only a simultaneously or previously finalized,
  non-invalidated promotion.

Rows may be assembled in one transaction before their finalization row exists.
That intermediate state is intentionally non-operational. Finalization, not
the initial artifact/promotion insert, is the authority that validates all
multirow relationships.

Legacy promotions created without this normalized evidence are invalidated,
including target-specific legacy promotions. Historical artifacts and events
remain queryable, but none is operational until a new evidence-backed
promotion is appended.

### 3.5 Artifact-to-training-run coherence

Every future `trained_models` insert must agree with its owning training run:

- compatible target and one completed candidate from the same run;
- exact feature contract ID and SHA-256;
- equal normalized included-run membership;
- semantically equal query-parameter JSON;
- equal seed, grouping strategy, Python version, and library-version JSON.

The training run and artifact must both carry the published V17 descriptor
hash
`56af955cadb90dda63b79f48dcec18bccaabc8eb33bcce07f5bf1874dcfbca8a`.
The artifact's `ordered_features_json` must equal the canonical 17-name array
in exact order; extra, missing, duplicated, or reordered names are invalid.

There are two valid artifact-write paths, and neither reopens a terminal run:

1. **Initial finalization:** while the training run is `RUNNING`, create and
   finalize the complete ranking, fit its winner, insert required artifact(s)
   and candidate link(s), register the ranking as the run's initial ranking,
   and transition the run to `COMPLETE`. Insert promotion, ranking link,
   promotion finalization, and selection in that same transaction.
2. **Post-completion reranking:** the run stays `COMPLETE`. In one transaction,
   insert the new ranking definition, complete universe, scores and ranking
   finalization; fit only its frozen winner; insert artifact(s) and candidate
   link(s); then insert promotion, ranking link, promotion finalization, and
   selection. No candidate, evaluation, fold, OOF row, or existing artifact
   changes. A rollback leaves no scores, artifacts, or events from the attempt.

`RUNNING -> COMPLETE` is allowed only when:

- every evaluation is terminal `COMPLETE` and none is `FAILED`;
- every declared candidate is finalized and the candidate catalog is frozen;
- the initial ranking is finalized over that complete catalog;
- the winner's target artifact exists and is linked, or both classifier and
  regressor artifacts exist for a system winner;
- no incomplete artifact/candidate/ranking assembly exists.

After the transition, no candidate or evaluation can be added. The separate
reranking path may add only rankings, scores, final winner artifacts, and their
promotion/selection events.

At the completion boundary the initial ranking universe is rechecked against
the final candidate catalog. Adding a candidate after initial ranking
finalization therefore makes `RUNNING -> COMPLETE` fail until a new complete
initial ranking is finalized; an incomplete universe can never be frozen by a
timing race.

For the second path, schema guards accept the artifact only if its candidate
coverage and frozen post-005 provenance are already complete. Parent status
`COMPLETE` is valid only for this reranking path; it never makes a pre-005
artifact valid.

JSON objects are compared structurally, not as raw text; object key order and
whitespace do not create false mismatches. Array order remains significant.
JSON containing duplicate object keys is rejected because it has no unique
portable interpretation. Comparison uses the complete JSON tree, including
node type and scalar value; it must not rely on a shallow top-level test.
All future Python writers also serialize stored JSON canonically with sorted
keys and compact separators.

Migration 005 never creates candidate links or finalizations for pre-005
artifacts: their evidence was mutable under the old schema and cannot be proven
afterward. They remain immutable and unlinked, their training runs/promotions
are invalidated, and operational views exclude them. Only evidence written
under 005 guards can become valid. Contradictory post-005 normalized evidence
always aborts its transaction.

### 3.6 Honest handling of legacy promotions

Migration 005 does not overwrite or delete historical promotions. It adds an
append-only invalidation event:

```sql
CREATE TABLE model_promotion_invalidations (
    promotion_id INTEGER PRIMARY KEY,
    reason TEXT NOT NULL,
    invalidated_at_utc TEXT NOT NULL,
    FOREIGN KEY (promotion_id)
        REFERENCES model_promotions(promotion_id) ON DELETE RESTRICT
);
```

Rows are immutable and no conflicting `REPLACE`/UPSERT can overwrite an
existing identity. Migration 005 invalidates every promotion present before
005 because none can honestly prove a complete frozen ranking universe under
the new schema. Rows inferred from `001_v17_initial` use specific reason codes
for fabricated system value or missing target evidence; later 002–004 rows use
`missing_normalized_ranking_evidence`. No pre-005 promotion is operational by
default, with no reconstruction exception.

Add `valid_model_promotions` and rebuild `active_model_selections`. The active
promotion view is exactly `model_promotions JOIN
model_promotion_finalizations JOIN model_promotion_rankings JOIN
model_ranking_finalizations LEFT ANTI JOIN model_promotion_invalidations`.
Merely existing and not being invalidated is insufficient. The active-selection
view first determines the unique leaf over the complete immutable selection
chain and only then joins that validity view. Filtering promotions before leaf
selection is forbidden because it would silently reactivate an older model. A
selection pointing at an invalidated or unfinalized promotion stays in history
but is not active. A later valid promotion may supersede that historical leaf
normally.

Consequently, an upgraded database may temporarily have no active selection
for any target. The only recovery is an explicit evidence-backed
re-ranking/refit through Plan C; the migration must not invent a replacement
value.

New selections cannot reference an invalidated promotion. Direct consumers of
raw `model_promotions` must join the invalidation state or use
`valid_model_promotions`.

### 3.7 Real-only feature contract

`TABULAR_V3_17.validate_frame()` rejects any complex dtype before numeric
coercion. This includes values with a zero imaginary component. Object columns
containing complex values remain invalid through the existing numeric-type
gate. No imaginary component may be discarded or bypass the finiteness check.

### 3.8 SQLite files remain local

Add Git ignore rules for:

```text
*.sqlite3
*.sqlite3-wal
*.sqlite3-shm
*.sqlite3-journal
*.sqlite
*.sqlite-journal
*.db-wal
*.db-shm
*.db-journal
*.workflow.lock
```

The existing `*.db` rule remains. A behavioral test invokes
`git check-ignore` on the configured database and sidecar names. Intentional
SQLite fixtures require an explicit negated rule in their fixture directory;
`git add -f` is not the documented workflow.

## 4. Preflight And Failure Behavior

SQLite 3.45 cannot raise a dynamic owner ID from a free-standing SQL
expression, so migration 005 has a versioned Python preflight hook. The
migration catalog associates version 005 with a validator that runs after
`BEGIN IMMEDIATE` and before the first DDL statement. Diagnostic queries return
`invariant`, `owner_kind`, and `owner_id`; the hook raises a typed exception
without BLOBs or dataset contents.

The validator source has its own SHA-256 recorded transactionally in
`schema_migration_validators(version INTEGER PRIMARY KEY REFERENCES
schema_migrations(version) ON DELETE RESTRICT, validator_name TEXT UNIQUE NOT
NULL, validator_sha256 TEXT NOT NULL CHECK(length(validator_sha256)=64))` and
verified on every later migration run. General UPDATE/DELETE and conflicting
REPLACE guards make this history immutable. The published migration manifest
freezes both the 005 SQL hash and validator hash.
SQL constraints, invalidation counts, and triggers provide defense in depth,
so bypassing the Python hook still cannot make pre-005 evidence operational or
commit new normalized rows that violate the same core invariants.

Migration 005 must fail before installing any new guard when it finds:

- a non-prefix migration history or unexpected checksum;
- an existing foreign-key or `integrity_check` violation;
- a schema object incompatible with the expected version-004 catalog;
- an active caller transaction or workflow owner;
- inability to enumerate every pre-005 training run and promotion for complete
  deterministic invalidation.

Malformed or incomplete legacy ML provenance is not parsed into the new
authority tables. It is handled by deterministic training-run/promotion
invalidation and therefore does not abort an otherwise structurally sound
upgrade. It remains historical and non-operational.

An on-disk database already at version 001–004 must use
`upgrade_database_with_backup()`. The operation first acquires the same
cross-process workflow lock used by training/recovery. The lock is an advisory
OS file lock at `<database>.workflow.lock`, implemented with `msvcrt` on Windows
and `fcntl` on POSIX; its held descriptor is released automatically on process
exit/crash. All managed writers must acquire it for training lifecycle or
migration operations.

Under that one lock, databases at 001–003 first advance to exactly 004. The
operation then checkpoints WAL, creates a version-004 backup, verifies its
integrity, and records an internal single-use receipt bound to the resolved
source/backup paths, checkpointed logical database fingerprint, backup
SHA-256, and schema version. It acquires `BEGIN IMMEDIATE`, revalidates the
logical fingerprint so an unmanaged WAL writer cannot create a stale receipt,
and applies 005 without releasing the workflow lock. A stale/mismatched receipt
aborts and is never accepted from a caller-constructed object. A live trainer
or recovery owner makes upgrade fail before checkpoint.

A fresh empty database may apply 001–005 directly; an in-memory database is
permitted only through the explicit test API.

Because the executor is atomic, any failure leaves schema history and data at
version 004. Tests restore the verified backup and prove its schema/data match
the preflight source.

### 4.1 Interrupted-run recovery

The system never guesses that a live trainer is stale and performs no automatic
startup recovery. The local CLI/UI exposes an explicit, user-confirmed recovery
operation. It first acquires the application's exclusive training workflow
lock and `BEGIN IMMEDIATE`; if either is already owned, recovery fails. It then
changes the selected abandoned `RUNNING` training run/evaluations to `FAILED`,
records a fixed interruption reason and timestamp, and preserves all partial
OOF, metrics, and artifacts as non-promotable evidence. Recovery never deletes
rows or reopens a terminal owner.

Migration 005 replaces the incompatible 003 status guards. A `RUNNING ->
FAILED` transition is allowed even when completed candidates, partial
artifacts, or candidate finalizations exist. Recovery also changes every
`PENDING` or `RUNNING` evaluation under that run to `FAILED`; already-complete
evaluations/candidates remain immutable evidence. Only a valid
`model_promotion_finalization` or selection blocks recovery. Partial artifacts
and links remain non-operational because their parent run is `FAILED`. A
promoted or terminal run cannot be recovered this way. New training uses a new
`training_run_id` rather than resuming the failed owner.

Metrics are append-only. Evaluation metrics may be inserted only while their
evaluation is `RUNNING`; candidate/ranking metrics may be appended after a
candidate and run are complete to support reporting and reranking. Every score
records its concrete candidate or candidate pair, and promotions reference the
exact score rows they use.

`model_ranking_scores` is the only metric authority for ranking, selection, and
promotion. Existing `model_metrics` rows owned by a training run or artifact
are diagnostic history and never satisfy a promotion guard. Operational
evaluation-metric views join a `COMPLETE` evaluation and finalized candidate;
failed, unlinked, and legacy evidence is excluded. The UI labels raw legacy
metrics as unverified instead of mixing them with valid ranking scores.

## 5. Performance Boundaries

- No JSON table-valued function runs per OOF insert.
- OOF validation uses primary-key or explicit indexed lookups.
- Migration-only JSON expansion may be linear in the number of persisted
  training/evaluation membership entries.
- Added triggers affect training writes, not simulation timestep ingestion or
  training-sample reads.
- The million-row timestep scale gate must remain within 120 seconds and
  500 MiB.
- Add an `EXPLAIN QUERY PLAN` gate for `active_model_selections` after the
  invalidation join; no full scan of OOF or model BLOB table is acceptable.
- Add indexed/query-plan gates for candidate-to-evaluation/fold,
  candidate-to-artifact, ranking-to-entry/score by metric ordinal,
  promotion-to-ranking/finalization, and OOF coverage on
  `(evaluation_id,run_id,node_pk)`.
- Add an opt-in scale fixture with one million OOF rows that measures candidate
  coverage finalization and reranking. Each must remain under 120 seconds and
  the total fixture under 500 MiB after checkpoint; the benchmark reports both
  time and query plan without choosing schema winners automatically.

## 6. Verification Requirements

Implementation follows strict RED-GREEN-REFACTOR and separate independent
review. Required regressions include:

1. `rowid`, `_rowid_`, and `oid` identity updates with and without descendants;
2. permitted and forbidden training/evaluation state transitions;
3. mutation, deletion, UPSERT, and REPLACE attempts for configuration,
   memberships, evaluations, OOF, candidates, scores, finalizations, and
   promotion invalidations;
4. OOF target/fold/membership/observed-value and numeric-domain checks, plus
   rejection of zero/partial OOF, missing folds, overlapping validation folds,
   and incomplete population coverage;
5. candidate/artifact mismatches for algorithm, hyperparameters, folds, and
   every persisted provenance dimension, including
   semantically equal JSON with different key order, and distinct
   scaler/imputer/target-transform/pipeline versions;
6. promotions rejected without finalized candidate links or normalized primary
   score evidence, and system scores rejected unless the classifier/regressor
   pair matches the promoted artifacts;
   ranking tests omit the true best candidate, mix scores from two ranking IDs,
   alter direction/definition/tie order, and attempt associations after
   finalization;
7. valid reranking and new final artifact/promotion under an already
   `COMPLETE` run without creating evaluations or reopening it;
8. post-membership mutation/insertion/deletion attempts against pinned runs,
   features, targets, grouping scenarios, nodes, and networks;
9. fresh migration, idempotence, and 004-to-005 upgrade invalidating every
   historical PENDING/RUNNING/terminal ML owner; typed structural-preflight
   rollback, SQL defense-in-depth rollback, verified backup, and restore;
10. five contiguous migration checksums plus the versioned validator checksum;
11. exact published SHA-256 constants for migrations 001 through 005 and the
   validator, frozen after 005 review approval;
12. deterministic invalidation of legacy evidence-free promotions, no active
   fallback, and successful later supersession by a valid promotion;
13. an active-selection chain `valid older -> invalid leaf` returning zero, so
   filtering before leaf selection is caught;
14. interrupted RUNNING recovery to FAILED, preservation of partial evidence,
   acceptance with candidate finalization and PENDING evaluations, rejection
   while an active trainer lock exists, and rejection once promoted;
15. complex feature dtypes, including finite, zero-imaginary, NaN, and infinite
   imaginary components;
16. behavioral Git-ignore checks for DB/WAL/SHM/rollback journals;
17. active-selection query plan, database suites, explicit scale suite,
    Python 3.11 dependency resolution, SHAP runtime smoke, compileall, and the
    broad non-scale compatibility suite;
18. build and install a wheel in an isolated temporary environment and verify
    `importlib.resources` discovers migrations 001–005, the validator module,
    and the published hash manifest;
19. upgrade rejection with a live trainer, stale receipt after a WAL write, or
    network SHA-256 different from `inp_bytes`;
20. an unfinalized but non-invalidated promotion absent from every operational
    view, plus the one-million-row OOF finalization/reranking scale gate.

Mutation checks must prove that the new tests fail if identity guards,
immutability guards, membership checks, invalidation filters, or complex-dtype
rejection are removed.

## 7. Documentation And Downstream Contract

Plan B must document `pinned by training provenance` cleanup failures and never
mutate a simulation run after it becomes a normalized training input. Plan C
must follow the PENDING/membership/RUNNING lifecycle, candidate/finalization
chain, exact score evidence, workflow lock, and valid-promotion views. Neither
plan begins until Plan A plus this addendum pass independent final review.

Invalidated legacy selections and any required re-training are
reported explicitly to the local operator. The system must never silently fall
back to an older or unverifiable model.

## 8. Non-goals

- Do not repair unknown historical provenance automatically.
- Do not rewrite migrations 001 through 004.
- Do not compute a replacement system metric inside SQL.
- Do not delete failed or invalidated evidence.
- Do not add a multi-user database server, remote storage, or background
  migration service.
- Do not begin model training, prediction, CNN-LSTM migration, or legacy
  pipeline deletion in this correction.

## 9. Acceptance

This design is accepted only when:

- all confirmed bypasses are covered by behavioral tests and independent
  adversarial review;
- structurally corrupt pre-005 databases fail atomically, all structurally
  readable pre-005 ML evidence becomes explicitly invalid history, and every
  contradictory post-005 normalized/source-provenance write fails atomically;
- selected simulation rows are pinned and every operational artifact traces
  through a finalized candidate to complete fold/OOF evidence;
- every operational promotion references normalized finite score evidence that
  matches its promoted candidate or candidate pair;
- legacy evidence-free promotions are auditable but never operational;
- valid initial training, interrupted-run recovery, and post-completion
  reranking remain possible without reopening terminal owners;
- the existing loader, query-plan, scale, SHAP, and compatibility gates remain
  green, apart from explicitly documented environmental Tcl/Tk limitations.
