# SQLite V17 Training Frame Cutover Design

**Date:** 2026-09-03
**Status:** Approved in conversation; ready for implementation planning
**Branch:** `feature/hydrograph-augmentation-sql-persistence`
**Continues:** `docs/FLUJO_ACTUAL.md` §12 ("Migración a SQLite como única
fuente de verdad"), steps 3-8 of §12.8. Steps 1-2 are already done (see
§12.2-12.3 of that document).
**Relationship to prior specs:** this is the *minimal-path* continuation.
It deliberately does **not** resume the bigger August roadmap (`SimulationStore`,
unified ML registry, legacy retirement) — those plan/spec files have been
marked superseded and point back here.

## 1. Purpose

Make `training_samples_v17` the read **and** write source of truth for the
24-column training frame that `data/training/dataset_final.csv` holds today.
CSV becomes an optional, on-demand export. Nothing else changes: simulation
execution, the 17-feature contract, XGBoost training, LOSO/GroupKFold5
evaluation, the legacy desktop GUI, and the `.joblib` artifacts in
`outputs/models/` all stay exactly as they are.

## 2. Non-goals

- No `SimulationStore` or per-timestep persistence.
- No unified model registry (`model_candidates`/`model_rankings`/
  `model_promotions`/`model_selections`) — out of scope, as already decided
  for `csv_backfill.py`.
- No retirement of the legacy desktop GUI, `ml/train.py`, or
  `swmm_resilience.db`.
- No change to the 17-feature contract, training algorithm, or evaluation
  methodology.

## 3. Baseline (verified against code, 2026-09-03)

- `training_samples_v17` view: exists (migration 001).
- Read loader: `swmm_resilience/database/training_queries.py::load_training_samples(conn, run_ids=None)` + `export_training_samples_csv(...)` — tested in
  `tests/database/test_training_view_v17.py`. Used only by that test today;
  no pipeline code calls it.
- Write path: `python main.py --persist-sql` → `csv_backfill.py` reads the
  CSV and backfills SQL (`backfill_networks_and_runs`) + trains and persists
  the evidence chain (`persist_training_run`). This is a separate, optional,
  manual step — the CSV remains the only thing the default pipeline reads
  or writes.
- Every other command (`--resilience-curve`, `--flood-volume-curve`,
  `--factor-comparison`, `--only-maps`, `--only-ml`/full pipeline,
  `--analyze-features`, `--evaluate-shapes`, `--evaluate-generalization`)
  calls `pd.read_csv(config.dataset.output_path)` directly. §12.8 of
  `FLUJO_ACTUAL.md` only listed the first five of these for cutover; the
  last three were missed and are added to scope here (see §5.3).

## 4. Key decisions

**4.1 — Loader stays at `database/training_queries.py`.** Do not move/rename
it to `dataset/loader.py` as the original §12.3 sketch proposed. It already
depends on `sqlite3.Connection` and lives next to `migrations.py`/
`connection.py`; the `dataset/` package is CSV/DataFrame-oriented
(`assembler.py`, `shape_selection.py`, `validator.py`). Moving it buys
nothing and would orphan its existing, passing test file.

**4.2 — No new SQL-side filters.** The original §12.3 sketch proposed
filtering by `network`/`shape_id`/`factor_mult`/`only_flooded`/
`sample_frac`/`chunksize` pushed into SQL. Checked every real consumer:
all of them read the *entire* frame and filter by factor/shape in pandas
afterward, exactly like they do with the CSV today. `load_training_samples(conn)`
called with no `run_ids` already returns that same full frame. Building
extra filters now would be speculative complexity with no caller — skip it.

**4.3 — One convenience wrapper.** Every one of the eight call sites in
`main.py` needs the same three lines: open a managed connection, apply
migrations, call the loader, close the connection. Add
`load_training_frame(db_path: Path) -> pd.DataFrame` to
`training_queries.py` (thin wrapper around `connect_managed_database` +
`apply_migrations` + `load_training_samples` + `conn.close()`) so `main.py`
doesn't repeat that boilerplate eight times. It raises the same
`ValueError("No COMPLETE v17 training samples found")` as
`load_training_samples` when the DB is empty — callers that currently
check `Path(...).exists()` before reading catch this instead (see §5.3).

**4.4 — `dataset.db_path` config field.** Add it to `config.yaml` under
`dataset:`, default `"outputs/training_v17.sqlite3"` (same value as the
`SQL_DB_PATH` constant `main.py` hardcodes today). Add `db_path: Path` to
`DatasetConfig` in `config.py`, following the same `base_dir / ...` pattern
`output_path` already uses. `output_path` keeps its field name but its role
narrows to "export path" once Phase 2 lands.

**4.5 — `--persist-sql` narrows, doesn't disappear.** Once `assembler.py`
writes `networks/nodes/scenarios/runs/node_features/node_results` directly
(Phase 2), the "backfill CSV → SQL" half of `--persist-sql`
(`backfill_networks_and_runs`) becomes redundant for the default pipeline.
Keep the flag but narrow it to just `persist_training_run` — a standalone
"train and persist the SQL evidence chain" action, distinct from the CLI's
normal `.joblib` training. `backfill_networks_and_runs` stays in
`csv_backfill.py` for the case of backfilling an *existing* CSV someone
generated before Phase 2 (or via `--export-csv` + a manual re-import) — it
just stops being what `--persist-sql` calls by default.

**4.6 — `validate_dataset` does not get a SQL-specific variant.**
`load_training_samples` already enforces cardinality (node_count vs.
feature/result rows) and the feature contract before returning a frame —
duplicating that check in `validate_dataset` for the loader path is
redundant. `validate_dataset(df, n_nodes, n_factors)` keeps validating the
in-memory frame exactly as today, and only runs on the fresh-assembly path
(where `assemble_dataset` just built `df` from this run's simulations) —
not after `load_training_frame`, whose guarantees are already loader-side.

## 5. Phase 1 — Read cutover

Migrate every `pd.read_csv(config.dataset.output_path)` call site in
`main.py` to `load_training_frame(config.dataset.db_path)`, one at a time,
running the full test suite after each. Order (dependency-free, so this
order is about risk, not necessity — cheapest/most-tested consumers first):

1. `--resilience-curve` — `compute_resilience_curve` already takes a
   DataFrame; only the `main.py` call site changes.
2. `--flood-volume-curve` — same shape as #1.
3. `--factor-comparison` — `generate_factor_comparisons(dataset_path, config, models_dir, output_dir)` in
   `analysis/factor_comparison.py` does its own `pd.read_csv` internally.
   Change its first parameter from `dataset_path: Path` to
   `frame: pd.DataFrame` (caller in `main.py` loads it first) — keeps the
   function itself DB-agnostic and easy to unit-test.
4. `--only-maps` — swap the `pd.read_csv` in `main.py`.
5. `--only-ml` / full pipeline's `use_existing_dataset` branch — swap the
   `pd.read_csv` in `main.py`. (The freshly-assembled path, `assemble_dataset(...)`,
   is untouched until Phase 2.)
6. `--analyze-features` — swap `pd.read_csv(dataset_path)`; replace the
   `dataset_path.exists()` precondition with a try/except around
   `load_training_frame` that reports the same user-facing error message.
7. `--evaluate-shapes` — same precondition swap; only reads the frame to
   build `base_inflows` (a `node_id -> base_inflow_lps` map).
8. `--evaluate-generalization` — same as #7.

**Gate:** CSV writing is untouched in this phase (the pipeline still writes
`dataset_final.csv` every full run) — if reading from SQL diverges from the
CSV for any reason, every one of these commands can be pointed back at
`pd.read_csv` with a one-line revert. Full suite must stay green after each
step.

## 6. Phase 2 — Write cutover

1. `assembler.py::assemble_dataset` gains a write path: after building the
   in-memory `dataset` DataFrame (unchanged), if a `db_path` is supplied,
   open a managed connection, apply migrations, call
   `backfill_networks_and_runs(conn, dataset, inp_path, network_name)`
   (reused from `csv_backfill.py` — no logic changes needed there), commit,
   close. CSV writing (`dataset.to_csv(output_path)`) stays — dual-write
   through the rest of this phase.
2. `main.py`'s full-pipeline branch passes `config.dataset.db_path` into
   `assemble_dataset(...)`.
3. Add `--export-csv` to `main.py`: calls `export_training_samples_csv(conn, config.dataset.output_path)`
   on demand. This is the tool that regenerates `dataset_final.csv` from
   SQL for anything still relying on the file directly (spreadsheets, ad
   hoc scripts, `docs/superpowers/...` historical tooling).
4. **Parity gate (blocking):** a test that runs the full pipeline once,
   captures `dataset_final.csv`, then runs `--export-csv` and diffs the two
   byte-for-byte (or via `pd.testing.assert_frame_equal` after
   `pd.read_csv` on both) — column order, dtypes, and values must match
   exactly. This is the `12.9` "`--export-csv` regenera el CSV idéntico al
   de referencia" checklist item.
5. Only once the parity gate is green: drop the default `dataset.to_csv(...)`
   call from `assemble_dataset` (CSV now only appears via `--export-csv`).
   This is the one irreversible step in the whole migration — everything
   before it is dual-write and revertible.

## 7. Phase 3 — Cleanup

1. Narrow `--persist-sql` per §4.5 (drop the now-redundant
   `backfill_networks_and_runs` call from its handler in `main.py`; keep
   `persist_training_run`).
2. Consider the `csv_backfill.py` → rename suggested in `FLUJO_ACTUAL.md`
   §12.4 ("ya no es backfill de CSV sino la ruta normal de escritura") —
   non-blocking, do it if it doesn't cost an extra review round.
3. Run `pytest -m scale` (`tests/database/test_scale_v17.py`) as the final
   gate — this was already true after Phase 1/2 land, but re-run explicitly
   before declaring done.
4. Update `docs/FLUJO_ACTUAL.md` and `COMANDOS.md` one more time to reflect
   the finished state (mirrors the existing `12.9` checklist).

## 8. Testing strategy

- Unit tests for `load_training_frame` (new): empty DB raises with the
  message callers expect; happy path matches `load_training_samples`.
- `generate_factor_comparisons` signature change: update its existing tests
  to pass a DataFrame fixture instead of a CSV path.
- `assemble_dataset` write path: reuse the fixture-building helpers already
  in `tests/database/test_training_view_v17.py` (`_insert_network`, etc. —
  or better, assert against `backfill_networks_and_runs`'s own existing
  test coverage) rather than duplicating them; add a focused test that
  `assemble_dataset(..., db_path=...)` produces rows queryable via
  `load_training_samples`.
- The Phase 2 parity gate (§6.4) is the highest-value new test in this
  entire spec — it is the thing that makes dropping CSV-by-default safe.
- Full `pytest -q` after every numbered step in Phases 1-2, `pytest -m scale`
  at the end of Phase 3.

## 9. Risk & rollback

- Phases 1 and most of Phase 2 are dual-read/dual-write: CSV keeps being
  produced and nothing destructive happens until §6.5. Any regression found
  before then reverts with a one-line change per call site.
- §6.5 (dropping default CSV write) is gated on the parity test passing,
  which is exactly the condition `FLUJO_ACTUAL.md` §12.9 already names as
  "done." No step after that point is planned to be reversible without
  re-adding the `to_csv` call and re-running the full pipeline.

## 10. Acceptance criteria

Same as `docs/FLUJO_ACTUAL.md` §12.9, restated for this spec's scope:

- [ ] No call site in `main.py` reads `dataset_final.csv` via `pd.read_csv`
      for the training frame (Phase 1).
- [ ] `assemble_dataset` writes `networks/.../node_results` directly
      (Phase 2.1-2.2).
- [ ] `--export-csv` reproduces `dataset_final.csv` byte-for-byte from SQL
      (Phase 2.4 gate).
- [ ] Default pipeline run writes only to `outputs/training_v17.sqlite3`
      (+ `.joblib` + `.json` + `.png`); CSV only appears via `--export-csv`
      (Phase 2.5).
- [ ] `pytest -q` and `pytest -m scale` green (Phase 3.3).
- [ ] `FLUJO_ACTUAL.md` and `COMANDOS.md` updated (Phase 3.4).
