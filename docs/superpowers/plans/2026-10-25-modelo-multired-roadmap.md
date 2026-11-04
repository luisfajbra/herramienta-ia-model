# Cross-Network Readiness Roadmap

This roadmap sequences three separately reviewable implementation plans. Finish each plan's exit gate before starting the next. A later plan may be revised after inspecting network B; do not silently expand the accepted hydraulic domain.

| Order | Plan | Deliverable | Exit gate |
|---|---|---|---|
| 1 | [Contract, identity, and provenance](2026-10-25-identidad-y-contrato-multired.md) | Reproducible v4_21 data and model artifacts with collision-free network identity. | Two synthetic networks sharing node and event names can be prepared and trained without contract or provenance mismatch. |
| 2 | [Hydraulic admission and truth](2026-10-25-admision-hidraulica-verdad.md) | Versioned network admission, input/label policy, and verified eligible SWMM runs. | The inspected B network is classified as supported or rejected with reasons; controlled simulations pass the configured QA policy. |
| 3 | [Spatial evaluation and shared models](2026-10-25-evaluacion-y-modelo-compartido.md) | Frozen A→B transfer experiment, followed by separate A+B known-network evaluation. | Independent folds, tuning and reports pass audit; candidates are reproducible, with no automatic promotion. |

The user confirmed the input for this phase is a hydrograph of flow at each node. Direct rainfall-to-runoff prediction is out of scope. MLP and XGBoost remain candidate families. The second INP is not yet in the project, so no plan assumes its units, boundary conditions, topology, calibration, or initial state.

## Dependency and stop sequence

1. Complete Plan 1 and review its two-network synthetic fixture. If its metadata, network keys, or data selection remain ambiguous, stop before creating a combined database.
2. Obtain network B and its associated event/inflow data. Run the read-only admission inventory in Plan 2. Resolve supported hydraulic conditions, feature availability, initial-state policy, flood threshold, and simulation-quality thresholds with the project owner. If B requires new physical features or a different response variable, update the design document before implementing that change.
3. Complete Plan 2 and review label agreement and SWMM quality per run. Do not use unverified runs to prepare the principal training dataset.
4. Complete the frozen A→B experiment from Plan 3 before any selection decision that uses B labels. Preserve it as the result for A→B.
5. Only after the A→B record is sealed, prepare A+B and run the distinct known-network/event experiments. Do not report this as generalization to a third network.
6. Reserve an actually unseen third network for a future test if the thesis claim is transfer to a network never used in any feature, threshold, architecture, or hyperparameter decision.

## Shared implementation controls

- Work in the approved Optuna worktree. Read its `git status` first; preserve pre-existing user files and changes.
- Each plan task uses TDD: write the regression test, run it to confirm the expected failure, implement the smallest change, run focused tests, then commit the task's own files.
- All generated SQLite databases, changed INPs, SWMM runs, caches, OOF files, and candidate models go under a new experiment-specific directory. Never overwrite the canonical dataset or active models.
- Keep the v17 contract and its existing CLI behavior available. A v4_21 command must select its contract, SQL view and feature table explicitly and reject inconsistent combinations.
- Record the actual INP SHA-256, event-series hashes, SWMM and package versions, units, initialization and closure policy, threshold, continuity value, drainage decision and exclusion reason for each run.
- Do not set engineering tolerances or a dry/steady initial condition by coding convenience. Add required configuration fields and block the experiment until their values and rationale are approved.
- Passing software tests establishes code behavior. It does not establish calibration, hydraulic adequacy, or network generalization; report those as separate gates.

## Definition of project readiness

Call the pipeline **ready to run a controlled pilot** when Plans 1 and 2 pass their exit gates and the owner's B-specific decisions are recorded. Call the two-network model **evaluated on a second network** only after frozen A→B predictions and hydraulic QA are complete. Call the approach **validated for unseen networks** only with an untouched network-level test that was not used to choose features, labels, thresholds, tuning, architecture or stopping rules.


## Progress note (2026-11-03)

The frozen Chico Sur-to-Chico Norte transfer report (2026-10-30) and a separate novel-synthetic-hydrograph evaluation (2026-11-02) now contain verified external predictions, metrics, maps, and presentation artifacts for the frozen MLP and XGBoost candidates. The 2026-11-02 evaluation is still a dedicated experiment workflow: running the ordinary project commands does not reproduce or surface it, it did not retrain/promote candidates, and it did not write labels to the canonical training database. This does not close hydraulic admission or the A+B nested-training plan. The next gate remains the documented Chico Norte hydraulic review; keep the approved 5% continuity rule provisional and retain InitFlow sensitivity separately. After that gate, provide an explicit opt-in entry point for reproducing frozen transfer evaluation, then proceed to A+B nested network/event evaluation only under the plan's admission and split rules. Reserve a third network for any claim about performance on an unseen network. See `docs/superpowers/reports/2026-11-02-evaluacion-hidrogramas-nuevos-chico-norte.md` for the detailed status and sequence.

The follow-up hydraulic review on 2026-11-03 verified both frozen report collections, ran nine isolated sensitivity simulations on three selected cases, and repeated six runs to inspect native routing steps in two windows. It found local time-series differences despite small aggregate flood-volume changes and reproduced a shared-grid incompatibility affecting closure of some historical events. The QA policy is unchanged; the closure fix, broader numerical sensitivity, and justification of the initial state remain open. See [the sensitivity report](../reports/2026-11-03-revision-hidraulica-sensibilidad-chico-norte.md) and its linked evidence before treating Chico Norte as admitted for shared-model training.

The owner requested a broader, thesis-ready hydraulic sensitivity study and selected routing-step and initial-state sensitivity for its first stage. The [exploratory protocol](../reports/2026-11-03-protocolo-sensibilidad-ampliada.md) covers all ten previously evaluated novel scenarios, while the [implementation plan](2026-11-03-sensibilidad-hidraulica.md) specifies the opt-in workflow, closure and time-reader prerequisites, traceability, QA, reports and preservation. Executing this plan and admitting Chico Norte remain separate open gates; no physical-parameter ranges or new automatic QA exclusions have been approved.

The [expanded exploratory report](../reports/2026-11-03-sensibilidad-hidraulica-ampliada.md) now records ten scenarios, thirty variants, verified inputs and outputs, QA, class changes and the comparison limits. This does not complete the integration plan, fix closure/time reading, or resolve network admission. Full SWMM OUT files remain local in Git-ignored storage and need a separately verified backup.

The [frozen-evaluation integration](../reports/2026-11-03-integracion-evaluacion-congelada.md) adds an opt-in `main.py` entry point for the documented ten Chico Norte scenarios. A real rerun reproduced the prior metrics and generated separate maps and parity figures. Arbitrary network/event selection, presentation figures for partially excluded scenarios, hydraulic admission and shared-network training remain open.
