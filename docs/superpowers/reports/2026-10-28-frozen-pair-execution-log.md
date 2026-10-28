# Frozen A/B SWMM rerun execution log

## Frozen run decisions

- Execute the pre-registered 1,000 A and 975 B scenarios with the existing synthetic nodal hydrographs.
- Round source timestamps to the nearest minute, as approved, and preserve nodal peaks.
- Keep B frozen as the transfer test; do not use B targets for fitting or tuning.
- Keep the 5% continuity limit and configured drainage/coverage tolerances provisional. Record ineligible cases and exclude them from training.
- Preserve every run root; do not overwrite partial generations.

## Execution interruptions and recovery

The first execution root, `outputs/experiments/frozen_pair_rerun_20261028/20261028T184001Z`, stopped after 464 A scenarios because Windows denied the atomic checkpoint replacement. Its partial artifacts remain preserved and are not eligible for staging.

The second root, `outputs/experiments/frozen_pair_rerun_20261028/20261028T191042Z`, stopped after the checkpoint JSON had reached 487 completed scenarios while its `.tmp` held 488. No simulator process remained, both files were readable, neither was read-only, and promoting the temporary checkpoint with `os.replace` then succeeded. This confirms a transient Windows file-lock/sharing condition at checkpoint publication rather than a SWMM or hydraulic QA error.

The runner now retries checkpoint replacement after bounded delays when it receives `PermissionError`. A regression test injects two transient permission failures and verifies the third atomic replacement succeeds. The focused runner suite passes (9 tests). The 488-scenario root is preserved as an interrupted generation; a new, cleanly anchored pair run is required for the authoritative A/B manifests.

## Authoritative execution

The current authoritative root is `outputs/experiments/frozen_pair_rerun_20261028/20261028T194035Z`, anchored before simulation at commit `4efc53a`. A completed all 1,000 scenarios with no technical errors. Recalculated QA currently excludes 134 runs (100 not drained under the frozen retry policy, 35 above the provisional 5% continuity limit, and one with both reasons); all 1,000 have finite coverage. The remaining 866 are provisionally eligible for A training, subject to successful manifest staging.

B remains in progress and is still sealed from model selection. At the last checkpoint, 63 of 975 scenarios had completed; 26 were hydraulically ineligible and none had a technical error. These counts are operational checkpoints, not final B results.

## Staging corrections found by the real A manifest

The first A staging attempt exposed a representational difference in the INP parser/writer: an empty optional `[TAGS]` section can be present in the source and omitted from the effective INP. Since the parsed section is empty, it carries no hydraulic objects or values. The hydraulic descriptor comparison now ignores only empty sections; a regression test covers this exact equivalence.

A second attempt exposed the distinct frozen event contracts. Synthetic shape events contain every source `FLOW` node with positive base peak. The `base` event preserves every source `FLOW` node, including 13 whose peak is zero. The verifier and feature constructor now accept exactly either set and reject any other node set. A regression fixture exercises a base event with a zero-peak `FLOW` node. The two staging suites pass: 43 tests.

The first two failed staging databases and their diagnostic artifacts are preserved and must not be reused. A third, fresh SQLite staging is underway with unique v3 manifest names. Until it completes and its final manifest is verified, A is not yet a prepared training dataset. No A model fitting or hyperparameter search has started; B targets have not been read.
