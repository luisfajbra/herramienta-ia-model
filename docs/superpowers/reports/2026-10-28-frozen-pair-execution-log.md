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

Pending. The next run must finish both batches and produce complete A and B manifests before either dataset is staged. The interrupted roots are retained for audit and comparison only.
