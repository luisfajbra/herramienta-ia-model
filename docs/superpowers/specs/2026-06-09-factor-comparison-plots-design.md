# Factor Comparison Plots Design

## Goal

Generate per-factor node-volume and parity plots for the original
multiplier-based workflow.

## Inputs

- SWMM node volumes from `dataset_final.csv`.
- XGBoost node predictions from the existing trained models.
- Every `factor_mult` present in the dataset.

## Outputs

Create `outputs/factor_comparisons` and write exactly two English-language
plots per factor:

- `volume_by_node_factor_<factor>.png`, with SWMM in blue and XGBoost in
  orange.
- `parity_factor_<factor>.png`, with XGBoost volume versus SWMM volume and a
  `y = x` reference.

Node labels omit numeric `C`, `I`, and `O` suffixes.

## Interface

Add `python main.py --factor-comparisons`. The command reuses existing SWMM
results and does not rerun SWMM.

## Testing

Test node merging, plot names, colors, English labels, factor iteration, and
the total of two files per factor.
