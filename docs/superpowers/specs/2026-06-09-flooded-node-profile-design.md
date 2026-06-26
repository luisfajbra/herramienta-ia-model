# Flooded Node Profile Design

## Goal

Generate an additional `V(m3) vs Node ID` plot for each factor containing only
nodes whose reference SWMM flood volume is greater than zero.

## Behavior

- Filter rows with `vol_swmm_m3 > 0`.
- Keep both SWMM and XGBoost volume curves for the retained nodes.
- Save the additional plot as
  `volume_by_node_flooded_swmm_factor_X.XX.png`.
- Keep all existing profile and parity PNG files unchanged.
- Skip the additional PNG when a factor has no SWMM-flooded nodes.

## Verification

Automated tests verify the strict SWMM filter, labels, output filename, and
integration into per-factor generation. Generated files are also checked for
count, non-zero size, and visual readability.
