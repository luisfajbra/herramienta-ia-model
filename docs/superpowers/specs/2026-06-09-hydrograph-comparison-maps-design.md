# Hydrograph Comparison Maps Design

## Goal

Generate one SWMM flood map and one ML flood map for every hydrograph validation
scenario, directly inside the selected validation output directory.

## Design

Add a comparison-map helper to the existing model comparison visualization
module. It will convert the canonical comparison DataFrame into the
`plot_flood_map` contract twice:

- SWMM uses `inunda_swmm` and `vol_swmm_m3`.
- ML uses `inunda_pred` and `vol_pred_m3`.

Both maps use the same maximum volume from both columns, preserving direct color
and marker-size comparability. Files are named
`flood_map_swmm_<scenario>.png` and `flood_map_ml_<scenario>.png` and are saved
directly in the validation output directory.

The batch validation coordinator will call the helper for future runs. Existing
`comparison_summary.csv` files can also be grouped by scenario and passed to the
same helper without rerunning SWMM.
