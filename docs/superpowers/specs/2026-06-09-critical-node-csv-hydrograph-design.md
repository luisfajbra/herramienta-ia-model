# Critical-Node CSV Hydrograph Design

## Goal

Generate one inflow hydrograph per evaluated CSV scenario, matching the
existing original-scenario hydrograph style.

## Selection Rule

For each scenario, select the node whose input series contains the highest
single `value_lps`. This is the same peak-inflow rule used by
`plot_hydrograph`.

## Output

Write `hydrograph_<scenario_id>.png` directly in the validation `out_dir`.
The chart uses:

- `Time (min)` on the x-axis.
- `Flow (L/s)` on the y-axis.
- A blue line and translucent fill.
- An English title with the visible node suffix removed.

## Integration

Add a renderer that accepts an already validated `HydrographScenario`.
Call it once per scenario from `run_batch_validation`. Existing validation
runs can be rendered from their source CSV files without rerunning SWMM.

## Testing

Verify peak-node selection, conversion from hours to minutes, visible node
formatting, root output location, and one call per batch scenario.
