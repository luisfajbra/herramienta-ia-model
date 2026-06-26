# English Plot Labels Design

## Goal

Make every generated chart display titles, axes, legends, color bars, annotations,
and node labels in English.

## Design

Add a small visualization-label helper with two responsibilities:

- Convert internal ML feature names into descriptive English labels for feature
  importance charts.
- Convert node IDs ending in `I` or `O` immediately after a number into their
  numeric display form, for example `123I` and `123O` become `123`.

The helper changes display text only. DataFrame values, joins, model inputs,
sorting, and SWMM identifiers remain unchanged.

All Matplotlib generators and their title-producing callers will use English
text. Existing output paths and file names remain unchanged.

## Feature Labels

The current tabular model features will be presented as:

- `elev_fondo`: Invert Elevation
- `prof_max`: Maximum Depth
- `n_tuberias_in`: Inlet Pipe Count
- `n_tuberias_out`: Outlet Pipe Count
- `diam_max_in`: Maximum Inlet Diameter
- `diam_max_out`: Maximum Outlet Diameter
- `pendiente_max_in`: Maximum Inlet Slope
- `pendiente_out`: Outlet Slope
- `base_inflow_lps`: Base Inflow
- `dist_outfall_m`: Distance to Outfall
- `n_nodos_aguas_arriba`: Upstream Node Count
- `q_pico_acum_base`: Base Accumulated Peak Flow
- `upstream_capacity_lps`: Upstream Capacity
- `factor_mult`: Flow Multiplier
- `q_pico_nodo`: Node Peak Inflow
- `q_pico_acum_escalado`: Scaled Accumulated Peak Flow

Unknown feature names will fall back to title-cased words split on underscores.

## Testing

Unit tests will cover both label helpers. Plot-level tests will inspect generated
Matplotlib text for representative charts, including feature importance,
hydrograph, flood map, resilience, and node profiles. The complete test suite
will then guard against integration regressions.
