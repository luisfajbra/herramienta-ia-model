# Temporal CNN Scaffold

This folder is reserved for the future 1D CNN workflow for hydrograph-based
failure prediction.

Current tabular models remain in `swmm_resilience/ml/train.py`. The temporal
workflow will be complementary, not a replacement.

## Intended Questions

- Given a recent hydrograph window, will a node fail soon?
- How many minutes before the hydrograph peak can we detect risk?
- Which temporal shape patterns are associated with failure?

## Planned Data Shape

The future dataset should be built from time series per node and per run:

```text
run_id
node_id
time_min
inflow_lps
depth_m
max_depth_ratio
flooding_lps or flooded
static node/link features
```

The CNN input will likely be a rolling window:

```text
X shape: [samples, timesteps, features]
y shape: [samples]
```

Example target:

```text
failure_within_horizon = 1 if the node floods within the next N minutes
```

## Planned Files

- `schemas.py`: shared dataclasses for temporal configuration.
- `dataset.py`: future rolling-window dataset builder.
- `train_cnn.py`: future CNN training entrypoint.
- `predict.py`: future temporal prediction entrypoint.

## Not Implemented Yet

No deep learning framework is required yet. We will decide later whether to use
PyTorch, TensorFlow/Keras, or another lightweight approach after the temporal
dataset is defined and validated.
