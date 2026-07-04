# Training Procedure Explained

This document explains how the project trains its machine learning models to
approximate SWMM hydraulic simulations. It is written in plain English and
describes the meaning of the training process rather than the internal code
names.

The main idea is simple: SWMM is treated as the trusted hydraulic simulator.
The project runs SWMM many times under controlled storm conditions, extracts
what happened at every junction, and uses those examples to train faster
machine learning models. After training, the machine learning models can
estimate flooding behavior without having to run SWMM again for every new
scenario.

## 1. What The Training Is Trying To Learn

The current production training procedure learns two related tasks:

1. **Flood occurrence**: whether a junction floods or does not flood.
2. **Flood volume**: how much flood volume is produced when a junction floods.

These are trained as two separate models because they answer different
questions. The first model is a decision model: "Does this junction flood?"
The second model is a magnitude model: "If flooding happens, how large is the
flood volume?"

This separation is important. Most junctions in many hydraulic scenarios do
not flood. If a single model tried to predict volume for every junction, it
would spend much of its effort learning that many cases are zero. By separating
the problem, the first model learns the boundary between safe and flooded
conditions, while the second model focuses only on the cases where flood volume
is meaningful.

## 2. Where The Training Data Comes From

The training data is generated from a SWMM network file. In the current
configuration, the network is the Chico Sur model, and the main input file is:

`data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp`

The pipeline does not invent labels manually. Instead, it creates hydraulic
scenarios, runs SWMM, reads the SWMM report, and uses the SWMM results as
ground truth.

Each training example represents:

**one junction in the drainage network under one simulated hydraulic scenario.**

For example, if the network has 160 junctions and the project simulates 175
storm scenarios, the resulting dataset has about 28,000 rows. Each row says:

- what the junction is like physically,
- what hydraulic demand was applied in that scenario,
- whether SWMM says the junction flooded,
- and how much flood volume SWMM reported.

## 3. How The Hydraulic Scenarios Are Created

The current configuration defines a sweep of inflow multipliers from 0.2 to
5.0, stepping by 0.2. That means the model is trained on a range from very mild
flow conditions to severe flow conditions.

For each multiplier, the base inflow hydrograph is scaled and SWMM is run. The
configuration also points to a folder of hydrograph shapes, so the project can
train not only on one storm shape but also on several temporal patterns. These
shapes describe different rainfall or inflow behaviors, such as shorter,
longer, more common, or more extreme storm profiles.

For every combination of storm shape and intensity level, the pipeline creates
a scenario-specific SWMM run. SWMM then computes the real hydraulic response of
the network.

This gives the machine learning model exposure to several types of variation:

- weak, moderate, and severe inflow intensity,
- different storm durations,
- different times at which the storm reaches its peak,
- different local inflow peaks at each junction,
- and different accumulated upstream hydraulic loads.

## 4. What Is Extracted Before Training

The training dataset combines three kinds of information.

### Static Network Information

Static information describes the physical and topological properties of each
junction. It does not change from scenario to scenario. It includes plain
engineering descriptors such as:

- bottom elevation of the junction,
- maximum physical depth of the junction,
- number of incoming pipes,
- number of outgoing pipes,
- largest incoming pipe diameter,
- largest outgoing pipe diameter,
- steepest incoming pipe slope,
- outgoing pipe slope,
- base inflow at the junction,
- distance to the outfall,
- number of upstream junctions,
- accumulated base inflow from upstream contributing nodes,
- and estimated upstream conveyance capacity.

These values help the model understand the local vulnerability of a junction.
For example, a shallow junction with high upstream load and limited downstream
capacity is hydraulically different from a deep junction near the outfall with
large outgoing capacity.

### Dynamic Scenario Information

Dynamic information changes from one scenario to another. It describes the
hydraulic demand imposed by a specific storm condition. It includes:

- peak inflow applied at the junction,
- accumulated peak inflow arriving from the junction and its upstream area,
- total storm duration,
- and time from the beginning of the event to the peak.

The model deliberately uses hydraulic demand values rather than relying on a
generic "scenario factor" as a direct model input. This matters because real
hydrograph validation scenarios may not have one clean global multiplier. Two
nodes can receive different hydrograph shapes or different peak values, so the
model needs physically interpretable inputs that still make sense outside the
original training sweep.

### SWMM-Derived Labels

After every SWMM run, the project reads the SWMM report and extracts flooding
results for every junction. The two labels used for training are:

- whether the junction flooded,
- and total flood volume in cubic meters.

If a junction does not appear in SWMM's flooding summary, the project treats
its flood volume as zero. A junction is considered flooded when its flood
volume is at least the configured threshold. In the current configuration, that
threshold is 1.0 cubic meter.

This threshold avoids treating tiny numerical traces as meaningful flooding.
It also keeps training and validation consistent, because the same flood
labeling rule is used when evaluating against new SWMM simulations.

## 5. Dataset Assembly

Once the static information, scenario information, and SWMM labels are ready,
the project joins them into a single table.

The assembly logic works like this:

1. Start with one row per junction using the static network information.
2. For a specific storm scenario, attach that scenario's hydraulic demand
   values to every junction.
3. Attach SWMM's flood occurrence and flood volume result for the same
   junction.
4. Repeat this for every simulated scenario.
5. Stack all scenario tables into one training dataset.

The final dataset is saved as:

`data/training/dataset_final.csv`

This file is the main training source for the current tabular models.

## 6. Dataset Validation Before Training

Before fitting the models, the pipeline performs sanity checks on the dataset.
It verifies that:

- the flood occurrence label has no missing values,
- the flood volume label has no missing values,
- flood volume is never negative,
- the flood occurrence label is always either "no flood" or "flood",
- the number of rows matches the expected number of junctions multiplied by
  the number of simulated scenarios,
- and at least some rows contain flooding.

If no junction floods anywhere in the training set, the model cannot learn the
flooding task, so training stops. If the percentage of flooded rows is very
low, the pipeline warns that the scenario range may not be severe enough.

## 7. The Final Production Models

The current production pipeline trains two final models and saves them under:

`outputs/models/`

The saved files are:

- `classifier.joblib`: the flood occurrence model,
- `regressor.joblib`: the flood volume model,
- `training_inp_hash.txt`: a fingerprint of the SWMM network file used for
  training.

The fingerprint is important because predictions are only trustworthy for the
same network geometry and topology. During hydrograph validation, the project
checks that the SWMM input file being evaluated matches the network used for
training.

## 8. Flood Occurrence Model

The flood occurrence model is trained on all rows in the dataset.

Its job is to learn the relationship between:

- physical properties of the junction,
- upstream network structure,
- hydraulic demand during the scenario,
- storm duration and peak timing,
- and the final yes/no flooding outcome reported by SWMM.

The configured default algorithm is gradient boosted decision trees through
XGBoost. The current configuration uses:

- 200 boosted trees,
- maximum tree depth of 6,
- learning rate of 0.05,
- 80 percent row subsampling per boosting step,
- and automatic class imbalance weighting.

Class imbalance weighting matters because flooded and non-flooded examples are
usually not equally common. If non-flooded examples dominate, a naive model
could appear accurate by predicting "no flood" too often. The automatic weight
increases the penalty for mistakes on the rarer flooded class.

Missing input values are handled with median imputation inside the model
pipeline. This is useful for network features where a value may be naturally
absent, such as an upstream pipe diameter for a headwater junction.

## 9. Flood Volume Model

The flood volume model is trained only on rows where SWMM reported flooding.

This is intentional. The volume model is not asked to learn the difference
between zero and non-zero flooding. That decision belongs to the occurrence
model. The volume model learns the magnitude of flooding once the system is in
a flooded condition.

Before training, flood volume is transformed with a logarithmic transform. In
plain terms, very large flood volumes are compressed during training so they do
not dominate the learning process. This often helps when target values span a
wide range, which is common in flood volume prediction.

At prediction time, the project reverses the transform and converts the model
output back to cubic meters. Negative reconstructed volumes are clipped to
zero, because negative flood volume has no physical meaning.

The default volume model is also XGBoost, with the same broad tree settings as
the occurrence model.

## 10. How A Final Prediction Is Produced

The two trained models are used together.

For each junction in a new scenario:

1. The project computes the same physical and hydraulic input descriptors used
   during training.
2. The occurrence model predicts whether the junction floods.
3. If the occurrence model predicts no flooding, the reported flood volume is
   treated as zero for the end-to-end system.
4. If the occurrence model predicts flooding, the volume model estimates the
   flood volume.

This produces both a flood map and a network-wide volume estimate.

The project also supports arbitrary hydrograph validation from CSV files. In
that mode, it reads time series for inflow at nodes, writes scenario-specific
SWMM input files, runs SWMM for comparison, computes the model inputs from the
same hydrographs, and compares machine learning predictions against SWMM.

## 11. Evaluation Procedure

Training the final saved models is not the same thing as evaluating model
quality. For evaluation, the project retrains temporary models across grouped
validation splits and measures how well they generalize.

The configured evaluation methods are:

- **Leave-one-scenario-level-out validation**: one intensity level is held out
  at a time, and the model is trained on the remaining levels.
- **Five-fold grouped validation**: scenario groups are split into five folds,
  keeping all rows from the same scenario level together.

Grouping is critical. Rows from the same scenario are highly related because
they come from the same SWMM run. If some junctions from a scenario were placed
in training and other junctions from the same scenario were placed in testing,
the evaluation would be too optimistic. Grouped validation asks a harder and
more honest question: can the model generalize to a hydraulic condition it did
not see during training?

The evaluation is reported at three levels.

### Occurrence Model Alone

This measures the yes/no flooding model independently. The main metrics are:

- precision: when the model predicts flooding, how often it is correct,
- recall: of all truly flooded nodes, how many it finds,
- F1 score: balance between precision and recall,
- and area under the ROC curve when both classes are present.

### Volume Model With Oracle Routing

This evaluates the volume model only on junctions that truly flooded according
to SWMM. It uses the real flooding labels to decide which rows should receive a
volume prediction.

This is an optimistic upper bound. It answers: "If the occurrence decision were
perfect, how good would the volume estimates be?"

The main metrics include:

- Nash-Sutcliffe efficiency,
- logarithmic Nash-Sutcliffe efficiency,
- root mean squared error,
- mean absolute error,
- and coefficient of determination.

### End-To-End System

This evaluates the complete operational workflow. The occurrence model first
decides which junctions flood, and only those predicted flooded junctions are
sent to the volume model.

This is the most realistic evaluation because it includes both possible error
types:

- missing a truly flooded junction,
- and assigning volume to a junction that SWMM says did not flood.

The end-to-end metrics include:

- percentage of correctly classified junctions,
- volume error across all junctions,
- total predicted flood volume,
- and total SWMM flood volume.

Evaluation files are saved under:

`outputs/metrics/`

## 12. Feature Importance And Interpretability

After training the final models, the pipeline generates feature importance
plots. These help explain which physical or hydraulic descriptors were most
useful to the tree models.

There is also a feature analysis workflow that can create:

- correlation heatmaps,
- an ablation comparison with and without storm shape descriptors,
- and SHAP-based explanations for the occurrence and volume models.

The purpose is to support technical interpretation, not just prediction. In a
hydraulic modeling project, it is not enough for a model to produce numbers;
the model should also be checked for whether it is relying on physically
reasonable signals.

## 13. Why The Scenario Multiplier Is Not A Model Input

The dataset keeps the scenario intensity multiplier as metadata, but the
current production model does not use it as a direct input.

This is a deliberate modeling choice. A single multiplier only makes sense for
the synthetic training sweep where every inflow is scaled uniformly. In real
validation hydrographs, different nodes may have different time series and
different peak values. There may be no single multiplier that honestly
describes the entire scenario.

Instead, the model receives interpretable hydraulic quantities:

- the actual peak inflow at the junction,
- the accumulated upstream peak inflow,
- storm duration,
- and time to peak.

These quantities are meaningful both for synthetic training scenarios and for
arbitrary validation hydrographs.

## 14. Hydrograph Validation After Training

The trained models can be tested against external hydrograph CSV scenarios.
This validation process is more demanding than the original multiplier sweep.

For each hydrograph scenario:

1. The CSV is loaded and checked.
2. A scenario-specific SWMM input file is written.
3. A drain-down period is appended so post-storm flooding volume is not
   truncated too early.
4. SWMM is run to obtain the reference answer.
5. The machine learning model computes the same kind of input descriptors from
   the hydrograph.
6. Model predictions are compared against SWMM at both node level and
   scenario-total level.

The validation output includes:

- per-node comparison tables,
- per-scenario total flood volume comparisons,
- timing comparisons between SWMM and machine learning,
- classification and volume metrics per scenario,
- and plots for parity, node profiles, flood maps, and hydrographs.

This is the main way to test whether the trained model is useful beyond the
exact synthetic scenarios used to build the training dataset.

## 15. Temporal Surrogate Training

In addition to the current production tabular models, the repository contains
an experimental temporal surrogate training path.

The temporal surrogate is designed to learn from time series, not only from
summary descriptors. Instead of giving the model a few engineered quantities
such as peak inflow and storm duration, it gives the model a resampled sequence
of inflow values over time, together with static junction descriptors.

The temporal surrogate creates one sample for each junction in each SWMM run.
For every sample, it stores:

- a time sequence of total inflow and local lateral inflow,
- static junction and network descriptors,
- a flood occurrence label,
- and a peak flooding rate target.

Because storm events can have different lengths, the sequences are padded to a
common length before training.

Two neural architectures are available:

- a one-dimensional convolutional model, which looks for temporal patterns in
  the inflow sequence,
- and a recurrent model, which processes the sequence through an LSTM.

Both architectures use two branches:

1. a temporal branch that processes the inflow sequence,
2. a static branch that processes junction and network descriptors.

The two branches are fused, and the model has two output heads:

- one output for flood occurrence,
- and one output for flood magnitude.

The temporal surrogate uses grouped cross-validation by SWMM run. That keeps
all junctions from the same simulation together in either training or
validation, reducing leakage between related examples.

The sequence values and static descriptors are standardized using only the
training fold during cross-validation. After validation, a final model is
trained on all available groups and saved with its standardization objects and
a manifest that records the training metadata.

The temporal path is useful for experiments where the actual shape of the
hydrograph matters more than summary descriptors. However, the README indicates
that the current main workflow is still the CSV-based tabular pipeline with
saved joblib models.

## 16. What Gets Saved

The main tabular training workflow saves:

- the final occurrence model,
- the final volume model,
- a network fingerprint for compatibility checking,
- evaluation metrics,
- feature importance figures,
- and flood maps for selected scenario levels.

The temporal surrogate workflow saves:

- neural network weights,
- standardization objects for temporal and static inputs,
- per-fold validation metrics,
- and a manifest describing the temporal features, static features, training
  groups, model type, and target transformation.

## 17. Practical Interpretation

The training procedure can be understood as a supervised learning replacement
for repeated SWMM simulations.

SWMM is used first because it is the trusted physics-based engine. Machine
learning is used afterward because it is much faster once trained. The quality
of the model depends on whether the training scenarios cover the kinds of
hydraulic conditions that will be requested later.

The model should be most reliable when:

- the same drainage network is used,
- the new scenario lies within the range of training intensities,
- the hydrograph shapes are similar to those seen during training,
- the flood threshold is interpreted consistently,
- and the model inputs are computed in the same way during training and
  inference.

The model is less reliable when:

- the network geometry changes,
- inflows are far outside the training range,
- a new storm has a shape not represented in training,
- the hydraulic behavior depends on effects not captured in the input
  descriptors,
- or the SWMM reference runs themselves have continuity problems.

That is why the pipeline includes network hash checks, extrapolation warnings,
continuity warnings, grouped validation, and direct SWMM-versus-model
hydrograph validation.

## 18. Short End-To-End Summary

The complete training story is:

1. Read the SWMM network.
2. Extract physical and topological properties for every junction.
3. Generate many storm scenarios by changing intensity and hydrograph shape.
4. Run SWMM for every scenario.
5. Read SWMM flood volume results from the report files.
6. Label each junction as flooded or not flooded using the configured volume
   threshold.
7. Assemble one row per junction per scenario.
8. Validate that the dataset is internally consistent.
9. Train a flood occurrence model on all rows.
10. Train a flood volume model only on flooded rows, using a logarithmic target
    transform.
11. Save the final models and the network fingerprint.
12. Evaluate with grouped validation so related rows from the same scenario are
    not split unrealistically.
13. Generate metrics, feature importance plots, maps, and optional hydrograph
    validation outputs.

In one sentence: the project teaches fast models to imitate SWMM by showing
them many SWMM-generated examples of how each junction behaves under different
storm intensities and shapes.
