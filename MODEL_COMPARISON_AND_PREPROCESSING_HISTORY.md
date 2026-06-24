# Model Comparison and Preprocessing History

This document explains the earlier model-selection work we did before settling
on the current XGBoost-based architecture. It focuses on the modeling process,
the preprocessing alternatives we evaluated, the validation logic, and why
XGBoost became the preferred choice. It intentionally does **not** list the
individual input variables used by the models.

## 1. Why We Tested Multiple Models

At the beginning of the ML stage, the goal was not to assume that one algorithm
would be best. The hydraulic problem is nonlinear, noisy, and structured by
network behavior, so we needed empirical evidence.

We evaluated several families of tabular machine-learning models:

- linear regularized regressors,
- kernel-based support vector models,
- logistic classification,
- gradient-boosted decision trees through XGBoost.

The comparison had two purposes:

1. Find a model that could estimate flood volume accurately.
2. Find a model that could classify flooded versus non-flooded nodes reliably.

That distinction mattered because the project has two related but different
prediction tasks. One task asks whether flooding occurs; the other asks how
severe flooding is when it occurs. A model can be good at one and weaker at the
other, so we evaluated both tasks separately.

## 2. Model Families We Tried

### 2.1 Regression Models

For the flood-volume prediction task, we compared:

- Ridge regression.
- Lasso regression.
- Support Vector Regression with an RBF kernel.
- XGBoost regression.

The linear models were useful baselines. They tested whether a relatively
simple weighted combination of the engineered descriptors could explain the
hydraulic response. Lasso also tested whether sparsity and coefficient
shrinkage could improve generalization.

The support vector model tested whether a nonlinear kernel method could capture
curved relationships in the data without explicitly using tree ensembles.

XGBoost tested a more flexible nonlinear approach based on boosted decision
trees. This was especially relevant because hydraulic behavior often changes by
thresholds and interactions: a node may behave normally until the combined
scenario load and local capacity cross a critical point, after which flooding
increases sharply.

### 2.2 Classification Models

For the flooded/non-flooded classification task, we compared:

- Logistic regression.
- Support Vector Classification with an RBF kernel.
- XGBoost classification.

Logistic regression gave a simple probabilistic linear baseline. The support
vector classifier gave a nonlinear margin-based baseline. XGBoost gave the
tree-boosting alternative, better suited to interactions, thresholds, and
heterogeneous behavior across the network.

## 3. Preprocessing Methods We Evaluated

The comparison was not just a model shootout. We also tested preprocessing
choices because different algorithms react differently to scaling,
missingness, redundancy, and dimensionality.

### 3.1 Numeric Feature Filtering

The historical comparison path selected a clean numeric modeling matrix from
the exported dataset. Non-numeric metadata, identifiers, and result columns
that would leak the answer were excluded before model fitting.

The reason was methodological: the model should only receive information that
would be available at prediction time. Any output produced by a SWMM run cannot
be used as an input for a surrogate that is supposed to avoid running SWMM.

### 3.2 Missing-Value Imputation

The model pipelines used median imputation.

This was important because some hydraulic descriptors are naturally absent for
some nodes. For example, boundary or headwater conditions can produce missing
values in descriptors that only exist when a certain local structure is
present. Dropping those rows would waste data and bias the dataset toward only
well-connected or structurally typical nodes.

Median imputation was a conservative choice:

- it is robust to outliers,
- it works well with skewed hydraulic quantities,
- it keeps the preprocessing inside the training pipeline,
- it avoids leaking test-set statistics into training.

### 3.3 Scaling

We evaluated pipelines with standard scaling.

Scaling was especially important for linear models, support vector models, and
PCA. Those methods are sensitive to the relative magnitude of each input
dimension. Without scaling, a large-unit quantity can dominate a distance,
margin, coefficient, or principal component simply because of its scale.

For tree-based models like XGBoost, scaling is not mathematically necessary in
the same way. Decision trees split by thresholds and are mostly insensitive to
monotonic rescaling. However, because the comparison path used a shared
preprocessing framework and PCA was enabled, scaling still entered the
preprocessing stack used in those experiments.

### 3.4 PCA Dimensionality Reduction

We also evaluated a compact PCA-based feature space.

The historical comparison used five principal components. The PCA analysis
artifact reports:

```text
rows: 4991
feature_count: 16
pca_components: 5
cumulative_variance_percent: 84.0978
```

This means the original numeric information was compressed into five
orthogonal components that retained about 84.1% of the variance.

The motivation for PCA was technical:

- reduce redundancy among correlated hydraulic descriptors,
- make model comparison more stable on a limited dataset,
- reduce dimensionality before fitting scale-sensitive models,
- create a compact baseline for comparing algorithms fairly.

PCA was not treated as a final claim that the original descriptors were
unimportant. It was a preprocessing experiment: a way to test whether a reduced
mathematical representation could improve stability and generalization.

### 3.5 Grouped Splitting

The comparison used grouped train/test splitting rather than naive row-level
random splitting.

This was one of the most important methodological choices. A single simulation
run produces many rows. If rows from the same run were randomly split between
training and testing, the test set would be too similar to the training set and
metrics would look artificially strong.

Instead, full groups were kept together. The held-out set contained complete
groups that the model had not seen during fitting. Cross-validation followed
the same principle.

This made the evaluation stricter and more honest.

## 4. Evaluation Methodology

The historical model-comparison path evaluated models on:

- a grouped hold-out test split,
- grouped cross-validation on the training split,
- scenario-specific breakdowns when scenario labels existed.

For regression, the comparison reported:

- MAE,
- RMSE,
- R2,
- cross-validated MAE,
- cross-validated RMSE,
- cross-validated R2.

For classification, it reported:

- accuracy,
- precision,
- recall,
- F1,
- cross-validated accuracy,
- cross-validated precision,
- cross-validated recall,
- cross-validated F1.

This combination gave a broader view than a single metric. In hydraulic
prediction, one number is rarely enough:

- MAE shows typical absolute error.
- RMSE penalizes large misses more heavily.
- R2 shows explained variance.
- Precision shows how reliable positive flood predictions are.
- Recall shows how many real flooded cases are captured.
- F1 balances precision and recall.
- Cross-validation checks whether the result is stable across grouped splits.

## 5. Regression Results

The flood-volume regression comparison produced the following results:

| Model | MAE | RMSE | R2 | CV MAE | CV RMSE | CV R2 |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 7.739 | 19.528 | 0.958 | 8.757 | 17.078 | 0.980 |
| SVR RBF | 26.080 | 72.093 | 0.423 | 46.696 | 107.220 | 0.356 |
| Ridge | 51.465 | 85.686 | 0.184 | 73.678 | 115.025 | 0.251 |
| Lasso | 51.469 | 85.688 | 0.184 | 73.682 | 115.026 | 0.251 |

XGBoost was the clear winner for regression.

It had:

- the lowest MAE,
- the lowest RMSE,
- the highest R2,
- the best cross-validated MAE,
- the best cross-validated RMSE,
- the best cross-validated R2.

The gap was not small. XGBoost reduced the test MAE by roughly 70% compared
with the support vector regressor and by roughly 85% compared with the linear
models. Its R2 was also dramatically higher, meaning it captured most of the
variance in the flood-volume response while the alternatives left much more of
the hydraulic behavior unexplained.

Technically, this makes sense. Flood volume is not a purely linear response.
Once hydraulic capacity is exceeded, volume can rise quickly, and the response
depends on combinations of scenario intensity, node condition, and network
position. Boosted trees are well suited to that kind of nonlinear,
interaction-heavy structure.

## 6. Classification Results

The flooded/non-flooded classification comparison produced:

| Model | Accuracy | Precision | Recall | F1 | CV Accuracy | CV Precision | CV Recall | CV F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost Classifier | 0.925 | 0.991 | 0.795 | 0.882 | 0.973 | 0.990 | 0.951 | 0.970 |
| SVC RBF | 0.893 | 0.846 | 0.848 | 0.847 | 0.895 | 0.867 | 0.913 | 0.889 |
| Logistic Regression | 0.812 | 0.802 | 0.615 | 0.696 | 0.829 | 0.810 | 0.829 | 0.819 |

XGBoost was also the best overall classifier.

It achieved:

- the highest accuracy,
- the highest precision,
- the highest F1,
- the highest cross-validated accuracy,
- the highest cross-validated precision,
- the highest cross-validated recall,
- the highest cross-validated F1.

There is one nuance: the RBF support vector classifier had slightly higher
recall on the single held-out test split. That means it caught a few more
positive cases in that particular split. However, it did so with substantially
lower precision and lower F1. In practical terms, it was more willing to call
nodes flooded, but those positive predictions were less reliable.

XGBoost gave the better balance. It produced very high precision while still
maintaining strong recall, and its cross-validation results were stronger
overall. That made it the better default choice for the project.

## 7. Why XGBoost Won

XGBoost won because the problem is a strong fit for boosted trees.

The hydraulic response is nonlinear. Flooding does not increase smoothly and
linearly under every condition. It often behaves like a threshold process:
below a certain hydraulic stress level, the node may remain dry; beyond that
point, flooding can appear and increase rapidly.

The response is also interaction-heavy. A node's outcome depends on multiple
conditions at the same time, not just one isolated quantity. Boosted trees are
good at learning such conditional interactions because each tree partitions the
data into regimes, and the ensemble combines many such partitions.

XGBoost also handles mixed distributions well. Hydraulic datasets often include
many zero or near-zero outcomes plus a smaller number of high-severity cases.
Linear models tend to smooth over those regimes. Kernel methods can capture
nonlinearity, but they often struggle to extrapolate robustly and can become
less transparent operationally. XGBoost gave the strongest combination of
accuracy, stability, and deployability.

## 8. What We Learned from the Other Models

The other models were not wasted work. They helped define the baseline and
showed what kind of structure the data had.

The linear models were useful because they tested a simple hypothesis: maybe
the engineered descriptors were enough for a mostly linear predictor. Their
results showed that the response was too nonlinear for that to be the best
approach.

The support vector models showed that nonlinear modeling helped. The RBF
regressor improved over the linear regressors on test error, and the RBF
classifier improved classification recall. But the support vector family still
did not match XGBoost's overall performance, especially in regression.

XGBoost showed that the best-performing model needed both nonlinearity and
interaction modeling, while still being stable under grouped validation.

## 9. Why the Preprocessing Work Mattered

The preprocessing experiments mattered because the winner should not depend on
an unfair pipeline.

Median imputation made the comparison robust to incomplete structural
descriptors.

Standard scaling made the comparison fair for models that depend on distances,
margins, coefficients, or PCA.

PCA tested whether a compact mathematical representation could preserve enough
information while reducing redundancy. The five-component PCA representation
preserved about 84.1% of the variance, which was enough to run a controlled
comparison across models.

Grouped splitting protected the evaluation from leakage across related rows.
This was essential because rows generated by the same simulation are not
independent in the same way that randomly sampled observations would be.

In short, we did not simply run XGBoost and declare it best. We built a
controlled comparison framework, evaluated multiple preprocessing choices, and
then selected XGBoost because the metrics supported it.

## 10. How This Led to the Current Architecture

After the comparison, the project moved toward XGBoost as the preferred
surrogate model family.

The historical artifact manifest records XGBoost as the latest saved model for
both tasks:

```text
regression:     xgboost
classification: xgboost_classifier
```

The current active pipeline is now more direct. Instead of rerunning the whole
model-comparison table every time, the main workflow trains the selected
XGBoost architecture and evaluates it with the current validation harness.

That is the normal evolution:

1. Explore several model families.
2. Compare them under consistent preprocessing and validation.
3. Select the best-performing family.
4. Promote that family into the active production-style pipeline.

## 11. Presentation-Friendly Summary

You can explain the history like this:

> Before choosing the final architecture, we benchmarked several tabular
> machine-learning families: regularized linear models, support vector models,
> and XGBoost. We also evaluated preprocessing choices such as missing-value
> imputation, scaling, and PCA-based dimensionality reduction. The comparison
> used grouped validation so that related rows from the same simulation did not
> leak across train and test. Across the metrics, XGBoost was the strongest
> option: it clearly won the regression task and had the best overall
> classification balance. That is why the current system uses XGBoost as the
> core surrogate model.

## 12. Technical Conclusion

The model-comparison phase showed that the hydraulic surrogate problem is not
well served by simple linear assumptions. The data contains nonlinear
thresholds and interactions that XGBoost captures much more effectively.

The preprocessing work improved the rigor of the comparison:

- imputation prevented structural missing values from breaking the pipeline,
- scaling made distance- and component-based methods viable,
- PCA tested a compact representation,
- grouped validation reduced leakage and gave more honest metrics.

Even under that more controlled setup, XGBoost won. That result justified
using XGBoost as the final tabular surrogate architecture for the project.

## 13. Source Evidence

This note is based on:

- `swmm_resilience/ml/train.py`
- `data/networks/chico_hydro-qx1/results/regression_comparison_flooding_volume_m3.csv`
- `data/networks/chico_hydro-qx1/results/classification_comparison_flooded.csv`
- `data/networks/chico_hydro-qx1/results/pca_analysis/pca_summary.txt`
- `data/networks/chico_hydro-qx1/results/model_artifacts/manifest.json`
- `DOCUMENTACION_COMPLETA_PROYECTO.md`
