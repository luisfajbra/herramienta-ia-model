# LOSO And GroupKFold With 175 Hydrograph Scenarios

This note explains how the current evaluation works now that the dataset has
175 hydrograph scenarios.

## Why There Are 175 Scenarios

The current training dataset is built from:

```text
7 hydrograph shapes x 25 intensity factors = 175 scenarios
```

Each scenario is one combination of:

- one hydrograph shape, such as a common storm or a rare extreme storm,
- and one intensity factor, such as 0.2, 0.4, 0.6, up to 5.0.

So a scenario is not only "factor 2.0". It is more specific:

```text
common storm at factor 2.0
rare extreme storm at factor 2.0
long moderate storm at factor 2.0
...
```

With 7 shapes and 25 factors, there are 175 total shape-factor combinations.

## The Important Detail

Even though the dataset has 175 scenarios, the current LOSO and GroupKFold
evaluation do not treat the 175 scenarios as 175 separate validation groups.

The current evaluation groups rows by **intensity factor**.

That means all hydrograph shapes that share the same intensity factor are kept
together during validation.

For example, all of these belong to the same validation group:

```text
base shape at factor 2.0
common storm at factor 2.0
rare extreme storm at factor 2.0
long moderate storm at factor 2.0
...
```

They are different hydrograph scenarios, but they share the same intensity
factor, so the current evaluation treats them as one group.

## How LOSO Works

LOSO means **leave one group out**.

In the current setup, the group is the intensity factor. Therefore, LOSO leaves
out one intensity factor at a time, not one individual hydrograph scenario.

If there are 25 intensity factors, LOSO creates 25 evaluation folds.

Example:

```text
Fold 1
Test:
  all 7 hydrograph shapes at factor 0.2
Train:
  all 7 hydrograph shapes at every other factor

Fold 2
Test:
  all 7 hydrograph shapes at factor 0.4
Train:
  all 7 hydrograph shapes at every other factor

Fold 3
Test:
  all 7 hydrograph shapes at factor 0.6
Train:
  all 7 hydrograph shapes at every other factor

...

Fold 25
Test:
  all 7 hydrograph shapes at factor 5.0
Train:
  all 7 hydrograph shapes at every other factor
```

So, even with 175 total scenarios:

```text
LOSO = 25 folds
```

because there are 25 intensity-factor groups.

## How GroupKFold5 Works

GroupKFold5 also uses intensity factor as the group.

Instead of leaving out one factor at a time, it splits the 25 factors into 5
larger groups.

Each fold tests on about 5 intensity factors and trains on the remaining 20
intensity factors.

Because each factor has 7 hydrograph shapes:

```text
Test set per fold:
  about 5 factors x 7 shapes = about 35 hydrograph scenarios

Training set per fold:
  about 20 factors x 7 shapes = about 140 hydrograph scenarios
```

So GroupKFold5 creates:

```text
5 folds
```

Each fold contains many scenarios, but the split is still organized by
intensity factor.

## What This Evaluation Actually Tests

The current evaluation mainly tests this question:

> Can the model generalize to intensity levels it did not train on?

For example, when factor 2.0 is held out, the model has not trained on any
hydrograph shape at factor 2.0. It must predict all 7 shapes at that intensity
using what it learned from the other intensity levels.

That is useful because it checks whether the model understands the relationship
between hydraulic severity and flooding.

## What This Evaluation Does Not Fully Test

The current evaluation does not fully test this question:

> Can the model generalize to a completely unseen hydrograph shape?

Why not?

Because every fold usually still contains the same hydrograph shape families in
training, just at other intensity factors.

For example, when testing:

```text
common storm at factor 2.0
```

the model may already have trained on:

```text
common storm at factor 0.2
common storm at factor 0.4
common storm at factor 0.6
...
common storm at factor 5.0
```

except for factor 2.0 itself.

So the model is being tested on an unseen intensity for a known shape family,
not on a totally unseen shape family.

## Simple Summary

The dataset has:

```text
175 hydrograph scenarios = 7 shapes x 25 factors
```

But the current validation groups by factor:

```text
25 factor groups
```

Therefore:

```text
LOSO = leaves out 1 factor group at a time
     = 25 folds
     = each test fold has 7 scenarios

GroupKFold5 = splits the 25 factor groups into 5 folds
            = each test fold has about 35 scenarios
```

In plain English:

> The current validation tests whether the model can predict new intensity
> levels across all hydrograph shapes. It does not fully test whether the model
> can predict a completely new hydrograph shape that was absent from training.

