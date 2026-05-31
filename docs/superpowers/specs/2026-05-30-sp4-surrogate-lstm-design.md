# SP4 — Surrogate LSTM Benchmark

## Goal

Add `SWMMSurrogateLSTM` as a fourth model in the unified comparison pipeline
(`compare_surrogate.py`), using the same dataset, same GroupKFold splits, and
same dual-head loss as the surrogate CNN. Directly answers: **does LSTM beat
CNN for surrogate flood prediction?**

---

## Context

The original SP4 spec (2026-05-29) was designed to compare against the early-warning
CNN (SP3) using sliding windows and SWMM-output features. That approach is now
obsolete — SP3 was redesigned as a surrogate model and the comparison framework
was rebuilt around `build_unified_dataset()` and `compare_surrogate.py`.

This redesigned SP4 plugs into the existing infrastructure with minimal new code:
one new model file + a small addition to `compare_surrogate.py`.

---

## Architecture: `SWMMSurrogateLSTM`

Mirrors `SWMMSurrogateCNN` exactly — only the temporal branch changes.

```
X_seq    [batch, T, 2]
    → LSTM(input_size=2, hidden_size=64, num_layers=2,
            batch_first=True, dropout=0.2)
    → last hidden state h_n[-1]                         → [batch, 64]

X_static [batch, 21]
    → Linear(21→32) → ReLU
    → Linear(32→32) → ReLU                              → [batch, 32]

concat([lstm_out, static]) → [batch, 96]
    → Linear(96→64) → ReLU → Dropout(0.3)              → [batch, 64]

Classification head: Linear(64→1)   ← raw logit, NO sigmoid
Regression head:    Linear(64→1)    ← unbounded float
```

`use_temporal=False` mode: LSTM branch disabled, `fusion_in=32`, static only.
Same convention as `SWMMSurrogateCNN` so the ablation comparison is consistent.

**File:** `swmm_resilience/ml/temporal/models/surrogate_lstm.py`

---

## Integration into compare_surrogate.py

Add `_train_eval_lstm()` helper and a fourth model block inside the existing
`compare_surrogate()` fold loop. The LSTM result columns follow the same naming
convention: `lstm_auc_roc`, `lstm_f1`, `lstm_rmse`, `lstm_mae`, etc.

The comparison table becomes:

| Model | Temporal branch | Static features |
|-------|----------------|-----------------|
| XGBoost | — | 21 |
| CNN full | Conv1d + AdaptiveAvgPool | 21 |
| CNN ablation | — | 21 |
| **LSTM** | LSTM (2 layers, hidden=64) | 21 |

---

## Loss function

Identical to CNN surrogate:
```
total_loss = α × BCEWithLogitsLoss(pos_weight=n_neg/n_pos) + β × MSELoss()
α = 1.0,  β = 0.01
```

---

## Training protocol

- Same `build_unified_dataset()` dataset (2576 samples)
- Same `GroupKFold(n_splits=5)` splits — **same fold indices as CNN and XGBoost**
- Same scalers (StandardScaler fitted on train fold only)
- Same metrics: AUC-ROC, F1, Precision, Recall, Accuracy, RMSE, MAE
- Optimizer: AdamW, lr=1e-3, `ReduceLROnPlateau(patience=5)`
- Epochs: 100 (same as CNN)

---

## Files modified / created

| File | Change |
|------|--------|
| `swmm_resilience/ml/temporal/models/surrogate_lstm.py` | **New** — `SWMMSurrogateLSTM` |
| `swmm_resilience/ml/temporal/compare_surrogate.py` | **Modify** — add `_train_eval_lstm()` + LSTM block in fold loop |
| `tests/ml/temporal/test_surrogate_lstm.py` | **New** — model forward-pass tests |
| `tests/ml/temporal/test_compare_surrogate.py` | **Modify** — add LSTM metric columns check |

---

## Out of scope

- Standalone `train_lstm.py` script (comparison happens inside `compare_surrogate.py`)
- Attention LSTM (future extension)
- Desktop integration (SP5)
- Bi-directional LSTM (YAGNI)
