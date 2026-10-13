# Correction 03 — Resolution of the flooding volume and semantics of the label (D-04 / H-06)

- **Date:** 2026-10-11
- **Type:** bounded *spike*. It is an investigation whose product is this document with figures and a recommendation. No project code, nor `config.yaml`, nor `data/`, nor `outputs/` was modified. Only this document and the dated correction notes of §8.1 enter the repository. The scripts and the copies of `.inp`, `.rpt` and `.out` remain only in the session scratchpad (§9), which is temporary. For that reason the complete code of the scripts goes in the Appendix.
- **Branch:** `worktree-optuna-tuning-familia-ganadora` (base `6e8579b`, with uncommitted changes from another phase that were not touched). The line numbers of `main.py` refer to the working tree.
- **Engine and libraries (project venv):** pyswmm 2.1.0, swmm-toolkit 0.17.0, swmm-api 0.4.73.
- **Network:** `data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp` (`FLOW_UNITS LPS`, `DYNWAVE`, `ROUTING_STEP` 5 s, `REPORT_STEP` 3 min, `ALLOW_PONDING NO`).
- **Evaluation data:** PreparedDataset `ca706e27654e0cf5` (`outputs/bench/prepared_v4_21/`) and OOFs from `outputs/bench/oof/ca706e27654e0cf5/` (154,080 rows per protocol), read in read-only mode.
- **SWMM simulations:** 18 in total, all on copies in the scratchpad and assembled with the same functions of the pipeline:
  - 6 control ones (`s01`);
  - 1 with `REPORT_STEP` of 5 s (`s02`);
  - 2 verification ones (`s06`);
  - 9 added in the correction round (`s07`).
- **Scope of every quantitative claim:** a single network, with 17 different simulated scenarios (shape, factor), of which 15 have overflow: 6 from `s01`, 2 from `s06` (`common_bogota` f = 3.4 and 3.6) and 9 from `s07`; the `s02` run repeats a scenario of `s01` with another `REPORT_STEP`. The node-by-node comparisons and the bound on the effective threshold use the 15 scenarios with per-node record (`s01` + `s07`, 13 with overflow), because `s06` did not save the volume of each node. Any generalization beyond that is inference and is flagged as such.
- **Citation rule:** each figure carries in brackets the script that produced it (§9). What was not measured is marked as **not verified**.

---

## 1. Summary

1. The current label is `inunda = (V_rpt ≥ 1 m³)`, where `V_rpt` is the *Total Flood Volume* of the `.rpt`. That volume is printed in 10⁶ L with 3 decimals, that is, with a resolution of 1 m³.
2. In the 15 scenarios with per-node record (`s01` + `s07`, 13 with overflow; of 17 simulated scenarios in total), `V_rpt` coincides with the rounding to the integer of the continuous volume that SWMM accumulates at each computation step in 1,407 of the 1,408 cases with 0 < V < 100 m³. There are 13 cases (8C with 83.5 m³ and 12 nodes with more than 283 m³) in which `V_rpt` is **below** the API value, by 0.5 to 0.86 m³; their origin is **not verified**. The current label coincides with `V_cont ≥ t` in the 2,400 node-scenario cases (874 are trivial dry nodes) for any threshold t in the interval (0.4988, 0.5002] m³ [s05, s08]. It is a **deterministic quantization, not noise**: the nominal threshold of 1 m³ works as a threshold of ≈ 0.5 m³ of continuous volume.
3. If the continuous volume is wanted, the source is `pyswmm` (`node.statistics['flooding_volume']`), which adds no appreciable cost. The integral of the `flooding` series of the `.out` **is not suitable** with the current `REPORT_STEP` of 3 min: it departs from the continuous volume by up to 9.5 m³ per node (\|out − API\|) and from the `.rpt` by up to 9.3 m³. With a `REPORT_STEP` of 5 s, the maximum difference with respect to the API goes down to 0.89 m³ (median 0.039). That is worse than the `.rpt` at the maximum (0.49) but better at the median (0.27) [s01, s02].
4. In the 6 control scenarios, switching to `V_cont ≥ 1 m³` changes 21 labels, all from 1 to 0. Switching to `V_cont > 0` changes 106, all from 0 to 1 [s01].
5. Re-simulating the 1000 runs costs about **17 min of SWMM**. Adding the construction of the `.inp`, the reading of labels and that of the drainage, without assembling features, it is about 20 min. It is an extrapolation with uncertainty (§5.1) [s03]. The real cost lies in the downstream chain (dataset, SQL, prep_id, benchmark, Optuna, tables and validation) and in re-tracing all the figures of the thesis.
6. In LeaveOneShapeOut, 333 of the 380 false negatives of XGBoost have `V_rpt = 1`, that is, a continuous volume of about 0.5 to 1.5 m³. If those 1,894 rows are excluded, the pooled F1 goes from 0.9817 to 0.9847 and the recall from 0.9918 to 0.9989 [s04]. In those rows, the median absolute error of the regressor (1.18 m³) doubles the maximum quantization error (0.5 m³) [s08].
7. 12 of the 72 cases of "stops flooding when the factor rises" were re-simulated. In all 12 the continuous volume really goes down, but in 2 it goes down by less than 0.1 m³ and in 3 between 0.1 and 0.3 m³ [s06, s07] (§4.1).
8. **Recommendation: path A.** The current resolution and threshold are kept, declared as an assumption (effective threshold of ≈ 0.5 m³ continuous) and the confusion matrix is reported with the band separated. Path B is only worthwhile if the author decides to regenerate the complete dataset for another reason. In that case, capturing the continuous volume has almost zero marginal cost (§7).

---

## 2. How the volume and the label are obtained today

### 2.1 Resolution chain

| Step | Where | What happens | Unit and resolution |
|---|---|---|---|
| 1. SWMM computation | SWMM 5.2 engine (swmm-toolkit 0.17.0) | The overflow of each node is accumulated at each routing step (actual dt of 1 to 5 s, annex 01, V-11) | Continuous volume in floating point |
| 2. Report | `.rpt`, section *Node Flooding Summary*, column *Total Flood Volume* | Printed with 3 decimals (observed format: `1.286`) | 10⁶ L; resolution 0.001·10⁶ L = **1 m³** |
| 3. Reading | `swmm_api_io.py:271-320` (`read_node_flooding_summary`) | `swmm_api.read_rpt_file(...).node_flooding_summary`. It takes the column whose name contains "volume" and "10^6" (l. 296) and multiplies by 1000 (l. 305-308) | m³, integers with exceptions |
| 3b. Fallback | `labels.py:26-90` (`_parse_node_flooding_text`) | Only if swmm_api fails. It takes `parts[-1]` (l. 79), which in SWMM 5.2 is *Maximum Ponded Depth* (H-07) | Gives **0** in all nodes (confirmed again here: Σ = 0 in the 4 scenarios with overflow [s01]) |
| 4. Labelling | `labels.py:122-139` (`extract_labels`) + `labels.py:19-23` (`flood_label`) | Nodes absent from the summary receive 0. `inunda = vol ≥ threshold` | Threshold `config.yaml:14` `dataset.flood_threshold_m3: 1.0` |
| 5. Generation | `main.py:1213` (shape `base`, via `batch.run_batch` → `runner.run_simulation_simple`) and `main.py:1280` (39 shapes, via `batch.run_batch_shapes`) | Simulates with `pyswmm.Simulation` (`for _ in sim: pass`) and reads the `.rpt` | Same |
| 6. Dataset | `dataset/assembler.py` → `data/training/dataset_final.csv` → `scripts/migrar_dataset_v4_21.py` → `dataset_final_v4_21.csv` | Columns `vol_inundacion_m3` and `inunda` | Same |
| 7. Persistence | `database/csv_backfill.py:207` (`--persist-sql`) → `outputs/training_v17.sqlite3`, table `node_results` | The two columns are copied | Same |
| 8. Benchmark | PreparedDataset `ca706e27654e0cf5`, `targets.parquet` | `inunda` and `vol_inundacion_m3` | 154,080 rows |

**Current formula:**

`vol_inundacion_m3 = 1000 · TFV_rpt`, with `TFV_rpt` = *Total Flood Volume* [10⁶ L] with 3 decimals; `inunda = 1 ⇔ vol_inundacion_m3 ≥ 1.0`.

### 2.2 What the persisted data say [s04]

| Count in the PreparedDataset (154,080 rows) | Value |
|---|---|
| Positive rows (`inunda = 1`) | 46,225 |
| `inunda == (vol ≥ 1)` in all rows | Yes |
| Rows with volume in (0, 1) m³ | 0 |
| Rows with volume exactly 1.0 m³ | 1,894 |
| Rows with volume in [0.5, 2) m³ | 1,894 (they are the same) |
| Non-integer positives | 6 (all ≥ 2 m³) |

The audit figures (46,242 positives, 1,902 rows with 1.0) correspond to the complete dataset of 160,000 rows. The PreparedDataset excludes 37 runs due to continuity.

---

## 3. Higher-resolution volume sources: comparison in control scenarios

### 3.1 Method

- **Scenarios:** the shape `base`, two short shapes (`shape_ultra_short_impulse`, 0.5 h; `shape_micro_burst_45min`) and a long one (`long_moderate`, 12 h) were combined with the factors 0.2, 1.0, 3.0 and 5.0.
- **Construction of the `.inp`:** `write_scaled_inp` for `base`; `apply_shape` + `apply_closure_policy` (tolerance 0.01 L/s) + `write_scenario_inp` (6 h drainage) for the shapes, as in `batch._prepare_closed_scenario_inp`.
- **Sources compared node by node:**
  - **rpt:** `extract_labels` (the pipeline route).
  - **out:** trapezoidal integral of the `flooding` variable (L/s, because `FLOW_UNITS = LPS`) of the `.out`, read with `swmm_api.SwmmOutput`; L/s·s/1000 = m³. Left and right rectangles were also computed. In `s01` the integral starts at the first record of the `.out` (t = 3 min); in `s02` q(0) = 0 is prepended.
  - **API:** `pyswmm` `node.statistics['flooding_volume']` in m³, read when the simulation loop ends.
- **Reproduction:** in the 4 scenarios that are in the DB, the replicated `.rpt` volume is **identical** to that of `targets.parquet` in 160 of 160 nodes of each scenario (640 of 640). The totals also coincide: `base` f = 3 gives 7,669 m³ and 61 flooded nodes, `shape_ultra_short_impulse` f = 5 gives 1,320 / 96, `shape_micro_burst_45min` f = 3 gives 1,704 / 60 and `long_moderate` f = 5 gives 123,097 / 91 [s01, s04]. `base` f = 0.2 reproduces the continuity error of 5.168 % of `outputs/continuity_excluded.csv`, which is precisely the reason why that run is excluded.

### 3.2 Volume per source and per-node differences [s01]

Absolute differences in m³, computed over the nodes with overflow according to any of the sources.

| Scenario | CE (%) | In DB | Nodes with overflow | Σ rpt | Σ API | Σ out (3 min) | *Flooding Loss* (.rpt) | max \|rpt−API\| | median \|rpt−API\| | max \|rpt−out\| | median \|rpt−out\| | max \|out−API\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `base` f = 0.2 | 5.168 | no (excluded) | 0 | 0 | 0 | 0 | 0 | — | — | — | — | — |
| `base` f = 3.0 | 0.459 | yes | 88 | 7,669 | 7,673.19 | 7,672.18 | 7,673 | 0.515 | 0.210 | 3.695 | 0.236 | 3.618 |
| `shape_ultra_short_impulse` f = 5.0 | 1.978 | yes | 128 | 1,320 | 1,327.26 | 1,303.11 | 1,328 | 0.494 | 0.270 | 9.305 | 0.788 | 9.535 |
| `shape_micro_burst_45min` f = 3.0 | 1.587 | yes | 101 | 1,704 | 1,713.09 | 1,705.42 | 1,713 | 0.500 | 0.183 | 5.492 | 0.293 | 5.339 |
| `long_moderate` f = 1.0 | 0.286 | yes | 0 | 0 | 0 | 0 | 0 | — | — | — | — | — |
| `long_moderate` f = 5.0 | 0.055 | yes | 97 | 123,097 | 123,111.53 | 123,100.71 | 123,099 | 0.857 | 0.280 | 1.343 | 0.240 | 1.790 |

| Global (414 node-scenarios with overflow) | maximum | median | p95 |
|---|---|---|---|
| \|rpt − API\| | 0.857 | 0.241 | 0.482 |
| \|rpt − out (3 min, trapezoid)\| | 9.305 | 0.313 | 3.014 |
| \|out (3 min, trapezoid) − API\| | 9.535 | 0.301 | 2.820 |
| \|out (3 min, right rect.) − API\| | 10.234 | 0.303 | 3.009 |
| \|out (3 min, left rect.) − API\| | 9.535 | 0.303 | 3.009 |

**Reading:**

- **rpt versus API.** In the 6 scenarios of `s01` (960 node-scenario cases: 546 dry nodes, 306 with 0 < V < 100 m³ and 108 with V ≥ 100 m³), `V_rpt = round(V_API)` in the 306 cases with 0 < V < 100 m³ [s05, s08]. In 12 of the 108 cases with V ≥ 100 m³ (all above 283 m³), `V_rpt` is **below** the API value, by 0.51 to 0.86 m³ (relative ≤ 0.18 %). In those 12 cases `V_rpt = floor(V_API)`, which looks like a truncation or a difference between accumulators; the origin is **not verified**. In the 9 scenarios of `s07` one more case of the same type appears with a smaller volume: `8C`, `rare_extreme` f = 4.6, with V_API = 83.508 and V_rpt = 83 [s08]. None of these differences affects the label.
- **The `.out` is not a higher-resolution source with the current configuration.** SWMM writes there instantaneous values every `REPORT_STEP` = 180 s (60 to 380 records per run; the first at 3 min [s01]). Integrating those values loses the short overflows. In `shape_ultra_short_impulse`, the integral of the `.out` departs from the continuous volume (API) by up to 9.5 m³ per node, a relative error of 344 % with respect to V_API in a node with V ≥ 1 m³. Compared with the `.rpt`, the difference reaches 9.3 m³. Using `out ≥ 1` instead of the `.rpt` would change 30 labels in the 6 scenarios [s01].
- **Test with `REPORT_STEP` = 5 s** (`shape_ultra_short_impulse` f = 5, 1 run) [s02]:

| Source versus API | max (m³) | median (m³) |
|---|---|---|
| out with `REPORT_STEP` 5 s | 0.891 | 0.039 |
| out with `REPORT_STEP` 180 s | 9.535 | 0.575 |
| rpt | 0.494 | 0.270 |

  With 5 s the `.out` goes from 0.89 to 31.8 MB (4,680 records) and still changes 1 label with respect to the API. At the maximum it is still worse than the `.rpt` (0.891 versus 0.494 m³), but at the median it is better (0.039 versus 0.270 m³). Changing `REPORT_STEP` does not alter the computation: the API and the `.rpt` are identical to those of `s01` (difference ≤ 3.6·10⁻¹⁵ m³).
- **`.rpt` with more decimals:** SWMM offers no option to change the format of the *Node Flooding Summary*. **Not verified** in the C source code. In the generated `.rpt` files the column always has 3 decimals.
- **Source conclusion:** if the continuous volume is wanted, the source is `pyswmm` `node.statistics['flooding_volume']`. It is already used in the venv (annex 01, V-04 and d11) and reading it at the end of the loop costs ≤ 0.023 s per run [s03].

---

## 4. Label changes according to the rule [s01, s05]

The rules are: (i) current, `V_rpt ≥ 1`; (ii) `V_continuo ≥ 1`; (iii) `V_continuo > 0`. `V_continuo` = API.

| Scenario | Overflow (V_cont > 0) | (i) positives | (ii) positives | (iii) positives | Changes (i)→(ii) | Changes (i)→(iii) | V_cont in (0, 0.5) | V_cont in [0.5, 1) | Overflowing with label 0 (%) |
|---|---|---|---|---|---|---|---|---|---|
| `base` f = 0.2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| `base` f = 3.0 | 88 | 61 | 60 | 88 | 1 | 27 | 27 | 1 | 30.7 |
| `shape_ultra_short_impulse` f = 5.0 | 128 | 96 | 83 | 128 | 13 | 32 | 32 | 13 | 25.0 |
| `shape_micro_burst_45min` f = 3.0 | 101 | 60 | 53 | 101 | 7 | 41 | 41 | 7 | 40.6 |
| `long_moderate` f = 1.0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — |
| `long_moderate` f = 5.0 | 97 | 91 | 91 | 97 | 0 | 6 | 6 | 0 | 6.2 |
| **Total** | **414** | **308** | **287** | **414** | **21** | **106** | **106** | **21** | — |

**Reading:**

- All the (i)→(ii) changes are from 1 to 0 and coincide with the nodes with `V_cont` in [0.5, 1). All the (i)→(iii) changes are from 0 to 1 and coincide with the nodes with `V_cont` in (0, 0.5) [s01].
- **Scope and bound of the effective threshold [s05, s08].** The interval of thresholds t for which `(V_rpt ≥ 1) ⇔ (V_cont ≥ t)` holds in all the simulated cases is sought. The lower bound is the largest V_cont with label 0 and the upper bound, the smallest V_cont with label 1. The cases with 0 < V < 100 m³ are the only informative ones for the threshold; the dry and the large nodes satisfy it trivially.

| Set | Scenarios (with overflow) | Node-scenario cases | Dry | 0 < V < 100 m³ | V ≥ 100 m³ | Label discrepancies | Effective threshold t |
|---|---|---|---|---|---|---|---|
| `s01` | 6 (4) | 960 | 546 | 306 | 108 | 0 | (0.4988, 0.5142] |
| `s07` | 9 (9) | 1,440 | 328 | 1,102 | 10 | 0 | (0.4987, 0.5002] |
| `s01` + `s07` | 15 (13) | 2,400 | 874 | 1,408 | 118 | 0 | **(0.4988, 0.5002]** |

  `s06` confirms it with a data point that was not saved per node: `66I` is worth 0.5002 m³ with label 1. The reviewer's independent repetition, on 3 other scenarios, gave a bound of (0.4643, 0.5262] (not re-run here). It is wider, but compatible: its intersection with ours is still (0.4988, 0.5002]. All this holds for **one network and the 13 scenarios with overflow that have a per-node record** (plus the isolated data point of `66I` in `s06`). That the effective threshold is ≈ 0.5 m³ in the rest of the dataset is an inference: the mechanism is the rounding to 3 decimals of the `.rpt`, but it was measured only in that sample.
- The overflows that today have label 0 are small: maximum 0.499 m³, median 0.153 m³, sum of 20.3 m³ in the 4 scenarios with overflow [s05]. In these controls they represent between 6 % and 41 % of the overflowing nodes. The audit reported between 27 % and 57 % in other scenarios (f = 1.6 and 2.0, closer to the onset of flooding). That figure comes from annex 01 (`d11`) and was not verified again here.
- The minimum positive volume seen by the API is 0.0027 m³. Rule (iii) is sensitive to those traces: with `V_cont > 0.01` there would be 412 positives instead of 414.
- **Projection to the dataset (estimate, not verified):** in the controls there are 28 nodes with `V_rpt = 1`, with V_cont between 0.514 and 1.455 m³ across the 28. Of them, 21 have `V_cont < 1`. Of the 7 nodes with `V_rpt = 2`, none. If that proportion (21/28 = 75 %, Wilson 95 % CI [56.6 %, 87.3 %]) were maintained, rule (ii) would turn to 0 about 1,420 of the 1,894 rows with `V_rpt = 1` (interval from 1,073 to 1,654 rows), that is, 3.1 % of the 46,225 positives (from 2.3 % to 3.6 %) [s08]. The interval assumes that the 28 nodes are a representative sample, which they are not (4 scenarios). Rule (iii) would turn to 1 about 32 % of the dry nodes of the scenarios with overflow (106 of 332 in the controls). Neither of the two projections was measured on the complete dataset.

### 4.1 The 72 cases "stops flooding when the factor rises" [s06, s07]

Annex 03 (D-04) reports 72 transitions in the DB (same shape and node, `inunda = 1` at f and `inunda = 0` at f + 0.2): 68 with `V_rpt = 1` and 4 with `V_rpt = 2`. It attributes them to "rounding noise".

**Deduced (without simulating).** In the measured cases, `V_rpt` is a monotonically non-decreasing function of `V_cont`: the rounding, or the truncation in the large cases. Therefore, every 1 → 0 transition of the label requires that the **continuous** volume goes down between f and f + 0.2. The rounding can expose a decrease, but not create it. What the rounding does determine is whether a small decrease crosses the threshold: a decrease of 0.03 m³ around 0.5 m³ changes the label, and the same decrease around 10 m³ does not.

**Measured.** 12 of the 72 transitions were re-simulated: 3 in `s06` and 9 in `s07`. This round asked for up to 8 new transitions; `rare_extreme` 4.0 → 4.2 entered without additional simulation, because it shares runs with the neighbouring pairs. 3 shapes (`common_bogota`, `rare_extreme` and `shape_ultra_short_impulse`) and 7 factor pairs are covered, including the 4 cases with `V_rpt = 2`. The remaining 60 were **not re-simulated (not verified)**. In all the runs, the DB coincides with the replicated `.rpt` in 160 of 160 nodes.

| Shape | f → f + 0.2 | Node | V_rpt before | V_cont before | V_rpt after | V_cont after | Decrease in V_cont (m³) |
|---|---|---|---|---|---|---|---|
| `common_bogota` | 3.4 → 3.6 | `107C` | 1 | 0.8623 | 0 | 0.0000 | 0.862 |
| `common_bogota` | 3.4 → 3.6 | `52C` | 1 | 0.8675 | 0 | 0.0197 | 0.848 |
| `common_bogota` | 3.4 → 3.6 | `66I` | 1 | 0.5002 | 0 | 0.4663 | 0.034 |
| `common_bogota` | 4.4 → 4.6 | `51C` | 1 | 0.8617 | 0 | 0.3936 | 0.468 |
| `rare_extreme` | 3.8 → 4.0 | `16I` | 1 | 0.5516 | 0 | 0.4738 | 0.078 |
| `rare_extreme` | 3.8 → 4.0 | `40I` | 1 | 0.5221 | 0 | 0.4202 | 0.102 |
| `rare_extreme` | 3.8 → 4.0 | `51C` | 2 | 2.1310 | 0 | 0.2586 | 1.872 |
| `rare_extreme` | 4.0 → 4.2 | `66I` | 1 | 0.6192 | 0 | 0.4561 | 0.163 |
| `rare_extreme` | 4.2 → 4.4 | `40C` | 2 | 2.4388 | 0 | 0.1498 | 2.289 |
| `rare_extreme` | 4.4 → 4.6 | `16I` | 1 | 0.6038 | 0 | 0.4372 | 0.167 |
| `rare_extreme` | 4.4 → 4.6 | `51C` | 2 | 2.3164 | 0 | 0.2789 | 2.037 |
| `shape_ultra_short_impulse` | 4.4 → 4.6 | `40C` | 2 | 1.7461 | 0 | 0.0000 | 1.746 |

**Measured distribution (n = 12):**

| Decrease in V_cont | Cases |
|---|---|
| ≥ 0.3 m³ | 7 (the 4 with `V_rpt = 2` and 3 with `V_rpt = 1`) |
| Between 0.1 and 0.3 m³ | 3 |
| < 0.1 m³ | 2 (`66I` 0.034 and `16I` 0.078) |

In the 12 cases the continuous volume really goes down, but only in 7 is the decrease large relative to the threshold.

**Reading:**

- The explanation of annex 03 is **partially correct**. In 5 of the 12 cases (those with a decrease < 0.3 m³) the continuous volume goes from slightly above to slightly below 0.5 m³. Those cases are borderline: a small variation that the threshold turns into a label change. In the other 7, the decrease is large and unrelated to the resolution.
- The physical cause of the decreases was not investigated. That the overflow is redistributed between nodes when the load increases is an **unverified interpretation**.
- **With a continuous threshold of 1 m³ [s07].** In the 6 pairs of `s07`, the current rule produces 9 non-monotonic transitions and the rule `V_cont ≥ 1` produces 5. Of those 5, 4 are the same nodes with `V_rpt = 2` (`51C` twice and `40C` twice) and 1 is new: `12C` in `rare_extreme` 4.0 → 4.2, which goes from ≥ 1 to < 1 m³ without changing its `V_rpt`. In the pair `common_bogota` 3.4 → 3.6 of `s06`, the rule `V_cont ≥ 1` produces no case (0 transitions). That is: changing the threshold reduces these cases in the sample, but does not eliminate them, and new ones appear around the new threshold.

---

## 5. Cost of re-simulating (path B)

### 5.1 Measured times [s01, s03]

| Scenario | Simulated hours | `.inp` construction (s) | SWMM (s) | Label reading (s) | Complete run* (s) | `.out` (MB) |
|---|---|---|---|---|---|---|
| `base` f = 0.2 | 3.00 | 0.118 | 0.371 | 0.044 | 0.533 | 0.415 |
| `base` f = 3.0 | 3.00 | 0.158 | 0.380 | 0.060 | 0.598 | 0.415 |
| `shape_ultra_short_impulse` f = 5.0 | 6.50 | 0.103 | 0.611 | 0.069 | 0.868 | 0.892 |
| `shape_micro_burst_45min` f = 3.0 | 6.75 | 0.131 | 0.595 | 0.049 | 0.859 | 0.926 |
| `long_moderate` f = 1.0 | 19.00 | 0.085 | 1.656 | 0.043 | 1.867 | 2.592 |
| `long_moderate` f = 5.0 | 19.00 | 0.058 | 1.398 | 0.051 | 1.591 | 2.592 |

\* Construction + SWMM + labels + drainage reading from the `.out` (0.084 s on average, only in the CSV shapes, as `run_batch_shapes` does).

**Extrapolation** (least squares `t = a + b·H`, maximum residual of 0.14 s; the 40 shapes add up to 478.2 simulated hours per factor, counting the closing of the inflow and the 6 h drainage):

| Concept | Model | 1000 runs |
|---|---|---|
| SWMM only | 0.142 s + 0.0726 s/h | 1,010 s ≈ **16.8 min** |
| Complete run | 0.366 s + 0.0720 s/h | 1,226 s ≈ **20.4 min** |

- **Scope of "complete run":** `.inp` construction, SWMM, label reading and drainage reading. It does not include feature assembly, the dataset or writing to the DB.
- **Uncertainty of the extrapolation:** the fit uses 6 runs with only 4 distinct horizons (3; 6.5; 6.75 and 19 h). The two 19 h runs differ by 18 % (1.656 versus 1.398 s). The figures of 16.8 and 20.4 min must be read as an order of magnitude (tens of minutes), not as precise estimates.
- It does not include the drainage retries of `run_batch_shapes` (up to 2 extensions of +2 h each). How many runs would retry is **not verified**. A coarse upper bound, unmeasured, is that each run is simulated 3 times with longer horizons, which would take the total to a bit more than triple (on the order of 1 h).
- These are wall-clock times on this machine, serial and with a single process.

### 5.2 What would have to be redone after re-simulating (nothing was executed)

| Step | What | Known cost |
|---|---|---|
| 1 | Change the code to read `node.statistics['flooding_volume']` in `runner.run_simulation_simple`, `batch.run_batch_shapes` **and** in the validation path (`validation/hydrograph_batch.py`), to keep the single rule of `labels.py`. Save the continuous volume next to that of the `.rpt`. Fix H-07 on the way. New tests | Development and tests; not measured |
| 2 | Regenerate the 1000 runs and `data/training/dataset_final.csv` (`python main.py`) | ≈ 17–20 min of SWMM (§5.1) plus assembly; not measured |
| 3 | `scripts/migrar_dataset_v4_21.py` → `dataset_final_v4_21.csv` (tributary features) | Not measured |
| 4 | `python main.py --persist-sql` → `outputs/training_v17.sqlite3` (D-06: it is neither atomic nor idempotent, so it is advisable to use a new DB) | Not measured |
| 5 | New PreparedDataset (new `prep_id`) and re-evaluation `scripts/ejecutar_evaluacion_v4_21.py` + `scripts/informe_evaluacion.py` | 330 s according to `runs/provenance.json` (88 preparation + 194 XGBoost + 28 references + 20 tables) |
| 6 | Re-run the Optuna tuning of the winning family (current branch) and the promotion | Not measured |
| 7 | Re-run the out-of-sample validation (`--evaluate-validation-shapes`), which labels with the same rule | Not measured |
| 8 | Regenerate all the tables of `docs/auditoria/correcciones/datos/` (`comparacion_protocolos.csv`, `matriz_confusion.csv`, `banda_limitrofe.csv`, `por_forma.csv`, `runs/*.json`). `datos/tables_provenance.json` (OOF and CSV hashes, prep_id) and `datos/README.md` (prep_id `ca706e27654e0cf5`, label definition, commands) would be invalidated. This document and the figures of the thesis would also have to be updated | Editorial work; not measured |

The bottleneck is not SWMM, but steps 1 and 4–8 and the risk of misaligning the traceability already built (D-01, D-06).

---

## 6. Effect on the evaluation (path A, only with the saved OOFs) [s04]

The matrix reconstructed from the OOFs coincides with `matriz_confusion.csv` in the 6 (model, protocol) pairs.

### 6.1 Confusion matrix with the band separated — XGBoost, LeaveOneShapeOut

| True volume (`.rpt`) | True label | n | Predicted 0 | Predicted 1 |
|---|---|---|---|---|
| 0 (V_cont < ≈ 0.5) | 0 | 107,855 | 106,526 (TN) | 1,329 (FP) |
| [1, 2) = only the integer 1 (V_cont ≈ [0.5, 1.5)) | 1 | 1,894 | 333 (FN) | 1,561 (TP) |
| [2, 5) | 1 | 2,416 | 34 (FN) | 2,382 (TP) |
| [5, 100) | 1 | 18,035 | 13 (FN) | 18,022 (TP) |
| ≥ 100 | 1 | 23,880 | 0 (FN) | 23,880 (TP) |

In the other XGBoost protocols, the FNs in the band are 443 of 879 (LOSO) and 435 of 844 (GroupKFold5).

### 6.2 Pooled metrics with and without the ambiguous band [0.5, 2) m³

| Model | Protocol | Rows | F1 | Recall | Precision | FN | FP |
|---|---|---|---|---|---|---|---|
| xgboost | LeaveOneShapeOut (all) | 154,080 | 0.9817 | 0.9918 | 0.9718 | 380 | 1,329 |
| xgboost | LeaveOneShapeOut (without band) | 152,186 | 0.9847 | 0.9989 | 0.9709 | 47 | 1,329 |
| xgboost | LOSO (all) | 154,080 | 0.9667 | 0.9810 | 0.9528 | 879 | 2,245 |
| xgboost | LOSO (without band) | 152,186 | 0.9704 | 0.9902 | 0.9513 | 436 | 2,245 |
| xgboost | GroupKFold5 (all) | 154,080 | 0.9681 | 0.9817 | 0.9549 | 844 | 2,144 |
| xgboost | GroupKFold5 (without band) | 152,186 | 0.9718 | 0.9908 | 0.9535 | 409 | 2,144 |
| baseline_forma_mas_cercana | LeaveOneShapeOut (all) | 154,080 | 0.9764 | 0.9808 | 0.9721 | 889 | 1,299 |
| baseline_forma_mas_cercana | LeaveOneShapeOut (without band) | 152,186 | 0.9824 | 0.9937 | 0.9714 | 278 | 1,299 |
| baseline_interpolacion_factor | LOSO / GroupKFold5 (all) | 154,080 | 0.9718 | 0.9910 | 0.9534 | 416 | 2,238 |
| baseline_interpolacion_factor | LOSO / GroupKFold5 (without band) | 152,186 | 0.9751 | 0.9995 | 0.9519 | 23 | 2,238 |

**Mandatory caveats when reporting the exclusion:**

- **The exclusion is asymmetric.** It can only remove the positive side of the boundary (`V_rpt = 1`). The negative side (overflows of 0 to 0.5 m³) stays mixed with the dry nodes in `V_rpt = 0` and cannot be separated without the continuous volume. For that reason the exclusion inflates the recall and leaves the precision almost the same (0.9718 → 0.9709). It must be presented as a sensitivity analysis, not as the main metric.
- **Many FPs are also close to the boundary.** Of the 1,329 FPs of LeaveOneShapeOut, the regressor predicts less than 2 m³ in 578 and less than 5 m³ in 1,127. `y_pred_reg` is the regressor prediction over all the test rows, not the one routed by the classifier (`evaluate._fit_predict_fold`). The mean classifier probability on the FNs of the band is 0.241.
- The FNs of the band are concentrated in out-of-range or transition shapes: `new_hydrograph` 77, `rare_extreme` 27, `shape_ultra_short_impulse` 27, `common_bogota` 26 and `shape_flash_flood_urban` 20.

### 6.3 What path B suggests for the metrics (not verified)

With continuous labels `V_cont ≥ 1`, about 1,070 to 1,650 rows of the band would become negative (projection of §4). That would move the boundary, but not eliminate it. It is an **interpretation** to expect that there will still be a band of nodes with V_cont near 1 m³ that is hard to classify. Without retraining it cannot be known whether F1 goes up or down. Evaluating the current OOFs against relabelled labels would make no sense, because the model was trained with the other boundary.

That the FNs are concentrated in the band is compatible with the difficulty coming from the closeness to the threshold. In the measured sample, the current label is a deterministic function of the continuous volume (§4), so the quantization does not add noise to the label. It was not demonstrated that this is the cause: it is an **interpretation**.

### 6.4 Effect on the regressor [s08]

The regression target (`vol_inundacion_m3`) is also quantized to integers. The quantization error is at most ±0.5 m³, that is, ±50 % at 1 m³ and ±10 % at 5 m³. To see whether that error weighs against the model error, the XGBoost error in LeaveOneShapeOut was measured on the flooded rows with small true volume, using the saved OOFs (`y_pred_reg`, regressor prediction on all test rows).

| True volume (m³) | n | median \|error\| (m³) | mean \|error\| (m³) | p90 (m³) | Median relative error | Mean bias (m³) | Rows with \|error\| ≤ 0.5 m³ | Maximum relative quantization error |
|---|---|---|---|---|---|---|---|---|
| = 1 | 1,894 | 1.181 | 2.002 | 4.702 | 1.18 | +1.94 | 30.6 % | 0.50 |
| = 2 | 969 | 1.821 | 2.839 | 6.955 | 0.91 | +2.73 | 20.0 % | 0.25 |
| [1, 5] | 4,925 | 1.626 | 2.828 | 6.882 | 0.77 | +2.57 | 22.0 % | 0.50 |
| [3, 5] | 2,062 | 2.035 | 3.581 | 9.045 | 0.52 | +3.07 | 15.0 % | 0.17 |
| [5, 20] | 6,372 | 3.866 | 6.356 | 15.162 | 0.36 | +3.33 | 7.1 % | 0.10 |
| [20, 100] | 12,059 | 13.853 | 19.909 | 44.067 | 0.31 | +3.74 | 2.0 % | 0.03 |

**Reading (interpretation):**

- In the range of 1 to 5 m³, the median absolute error of the model (1.6 m³) is more than 3 times the maximum quantization error (0.5 m³). Only 22 % of the rows fall within ±0.5 m³.
- Moreover, the model **overestimates** systematically (bias of +2 to +3 m³). That is not explained by a symmetric quantization of ±0.5 m³.
- In this range, the regressor error is dominated by the model and not by the label resolution. A continuous target would reduce at most a component of ≤ 0.5 m³ per row, and it cannot be known how much of that component the model would exploit without retraining.

---

## 7. Recommendation

**Path A: keep the resolution and the threshold, declare them and report the matrix with the band separated.**

**Reasons:**

1. **The current label is a deterministic quantization, not noise.** In the sample with per-node record (one network, 13 scenarios with overflow, 2,400 node-scenario cases of which 1,408 are informative) it is `V_cont ≥ t` with t in (0.4988, 0.5002] m³, without any discrepancy [s05, s08]. The problem of D-04/H-06 is one of **semantics**: the nominal threshold of 1 m³ works as ≈ 0.5 m³ continuous. It is solved by declaring it, without touching the data.
2. **Path B does not eliminate the phenomenon that motivated the finding.** With a continuous threshold of 1 m³, the boundary shifts about 0.5 m³ and changes the label of between 2.3 % and 3.6 % of the positives (projection, §4). That the FNs of the band reflect the difficulty inherent to any boundary is an **interpretation**: no retraining verified it. The non-monotonic transitions were reduced, but did not disappear, with `V_cont ≥ 1` in the re-simulated pairs: 5 instead of 9, one of them new (§4.1).
3. **The regressor does not justify it either.** In the rows of 1 to 5 m³, the median error of the model (1.6 m³) and its bias (+2.6 m³) clearly exceed the maximum quantization error (0.5 m³) (§6.4).
4. **Cost/benefit ratio.** SWMM is cheap (on the order of 17 to 20 min), but path B forces redoing the dataset, the DB, the `prep_id`, the benchmark, Optuna, the validation and all the already traced figures of the thesis (§5.2). All that for a change of definition that path A expresses at no cost.
5. **Condition that reverses the recommendation (conditional hypothesis).** Only if the author decides to **regenerate the complete dataset** for another reason, it is advisable to incorporate path B in that same regeneration. Capturing `node.statistics['flooding_volume']` costs ≤ 0.023 s per run. Both columns (`vol_rpt_m3` and `vol_continuo_m3`) would have to be saved and the label defined with an explicitly chosen threshold.
   - One of those possible causes is H-01 (InitFlow fixed at factor 1). Its impact is the auditor's figure (annex 01) and it was **not re-simulated here**. Moreover, H-01 shares the 🟠 severity with H-02, which does not require re-simulating.
   - The consolidated report proposes for H-01 and H-05 a re-simulation of a **subset**, to measure sensitivity, and not the regeneration of the 1000 runs.
   - A re-simulation of a subset does **not** activate path B: it does not change the labels of the dataset. It would allow, at no additional cost, recording the continuous volume in that subset and enlarging the sample of §4.
6. **Discarded source.** The integral of the `.out` with `REPORT_STEP` = 3 min must not be used. It departs from the continuous volume by up to 9.5 m³ per node, much more than the quantization to be corrected (≤ 0.51 m³; the largest case below 100 m³ is `8C`, which differs by 0.508 m³) [s01, s02, s08]. The same correction is pending in the note of `anexos/01-hidraulica-swmm.md` (see `04-informe-consolidado.md`).

**Cost of each path:**

| Path | Simulations | Code | Re-evaluation | Thesis figures |
|---|---|---|---|---|
| A | 0 | None mandatory. Optional: add the table of §6.1 to `scripts/informe_evaluacion.py` (`banda_limitrofe.csv` already exists). Advisable separately: fix H-07 (fallback parser), which does not change data | None: §6 comes from the existing OOFs | Only add the declaration and the matrix by band |
| B | 1000 (on the order of 17–20 min; with drainage retries, not verified) | Capture of the continuous volume in 3 paths, plus tests | Dataset, DB, `prep_id`, benchmark (≈ 5.5 min), Optuna, validation | All the evaluation ones |

---

## 8. Pending decision for the author

1. **Choose path A or B.** The recommendation is A, unless it is decided to regenerate the complete dataset for another reason (§7.5); in that case, B within that same regeneration.
2. **Wording of the threshold (path A).** Keep "≥ 1 m³ of the `.rpt`" and state that it is consistent with "≈ 0.5 m³ of continuous volume", or state the threshold directly as ≈ 0.5 m³ continuous. The data are the same in both cases.
3. **Main metric versus sensitivity.** Confirm three things:
   - the main metric remains the complete matrix (F1 0.9817 in LeaveOneShapeOut);
   - the matrix by band (§6.1) is always reported;
   - the exclusion of the band (§6.2) goes only as sensitivity, with the asymmetry warning.
4. **Path B: continuous threshold.** If B is chosen, decide the threshold (0.5, 1 or 5 m³, or a sensitivity sweep as H-06 proposes) **before** looking at the metrics.
5. **H-07.** Fix the fallback parser (reading by header and explicit failure). It is independent of the chosen path.
6. **Review the correction notes applied** to the earlier documents (§8.1) and decide whether their text is integrated into the final version of the audit.

### 8.1 Corrections to earlier documents

They were applied in the same commit as this review, as dated notes added without rewriting the original text (script `s09_notas.py`).

| Document and place | Original claim | Correction | Status |
|---|---|---|---|
| `anexos/03-datos-persistencia-sql.md`, row D-04 (l. 54; note after the table) | "They are rounding and boundary calculation noise, not physics" | 12 of the 72 transitions re-simulated. In all 12 the continuous volume goes down: 7 with decreases ≥ 0.3 m³ and 2 with decreases < 0.1 m³ (borderline). The other 60 are unverified | Applied |
| Same | "the threshold filters nothing other than the rounding of the report" | Deterministic quantization equivalent to `V_cont ≥ t`, t ∈ (0.4988, 0.5002]. It filters real overflows < ≈ 0.5 m³ | Applied |
| Same | "read the volume from the binary `.out`, which is not rounded" | Discarded with `REPORT_STEP` of 3 min (up to 9.5 m³ of error). The source is the pyswmm statistic | Applied |
| Same | "threshold with a margin over the resolution (e.g. 5-10 m³)" | It shifts the band, it does not eliminate it. The threshold is a semantic decision | Applied |
| `2026-10-10-auditoria-completa.md` l. 125, row D-04 (note after table 4.1) | "72 cases … rounding noise" | Same: partially correct, 12 re-simulated | Applied |
| Same l. 186 | "unstable by construction" | Deterministic label. The relation of the FNs with the closeness to the threshold is an interpretation | Applied |
| Same l. 197 | "borderline band (true volume from 1 to 2 m³)" | The only integer in [1, 2) is 1, which corresponds to V_cont ≈ [0.5, 1.5) | Applied |
| Same l. 229, item 10 | "Read the flooded volume with a higher resolution than the `.rpt`" | The integral of the `.out` is not a better source with the current `REPORT_STEP`; only the API statistic is | Applied |
| Same l. 263, assumption 3 | "the label is `volume ≥ 1 m³`" | The effective threshold of ≈ 0.5 m³ continuous, bounded, is added | Applied |
| `anexos/01-hidraulica-swmm.md` l. 111, recommendation of H-06 | "or the integration of the `.out`" | Discarded with the current `REPORT_STEP` | Applied |
| `correcciones/datos/README.md` l. 20 | "A row is flooded if its volume is ≥ 1 m³" | The effective threshold of ≈ 0.5 m³ continuous is added | Applied |
| `2026-10-10-auditoria-completa.md` l. 215 (decision 4: "Is the 1–2 m³ band treated separately?") | Uses "1–2 m³ band" without specifying that it is only the integer 1 | Same clarification as in l. 197 | **Not applied**: outside the scope authorized in this round. The note of l. 197 covers it |
| `anexos/01-hidraulica-swmm.md` H-06 "What happens" and summary table; `anexos/03` l. 195 | "≥ 1 m³ is equivalent to ≥ 0.5 m³ continuous"; count of 72 transitions | Reviewed: they are consistent with what was measured. They require no correction | — |

### 8.2 Text for the thesis (path A)

> The classification target variable is defined from the overflow volume per node reported by SWMM 5.2 in the *Node Flooding Summary* table of the report file (`.rpt`). That table expresses the volume in 10⁶ L with three decimals, so the resolution of the available volume is 1 m³. A node is labelled as flooded (`inunda = 1`) if its reported volume is at least 1 m³. In control re-simulations on the same network (17 scenarios, 15 of them with overflow) the reported volume was compared with the volume that SWMM accumulates at each computation step; the node-by-node comparison was recorded in 15 of those scenarios (13 with overflow). In the nodes with overflows below 100 m³, the value of the report coincided with the rounding to the integer of that continuous volume, except for one case of 83.5 m³. In some nodes with large volumes the report was up to 0.86 m³ below, for an unidentified cause that does not affect the label. Consequently, in that sample the rule is consistent with a threshold of ≈ 0.5 m³ on the continuous volume, bounded to the interval (0.4988, 0.5002] m³. Overflows below that threshold (median of 0.15 m³ in the control scenarios) are considered non-flooding. The label is, therefore, a deterministic quantization of the simulated volume, not a source of noise. That this behaviour holds across the whole dataset is an inference based on the rounding mechanism of the report and was not verified case by case. The nodes with a volume close to the threshold are, moreover, the hardest to classify: in the held-out-shape validation (LeaveOneShapeOut), 333 of the 380 false negatives of XGBoost correspond to nodes with a reported volume of 1 m³ (continuous volume of approximately 0.5 to 1.5 m³). For that reason, in addition to the pooled confusion matrix, the matrix separated by true volume class is reported and, as a sensitivity analysis, the metrics without that band (F1 of 0.9817 with all rows and 0.9847 without the band). This sensitivity excludes only the positive side of the boundary, because overflows of 0 to 0.5 m³ cannot be distinguished from dry nodes in the available data.

---

## 9. Scripts used and exact commands

All of them are in the session scratchpad (`<scratch>` = `C:/Users/LF409F~1.FAJ/AppData/Local/Temp/claude/c--Users-lf-fajardo-Downloads-Tesis-herramienta-ia-model/288ddaf2-bd7c-4ac1-bbe9-c4293e922cdc/scratchpad/spike_d04`). The copies of `.inp`, `.rpt` and `.out` remained in `<scratch>/runs/`. They are read-only with respect to the repository and their complete code is in the Appendix. They are run from `<scratch>`, with `<repo>` = `C:/Users/lf.fajardo/Downloads/Tesis/herramienta-ia-model/.claude/worktrees/optuna-tuning-familia-ganadora` and `<py>` = `C:/Users/lf.fajardo/Downloads/Tesis/herramienta-ia-model/venv/Scripts/python.exe`.

| Script | Command | Simulations | What it produces |
|---|---|---|---|
| `s01_control.py` | `<py> s01_control.py <repo>` | 6 | rpt, out and API per node; reproduction against the DB; times (`s01_nodos.csv`, `s01_resumen.csv`, `s01_tiempos.csv`) |
| `s02_reportstep.py` | `<py> s02_reportstep.py <repo>` | 1 | Convergence of the `.out` with `REPORT_STEP` of 5 s (`s02_nodos.csv`) |
| `s03_costo.py` | `<py> s03_costo.py <repo>` | 0 | Extrapolation of the cost to 1000 runs |
| `s04_oof_banda.py` | `<py> s04_oof_banda.py <repo>` | 0 | Matrix by band and metrics with and without band from the OOFs (`s04_metricas.csv`, `s04_matriz_por_banda.csv`) |
| `s05_equivalencia.py` | `<py> s05_equivalencia.py` | 0 | Equivalence `rpt ≥ 1` ⇔ `V_cont ≥ 0.5`; `rpt = round(V_cont)` |
| `s06_no_monotonia.py` | `<py> s06_no_monotonia.py <repo>` | 2 | Non-monotonic cases of `common_bogota` f = 3.4 → 3.6 with continuous volume |
| `s07_no_monotonia_ampliada.py` | `<py> s07_no_monotonia_ampliada.py <repo>` | 9 | 9 more transitions (including the 4 cases with 2 m³); transitions with `V_cont ≥ 1` (`s07_nodos.csv`, `s07_transiciones.csv`) |
| `s08_cota_y_regresor.py` | `<py> s08_cota_y_regresor.py <repo>` | 0 | Breakdown of cases and bound of the effective threshold; regressor error per band from the OOFs; Wilson CI |
| `s09_notas.py` | `<py> s09_notas.py <repo>` | 0 | Inserts the correction notes of §8.1 into the earlier documents (writes only in `docs/auditoria/`) |

Execution order: `s01` → `s02` → `s03` → `s04` → `s05` → `s06` → `s07` → `s08` → `s09`. `s02`, `s03`, `s04`, `s05` and `s08` read outputs of `s01` (and `s08` also of `s07`). Total: 18 SWMM simulations, each of less than 2 s.

---

## Appendix — Code of the scripts

### `s01_control.py`

```python
"""Spike D-04/H-06: volumen de inundacion por nodo segun tres fuentes en escenarios de control.

Fuentes comparadas, nodo a nodo:
  rpt  : Node Flooding Summary del .rpt leido por el pipeline (extract_labels -> swmm_api), m3
  out  : integral temporal de la serie 'flooding' (L/s) de los nodos en el .out binario
         (swmm_api.SwmmOutput), con paso REPORT_STEP; trapecio, rectangulo izq. y der.
  api  : pyswmm node.statistics['flooding_volume'] (acumulado interno de SWMM en cada paso de routing)

Construye los .inp con las MISMAS funciones del pipeline (write_scaled_inp para 'base';
apply_shape + apply_closure_policy + write_scenario_inp(drain_down 6 h) para las formas CSV),
sobre COPIAS en ./runs del directorio de trabajo (scratchpad). Solo lectura sobre el repo.

Uso: python s01_control.py <repo>   (ejecutar desde spike_d04)
Salida: s01_nodos.csv (una fila por escenario y nodo), s01_resumen.csv, s01_tiempos.csv
"""
import sys, time, re, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
repo = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(repo))
import numpy as np, pandas as pd
from pyswmm import Simulation, Nodes
from swmm_api import SwmmOutput, read_inp_file
from swmm_resilience.simulation.swmm_api_io import write_scaled_inp, read_continuity_error
from swmm_resilience.simulation.hydrograph_shapes import load_shape, apply_shape
from swmm_resilience.simulation.inflow_closure import apply_closure_policy
from swmm_resilience.simulation.timeseries_scenario import write_scenario_inp
from swmm_resilience.validation.hydrograph_csv import HydrographScenario
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.extraction.labels import extract_labels, _parse_node_flooding_text

INP = repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
SHAPES = repo / "data/hydrograph_shapes"
PREP = repo / "outputs/bench/prepared_v4_21/ca706e27654e0cf5"
OUT = Path("runs"); OUT.mkdir(exist_ok=True)

opts = read_inp_file(str(INP), encoding="utf-8")["OPTIONS"]
print("FLOW_UNITS:", opts["FLOW_UNITS"], " REPORT_STEP:", opts["REPORT_STEP"], " ROUTING_STEP:", opts["ROUTING_STEP"], " ALLOW_PONDING:", opts["ALLOW_PONDING"])
assert str(opts["FLOW_UNITS"]).upper() == "LPS"

static = extract_static_features(INP)
base_inflows = dict(zip(static.node_id.astype(str), static.base_inflow_lps))
nodes = static.node_id.astype(str).tolist()

keys = pd.read_parquet(PREP / "keys.parquet"); tg = pd.read_parquet(PREP / "targets.parquet")
db = pd.concat([keys[["shape_id", "factor_mult", "node_id"]], tg], axis=1)
db["factor_mult"] = db.factor_mult.round(3)

CASES = [("base", 0.2), ("base", 3.0), ("shape_ultra_short_impulse", 5.0),
         ("shape_micro_burst_45min", 3.0), ("long_moderate", 1.0), ("long_moderate", 5.0)]


def build(sid, f):
    if sid == "base":
        p = OUT / f"base_f{f:.3f}.inp"
        write_scaled_inp(str(INP), f, None, p, scenario_mode="timeseries")
        return p
    shape = load_shape(SHAPES / f"{sid}.csv")
    sc = HydrographScenario(f"{sid}_f{f:.3f}", apply_shape(shape, base_inflows, f), [t for t, _ in shape], shape[-1][0])
    sc, _ = apply_closure_policy(sc, 0.01)  # config.validation.inflow_closure_tolerance_lps
    return write_scenario_inp(INP, sc, OUT, drain_down_hours=6.0)


def out_integrals(out_path):
    with SwmmOutput(str(out_path)) as o:
        fl = o.get_part(kind="node", variable="flooding")  # L/s (FLOW_UNITS LPS)
        idx = o.index
    t = np.array([(x - idx[0]).total_seconds() for x in idx])
    step = float(np.median(np.diff(t)))
    # SWMM escribe el primer registro en START + REPORT_STEP? se reporta t0 para documentarlo
    q = fl[[c for c in fl.columns if c in nodes]].to_numpy(dtype=float)
    # se antepone q(t0 - step)=0 solo si el primer registro no es t=0 de la simulacion: se documenta abajo
    trap = np.trapz(q, t, axis=0) / 1000.0
    rect_r = (q[1:] * np.diff(t)[:, None]).sum(0) / 1000.0
    rect_l = (q[:-1] * np.diff(t)[:, None]).sum(0) / 1000.0
    cols = [c for c in fl.columns if c in nodes]
    return (pd.DataFrame({"vol_out_trap": trap, "vol_out_rect_der": rect_r, "vol_out_rect_izq": rect_l}, index=cols),
            dict(n_registros=len(t), report_step_s=step, primer_registro=str(idx[0]), ultimo_registro=str(idx[-1])))


def flooding_loss(rpt):
    t = Path(rpt).read_text(errors="replace")
    m = re.search(r"Flooding Loss \.+\s+(-?[\d.]+)\s+(-?[\d.]+)", t)
    return float(m.group(2)) * 1000 if m else float("nan")


allrows, summ, times = [], [], []
for sid, f in CASES:
    t0 = time.perf_counter(); p = build(sid, f); t_build = time.perf_counter() - t0
    api = {}
    t1 = time.perf_counter()
    with Simulation(str(p)) as sim:
        for _ in sim:
            pass
        t_loop = time.perf_counter() - t1
        for n in Nodes(sim):
            if n.is_junction():
                api[n.nodeid] = n.statistics["flooding_volume"]
    t_sim = time.perf_counter() - t1
    rpt = p.with_suffix(".rpt"); outp = p.with_suffix(".out")
    t2 = time.perf_counter(); lab = extract_labels(rpt, nodes, 1.0).set_index("node_id"); t_lab = time.perf_counter() - t2
    txt = _parse_node_flooding_text(rpt).set_index("node_id")["flooding_volume_m3"]
    oi, meta = out_integrals(outp)
    d = lab.rename(columns={"vol_inundacion_m3": "vol_rpt", "inunda": "inunda_actual"})
    d["vol_api"] = pd.Series(api)
    d = d.join(oi)
    d["vol_parser_texto"] = txt.reindex(d.index).fillna(0.0)
    d = d.fillna({"vol_out_trap": 0.0, "vol_out_rect_der": 0.0, "vol_out_rect_izq": 0.0})
    dbs = db[(db.shape_id == sid) & (db.factor_mult == round(f, 3))].set_index("node_id")
    d["vol_bd"] = dbs.vol_inundacion_m3.reindex(d.index)
    d.insert(0, "shape_id", sid); d.insert(1, "factor", f)
    allrows.append(d.reset_index().rename(columns={"index": "node_id"}))
    ce = read_continuity_error(rpt)
    times.append(dict(shape_id=sid, factor=f, t_build_s=t_build, t_sim_loop_s=t_loop, t_sim_total_s=t_sim,
                      t_labels_s=t_lab, inp_kb=p.stat().st_size / 1024, out_mb=outp.stat().st_size / 2**20,
                      rpt_kb=rpt.stat().st_size / 1024, **meta))
    print(f"{sid} f={f}: CE={ce}  sim={t_sim:.2f}s  build={t_build:.2f}s  out={meta}", flush=True)
    cont = d.vol_api
    lab_i = (d.vol_rpt >= 1.0); lab_ii = (cont >= 1.0); lab_iii = (cont > 0)
    any_fl = (d.vol_rpt > 0) | (cont > 0) | (d.vol_out_trap > 0)
    e_ro = (d.vol_rpt - d.vol_out_trap)[any_fl].abs(); e_ra = (d.vol_rpt - d.vol_api)[any_fl].abs()
    e_oa = (d.vol_out_trap - d.vol_api)[any_fl].abs()
    summ.append(dict(
        shape_id=sid, factor=f, continuity_error_pct=ce, en_bd=bool(len(dbs)),
        bd_igual_rpt=(int((d.vol_bd == d.vol_rpt).sum()) if len(dbs) else None),
        n_nodos_con_desborde=int(any_fl.sum()),
        sum_rpt=d.vol_rpt.sum(), sum_api=cont.sum(), sum_out_trap=d.vol_out_trap.sum(), flooding_loss_rpt=flooding_loss(rpt),
        sum_parser_texto=d.vol_parser_texto.sum(),
        max_abs_rpt_out=e_ro.max() if len(e_ro) else 0, med_abs_rpt_out=e_ro.median() if len(e_ro) else 0,
        max_abs_rpt_api=e_ra.max() if len(e_ra) else 0, med_abs_rpt_api=e_ra.median() if len(e_ra) else 0,
        max_abs_out_api=e_oa.max() if len(e_oa) else 0, med_abs_out_api=e_oa.median() if len(e_oa) else 0,
        max_rel_out_api_v_ge1=((d.vol_out_trap - cont).abs() / cont)[cont >= 1].max() if (cont >= 1).any() else 0,
        n_api_0_05=int(((cont > 0) & (cont < 0.5)).sum()), n_api_05_1=int(((cont >= 0.5) & (cont < 1)).sum()),
        n_api_1_2=int(((cont >= 1) & (cont < 2)).sum()),
        n_lab_i=int(lab_i.sum()), n_lab_ii=int(lab_ii.sum()), n_lab_iii=int(lab_iii.sum()),
        n_lab_iii_gt_0p01=int((cont > 0.01).sum()),
        cambia_i_vs_ii=int((lab_i != lab_ii).sum()), i1_ii0=int((lab_i & ~lab_ii).sum()), i0_ii1=int((~lab_i & lab_ii).sum()),
        cambia_i_vs_iii=int((lab_i != lab_iii).sum()),
        cambia_i_vs_out_ge1=int((lab_i != (d.vol_out_trap >= 1.0)).sum()),
        min_api_positivo=cont[cont > 0].min() if (cont > 0).any() else None,
    ))

nd = pd.concat(allrows, ignore_index=True); nd.to_csv("s01_nodos.csv", index=False)
sm = pd.DataFrame(summ); sm.to_csv("s01_resumen.csv", index=False)
tm = pd.DataFrame(times); tm.to_csv("s01_tiempos.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60)
print(sm.T); print(tm.T)
fl = nd[(nd.vol_rpt > 0) | (nd.vol_api > 0) | (nd.vol_out_trap > 0)]
print("GLOBAL nodos con desborde:", len(fl))
for a, b in [("vol_rpt", "vol_api"), ("vol_rpt", "vol_out_trap"), ("vol_out_trap", "vol_api"), ("vol_out_rect_der", "vol_api"), ("vol_out_rect_izq", "vol_api")]:
    e = (fl[a] - fl[b]).abs(); print(f"  |{a}-{b}|: max={e.max():.4f} mediana={e.median():.4f} p95={e.quantile(.95):.4f}")
print("  |rpt-api| > 0.5:", int(((fl.vol_rpt - fl.vol_api).abs() > 0.5).sum()))
print("  rpt - round(api) distinto de 0 (nodos):", int((fl.vol_rpt != fl.vol_api.round(0)).sum()))
```

### `s02_reportstep.py`

```python
"""Spike D-04: la integral del .out converge al volumen interno de SWMM si REPORT_STEP baja a 5 s?
Toma la COPIA runs/shape_ultra_short_impulse_f5.000.inp creada por s01, cambia solo REPORT_STEP a 0:00:05
(copia nueva runs/rs5_ultra_f5.inp), simula una vez y compara rpt / out(5 s) / api nodo a nodo.
Uso: python s02_reportstep.py <repo>   (despues de s01, desde spike_d04)"""
import sys, time, warnings, datetime
from pathlib import Path
warnings.filterwarnings("ignore")
repo = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(repo))
import numpy as np, pandas as pd
from pyswmm import Simulation, Nodes
from swmm_api import SwmmOutput, read_inp_file
from swmm_resilience.extraction.labels import extract_labels

src = Path("runs/shape_ultra_short_impulse_f5.000.inp"); dst = Path("runs/rs5_ultra_f5.inp")
inp = read_inp_file(str(src), encoding="utf-8")
inp["OPTIONS"]["REPORT_STEP"] = datetime.time(0, 0, 5)
inp.write_file(str(dst))
nodes = [str(n) for n in inp["JUNCTIONS"].keys()]
api = {}
t1 = time.perf_counter()
with Simulation(str(dst)) as sim:
    for _ in sim:
        pass
    for n in Nodes(sim):
        if n.is_junction():
            api[n.nodeid] = n.statistics["flooding_volume"]
t_sim = time.perf_counter() - t1
with SwmmOutput(str(dst.with_suffix(".out"))) as o:
    fl = o.get_part(kind="node", variable="flooding"); idx = o.index
t = np.array([(x - idx[0]).total_seconds() for x in idx]) + (idx[0] - datetime.datetime.combine(idx[0].date(), datetime.time(0))).total_seconds()
q = fl[nodes].to_numpy(float)
# se antepone t=0 con q=0 (SWMM no escribe el instante inicial)
t = np.r_[0.0, t]; q = np.vstack([np.zeros(q.shape[1]), q])
vout = pd.Series(np.trapz(q, t, axis=0) / 1000.0, index=nodes)
lab = extract_labels(dst.with_suffix(".rpt"), nodes, 1.0).set_index("node_id")
d = pd.DataFrame({"vol_rpt": lab.vol_inundacion_m3, "vol_api": pd.Series(api), "vol_out5s": vout})
prev = pd.read_csv("s01_nodos.csv"); prev = prev[prev.shape_id == "shape_ultra_short_impulse"].set_index("node_id")
d["vol_out180s"] = prev.vol_out_trap
d["vol_api_s01"] = prev.vol_api
m = (d[["vol_rpt", "vol_api", "vol_out5s"]] > 0).any(axis=1)
print(f"REPORT_STEP=5 s: t_sim={t_sim:.2f}s  registros={len(idx)}  out_MB={dst.with_suffix('.out').stat().st_size/2**20:.1f}")
print("api identico a s01 (REPORT_STEP no altera el calculo):", float((d.vol_api - d.vol_api_s01).abs().max()))
print("rpt identico a s01:", float((d.vol_rpt - prev.vol_rpt).abs().max()))
for a in ["vol_out5s", "vol_out180s", "vol_rpt"]:
    e = (d[a] - d.vol_api)[m].abs(); print(f"|{a}-api|: max={e.max():.4f} mediana={e.median():.4f}")
print("cambios de etiqueta (>=1) out5s vs api:", int(((d.vol_out5s >= 1) != (d.vol_api >= 1)).sum()))
d.to_csv("s02_nodos.csv")
```

### `s03_costo.py`

```python
"""Spike D-04: extrapola el costo de re-simular las 1000 corridas (40 formas x 25 factores) a partir
de los tiempos medidos en s01 (s01_tiempos.csv). No simula nada.
Modelo: t_corrida = a + b * horas_simuladas (minimos cuadrados sobre las 6 corridas de control).
Ademas mide (sin simular) el costo de la lectura de drenaje del .out que hace run_batch_shapes.
Uso: python s03_costo.py <repo>   (desde spike_d04, despues de s01)"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
repo = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(repo))
import numpy as np, pandas as pd
from swmm_resilience.simulation.hydrograph_shapes import load_all_shapes
from swmm_resilience.validation.drainage_state import read_tail_window_observables

tm = pd.read_csv("s01_tiempos.csv")
H = {"base": 3.0}
shapes = load_all_shapes(repo / "data/hydrograph_shapes")
from swmm_resilience.simulation.inflow_closure import close_series_to_zero
for sid, sh in shapes.items():
    # la politica de cierre (inflow_closure) agrega un paso de malla si la forma no termina en 0
    closed, _ = close_series_to_zero(sh, 1e-9)
    H[sid] = closed[-1][0] + 6.0
tm["horas_sim"] = tm.shape_id.map(H)
# costo de lectura de drenaje del .out (lo hace run_batch_shapes en cada corrida de forma CSV)
t_dr = []
for _, r in tm.iterrows():
    if r.shape_id == "base":
        continue
    p = Path("runs") / f"{r.shape_id}_f{r.factor:.3f}.out"
    t0 = time.perf_counter(); read_tail_window_observables(p, 6.0, 1.0); t_dr.append(time.perf_counter() - t0)
t_drain = float(np.mean(t_dr))
tm["t_corrida_s"] = tm.t_build_s + tm.t_sim_total_s + tm.t_labels_s + np.where(tm.shape_id == "base", 0.0, t_drain)
X = np.c_[np.ones(len(tm)), tm.horas_sim]
for col in ("t_sim_total_s", "t_corrida_s"):
    a, b = np.linalg.lstsq(X, tm[col], rcond=None)[0]
    pred = a + b * np.array([H[s] for s in H])
    tot = 25 * pred.sum()
    print(f"{col}: a={a:.3f}s b={b:.4f}s/h  residuo max={np.abs(X@[a,b]-tm[col]).max():.3f}s  -> 1000 corridas: {tot:.0f} s = {tot/60:.1f} min")
print("horas simuladas por forma: min %.2f med %.2f max %.2f; suma 40 formas = %.1f h" % (min(H.values()), np.median(list(H.values())), max(H.values()), sum(H.values())))
print(f"lectura de drenaje del .out por corrida: {t_drain:.3f}s (media de {len(t_dr)})")
print("tiempo medio de simulacion de las 6 corridas: %.2f s (min %.2f, max %.2f)" % (tm.t_sim_total_s.mean(), tm.t_sim_total_s.min(), tm.t_sim_total_s.max()))
print("overhead de leer node.statistics tras el bucle: max %.3f s" % (tm.t_sim_total_s - tm.t_sim_loop_s).max())
print(tm[["shape_id", "factor", "horas_sim", "t_build_s", "t_sim_total_s", "t_labels_s", "t_corrida_s", "out_mb"]].round(3).to_string(index=False))
```

### `s04_oof_banda.py`

```python
"""Spike D-04, ruta A: efecto de la banda ambigua en la evaluacion, usando SOLO los OOF persistidos
(sin reentrenar). Solo lectura sobre outputs/.
- Verifica que la matriz de confusion reconstruida coincide con docs/.../matriz_confusion.csv.
- Matriz de confusion separada por clase de volumen real (.rpt): 0 | [1,2) | [2,5) | [5,100) | >=100.
- Metricas agregadas excluyendo las filas con volumen real en [0.5, 2) m3 (banda ambigua).
- Ruta B (proyeccion): con s01_nodos.csv, fraccion de nodos con vol .rpt = 1 cuyo volumen continuo es < 1.
Uso: python s04_oof_banda.py <repo>   (desde spike_d04)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
repo = Path(sys.argv[1]).resolve()
PREP = repo / "outputs/bench/prepared_v4_21/ca706e27654e0cf5"
OOF = repo / "outputs/bench/oof/ca706e27654e0cf5"
CM = pd.read_csv(repo / "docs/auditoria/correcciones/datos/matriz_confusion.csv")
keys = pd.read_parquet(PREP / "keys.parquet"); tg = pd.read_parquet(PREP / "targets.parquet")
base = pd.concat([keys[["shape_id", "factor_mult", "node_id"]], tg], axis=1)
base["sample_idx"] = np.arange(len(base))
v = base.vol_inundacion_m3
print("filas:", len(base), " positivas:", int(tg.inunda.sum()), " vol==1.0:", int((v == 1.0).sum()),
      " vol en [0.5,2):", int(((v >= 0.5) & (v < 2)).sum()), " vol en (0,1):", int(((v > 0) & (v < 1)).sum()),
      " no enteros >0:", int(((v > 0) & (v != v.round())).sum()))
print("inunda == (vol>=1):", bool((tg.inunda == (v >= 1.0)).all()))
# comprobacion de reproduccion de las 4 corridas de control presentes en la BD
for sid, f in [("base", 3.0), ("shape_ultra_short_impulse", 5.0), ("shape_micro_burst_45min", 3.0), ("long_moderate", 5.0)]:
    s = base[(base.shape_id == sid) & (base.factor_mult.round(3) == f)]
    print(f"  BD {sid} f={f}: suma vol={s.vol_inundacion_m3.sum():.0f} m3, n_inunda={int(s.inunda.sum())}")

bins = [-np.inf, 0.5, 2.0, 5.0, 100.0, np.inf]
labs = ["0", "[1,2)", "[2,5)", "[5,100)", ">=100"]
base["clase_vol"] = pd.cut(v, bins, right=False, labels=labs)


def metr(y, p):
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum()); fn = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    pr = tp / (tp + fp) if tp + fp else float("nan"); rc = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, f1=f1, recall=rc, precision=pr)


rows, bandrows = [], []
for model in ["xgboost", "baseline_interpolacion_factor", "baseline_forma_mas_cercana"]:
    o = pd.read_parquet(OOF / f"{model}.parquet")
    for prot, g in o.groupby("protocol"):
        d = g.merge(base, on="sample_idx", how="left")
        assert len(d) == len(base) and d.sample_idx.is_unique
        y = d.inunda.to_numpy(); p = d.y_pred_clf.to_numpy()
        full = metr(y, p)
        ref = CM[(CM.modelo == model) & (CM.protocolo == prot)]
        ok = (len(ref) == 1) and all(int(ref[k].iloc[0]) == full[k] for k in ("tn", "fp", "fn", "tp"))
        amb = ((d.vol_inundacion_m3 >= 0.5) & (d.vol_inundacion_m3 < 2.0)).to_numpy()
        excl = metr(y[~amb], p[~amb])
        rows.append(dict(modelo=model, protocolo=prot, variante="todas", coincide_csv=ok, n=len(y), **full))
        rows.append(dict(modelo=model, protocolo=prot, variante="sin_banda_[0.5,2)", coincide_csv=None, n=int((~amb).sum()), **excl))
        for c in labs:
            m = (d.clase_vol == c).to_numpy()
            bandrows.append(dict(modelo=model, protocolo=prot, clase_vol_real=c, n=int(m.sum()),
                                 pred_0=int((p[m] == 0).sum()), pred_1=int((p[m] == 1).sum())))
        if model == "xgboost":
            fpm = (y == 0) & (p == 1); fnm = (y == 1) & (p == 0)
            pr = d.y_pred_reg.to_numpy()
            print(f"{model} {prot}: FP={int(fpm.sum())}, de ellos con vol predicho <2 m3: {int((fpm & (pr < 2)).sum())}, <5 m3: {int((fpm & (pr < 5)).sum())}; "
                  f"FN={int(fnm.sum())}, en [1,2): {int((fnm & amb).sum())}; prob media de FN en banda={d.y_prob_clf[fnm & amb].mean():.3f}")
            # por forma: cuantos FN de banda por forma (top 5) en LeaveOneShapeOut
            if prot == "LeaveOneShapeOut":
                print("   FN en banda por forma (top 5):", d[fnm & amb].shape_id.value_counts().head(5).to_dict())
                print("   FP por forma (top 5):", d[fpm].shape_id.value_counts().head(5).to_dict())

res = pd.DataFrame(rows); bm = pd.DataFrame(bandrows)
res.to_csv("s04_metricas.csv", index=False); bm.to_csv("s04_matriz_por_banda.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(res.round(4).to_string(index=False)); print(bm.to_string(index=False))

# --- Ruta B: proyeccion con los escenarios de control (s01) ---
nd = pd.read_csv("s01_nodos.csv")
r1 = nd[nd.vol_rpt == 1.0]; r0 = nd[nd.vol_rpt == 0.0]; r2 = nd[nd.vol_rpt == 2.0]
print(f"Control: nodos con rpt=1: {len(r1)}, de ellos continuo<1: {int((r1.vol_api < 1).sum())} "
      f"(rango continuo {r1.vol_api.min():.3f}-{r1.vol_api.max():.3f}); rpt=2: {len(r2)}, continuo<1: {int((r2.vol_api < 1).sum())}; "
      f"rpt=0: {len(r0)}, continuo>0: {int((r0.vol_api > 0).sum())}, continuo en [0.5,1): {int(((r0.vol_api >= .5) & (r0.vol_api < 1)).sum())}")
print("Control: nodos con rpt=0 y continuo>0, por escenario:", r0[r0.vol_api > 0].groupby(["shape_id", "factor"]).size().to_dict())
print("Control: nodos con rpt=0 (en escenarios con algun desborde):", r0[r0.groupby(["shape_id", "factor"]).vol_api.transform("max") > 0].groupby(["shape_id", "factor"]).size().to_dict())
```

### `s05_equivalencia.py`

```python
"""Spike D-04: la etiqueta actual (rpt >= 1) es una funcion determinista del volumen continuo?
Comprueba sobre s01_nodos.csv (6 escenarios x 160 nodos) que (rpt>=1) == (continuo>=0.5) y que rpt == round(continuo).
Uso: python s05_equivalencia.py   (desde spike_d04, despues de s01)"""
import pandas as pd, numpy as np
d = pd.read_csv("s01_nodos.csv")
print("filas nodo-escenario:", len(d))
print("discrepancias (rpt>=1) vs (api>=0.5):", int(((d.vol_rpt >= 1) != (d.vol_api >= 0.5)).sum()))
for lim in (10, 100, 1e9):
    m = d.vol_api < lim
    print(f"api<{lim:g}: n={int(m.sum())}, rpt != round(api): {int((d.vol_rpt[m] != np.round(d.vol_api[m])).sum())}")
big = d[d.vol_api >= 100]
print("api>=100: |rpt-api|/api max = %.2e" % ((big.vol_rpt - big.vol_api).abs() / big.vol_api).max())
g = d.groupby(["shape_id", "factor"])
t = pd.DataFrame({"desbordan_api_gt0": g.apply(lambda x: int((x.vol_api > 0).sum())),
                  "etiqueta_actual_1": g.apply(lambda x: int((x.vol_rpt >= 1).sum())),
                  "desbordan_con_etiqueta_0": g.apply(lambda x: int(((x.vol_api > 0) & (x.vol_rpt < 1)).sum()))})
t["pct_ocultos"] = (100 * t.desbordan_con_etiqueta_0 / t.desbordan_api_gt0).round(1)
print(t.to_string())
oc = d[(d.vol_api > 0) & (d.vol_rpt < 1)]
print("volumen continuo de los ocultos: max %.3f, suma %.1f m3, mediana %.3f" % (oc.vol_api.max(), oc.vol_api.sum(), oc.vol_api.median()))
```

### `s06_no_monotonia.py`

```python
"""Spike D-04: los 72 casos 'deja de inundar al subir el factor' (D-04, anexo 03) son ruido de redondeo
o el volumen continuo tambien baja? Simula common_bogota f=3.4 y f=3.6 (2 corridas, copias en ./runs)
y compara nodo a nodo rpt vs volumen continuo (pyswmm node.statistics['flooding_volume']).
Uso: python s06_no_monotonia.py <repo>   (desde spike_d04)"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
repo = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(repo))
import pandas as pd
from pyswmm import Simulation, Nodes
from swmm_resilience.simulation.hydrograph_shapes import load_shape, apply_shape
from swmm_resilience.simulation.inflow_closure import apply_closure_policy
from swmm_resilience.simulation.timeseries_scenario import write_scenario_inp
from swmm_resilience.validation.hydrograph_csv import HydrographScenario
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.extraction.labels import extract_labels
INP = repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
PREP = repo / "outputs/bench/prepared_v4_21/ca706e27654e0cf5"
static = extract_static_features(INP); nodes = static.node_id.astype(str).tolist()
base_inflows = dict(zip(static.node_id.astype(str), static.base_inflow_lps))
db = pd.concat([pd.read_parquet(PREP / "keys.parquet")[["shape_id", "factor_mult", "node_id"]], pd.read_parquet(PREP / "targets.parquet")], axis=1)
sid = "common_bogota"; res = {}
for f in (3.4, 3.6):
    shape = load_shape(repo / "data/hydrograph_shapes" / f"{sid}.csv")
    sc = HydrographScenario(f"{sid}_f{f:.3f}", apply_shape(shape, base_inflows, f), [t for t, _ in shape], shape[-1][0])
    sc, _ = apply_closure_policy(sc, 0.01)
    p = write_scenario_inp(INP, sc, Path("runs"), drain_down_hours=6.0)
    api = {}
    with Simulation(str(p)) as sim:
        for _ in sim:
            pass
        for n in Nodes(sim):
            if n.is_junction():
                api[n.nodeid] = n.statistics["flooding_volume"]
    lab = extract_labels(p.with_suffix(".rpt"), nodes, 1.0).set_index("node_id")
    bd = db[(db.shape_id == sid) & (db.factor_mult.round(3) == f)].set_index("node_id").vol_inundacion_m3
    res[f] = pd.DataFrame({"rpt": lab.vol_inundacion_m3, "api": pd.Series(api), "bd": bd})
    print(f"f={f}: BD==rpt en {int((res[f].bd == res[f].rpt).sum())}/160 nodos; suma rpt={res[f].rpt.sum():.0f}")
a, b = res[3.4], res[3.6]
cand = a[(a.rpt >= 1) & (b.rpt < 1)].index
print("nodos con etiqueta 1 a f=3.4 y 0 a f=3.6:", list(cand))
print(pd.DataFrame({"rpt_3.4": a.rpt[cand], "cont_3.4": a.api[cand].round(4), "rpt_3.6": b.rpt[cand], "cont_3.6": b.api[cand].round(4)}).to_string())
dec = (b.api < a.api) & (a.api > 0)
print("nodos cuyo volumen continuo BAJA de f=3.4 a f=3.6:", int(dec.sum()), "de", int((a.api > 0).sum()), "con desborde a f=3.4")
```

### `s07_no_monotonia_ampliada.py`

```python
"""Spike D-04, ronda de correcciones: re-simula mas transiciones 'deja de inundar al subir el factor'
(las 72 de la BD: inunda=1 a f y vol<1 a f+0.2, misma forma y nodo) y mide el volumen continuo
(pyswmm node.statistics['flooding_volume']) antes y despues. 9 corridas SWMM sobre copias en ./runs:
  rare_extreme f = 3.8, 4.0, 4.2, 4.4, 4.6  (transiciones 3.8->4.0, 4.0->4.2, 4.2->4.4, 4.4->4.6)
  shape_ultra_short_impulse f = 4.4, 4.6     (4.4->4.6)
  common_bogota f = 4.4, 4.6                 (4.4->4.6)
Incluye los 4 casos con vol .rpt = 2.0. Tambien guarda todos los nodos (s07_nodos.csv) para acotar el
umbral efectivo y contar transiciones no monotonas con el umbral continuo de 1 m3.
Uso: python s07_no_monotonia_ampliada.py <repo>   (desde spike_d04)"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
repo = Path(sys.argv[1]).resolve(); sys.path.insert(0, str(repo))
import pandas as pd
from pyswmm import Simulation, Nodes
from swmm_resilience.simulation.hydrograph_shapes import load_shape, apply_shape
from swmm_resilience.simulation.inflow_closure import apply_closure_policy
from swmm_resilience.simulation.timeseries_scenario import write_scenario_inp
from swmm_resilience.validation.hydrograph_csv import HydrographScenario
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.extraction.labels import extract_labels
INP = repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
PREP = repo / "outputs/bench/prepared_v4_21/ca706e27654e0cf5"
static = extract_static_features(INP); nodes = static.node_id.astype(str).tolist()
base_inflows = dict(zip(static.node_id.astype(str), static.base_inflow_lps))
db = pd.concat([pd.read_parquet(PREP / "keys.parquet")[["shape_id", "factor_mult", "node_id"]],
                pd.read_parquet(PREP / "targets.parquet")], axis=1)
db["f5"] = (db.factor_mult * 5).round().astype(int)

PLAN = {"rare_extreme": [3.8, 4.0, 4.2, 4.4, 4.6], "shape_ultra_short_impulse": [4.4, 4.6], "common_bogota": [4.4, 4.6]}
res = {}
for sid, fs in PLAN.items():
    shape = load_shape(repo / "data/hydrograph_shapes" / f"{sid}.csv")
    for f in fs:
        sc = HydrographScenario(f"{sid}_f{f:.3f}", apply_shape(shape, base_inflows, f), [t for t, _ in shape], shape[-1][0])
        sc, _ = apply_closure_policy(sc, 0.01)
        p = write_scenario_inp(INP, sc, Path("runs"), drain_down_hours=6.0)
        api = {}
        with Simulation(str(p)) as sim:
            for _ in sim:
                pass
            for n in Nodes(sim):
                if n.is_junction():
                    api[n.nodeid] = n.statistics["flooding_volume"]
        lab = extract_labels(p.with_suffix(".rpt"), nodes, 1.0).set_index("node_id")
        bd = db[(db.shape_id == sid) & (db.f5 == round(f * 5))].set_index("node_id").vol_inundacion_m3
        d = pd.DataFrame({"vol_rpt": lab.vol_inundacion_m3, "vol_api": pd.Series(api), "vol_bd": bd.reindex(lab.index)})
        res[(sid, f)] = d
        print(f"{sid} f={f}: BD==rpt en {int((d.vol_bd == d.vol_rpt).sum())}/160; suma rpt={d.vol_rpt.sum():.0f}", flush=True)

allnodes = pd.concat({k: v for k, v in res.items()}, names=["shape_id", "factor", "node_id"]).reset_index()
allnodes.to_csv("s07_nodos.csv", index=False)

rows, nm_cont = [], []
for sid, fs in PLAN.items():
    for f1, f2 in zip(fs[:-1], fs[1:]):
        a, b = res[(sid, f1)], res[(sid, f2)]
        for n in a.index[(a.vol_bd >= 1) & (b.vol_bd < 1)]:  # transiciones de la BD
            rows.append(dict(shape_id=sid, f1=f1, f2=f2, node_id=n, rpt_f1=a.vol_rpt[n], cont_f1=a.vol_api[n],
                             rpt_f2=b.vol_rpt[n], cont_f2=b.vol_api[n], caida=a.vol_api[n] - b.vol_api[n]))
        m = (a.vol_api >= 1) & (b.vol_api < 1)
        nm_cont.append(dict(shape_id=sid, f1=f1, f2=f2, transiciones_rpt=int(((a.vol_rpt >= 1) & (b.vol_rpt < 1)).sum()),
                            transiciones_cont_ge1=int(m.sum()), nodos=list(a.index[m]),
                            nodos_cont_baja=int(((b.vol_api < a.vol_api) & (a.vol_api > 0)).sum()),
                            nodos_desbordan_f1=int((a.vol_api > 0).sum())))
t = pd.DataFrame(rows); t.to_csv("s07_transiciones.csv", index=False)
pd.set_option("display.width", 250)
print(t.round(4).to_string(index=False))
print("n transiciones re-simuladas:", len(t), " caida>=0.3:", int((t.caida >= 0.3).sum()), " caida<0.1:", int((t.caida < 0.1).sum()),
      " 0.1<=caida<0.3:", int(((t.caida >= 0.1) & (t.caida < 0.3)).sum()))
print(pd.DataFrame(nm_cont).to_string(index=False))
```

### `s08_cota_y_regresor.py`

```python
"""Spike D-04, ronda de correcciones (sin simular):
(1) Alcance de la equivalencia rpt = round(V_cont): desglose de los casos nodo-escenario (secos, 0<V<100, >=100)
    y cota del umbral efectivo t tal que (V_rpt >= 1) <=> (V_cont >= t), con s01 (6 escenarios) y s07 (9 escenarios).
(2) Efecto de la cuantizacion sobre el regresor: OOF de XGBoost en LeaveOneShapeOut (solo lectura), filas con
    volumen real en [1, 5] m3.
(3) Intervalo de Wilson al 95 % para la proporcion 21/28 de s01.
Uso: python s08_cota_y_regresor.py <repo>   (desde spike_d04, despues de s01 y s07)"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
repo = Path(sys.argv[1]).resolve()

# (1) alcance y cota
s01 = pd.read_csv("s01_nodos.csv")[["shape_id", "factor", "node_id", "vol_rpt", "vol_api"]]
s07 = pd.read_csv("s07_nodos.csv")[["shape_id", "factor", "node_id", "vol_rpt", "vol_api"]]
for name, d in [("s01", s01), ("s07", s07), ("s01+s07", pd.concat([s01, s07]))]:
    seco = d.vol_api == 0; mid = (d.vol_api > 0) & (d.vol_api < 100); big = d.vol_api >= 100
    neg = d[d.vol_rpt < 1]; pos = d[d.vol_rpt >= 1]
    lo = neg.vol_api.max(); hi = pos.vol_api.min()
    print(f"[{name}] casos={len(d)} escenarios={d.groupby(['shape_id','factor']).ngroups} con desborde={int((d.groupby(['shape_id','factor']).vol_api.max()>0).sum())}; "
          f"secos={int(seco.sum())} 0<V<100={int(mid.sum())} V>=100={int(big.sum())}; "
          f"rpt!=round(V) en 0<V<100: {int((d.vol_rpt[mid] != np.round(d.vol_api[mid])).sum())}; "
          f"en V>=100: {int((d.vol_rpt[big] != np.round(d.vol_api[big])).sum())} de {int(big.sum())}; "
          f"cota umbral: ({lo:.4f}, {hi:.4f}]  [max V_cont con etiqueta 0, min V_cont con etiqueta 1]")
    b = d[big]; e = b.vol_rpt - b.vol_api
    print(f"    V>=100: rpt-api min={e.min():.3f} max={e.max():.3f}; nodos con rpt-api < -0.5: {int((e < -0.5).sum())}; "
          f"con rpt-api > +0.5: {int((e > 0.5).sum())}; floor(api)==rpt en esos: "
          f"{int((np.floor(b.vol_api[e < -0.5]) == b.vol_rpt[e < -0.5]).sum())}")

# (2) regresor
PREP = repo / "outputs/bench/prepared_v4_21/ca706e27654e0cf5"
tg = pd.read_parquet(PREP / "targets.parquet"); tg["sample_idx"] = np.arange(len(tg))
o = pd.read_parquet(repo / "outputs/bench/oof/ca706e27654e0cf5/xgboost.parquet")
o = o[o.protocol == "LeaveOneShapeOut"].merge(tg, on="sample_idx")
print("\n[regresor] y_pred_reg = expm1 de la prediccion del regresor sobre TODAS las filas de test (evaluate._fit_predict_fold)")
for lo_, hi_ in [(1, 1), (2, 2), (1, 5), (3, 5), (5, 20), (20, 100)]:
    m = (o.vol_inundacion_m3 >= lo_) & (o.vol_inundacion_m3 <= hi_) & (o.inunda == 1)
    x = o[m]; ae = (x.y_pred_reg - x.vol_inundacion_m3).abs(); re_ = ae / x.vol_inundacion_m3
    sesgo = (x.y_pred_reg - x.vol_inundacion_m3).mean()
    print(f"  V real en [{lo_},{hi_}] (n={len(x)}): |err| mediana={ae.median():.3f} media={ae.mean():.3f} p90={ae.quantile(.9):.3f}; "
          f"rel mediana={re_.median():.3f} media={re_.mean():.3f}; sesgo medio={sesgo:+.3f}; "
          f"frac |err|<=0.5: {(ae <= 0.5).mean():.3f}; cuantizacion max relativa = {0.5/lo_:.2f}")

# (3) Wilson
k, n, z = 21, 28, 1.959964
p = k / n; den = 1 + z**2 / n; c = (p + z**2 / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
lo_w, hi_w = c - h, c + h
print(f"\n[Wilson] {k}/{n}={p:.3f}; IC95 % = [{lo_w:.3f}, {hi_w:.3f}] -> filas de 1894: [{lo_w*1894:.0f}, {hi_w*1894:.0f}] "
      f"(punto {p*1894:.0f}); % de 46 225 positivos: [{100*lo_w*1894/46225:.1f}, {100*hi_w*1894/46225:.1f}] (punto {100*p*1894/46225:.1f})")
```

### `s09_notas.py`

```python
"""Inserta (sin reescribir texto existente) las notas de correccion fechadas del spike D-04 en los documentos
de auditoria anteriores. Cada insercion se ancla al texto exacto de la linea previa y conserva el fin de linea.
Uso: python s09_notas.py <repo>"""
import sys
from pathlib import Path
repo = Path(sys.argv[1]).resolve()
P = "> **Nota de corrección (2026-10-11, spike D-04):** "

NOTES = [
 ("docs/auditoria/anexos/03-datos-persistencia-sql.md", "| **D-22** |", "", [
  P + "ver `../correcciones/03-resolucion-de-la-etiqueta.md`. Sobre **D-04**:",
  ">",
  "> 1. \"Son ruido de redondeo\" no se sostiene en general. En la muestra re-simulada (una red, 13 escenarios con desborde), el volumen del `.rpt` es una función monótona del volumen continuo, así que toda transición 1→0 exige que el volumen continuo baje. De las 72 transiciones se re-simularon 12, y en las 12 el volumen continuo baja: 7 con bajadas ≥ 0,3 m³ (entre ellas las 4 con 2 m³), 3 con bajadas de 0,1 a 0,3 m³ y 2 con bajadas < 0,1 m³ (casos de borde alrededor de 0,5 m³). Las otras 60 no se verificaron.",
  "> 2. \"El umbral no filtra nada distinto del redondeo\": en la muestra, la etiqueta es determinista y equivale a un volumen continuo ≥ t, con t en (0,4988; 0,5002] m³. Sí filtra desbordes reales menores de ≈ 0,5 m³: 106 de los 414 nodos con desborde en 6 escenarios de control, con un máximo de 0,499 m³. Es una cuantización determinista, no ruido.",
  "> 3. \"Leer el volumen desde el `.out` binario\" queda descartado con el `REPORT_STEP` actual de 3 min: la integral del `.out` se aleja del volumen continuo hasta 9,5 m³ por nodo. La fuente adecuada es la estadística `node.statistics['flooding_volume']` de pyswmm.",
  "> 4. Un \"umbral con margen (5-10 m³)\" no elimina la banda limítrofe, solo la desplaza. Elegir el umbral es una decisión semántica pendiente del autor y no corrige la resolución.",
 ]),
 ("docs/auditoria/2026-10-10-auditoria-completa.md", "| T-06 | 🟠 |", "", [
  P + "en la fila D-04, atribuir los 72 casos a \"ruido de redondeo\" es solo parcialmente correcto. Se re-simularon 12: en los 12 el volumen continuo baja, en 7 con bajadas ≥ 0,3 m³ y en 2 con bajadas < 0,1 m³ (casos de borde). Las otras 60 no se verificaron. Ver `correcciones/03-resolucion-de-la-etiqueta.md` §4.1.",
 ]),
 ("docs/auditoria/2026-10-10-auditoria-completa.md", "Lectura: en LeaveOneShapeOut ningún falso negativo supera 67 m³", "", [
  P + "\"inestables por construcción\" exagera. En la muestra re-simulada, la etiqueta es una función determinista del volumen continuo, sin ruido, con un umbral efectivo en (0,4988; 0,5002] m³. Lo que sí se midió es que la banda `V_rpt = 1` (volumen continuo de ≈ 0,5 a 1,5 m³) es la más próxima al umbral y concentra 333 de los 380 FN. Que esa cercanía explique los FN es una interpretación. Ver `correcciones/03-resolucion-de-la-etiqueta.md` §6.",
 ]),
 ("docs/auditoria/2026-10-10-auditoria-completa.md", "   - la matriz por forma para las formas fuera de rango", "   ", [
  P + "con la resolución de 1 m³, el único volumen reportado en [1, 2) m³ es el entero 1, que corresponde a un volumen continuo de ≈ 0,5 a 1,5 m³. La matriz por banda está en `correcciones/03-resolucion-de-la-etiqueta.md` §6.1.",
 ]),
 ("docs/auditoria/2026-10-10-auditoria-completa.md", "10. Leer el volumen inundado con mayor resolución que el `.rpt`", "    ", [
  P + "con el `REPORT_STEP` actual (3 min), la integral de la serie `flooding` del `.out` **no** es una fuente de mayor resolución: se aleja del volumen continuo hasta 9,5 m³ por nodo, y hasta 0,89 m³ incluso con 5 s. La única fuente medida que da el volumen continuo es `node.statistics['flooding_volume']` de pyswmm. El spike recomienda la ruta A (declarar, sin re-simular). Ver `correcciones/03-resolucion-de-la-etiqueta.md`.",
 ]),
 ("docs/auditoria/2026-10-10-auditoria-completa.md", "3. La inundación es el volumen que reporta SWMM por nodo, con **resolución de 1 m³**", "   ", [
  P + "en la muestra re-simulada (una red, 13 escenarios con desborde), \"volumen reportado ≥ 1 m³\" es consistente con un umbral efectivo de ≈ 0,5 m³ sobre el volumen continuo, acotado a (0,4988; 0,5002] m³. Los desbordes menores cuentan como \"no inunda\".",
 ]),
 ("docs/auditoria/anexos/01-hidraulica-swmm.md", "- **Recomendación:** tomar el volumen continuo (`flooding_volume` de PySWMM o la integración del `.out`)", "  ", [
  P + "queda descartada la \"integración del `.out`\" con el `REPORT_STEP` actual de 3 min. Se aleja del volumen continuo hasta 9,5 m³ por nodo, más que la cuantización del `.rpt` (≤ 0,5 m³ por debajo de 100 m³). Si se quiere el volumen continuo, la fuente es `flooding_volume` de pyswmm. Ver `../correcciones/03-resolucion-de-la-etiqueta.md`.",
 ]),
 ("docs/auditoria/correcciones/datos/README.md", "  37 celdas (forma, factor). Una fila está inundada si su volumen es ≥ 1 m³.", "  ", [
  P + "el volumen es el del `.rpt`, con resolución de 1 m³. En la muestra re-simulada (una red, 13 escenarios con desborde), \"≥ 1 m³\" es consistente con un umbral efectivo de ≈ 0,5 m³ de volumen continuo, acotado a (0,4988; 0,5002] m³. Ver `../03-resolucion-de-la-etiqueta.md`.",
 ]),
]

for rel, anchor, indent, lines in NOTES:
    path = repo / rel
    raw = path.read_bytes().decode("utf-8")
    eol = "\r\n" if "\r\n" in raw else "\n"
    L = raw.split(eol)
    idx = [i for i, l in enumerate(L) if l.startswith(anchor)]
    assert len(idx) == 1, (rel, anchor, idx)
    i = idx[0]
    if any("spike D-04" in l for l in L[i + 1:i + 4]):
        print("ya existe:", rel, anchor[:40]); continue
    block = [""] + [indent + l if l else indent.rstrip() for l in lines]
    if i + 1 < len(L) and L[i + 1].strip() != "":
        block = block + [""]
    L[i + 1:i + 1] = block
    path.write_bytes(eol.join(L).encode("utf-8"))
    print(f"insertada: {rel} tras la linea {i + 1} (eol={'CRLF' if eol == chr(13)+chr(10) else 'LF'})")
```
