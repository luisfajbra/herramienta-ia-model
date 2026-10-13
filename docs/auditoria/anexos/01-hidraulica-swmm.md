# Annex 01 — Hydraulic / SWMM audit

- **Date:** 2026-10-10
- **Branch / base commit:** `fix/cierre-auditoria-hidrogramas-20260927` @ `6e8579b`. The tree has uncommitted changes from another phase (bench/Optuna); they were not touched. The `main.py` line numbers refer to the **working tree**: in `6e8579b` they are 12 lines higher (the uncommitted diff removes 12 lines above line 225). In the other files cited there are no uncommitted changes.
- **Engine and libraries (project venv):** pyswmm 2.1.0, swmm-toolkit 0.17.0, swmm-api 0.4.73.
- **Network audited:** `data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp` (the one used by `config.yaml`): 160 junctions, 1 outfall (`109C`, FREE), 160 circular PVC conduits (n = 0.0082), `FLOW_UNITS LPS`, `DYNWAVE`, `ALLOW_PONDING NO`.
- **Dataset audited:** `outputs/training_v17.sqlite3` (opened only with `mode=ro`): 1000 runs = 40 shapes (`base` + 39 CSV) × 25 factors (0.2–5.0), with 160,000 node-scenario rows.

---

## 1. Scope and method

**Scope (hydraulic/SWMM area):** `swmm_resilience/simulation/` (runner, swmm_api_io, hydrograph_shapes, timeseries_scenario, batch), `swmm_resilience/extraction/` (static_features, topology, dynamic_features, labels), `swmm_resilience/validation/` (hydrograph_csv, hydrograph_batch), the feature construction and inference in `swmm_resilience/ml/scenario_predict.py`, `predict_from_inp.py` and `predict_tabular.py`, `swmm_resilience/analysis/flood_volume.py` and `resilience.py`, `swmm_resilience/dataset/`, `config.yaml` (hydraulic parameters) and the `.inp` files of `data/networks/` (read only).

**Method:**
1. Line-by-line reading of the cited code and of the `[OPTIONS]`, `[JUNCTIONS]`, `[CONDUITS]`, `[XSECTIONS]`, `[INFLOWS]` and `[TIMESERIES]` sections of the `.inp`.
2. Read-only queries to the v17 DB (`file:...?mode=ro`).
3. About 150 control SWMM simulations (1 to 3 s each), all on **copies** in the scratchpad and built with **the same pipeline functions** (`write_scaled_inp`, `apply_shape` + `write_scenario_inp`, `extract_labels`). Controlled variants were applied to those copies: InitFlow = 0, extended END_TIME and a cut to zero instead of a ramp. It was first verified that the replicas reproduce **exactly** the labels stored in the DB (§3, V-03).
4. Scripts `d01`–`d17`, included in full in §5. No project code, `config.yaml`, `outputs/` or any `.inp` was modified.

**Severity scale:** 🔴 CRITICAL (compromises scientific validity) · 🟠 MAJOR (biases results) · 🟡 MODERATE (limitation or latent risk) · 🔵 MINOR (quality).

**Overall result:** 0 🔴 · 2 🟠 · 6 🟡 · 6 🔵 (14 findings). I found no defect that invalidates the SWMM labels of the dataset: independent replicas reproduce them to the m³. The two 🟠 findings are the initial condition of the `.inp` (InitFlow) and the absence of a hydrograph-volume descriptor among the features.

---

## 2. Findings

### 2.1 Summary table

| ID | Sev. | File:line | Description | Quantified hydraulic impact |
|---|---|---|---|---|
| H-01 | 🟠 | `.inp` Qx1 `[CONDUITS]` InitFlow col. (lines 220–379); `swmm_api_io.py:229-263`; `timeseries_scenario.py:47-136` | All conduits start with InitFlow = sum of upstream peaks **at factor 1**, in all 1000 runs, regardless of the factor or the shape | 855 m³ stored at t = 0 (up to 5.9 times the event's input volume). Continuity error of up to 12.4% (15 of 80 control runs > 5%; with InitFlow = 0 all are at \|CE\| < 1%). In short events at f = 5: total flooded volume +0.7% to +10.5% and 2–5 fewer flooded nodes |
| H-02 | 🟠 | `hydrograph_shapes.py:31-37`; `dynamic_features.py:5-29`; `contracts.py:46-49` | Shapes are normalized **by peak** and the only shape features are `duracion_horas` and `tiempo_al_pico_h`. The event volume does not appear in any feature | ∫q_norm·dt varies 72 times (0.133–9.57 h). At equal duration and factor, the flooded volume differs up to 13.3 times (`shape_2h` versus `common_bogota`, f = 2: 1593 versus 120 m³) |
| H-03 | 🟡 | `scenario_predict.py:113-124`, `75-95`; `dynamic_features.py:48-54`; `hydrograph_batch.py:54-61`; `main.py:704-726` | Training is only spatially uniform and synchronous, but inference accepts arbitrary per-node CSVs. `tiempo_al_pico_h` is taken from the first series of the CSV (it can be null and give 0 h). The extrapolation flag only looks at peak/base | In training, q_pico_nodo/base = factor with a deviation of 8.9·10⁻¹⁶ in all 147 nodes (total uniformity). With a CSV whose first node is null, t_pico = 0.0 h even though the actual one is 1.0 h |
| H-04 | 🟡 | `batch.py:7-73`; `csv_backfill.py:147-156` | Training neither reads nor stores the SWMM continuity error (only validation does) | `runs.continuity_error_pct` is NULL in 1000 of 1000 runs. Control: 15 of 80 runs (19%) with \|CE\| > 5%, which is the threshold the project itself uses in validation |
| H-05 | 🟡 | `timeseries_scenario.py:120-123` (docstring 58-61) | The 6 h "drain" adds the point (t_end + 6 h, 0). If the shape does not end at 0, SWMM interpolates a **linear ramp** and not a cut to zero | Input volume +24.5% (`shape_2h`), +13.8% (`long_moderate`) and +9.8% (`shape_5h`) versus the declared hydrograph. Flooded volume +2.2% (`long_moderate`, f = 5) and +0.7% (f = 4); no change in the other cases tested |
| H-06 | 🟡 | `labels.py:19-23`, `122-139`; `swmm_api_io.py:304-307`; `config.yaml:14` | The label comes from the `.rpt`, with 3 decimals in 10⁶ L, so the resolution is 1 m³. The threshold "≥ 1 m³" is equivalent to "≥ 0.5 m³ continuous" | No DB volume is in (0, 1); 1902 rows (4.1% of the positives) are exactly 1.0, so changing `>=` to `>` turns them negative. In 6 control runs, between 10 and 28 nodes per scenario overflow (27–57% of those that overflow) but are labeled 0 |
| H-07 | 🟡 | `labels.py:75-86` (`parts[-1]`, line 79); `swmm_api_io.py:293-298` | The fallback text parser reads the last column, which in SWMM 5.2 is "Maximum Ponded Depth" and not the volume | If activated, **all** labels become 0 without warning. Verified: 0.0 m³ versus 7669 m³ by swmm_api in `base` f = 3 |
| H-08 | 🟡 | `topology.py:90-118` | The only capacity feature is that of the **upstream** conduits. The one that governs the surcharge of the node is that of the **outlet** conduit | `upstream_capacity_lps` is NaN in 57 headwater nodes (35.6% of the rows, imputed). AUC of q_acum/upstream_cap = 0.732 versus 0.884 for q_acum/outlet_cap |
| H-09 | 🔵 | `static_features.py:82-87`, `109` | `compute_slope` only uses JUNCTION elevations, so `pendiente_out` is NaN at the node that discharges to the outfall | 1 node (`108C`, 0.6% of the rows) with imputed outlet slope |
| H-10 | 🔵 | `runner.py:646-667`; `batch.py:7-16`; `main.py:992-1020` | The `base` shape is simulated with END = 3 h, without drain and with the original series; the other 39 shapes carry +6 h | Verified: it does not change any label (END = 9 h gives the same result at f = 0.2, 1, 3 and 5). 147–449 m³ of storage and about 7.4% of the recession remain unsimulated |
| H-11 | 🔵 | `labels.py:122-139`; `static_features.py:7-26`; `hydrograph_shapes.py:40-53`; `csv_backfill.py:88` | The training path does not check `FLOW_UNITS`, ignores Sfactor and Baseline of `[INFLOWS]` and fixes `flow_units = 'LPS'` in the DB | No current impact (LPS, Sfactor = 1 and Baseline = 0 verified in all 160 inflows). With US units the volume would be off by a factor of 3.785 |
| H-12 | 🔵 | `topology.py:14-32`, `83-88`; `dynamic_features.py:48-54` | `q_pico_acum` sums peaks without lag or attenuation, only over CONDUITS. With dividers or cycles it would count the flow twice | No current impact: the network is an acyclic tree with maximum out-degree 1. Risk when transferring to other networks |
| H-13 | 🔵 | `runner.py:427-429`, `456-462`, `515-530`; `swmm_api_io.py:131-134` | In the legacy path (`swmm_resilience/main.py` and desktop app) PySWMM keys that do not exist (`time_flooded`, `time_max_depth`) are read, a constant dt is assumed and `base_inflow_lps` is the Baseline (homonymous with the v17 feature, which is the peak) | Flooding duration 0 and `time_to_peak` None except through the `.rpt` overlay. The actual dt varies between 1 and 5 s (5 distinct values). It does not affect the thesis's v17 pipeline |
| H-14 | 🔵 | `hydrograph_csv.py:127-129`, `153-156` | `write_shape_validation_csv` rounds times to whole minutes (H:MM) | Shift ≤ 0.5 min and volume ≤ 1.22% (`shape_ultra_short_impulse`). In the validation shapes ≤ 0.01%. Training and inference t_pico differ by ≤ 3.3·10⁻⁴ h |

### 2.2 Detailed sheets

#### H-01 🟠 — Initial condition (InitFlow) fixed at factor 1 in all scenarios

- **Where:** `.inp` Qx1, `[CONDUITS]`, `InitFlow` column (lines 220–379). The functions `write_scaled_inp` (`swmm_api_io.py:229-263`) and `write_scenario_inp` (`timeseries_scenario.py:47-136`) only rewrite `[TIMESERIES]` and `[OPTIONS]`, so the InitFlow is inherited unchanged. Qx2 and Qx3 have the same InitFlow as Qx1 (ratio 1.000).
- **What happens:** the 160 conduits have InitFlow > 0, and the value is **exactly** the sum of the peaks of the series upstream of the inlet node (maximum difference 0.0 L/s; 1525 L/s at `T160`). Since the nodes have InitDepth = 0, SWMM initializes their depths from those of the conduits. The network starts "full" with the factor-1 peak flow in all runs, even if the scenario is f = 0.2 or a shape that starts at q = 0 (33 of the 39 shapes have q0 = 0).
- **Quantified impact (d05, d07, d09):**
  - Initial Stored Volume = 855 m³ in all runs. It is 5.9 times the input volume of `shape_ultra_short_impulse` at f = 0.2 (146 m³).
  - Routing continuity error with the pipeline configuration: up to **12.38%** (ultra_short, f = 0.2). At f = 0.2, 12 of 40 shapes exceed 5%; at f = 1.0, 3 of 40. The correlation between CE and (initial storage / input) is 0.857. With InitFlow = 0, the CE stays between −0.92% and +0.03% in all cases tested.
  - Labels (pipeline versus InitFlow = 0):

    | Scenario | Pipeline flooded vol. | Vol. with InitFlow = 0 | Δ | Flooded nodes (pipeline / InitFlow = 0) |
    |---|---|---|---|---|
    | `shape_ultra_short_impulse` f = 5 | 1320 m³ | 1195 m³ | +10.5% | 96 / 101 |
    | `rare_extreme` f = 5 | 2099 | 2054 | +2.2% | 98 / 100 |
    | `common_bogota` f = 5 | 2864 | 2825 | +1.4% | 98 / 100 |
    | `shape_flash_flood_urban` f = 5 | 4162 | 4121 | +1.0% | 95 / 99 |
    | `shape_micro_burst_45min` f = 5 | 4908 | 4875 | +0.7% | 95 / 97 |
    | `shape_2h` f = 3 | 5033 | 5038 | −0.1% | 60 / 62 |
    | `base` f = 3 / f = 5 | 7669 / 19940 | 7668 / 19941 | ≈ 0 | 61 / 61 · 93 / 93 |

  - Conclusion: the bias is concentrated in short, intense events (≤ 1 h), precisely the "flash" regime that the thesis wants to cover. In long events it is negligible.
- **Evidence:** `d03_base_inp.py` (InitFlow = sum of peaks, 160/160), `d05_experiments.py`, `d07_more.py`, `d09_continuity_sweep.py`. External corroboration: `validation_output/val_extrap_ultrafast_20min/metrics_per_scenario.csv` has 8 of 24 scenarios with CE > 5% (maximum 11.37%), and `AUDITORIA_SURROGATE_HIDROGRAMAS.md:309` had already observed it without identifying the cause.
- **Recommendation:** generate the scenarios with InitFlow = 0, or with InitFlow = factor × q(t = 0) accumulated and consistent with the shape. Another option is a dry-weather *spin-up* or hotstart common to all runs. Then the dataset must be regenerated, or at least the shapes ≤ 1.5 h, and the criterion declared in the thesis.

#### H-02 🟠 — No feature describes the hydrograph volume

- **Where:** `hydrograph_shapes.py:31-37` (`get_shape_stats` returns only duration and t_pico), `dynamic_features.py:5-29`, `contracts.py:46-49`.
- **What happens:** the 39 training shapes and the 7 validation shapes are normalized to peak = 1 (`d02`). The factor scales the **peak**, and the event volume per node (base·f·∫q_norm) depends on the shape: ∫q_norm·dt goes from 0.133 h (`shape_ultra_short_impulse`) to 9.57 h (`shape_quasi_uniform_block`), i.e. 72 times. In drainage hydraulics, the overflowed volume depends mainly on the input volume that exceeds the capacity, but the model only sees (duration, t_pico).
- **Quantified impact (`d12_db_shapes.py`, v17 DB):**
  - With duration 10 h: `shape_slow_broad_rise` (t_pico 6.0) gives 9716 m³ and `shape_long_recession_tail` (t_pico 0.5) gives 1366 m³ at f = 2, a ratio of 7.1. At f = 5 the ratio is 77,404 versus 15,668 (4.9).
  - With duration 2 h: `shape_2h` gives 1593 m³ and `common_bogota` 120 m³ at f = 2 (13.3 times).
  - With 39 shapes, the (duration, t_pico) pair can serve as an "identifier" of the shape in training, but it does not determine the volume of a new shape. This physically explains the volume underestimation on unseen shapes reported in `AUDITORIA_SURROGATE_HIDROGRAMAS.md:309`.
- **Recommendation:** add volume features: `vol_evento_nodo = q_pico·∫q_norm` (m³), its upstream accumulation, the duration above a threshold (for example q > 0.5·q_pico) and the temporal centroid. They must be computed the same way in `compute_dynamic_features` and in `compute_scenario_dynamic_features`.

#### H-03 🟡 — Uniform and synchronous training, inference on arbitrary hydrographs

- **Where:** `scenario_predict.py:113-124` (`peak_map` = max per node; `tiempo_al_pico_h` from the **first** series, lines 118-119); `dynamic_features.py:48-54` (sum of upstream peaks); `hydrograph_batch.py:54-61` (`expected_nodes` includes the 13 nodes with a null series); `scenario_predict.py:75-95` (extrapolation flag only by peak/base); `main.py:704-726` (interp/extrap regime only by duration).
- **What happens:** in all 1000 runs, the 147 nodes with inflow receive the **same** shape and the **same** factor, with synchronous peaks (`d01`: |q_pico_nodo/base − factor| ≤ 8.9·10⁻¹⁶). The `--evaluate-hydrographs` mode accepts CSVs with different series per node, and there: (a) the sum of non-coincident peaks is not comparable with that of training; (b) `tiempo_al_pico_h` depends on the order of the nodes in the CSV, and if the first one is one of the 13 null nodes (`3C`, `4C`, `20C`, …) it gives 0 h (`d17`: 0.0 h versus 1.0 h actual); (c) nothing warns that the scenario is outside the training distribution.
- **Impact:** it does not bias the results already reported, because the out-of-sample validation (`--evaluate-validation-shapes`) also uses uniform shapes. It does limit what can be claimed: the surrogate is not validated for rainfall with spatial variability.
- **Recommendation:** compute t_pico and duration on the **aggregated** hydrograph (sum of the nodes) both in training and in inference, flag as extrapolation any CSV with heterogeneous shapes or lagged peaks, and declare the limitation in the thesis.

#### H-04 🟡 — No mass continuity control in the dataset generation

- **Where:** `batch.py:7-73` (run_batch and run_batch_shapes do not read the `.rpt` beyond the labels); `csv_backfill.py:147-156` (inserts `runs` without `continuity_error_pct`). Validation does do it (`hydrograph_batch.py:175-193`, `314-321`).
- **Evidence:** `d01_db.py`: `runs` has 1000 COMPLETE runs and `continuity_error_pct IS NULL` = 1000. `d09_continuity_sweep.py` (exact pipeline configuration, 80 runs): at f = 0.2, 12 of 40 with |CE| > 5%, 40 of 40 with |CE| > 1% and median 3.67%; at f = 1.0, 3 of 40 with > 5% and median 0.93%. All runs with CE > 5% have 0 flooded nodes, so they **do not contaminate positives**, but they are part of the 20% of negative rows (factor ≤ 1.0).
- **Recommendation:** read and store the routing CE (and the *Highest Continuity Errors* per node) for each run, set an acceptance criterion (for example |CE| ≤ 1%) and report it. The main cause is fixed with H-01.

#### H-05 🟡 — The "drain-down" is a linear ramp, not a cut to zero

- **Where:** `timeseries_scenario.py:120-123`. The point `(last_time + drain_down_hours, 0.0)` is added. SWMM interpolates linearly between points, so if q(t_end) ≠ 0 the flow decreases **linearly for 6 h**. The docstring (lines 58-61) presents it as a zero-flow point "para que la red drene" (so that the network drains), and moreover the flow does not reach 0 until exactly END_TIME, so these shapes have no real drain period.
- **Shapes affected:** `shape_2h` (q_end = 0.075), `shape_5h` (0.074) and `long_moderate` (0.35). They are 3 of 39 training shapes and 0 of 7 validation shapes. It also affects the `base` shape when re-simulated in `--evaluate-shapes` (`main.py:495-509`, q_end/q_peak = 0.074).
- **Quantified impact (d05, d07):** SWMM external inflow 6259 m³ versus 5032 m³ with cut (`shape_2h`, f = 1, +24.4%, equal to the analytical prediction of 0.5·q_end·6 h / ∫ = 24.5%); `long_moderate` +13.8%; `shape_5h` +9.8%. Flooded volume: `long_moderate` f = 5 gives 125,350 versus 122,624 m³ (+2.2%), f = 4 gives 83,237 versus 82,654 (+0.7%), f = 3 and f = 1 unchanged; `shape_2h` and `shape_5h` unchanged at f = 1, 3, 4 and 5 (the tail remains below capacity). The `duracion_horas` feature (12 h in `long_moderate`) does not include the 6 additional hours of inflow.
- **Recommendation:** write `(t_end + 1 min, 0)` and `(t_end + dd, 0)`, or require q_end = 0 in the shapes. If the tail is intentional, include it in `duracion_horas` and in the event volume (H-02).

#### H-06 🟡 — Label quantized to 1 m³ and the actual semantics of the threshold

- **Where:** `labels.py:19-23` (`vol >= threshold`), `122-139`; `swmm_api_io.py:304-307` (×1000 from 10⁶ L); `config.yaml:14` (`flood_threshold_m3: 1.0`).
- **What happens:** the *Node Flooding Summary* prints the volume in 10⁶ L with 3 decimals, so the resolution is 0.001·10⁶ L = **1 m³**. A real overflow smaller than 0.5 m³ is reported as 0.000 and one of 0.5 to 1.5 m³ as 1 m³. The nominal threshold "≥ 1 m³" is therefore equivalent to "≥ 0.5 m³ of continuous volume".
- **Evidence:**
  - `d01`/`d12`: in the DB no volume falls in (0, 1); the minimum positive is 1.0; 1902 rows are exactly 1.0 (4.1% of the 46,242 positives); only 6 rows are non-integer.
  - `d11_quantization.py` (continuous PySWMM volume, `node.statistics['flooding_volume']` in m³, compared with the `.rpt`): `base` f = 1.6 has 45 nodes that overflow and 27 labeled (18 hidden, maximum 0.347 m³); `base` f = 2.0, 63 versus 39; `shape_2h` f = 2.0, 67 versus 39; `rare_extreme` f = 2.0, 37 versus 16. The quantization error reaches ±0.498 m³ (±50% for a volume of 1 m³).
  - Node `2I` in `base` f = 3: "Hours Flooded 0.01, Max Rate 11.67 LPS, Volume 0.000", i.e. it overflows but is labeled 0.
- **Recommendation:** take the continuous volume (`flooding_volume` from PySWMM or the integration of the `.out`), document the threshold as a continuous volume and do a sensitivity analysis of the threshold (0, 0.5, 1, 5 m³) on the metrics. The log1p regressor is especially sensitive in the 1–5 m³ range.

  > **Correction note (2026-10-11, spike D-04):** the "integration of the `.out`" is ruled out with the current 3 min `REPORT_STEP`. It departs from the continuous volume by up to 9.5 m³ per node, more than the `.rpt` quantization (≤ 0.51 m³ below 100 m³; node `8C` differs by 0.508 m³). If the continuous volume is wanted, the source is pyswmm's `flooding_volume`. See `../correcciones/03-resolucion-de-la-etiqueta.md`.

#### H-07 🟡 — Fallback text parser reads the wrong column

- **Where:** `labels.py:75-86`. Line 79 takes `float(parts[-1])` as the volume. In SWMM 5.2 the table always has the final column **"Maximum Ponded Depth (Meters)"**, also with `ALLOW_PONDING NO` (see the header in `d04`). The docstring assumes that this column only appears with ponding.
- **Evidence:** `d04_base_run.py` on `base` f = 3 gives 7669 m³ (88 nodes) with swmm_api and 0.0 m³ with the text parser (maximum per-node difference 602 m³).
- **Impact:** today it is latent, because the swmm_api path works and the DB labels match my replicas. If swmm_api fails (version or encoding change), `read_flooding_volumes` falls back to the fallback text parser and generates a dataset with all labels at zero, without error. The heuristic of `swmm_api_io.py:293-298` (`"volume" in cl and (... "vol" in cl ...)`) is also fragile: any column containing "volume" is mapped to the volume.
- **Recommendation:** parse by header (column "Total Flood Volume"), fail loudly instead of degrading, and cross-check Σ node vol with "Flooding Loss" from the continuity (7669 versus 7673 m³ in the control; the difference comes from rounding and from nodes with less than 0.5 m³).

#### H-08 🟡 — The hydraulic capacity used is the upstream one, not the outlet one

- **Where:** `topology.py:90-118` (`upstream_capacity_lps` = Σ of the full-pipe Manning capacity of the conduits that **enter** the node).
- **What happens:** the surcharge and overflow of a junction depend on the accumulated flow exceeding the capacity of the **outlet** conduit, in addition to downstream backwater. The inlet capacity is a weak proxy and does not exist at headwater nodes.
- **Evidence (`d16_capacity_index.py`, v17 DB):** `upstream_capacity_lps` is NaN in 57 nodes (57,000 rows, 35.6%), which are then imputed with the median. Univariate AUC against `inunda`: q_pico_acum_escalado gives 0.753; q_pico_acum_escalado/upstream_capacity gives 0.732 (only in 103,000 rows); q_pico_acum_escalado/outlet_cap gives **0.884** (in 160,000 rows). The Manning formula is well implemented (§3, V-05).
- **Recommendation:** add `cap_salida_lps` and the index `q_pico_acum_escalado / cap_salida`, computed the same way in training and inference.

#### H-09 🔵 — `pendiente_out` NaN at the node that discharges to the outfall

- **Where:** `static_features.py:82-87`. `compute_slope` looks up elevations only in `junctions`; `109C` is an OUTFALL, so the slope of `T160` is left `None` and `pendiente_out` of `108C` is NaN even though `n_tuberias_out = 1`.
- **Evidence:** `d13_manning.py` shows that the only node with NaN `pendiente_out` and an outlet is `['108C']`. It is 1 node (1000 rows). The same pattern is in `topology.py:99-108`, but there it does not matter because an outfall is never a predecessor.
- **Recommendation:** include `[OUTFALLS]` in the elevation dictionary.

#### H-10 🔵 — Different protocol for the `base` shape

- **Where:** `batch.py:7-16` and `runner.py:646-667` (original END_TIME of 3 h, without drain and with the 160 original series); the remaining 39 shapes go through `write_scenario_inp` with +6 h. In `main.py:1002`, `duracion_horas` of `base` comes from the **first** series of the `.inp` (`1C`).
- **Evidence:** `d03`: all series end at 3 h with q/q_peak = 0.074–0.075, and the unsimulated recession (exponential extrapolation) is 6.8–8.3% of the volume. `d05`: END = 9 h leaves Flooding Loss and labels **identical** to END = 3 h for f = 0.2, 1, 3 and 5; only the final storage (147–449 m³ versus 0) and the outflow change. The normalized shapes of the 147 nodes coincide with a deviation ≤ 0.00075, so taking `1C` as representative is correct.
- **Recommendation:** unify the simulation protocol of the 40 shapes for methodological consistency. It is not necessary to regenerate the labels for this cause.

#### H-11 🔵 — Unit and `[INFLOWS]` assumptions not checked in the training path

- **Where:** `labels.py:122-139` and `swmm_api_io.py:304-307` assume 10⁶ **liters**, but only validation checks `FLOW_UNITS = LPS` (`hydrograph_batch.py:102-107`). `static_features.py:7-26` takes the series maximum without applying Sfactor or Baseline; `hydrograph_shapes.apply_shape` (40-53) replaces the series but leaves Sfactor and Baseline in `[INFLOWS]`, so they would be applied twice. `csv_backfill.py:88` writes a fixed `flow_units='LPS'`, without reading it from the `.inp`.
- **Evidence:** `d03` confirms `FLOW_UNITS LPS` and 0 inflows with Sfactor ≠ 1 or Baseline ≠ 0. There is no current impact.
- **Recommendation:** check units and `[INFLOWS]` at the start of the training pipeline with the same rules as validation, and abort if they are not met.

#### H-12 🔵 — Topological accumulation of peaks (risk when transferring to other networks)

- **Where:** `topology.py:14-32` (graph with CONDUITS only), `83-88` and `dynamic_features.py:48-54` (Σ of the ancestors' peaks).
- **Evidence:** `d03` shows an acyclic graph with maximum out-degree 1, a single sink (`109C`) and no isolated node. In this network, the sum is exact as a synchronous-inflow index (in training all nodes have their peak at the same hour) and ignores the attenuation and lag of the routing.
- **Recommendation:** document that `q_pico_acum` is an upper bound without routing. If it is transferred to networks with dividers, pumps or weirs, include all link types and split the flow at bifurcations.

#### H-13 🔵 — Errors in the legacy path (`runner.run_simulation`)

- **Where:** `runner.py:515-517` (`stats.get("time_flooded")`) and `521-530` (`stats.get("time_max_depth")`): **neither** key exists in PySWMM 2.1 (`d08` lists the real keys: `flooding_duration`, `max_depth_date`, …). `runner.py:427-429` and `456-462` compute the depth rate with the dt of the first step, when the real dt varies (`d08`: 1–5 s, 5 distinct values, 2205 steps). `swmm_api_io.py:131-134` and `runner.py:325`: `base_inflow_lps` = Baseline of `[INFLOWS]` (0 in this network), whereas in v17 the homonymous column is the series peak.
- **Impact:** only in `swmm_resilience/main.py` and in the desktop app. The flooding duration is left at 0 for the nodes that do not appear in the `.rpt` and `time_to_peak_min` always None. None of these columns enters the v17 model (`ML_DROP_COLUMNS`). Not verified empirically: that the default `steady` mode (`config.py:69`) applied to a hydrographic `.inp` with Baseline = 0 aborts in `write_scaled_inp`; it is deduced from reading `swmm_api_io.py:217-226` and `256-260`.
- **Recommendation:** fix the keys (`flooding_duration`, `max_depth_date`), use the real dt of each step and rename the legacy `base_inflow_lps`, or retire that path.

#### H-14 🔵 — Rounding to minutes when converting shape → validation CSV

- **Where:** `hydrograph_csv.py:127-129` and `153-156`.
- **Evidence (`d14_parity.py`):** the maximum shift is 0.5 min; ΔVol = −1.22% (`shape_ultra_short_impulse`), −0.26% (`shape_short_flat_top`) and ≤ 0.05% in the rest. In the 7 validation shapes, ≤ 0.01%.
- **Recommendation:** allow H:MM:SS in the CSV contract, or validate that the shape is already in whole minutes.

**Out-of-area note:** `--evaluate-shapes` (`main.py:480-555`) re-simulates the **same** shapes and factors of the training with the final model, so it is not an out-of-sample validation. I refer it to the ML annex; here I only state the fact.

---

## 3. What is done right (verified)

| ID | Verification | Evidence |
|---|---|---|
| V-01 | **The scaling preserves volume proportionally and SWMM integrates the declared hydrograph.** The `.rpt` external inflow matches the trapezoidal integral: `base` f = 1 gives 7508 m³ versus 7510 analytical (−0.03%); `shape_2h` with cut, 5032 versus 5027.8 (+0.08%); ultra_short f = 1, 731 versus 730.8. f = 3 gives 22,525 = 3 × 7508.3. This confirms the linear interpolation between points and the flow of 0 after the last point. | `d05` |
| V-02 | **Scaling identity.** Pipeline f = 1.0 ≡ original `.inp` (same inflow, CE of 1.439% in both, identical labels). The original Qx2 and Qx3 `.inp` files ≡ Qx1 × 2 and × 3 through the pipeline: 0 different labels and maximum per-node \|ΔVol\| of 1 m³. The geometry of Qx1, Qx2 and Qx3 is identical (ΔElev = ΔMaxDepth = ΔDiam = 0). | `d06`, `d15` |
| V-03 | **Reproducibility of the DB labels.** Independent replicas reproduce exactly vol/n_inunda: `base` f = 3 (7669/61) and f = 5 (19,940/93), `long_moderate` f = 3 (44,540/61) and f = 5 (125,348/91), `shape_2h` f = 3 (5033/60), `shape_ultra_short_impulse` f = 5 (1320/96), `shape_micro_burst_45min` f = 5 (4908/95), `rare_extreme` f = 2 and 5, `common_bogota` f = 5 and `shape_flash_flood_urban` f = 5. | `d05`, `d07`, `d12`, `d18` |
| V-04 | **Volume units.** `.rpt` in 10⁶ L × 1000 = m³ is correct in SI. The swmm_api path picks the right column (Σ = 7669 m³ versus "Flooding Loss" 7673 m³; the difference is rounding). PySWMM `flooding_volume` (95.67 m³) matches the `.rpt` (0.096·10⁶ L) for `1C`. | `d04`, `d08`, `d11` |
| V-05 | **Full-pipe Manning capacity** (`utils.py:66-76`): Q = (1/n)·A·(D/4)^(2/3)·S^(1/2)·1000 [L/s] is correct. Compared with SWMM's implicit capacity (Max Flow / Max-Full Flow) in the 160 conduits, the ratio goes from 0.972 to 1.037 (median 0.9999); the dispersion is due to the 2-decimal ratio of the `.rpt`. | `d13` |
| V-06 | **Slopes and offsets.** There are no adverse or zero slopes (0.00123–0.0923 m/m) nor zero lengths (58.9–218.3 m). The offsets (34 conduits) are correctly added to the invert elevation with `LINK_OFFSETS DEPTH`. Δz/L versus SWMM's Δz/√(L²−Δz²) differs by < 0.43% for S ≤ 0.092. | `d03`, `d13` |
| V-07 | **Topology.** Acyclic directed tree, 1 outfall, no isolated node, `dist_outfall_m` without NaN; q_pico_acum_base of the outfall = 1525 L/s = InitFlow of `T160`. | `d03`, `d13` |
| V-08 | **Training/inference parity of the dynamic features.** DB versus the inference path (`write_shape_validation_csv` → `load_scenario` → `compute_scenario_dynamic_features`) in the 39 shapes at f = 2: q_pico_nodo, q_pico_acum_escalado and duracion_horas are **identical**; tiempo_al_pico_h differs by ≤ 3.3·10⁻⁴ h. | `d14` |
| V-09 | **The truncation of the recession in `base` (3 h) does not alter the labels** (END = 9 h gives the same result at f = 0.2, 1, 3 and 5). | `d05` |
| V-10 | **Shapes.** The 39 training and the 7 validation shapes start at t = 0 (enforced by `load_shape`) and have peak 1.000. The 7 validation shapes end at q = 0, without ramp (H-05 does not affect them). The overlap guard between training and validation exists (`main.py:664-675`). | `d02` |
| V-11 | **Computation step.** DYNWAVE with ROUTING_STEP of 5 s, VARIABLE_STEP 0.75 and MINIMUM_STEP 0.5; the real dt is 1–5 s. The *Node Flooding Summary* integrates at every routing step, not at the 3 min REPORT_STEP. The PySWMM (step by step) and `.rpt` volumes match within ±0.5 m³ (pure quantization). | `d08`, `d11` |
| V-12 | **Physical threshold of the network.** No shape floods at f ≤ 1.0 (200 runs, 20% of the dataset); the onset of flooding is between f = 1.2 (`base`, `common_storm`, `long_moderate`) and f = 1.6 (`shape_ultra_short_impulse`). It is consistent with a network sized for the Qx1 design flow. | `d18` |
| V-13 | **Validation of the scenario CSVs** (`hydrograph_csv.py:33-124`): requires exact columns, exact nodes, H:MM format, absence of duplicates, finite values ≥ 0, common grid and t0 = 0. Moreover, the node↔series correspondence is 1:1 (`timeseries_scenario.py:19-44`). They are solid rules. | reading |

---

## 4. Hydraulic assumptions and limitations that the thesis must declare

1. **1D dynamic-wave model without dual drainage.** SWMM 5.2 (swmm-toolkit 0.17.0) with `DYNWAVE`, `ALLOW_PONDING NO`, SurDepth = 0 and Aponded = 0. The "flooding volume" is the **volume overflowing through the manhole cover** (MaxDepth of 1.4–3.3 m). That volume leaves the system: it does not accumulate nor re-enter downstream, and there is no 2D surface runoff. It does not represent depth, extent or damage.
2. **Downstream boundary condition:** a single FREE-type outfall `109C`, without backwater from the receiving body.
3. **Forcing:** there are no subcatchments or rainfall. The inflows are hydrographs imposed as direct inflow at 147 nodes (13 nodes without inflow). The "factor" scales **flows**, not precipitation, so it leaves out the rainfall-runoff nonlinearity.
4. **Uniform and synchronous scenarios:** the same normalized shape and the same factor at all nodes (H-03). The model is not validated for storms with spatial variability or with lag between subcatchments.
5. **Peak normalization:** the factor fixes the peak, and the event volume depends on the shape (H-02).
6. **Initial condition:** if the current InitFlow is kept, it must be declared as "network initially at the design-flow regime (factor 1)" and its effect on continuity reported (H-01).
7. **Label definition:** `inunda = 1` if the `.rpt` volume, quantized to 1 m³, is ≥ 1 m³, which is equivalent to ≥ 0.5 m³ continuous. Smaller overflows count as "no flooding" (H-06).
8. **Simulated duration:** the shape duration plus 6 h for the 39 CSV shapes, and exactly 3 h for `base`. Shapes that do not end at zero receive a ramp (H-05).
9. **A single network:** Chico Sur, PVC with n = 0.0082, 160 circular conduits in a tree without special structures (no pumps, weirs, orifices, storage or dividers). It has not been shown that the model is transferable to other networks (H-12).
10. **Factor range:** 0.2–5.0 with step 0.2. 20% of the runs (f ≤ 1.0) do not flood at any node.
11. **"Resilience" index** (`analysis/resilience.py:32`): 1 − fraction of flooded nodes. It is a binary count that does not weight magnitude, duration or recovery, so it does not correspond to the hydraulic resilience indices of the literature (severity × duration). It must be named precisely.
12. **Conduit capacity:** the nominal full-pipe Manning capacity is used, which does not consider pressurization, backwater or local losses.
13. **Numerical quality:** with the current configuration, the continuity error reaches 12.4% in short, low-intensity events (H-01, H-04). The acceptance criterion of the runs must be declared.

---

## 5. Reproducible diagnostic scripts

All the scripts are **read-only** on the repository and the DB (`mode=ro`). The simulations are done on copies in `./runs/` of the working directory. They are run from a single directory with the project's Python:

```
<repo>\venv\Scripts\python.exe dNN_*.py "<ruta-al-repo>" [args]
```

`d07`, `d09`, `d11` and `d15` reuse the helper functions of `d05_experiments.py`, which must be in the same directory. It must be run first, because it creates `runs/exp`. The total time is about 6 minutes (around 150 SWMM runs of 1–3 s).

| Script | Arguments | What it shows |
|---|---|---|
| `d01_db.py` | path to the DB | Counts, continuity NULLs, label quantization, q_pico/base uniformity |
| `d02_shapes.py` | shapes folder | Peak normalization, q_end, ∫q_norm, drain ramp |
| `d03_base_inp.py` | `.inp` | OPTIONS, InitFlow = Σ peaks, truncated recession, slopes, topology |
| `d04_base_run.py` | repo, factor | Format of the *Node Flooding Summary*, continuity, text parser versus swmm_api |
| `d05_experiments.py` | repo | InitFlow = 0, extended END, ramp versus cut |
| `d06_qx_compare.py` | repo | Qx2/Qx3 with the same geometry and InitFlow as Qx1 (ends with a KeyError on reaching Steady, which is another network; the Qx2/Qx3 results are printed before) |
| `d07_more.py` | repo | Ramp at f = 4–5 and InitFlow in short shapes |
| `d08_pyswmm_keys.py` | control `.inp` | Actual PySWMM keys and variable dt |
| `d09_continuity_sweep.py` | repo | CE sweep over 80 runs |
| `d10_validation_csvs.py` | repo | Format of the validation shapes |
| `d11_quantization.py` | repo | `.rpt` volume versus PySWMM (quantization) |
| `d12_db_shapes.py` | path to the DB | Volume per shape at equal duration; rows with vol = 1.0 |
| `d13_manning.py` | repo, `.rpt` | Manning capacity versus SWMM; slopes; static NaNs |
| `d14_parity.py` | repo | Feature parity between training and inference; rounding to minutes |
| `d15_identity.py` | repo | Pipeline f = 1 ≡ original; Qx2/Qx3 ≡ Qx1 × 2/× 3 |
| `d16_capacity_index.py` | repo | AUC of upstream capacity versus outlet capacity |
| `d17_tpico_inferencia.py` | repo | Inference t_pico taken from the first series |
| `d18_db_checks.py` | path to the DB | DB totals that are cross-checked with the re-simulations (V-03); physical threshold (V-12) |

Full code:


### `d01_db.py`

```python
import sqlite3, sys
db = sys.argv[1]
c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
q = lambda s: c.execute(s).fetchall()
print("networks:", q("select network_id,name,source_filename,flow_units,length(inp_bytes) from networks"))
print("n scenarios:", q("select count(*) from scenarios"), "runs:", q("select status,count(*),sum(continuity_error_pct is null) from runs group by status"))
print("shapes:", q("select count(distinct shape_id) from scenarios"), "factors:", q("select count(distinct factor_mult), min(factor_mult), max(factor_mult) from scenarios"))
print("dur/tpico por shape (base):", q("select shape_id, duracion_horas, tiempo_al_pico_h from scenarios where shape_id='base' limit 3"))
print("dur/tpico distintos:", q("select shape_id, min(duracion_horas),min(tiempo_al_pico_h) from scenarios group by shape_id order by 2"))
print("nodos:", q("select count(*), sum(base_inflow_lps>0) from nodes"))
print("filas training:", q("select count(*), sum(inunda), avg(inunda) from training_samples_v17"))
print("vol en (0,1):", q("select count(*) from training_samples_v17 where vol_inundacion_m3>0 and vol_inundacion_m3<1"))
print("vol en [1,2):", q("select count(*) from training_samples_v17 where vol_inundacion_m3>=1 and vol_inundacion_m3<2"))
print("vol==0 & inunda=1:", q("select count(*) from training_samples_v17 where vol_inundacion_m3<1 and inunda=1"))
print("min vol positivo:", q("select min(vol_inundacion_m3) from training_samples_v17 where vol_inundacion_m3>0"))
print("cuantiles vol>0 (n, sum):", q("select count(*), sum(vol_inundacion_m3) from training_samples_v17 where vol_inundacion_m3>0"))
print("vol perdida por umbral (suma vol 0<v<1):", q("select sum(vol_inundacion_m3) from training_samples_v17 where vol_inundacion_m3>0 and vol_inundacion_m3<1"))
print("ratio q_pico_nodo/base vs factor (max abs dev):", q("select max(abs(q_pico_nodo/base_inflow_lps - factor_mult)) from training_samples_v17 where base_inflow_lps>0"))
print("node_timeseries rows:", q("select count(*) from node_timeseries"), "scenario_inflows:", q("select count(*) from scenario_inflows"))
```

### `d02_shapes.py`

```python
import csv, glob, os, sys
import numpy as np
root = sys.argv[1]
print(f"{'shape':40s} {'n':>4s} {'dt_min':>12s} {'dur_h':>6s} {'max':>6s} {'q0':>6s} {'qlast':>6s} {'int_qn_h':>8s} {'dd_ramp_h':>9s} {'sub_min':>7s}")
for p in sorted(glob.glob(os.path.join(root, "*.csv"))):
    rows = list(csv.DictReader(open(p)))
    t = np.array([float(r["time_h"]) for r in rows]); q = np.array([float(r["q_norm"]) for r in rows])
    dt = np.diff(t)*60
    integ = np.trapz(q, t)
    # extra volume from drain-down ramp (last -> 0 over 6 h), in units of peak*hour
    ramp = 0.5*q[-1]*6.0
    submin = np.any(np.abs(t*60 - np.round(t*60)) > 1e-6)
    print(f"{os.path.basename(p)[:-4]:40s} {len(t):4d} {dt.min():5.2f}-{dt.max():5.2f} {t[-1]:6.2f} {q.max():6.3f} {q[0]:6.3f} {q[-1]:6.3f} {integ:8.3f} {ramp:9.3f} {str(submin):>7s}")
```

### `d03_base_inp.py`

```python
"""Analiza las series embebidas y la geometria del .inp base (forma 'base'). Solo lectura."""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
from swmm_api import read_inp_file
inp = read_inp_file(sys.argv[1], encoding="utf-8")
o = inp["OPTIONS"]
print("OPTIONS:", {k: o.get(k) for k in ["FLOW_UNITS","FLOW_ROUTING","ALLOW_PONDING","START_TIME","END_TIME","REPORT_STEP","ROUTING_STEP","VARIABLE_STEP","MIN_SLOPE","LINK_OFFSETS","SKIP_STEADY_STATE","NORMAL_FLOW_LIMITED","MINIMUM_STEP"]})
print("n junctions", len(inp["JUNCTIONS"]), "outfalls", list(inp["OUTFALLS"].keys()), "conduits", len(inp["CONDUITS"]))
print("tipo de tiempo en TIMESERIES:", type(next(iter(inp["TIMESERIES"].values())).data[0][0]))
mf = {k: (v.scale_factor, v.base_value) for k, v in inp["INFLOWS"].items()}
print("INFLOWS con Mfactor!=1 o Baseline!=0:", [k for k,(m,b) in mf.items() if (m not in (1,1.0,None)) or (b not in (0,0.0,None))])
P={}; R=[]; TP=[]; VP=[]; TR=[]; SH=[]
for name, ts in inp["TIMESERIES"].items():
    t = np.array([float(a) for a,_ in ts.data]); q = np.array([float(b) for _,b in ts.data])
    P[name]=q.max()
    if q.max() <= 0: continue
    R.append(q[-1]/q.max()); TP.append(t[q.argmax()]); VP.append(np.trapz(q,t)/q.max()); SH.append(q/q.max())
    k = np.log(q[-2]/q[-1])/(t[-1]-t[-2]); TR.append((q[-1]/k)/np.trapz(q,t))
R,TP,VP,TR=map(np.array,(R,TP,VP,TR))
pk=np.array(list(P.values()))
print("n series", len(pk), "con pico 0:", int((pk<=0).sum()), " pico>0 [min,med,max] L/s:", pk[pk>0].min(), np.median(pk[pk>0]), pk.max(), " suma picos:", pk.sum())
print("q_final(3h)/q_pico: min %.3f med %.3f max %.3f" % (R.min(), np.median(R), R.max()))
print("t_pico (h) unique:", np.unique(np.round(TP,3)))
print("integral/pico (h): min %.3f med %.3f max %.3f" % (VP.min(), np.median(VP), VP.max()))
S=np.vstack(SH); print("max desviacion entre formas normalizadas de distintos nodos:", float(np.abs(S-S[0]).max()))
print("recesion no simulada tras 3 h (extrap. exponencial) / volumen simulado: min %.3f med %.3f max %.3f" % (TR.min(), np.median(TR), TR.max()))
init = {lid: float(c.flow_initial or 0) for lid,c in inp["CONDUITS"].items()}
print("conduits con InitFlow>0:", sum(v>0 for v in init.values()), "de", len(init), " max InitFlow:", max(init.values()))
lens = [float(c.length) for c in inp["CONDUITS"].values()]
print("longitud min/max:", min(lens), max(lens))
print("conduits con offsets != 0:", sum(1 for c in inp["CONDUITS"].values() if (c.offset_upstream or 0) or (c.offset_downstream or 0)))
print("formas XSECTIONS:", {x.shape for x in inp["XSECTIONS"].values()}, " barrels!=1:", sum(1 for x in inp["XSECTIONS"].values() if (x.n_barrels or 1)!=1))
print("rugosidades:", {float(c.roughness) for c in inp["CONDUITS"].values()})
J = inp["JUNCTIONS"]
print("SurDepth>0:", sum(1 for j in J.values() if (j.depth_surcharge or 0)>0), " Aponded>0:", sum(1 for j in J.values() if (j.area_ponded or 0)>0), " InitDepth>0:", sum(1 for j in J.values() if (j.depth_init or 0)>0))
print("MaxDepth range:", min(j.depth_max for j in J.values()), max(j.depth_max for j in J.values()))
for sec in ["STORAGE","DIVIDERS","PUMPS","ORIFICES","WEIRS","OUTLETS","SUBCATCHMENTS","DWF","RDII","LOSSES","CONTROLS","CURVES"]:
    if sec in inp: print("seccion presente:", sec, len(inp[sec]))
# InitFlow vs suma de picos aguas arriba
import networkx as nx
G=nx.DiGraph()
for lid,c in inp["CONDUITS"].items(): G.add_edge(str(c.from_node),str(c.to_node),lid=lid)
dev=[]
for lid,c in inp["CONDUITS"].items():
    n=str(c.from_node); acc=sum(P.get(a,0) for a in nx.ancestors(G,n)|{n})
    dev.append(abs(acc-init[lid]))
print("max |InitFlow - suma picos aguas arriba del nodo de entrada| (L/s):", max(dev))
print("grafo: aciclico:", nx.is_directed_acyclic_graph(G), " max out_degree:", max(d for _,d in G.out_degree()), " nodos con out_degree 0:", [n for n,d in G.out_degree() if d==0])
print("nodos sin conductos:", [n for n in J if str(n) not in G])
```

### `d04_base_run.py`

```python
"""E1: corre la forma 'base' exactamente como run_batch (run_simulation_simple) sobre una COPIA
en el scratchpad, y compara parsers de volumen de inundacion. Uso: python d04_base_run.py <repo> <factor>"""
import sys, shutil, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
repo = Path(sys.argv[1]); factor = float(sys.argv[2])
sys.path.insert(0, str(repo))
from swmm_resilience.simulation.runner import run_simulation_simple
from swmm_resilience.extraction.labels import _parse_node_flooding_text, read_flooding_volumes
inp = repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
rd = Path("runs") / f"base_f{factor:.2f}"
rpt = run_simulation_simple(inp, factor, rd)
txt = rpt.read_text(errors="replace")
i = txt.find("Node Flooding Summary"); print(txt[i-10:i+900] if i>=0 else "SIN SECCION DE INUNDACION")
for key in ["Flow Routing Continuity", "Continuity Error"]:
    j = txt.find(key); print(txt[j:j+1100] if key=="Flow Routing Continuity" else "")
a = read_flooding_volumes(rpt); b = _parse_node_flooding_text(rpt)
m = a.merge(b, on="node_id", suffixes=("_swmmapi","_texto"), how="outer")
print("nodos inundados:", len(a), " suma vol swmm_api:", a.flooding_volume_m3.sum(), " suma vol parser texto:", b.flooding_volume_m3.sum())
print("max |dif| parsers:", (m.flooding_volume_m3_swmmapi - m.flooding_volume_m3_texto).abs().max())
```

### `d05_experiments.py`

```python
"""Experimentos de control (copias en scratchpad, nunca toca el repo).
Construye los .inp con las MISMAS funciones del pipeline (write_scaled_inp / apply_shape +
write_scenario_inp) y luego aplica variantes (InitFlow=0, END_TIME extendido, rampa vs corte a 0).
Uso: python d05_experiments.py <repo>"""
import sys, re, warnings, datetime, shutil
from pathlib import Path
warnings.filterwarnings("ignore")
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo))
import pandas as pd
from pyswmm import Simulation
from swmm_api import read_inp_file
from swmm_resilience.simulation.swmm_api_io import write_scaled_inp
from swmm_resilience.simulation.hydrograph_shapes import load_shape, apply_shape
from swmm_resilience.simulation.timeseries_scenario import write_scenario_inp
from swmm_resilience.validation.hydrograph_csv import HydrographScenario
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.extraction.labels import extract_labels

INP = repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
SHAPES = repo / "data/hydrograph_shapes"
OUT = Path("runs/exp"); OUT.mkdir(parents=True, exist_ok=True)
static = extract_static_features(INP)
base_inflows = dict(zip(static.node_id.astype(str), static.base_inflow_lps))
nodes = static.node_id.astype(str).tolist()

def rpt_metrics(rpt):
    t = Path(rpt).read_text(errors="replace")
    g = lambda lab: float(re.search(lab + r" \.+\s+(-?[\d.]+)\s+(-?[\d.]+)", t).group(2)) * 1000  # 10^6 L -> m3
    ce = re.findall(r"Continuity Error \(%\) \.+\s+(-?[\d.]+)", t)
    lab = extract_labels(Path(rpt), nodes, 1.0)
    return dict(ext_in_m3=g("External Inflow"), out_m3=g("External Outflow"), flood_loss_m3=g("Flooding Loss"),
                init_stor_m3=g("Initial Stored Volume"), final_stor_m3=g("Final Stored Volume"),
                cont_err_pct=float(ce[-1]), n_inunda=int(lab.inunda.sum()), vol_labels_m3=float(lab.vol_inundacion_m3.sum()))

def run(inp_path):
    with Simulation(str(inp_path)) as sim:
        for _ in sim: pass
    return rpt_metrics(Path(inp_path).with_suffix(".rpt"))

def variant(src, dst, init_zero=False, end_hours=None, cut_after=None):
    inp = read_inp_file(str(src), encoding="utf-8")
    if init_zero:
        for c in inp["CONDUITS"].values(): c.flow_initial = 0.0
    if cut_after is not None:  # reemplaza el punto de drenaje (rampa) por un corte a 0 un minuto despues
        for ts in inp["TIMESERIES"].values():
            d = list(ts.data)
            if len(d) >= 2 and d[-1][1] == 0.0 and d[-2][0] == cut_after:
                d[-1] = (cut_after + 1/60, 0.0); ts.data = d
    if end_hours is not None:
        o = inp["OPTIONS"]; st = datetime.datetime.combine(o["START_DATE"], o["START_TIME"]); e = st + datetime.timedelta(hours=end_hours)
        o["END_DATE"] = e.date(); o["END_TIME"] = e.strftime("%H:%M:%S")
    inp.write_file(str(dst)); return dst

rows = []
def rec(name, inp_path): 
    m = run(inp_path); m["caso"] = name; rows.append(m); print(name, m, flush=True)

# --- Forma base (run_batch): InitFlow y truncamiento a 3 h
for f in (0.2, 1.0, 3.0, 5.0):
    p = OUT / f"base_f{f}.inp"; write_scaled_inp(str(INP), f, None, p, scenario_mode="timeseries")
    rec(f"base f={f} (pipeline)", p)
    rec(f"base f={f} InitFlow=0", variant(p, OUT / f"base_f{f}_i0.inp", init_zero=True))
    rec(f"base f={f} END=9h", variant(p, OUT / f"base_f{f}_e9.inp", end_hours=9.0))
    rec(f"base f={f} InitFlow=0 END=9h", variant(p, OUT / f"base_f{f}_i0e9.inp", init_zero=True, end_hours=9.0))

# --- Formas con drenaje: rampa lineal del ultimo valor a 0 en 6 h vs corte a 0
for shape_id, f in (("shape_2h", 1.0), ("shape_2h", 3.0), ("long_moderate", 1.0), ("long_moderate", 3.0), ("shape_ultra_short_impulse", 1.0), ("shape_ultra_short_impulse", 5.0)):
    shape = load_shape(SHAPES / f"{shape_id}.csv")
    sc = HydrographScenario(f"{shape_id}_f{f:.3f}", apply_shape(shape, base_inflows, f), [t for t,_ in shape], shape[-1][0])
    p = write_scenario_inp(INP, sc, OUT / "shapes", drain_down_hours=6.0)
    rec(f"{shape_id} f={f} (pipeline)", p)
    rec(f"{shape_id} f={f} InitFlow=0", variant(p, OUT / f"{shape_id}_f{f}_i0.inp", init_zero=True))
    if shape[-1][1] > 0:
        rec(f"{shape_id} f={f} corte a 0 (sin rampa)", variant(p, OUT / f"{shape_id}_f{f}_cut.inp", cut_after=shape[-1][0]))
    # volumen nominal del hidrograma (trapecio) sin y con rampa
    import numpy as np
    vn = sum(np.trapz([v for _,v in s], [t for t,_ in s]) for s in sc.node_series.values()) * 3.6  # L/s*h -> m3
    vr = sum(0.5 * s[-1][1] * 6.0 for s in sc.node_series.values()) * 3.6
    print(f"   volumen nominal {vn:.1f} m3; volumen anadido por la rampa de drenaje {vr:.1f} m3 ({100*vr/vn:.1f} %)")

pd.DataFrame(rows).to_csv("d05_results.csv", index=False)
```

### `d06_qx_compare.py`

```python
"""Compara Qx2/Qx3/Steady contra Qx1 (geometria, InitFlow, series) y corre Qx2 vs Qx1*2 (pipeline)."""
import sys, warnings, re
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo))
from swmm_api import read_inp_file
N = repo / "data/networks"
q1 = read_inp_file(str(N/"chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"), encoding="utf-8")
for name, rel, k in [("Qx2","chico_hydro-qx2/SWMM - Chico (PVC) Prueba 5 - Qx2.00.inp",2),("Qx3","chico_hydro-qx3/SWMM - Chico (PVC) Prueba 5 - Qx3.00.inp",3),("Steady","chico_steady/SWMM - Chico (PVC) Prueba 1 - Steady.inp",None)]:
    q = read_inp_file(str(N/rel), encoding="utf-8")
    dz = max(abs(q["JUNCTIONS"][j].elevation - q1["JUNCTIONS"][j].elevation) for j in q1["JUNCTIONS"])
    dd = max(abs(q["JUNCTIONS"][j].depth_max - q1["JUNCTIONS"][j].depth_max) for j in q1["JUNCTIONS"])
    dg = max(abs(q["XSECTIONS"][l].height - q1["XSECTIONS"][l].height) for l in q1["XSECTIONS"])
    ri = [q["CONDUITS"][l].flow_initial / q1["CONDUITS"][l].flow_initial for l in q1["CONDUITS"] if q1["CONDUITS"][l].flow_initial]
    print(f"{name}: max|dElev|={dz} max|dMaxDepth|={dd} max|dDiam|={dg} InitFlow ratio vs Qx1: {min(ri):.3f}-{max(ri):.3f} END={q['OPTIONS'].get('END_TIME')}")
    if "TIMESERIES" in q and k:
        r = []
        for ts in q1["TIMESERIES"]:
            a = np.array([v for _,v in q1["TIMESERIES"][ts].data]); b = np.array([v for _,v in q["TIMESERIES"][ts].data])
            m = a>0; r += list(b[m]/a[m])
        print(f"   serie Qx/Qx1: min {min(r):.4f} max {max(r):.4f}")
    if "INFLOWS" in q and not k:
        print("   Steady INFLOWS baseline sum:", sum((v.base_value or 0) for v in q["INFLOWS"].values()), " timeseries:", "TIMESERIES" in q)
```

### `d07_more.py`

```python
"""E-rampa a factor alto y E-InitFlow en formas cortas (copias, scratchpad). Reusa helpers de d05."""
import sys, time
sys.argv = [sys.argv[0], sys.argv[1]]
src = open("d05_experiments.py", encoding="utf-8").read().split("rows = []")[0]
exec(src)
rows = []
def rec(name, p):
    t0 = time.perf_counter(); m = run(p); m["caso"] = name; m["t_s"] = round(time.perf_counter()-t0, 1); rows.append(m)
    print(name, {k: m[k] for k in ("ext_in_m3","flood_loss_m3","vol_labels_m3","n_inunda","cont_err_pct","t_s")}, flush=True)
def build(shape_id, f):
    shape = load_shape(SHAPES / f"{shape_id}.csv")
    sc = HydrographScenario(f"{shape_id}_f{f:.3f}", apply_shape(shape, base_inflows, f), [t for t,_ in shape], shape[-1][0])
    return shape, write_scenario_inp(INP, sc, OUT / "shapes2", drain_down_hours=6.0)
for shape_id in ("long_moderate", "shape_2h", "shape_5h"):
    for f in (4.0, 5.0):
        shape, p = build(shape_id, f)
        rec(f"{shape_id} f={f} rampa (pipeline)", p)
        rec(f"{shape_id} f={f} corte a 0", variant(p, OUT / f"{shape_id}_f{f}_cut.inp", cut_after=shape[-1][0]))
for shape_id in ("shape_micro_burst_45min", "rare_extreme", "shape_flash_flood_urban", "common_bogota"):
    for f in (2.0, 5.0):
        shape, p = build(shape_id, f)
        rec(f"{shape_id} f={f} (pipeline)", p)
        rec(f"{shape_id} f={f} InitFlow=0", variant(p, OUT / f"{shape_id}_f{f}_i0.inp", init_zero=True))
pd.DataFrame(rows).to_csv("d07_results.csv", index=False)
```

### `d08_pyswmm_keys.py`

```python
"""Verifica claves de estadisticas PySWMM usadas por runner.run_simulation (ruta legacy) y el paso de tiempo."""
import sys, warnings
warnings.filterwarnings("ignore")
from pyswmm import Simulation, Nodes, Links
p = sys.argv[1]
dts = []; prev = None
with Simulation(p) as sim:
    st = sim.start_time
    for _ in sim:
        t = (sim.current_time - st).total_seconds()
        if prev is not None: dts.append(t - prev)
        prev = t
    n = Nodes(sim)["1C"]; l = list(Links(sim))[0]
    print("node.statistics keys:", sorted(n.statistics.keys()))
    print("  'time_flooded' presente:", "time_flooded" in n.statistics, " 'time_max_depth' presente:", "time_max_depth" in n.statistics)
    print("  flooding_volume (pyswmm) 1C:", n.statistics.get("flooding_volume"), " flooding_duration:", n.statistics.get("flooding_duration"))
    print("conduit_statistics keys:", sorted(l.conduit_statistics.keys()))
import numpy as np
dts = np.array(dts); print("pasos de rutina: n=%d dt min=%.2f s, mediana=%.2f s, max=%.2f s, distintos=%d" % (len(dts), dts.min(), np.median(dts), dts.max(), len(np.unique(np.round(dts,3)))))
```

### `d09_continuity_sweep.py`

```python
"""Barrido de control del error de continuidad (configuracion EXACTA del pipeline) para las 41 formas
(base + 40 CSV) a factores 0.2 y 1.0 (82 corridas, ~1.5 s c/u). Copias en scratchpad."""
import sys
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open("d05_experiments.py", encoding="utf-8").read().split("rows = []")[0])
import glob, os
rows = []
for f in (0.2, 1.0):
    p = OUT / f"sweep_base_f{f}.inp"; write_scaled_inp(str(INP), f, None, p, scenario_mode="timeseries")
    m = run(p); m.update(shape="base", factor=f); rows.append(m)
    for csvp in sorted(glob.glob(str(SHAPES / "*.csv"))):
        sid = os.path.basename(csvp)[:-4]; shape = load_shape(Path(csvp))
        sc = HydrographScenario(f"{sid}_f{f:.3f}", apply_shape(shape, base_inflows, f), [t for t,_ in shape], shape[-1][0])
        p = write_scenario_inp(INP, sc, OUT / "sweep", drain_down_hours=6.0)
        m = run(p); m.update(shape=sid, factor=f); rows.append(m)
df = pd.DataFrame(rows); df.to_csv("d09_continuity.csv", index=False)
df["ratio_init_ext"] = df.init_stor_m3 / df.ext_in_m3
for f, g in df.groupby("factor"):
    print(f"factor {f}: n={len(g)}  |CE|>5%: {(g.cont_err_pct.abs()>5).sum()}  |CE|>1%: {(g.cont_err_pct.abs()>1).sum()}  CE max={g.cont_err_pct.max():.2f}%  mediana={g.cont_err_pct.median():.2f}%")
print(df.sort_values("cont_err_pct", ascending=False)[["shape","factor","ext_in_m3","init_stor_m3","ratio_init_ext","cont_err_pct","n_inunda"]].head(12).to_string(index=False))
print("corr(CE, init/ext):", df[["cont_err_pct","ratio_init_ext"]].corr().iloc[0,1].round(3))
```

### `d10_validation_csvs.py`

```python
"""Caracteriza los CSV de validacion out-of-sample: forma por nodo, pico/base, tiempo al pico por nodo, fin != 0."""
import sys, glob, os, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo))
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.validation.hydrograph_csv import load_scenario
inp = repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
st = extract_static_features(inp); base = dict(zip(st.node_id.astype(str), st.base_inflow_lps))
exp = {n for n,v in base.items() if v > 0}
for p in sorted(glob.glob(str(repo / "data/hydrograph_validation/*.csv"))):
    try:
        sc = load_scenario(Path(p), exp)
    except Exception as e:
        df = pd.read_csv(p); print(os.path.basename(p), "NO CARGA con expected_nodes(base>0):", str(e)[:120], "cols:", list(df.columns)[:5]); continue
    ratios = []; tps = []; ends = []; forms = []; vols = []
    for n, s in sc.node_series.items():
        t = np.array([a for a,_ in s]); q = np.array([b for _,b in s])
        ratios.append(q.max()/base[n]); tps.append(t[q.argmax()]); ends.append(q[-1]/q.max()); forms.append(q/q.max()); vols.append(np.trapz(q,t)/q.max())
    F = np.vstack(forms); first = next(iter(sc.node_series))
    print(f"{sc.scenario_id:28s} dur={sc.last_time_hours:5.2f}h dt={60*np.diff(sc.time_grid_hours).min():4.1f}min pico/base=[{min(ratios):.2f},{max(ratios):.2f}] "
          f"t_pico nodos=[{min(tps):.2f},{max(tps):.2f}] (feature usa nodo '{first}': {tps[0]:.2f}) q_fin/pico max={max(ends):.3f} "
          f"desv. forma entre nodos={np.abs(F-F[0]).max():.3f} integral/pico=[{min(vols):.2f},{max(vols):.2f}]")
```

### `d11_quantization.py`

```python
"""Compara el volumen de inundacion del .rpt (3 decimales en 10^6 L => resolucion 1 m3) con el
volumen continuo de PySWMM (node.statistics['flooding_volume'], m3) en corridas de control."""
import sys
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open("d05_experiments.py", encoding="utf-8").read().split("rows = []")[0])
import numpy as np
from pyswmm import Nodes
def both(inp_path):
    prec = {}
    with Simulation(str(inp_path)) as sim:
        for _ in sim: pass
        for n in Nodes(sim):
            if n.is_junction(): prec[n.nodeid] = n.statistics["flooding_volume"]
    lab = extract_labels(Path(inp_path).with_suffix(".rpt"), nodes, 1.0).set_index("node_id")
    d = lab.join(pd.Series(prec, name="vol_pyswmm"))
    return d
tot = []
cases = [("base", 1.6), ("base", 2.0), ("shape_2h", 1.6), ("shape_2h", 2.0), ("rare_extreme", 2.0), ("common_storm", 1.6)]
for sid, f in cases:
    if sid == "base":
        p = OUT / f"q_base_f{f}.inp"; write_scaled_inp(str(INP), f, None, p, scenario_mode="timeseries")
    else:
        shape = load_shape(SHAPES / f"{sid}.csv")
        sc = HydrographScenario(f"q_{sid}_f{f:.3f}", apply_shape(shape, base_inflows, f), [t for t,_ in shape], shape[-1][0])
        p = write_scenario_inp(INP, sc, OUT / "quant", drain_down_hours=6.0)
    d = both(p)
    flooded_hidden = d[(d.vol_pyswmm > 0) & (d.inunda == 0)]
    err = (d.vol_inundacion_m3 - d.vol_pyswmm)[d.vol_pyswmm > 0]
    small = d[(d.vol_pyswmm > 0) & (d.vol_pyswmm < 5)]
    print(f"{sid} f={f}: nodos con inundacion>0 (pyswmm)={int((d.vol_pyswmm>0).sum())}, inunda=1 (rpt>=1 m3)={int(d.inunda.sum())}, "
          f"inundan pero etiqueta 0={len(flooded_hidden)} (vol max oculto {flooded_hidden.vol_pyswmm.max() if len(flooded_hidden) else 0:.3f} m3); "
          f"|error cuantizacion| max={err.abs().max():.3f} m3; error relativo max en nodos 0<v<5 m3: {((small.vol_inundacion_m3-small.vol_pyswmm).abs()/small.vol_pyswmm).max() if len(small) else float('nan'):.2f}")
```

### `d12_db_shapes.py`

```python
"""Lectura (mode=ro) de training_v17: volumen total por forma/factor; formas con igual duracion."""
import sqlite3, sys
import pandas as pd
c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
df = pd.read_sql("select shape_id, factor_mult, duracion_horas d, tiempo_al_pico_h tp, sum(vol_inundacion_m3) vol, sum(inunda) n from training_samples_v17 group by shape_id, factor_mult", c)
print("filas con vol exactamente 1.0:", c.execute("select count(*) from training_samples_v17 where vol_inundacion_m3=1.0").fetchone()[0])
print("vol no entero:", c.execute("select count(*) from training_samples_v17 where vol_inundacion_m3 != cast(vol_inundacion_m3 as integer)").fetchone()[0])
for f in (2.0, 3.0, 5.0):
    sub = df[(df.factor_mult - f).abs() < 1e-6].set_index("shape_id")
    for a, b in (("shape_slow_broad_rise", "shape_long_recession_tail"), ("new_hydrograph", "shape_symmetric_moderate"), ("rare_extreme", "shape_flash_flood_urban"), ("shape_2h","common_bogota")):
        print(f"f={f}: {a} (d={sub.loc[a,'d']}, tp={sub.loc[a,'tp']}) vol={sub.loc[a,'vol']:.0f} n={sub.loc[a,'n']} | {b} (d={sub.loc[b,'d']}, tp={sub.loc[b,'tp']}) vol={sub.loc[b,'vol']:.0f} n={sub.loc[b,'n']}")
# mismo factor 1.0: hay inundacion?
print(df[(df.factor_mult-1.0).abs()<1e-6].query("n>0")[["shape_id","vol","n"]].to_string(index=False))
```

### `d13_manning.py`

```python
"""Compara capacidad Manning a tubo lleno de utils.circular_full_flow_lps (via topology/static) con la
capacidad implicita de SWMM (Max Flow / (Max/Full Flow)) del Link Flow Summary; revisa pendientes."""
import sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo))
from swmm_api import read_inp_file, read_rpt_file
from swmm_resilience.utils import circular_full_flow_lps
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.extraction.topology import compute_topology_features
INP = repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
inp = read_inp_file(str(INP), encoding="utf-8")
rpt = read_rpt_file(sys.argv[2])
lfs = rpt.link_flow_summary
print("columnas link_flow_summary:", list(lfs.columns))
J = inp["JUNCTIONS"]; O = inp["OUTFALLS"]
def elev(n): return (J[n].elevation if n in J else O[n].elevation)
rows = []
for lid, c in inp["CONDUITS"].items():
    z1 = elev(c.from_node) + (c.offset_upstream or 0); z2 = elev(c.to_node) + (c.offset_downstream or 0)
    s = (z1 - z2) / c.length
    cap = circular_full_flow_lps(inp["XSECTIONS"][lid].height, abs(s) if s else None, c.roughness)
    r = lfs.loc[lid]
    qmax = float(r.iloc[[i for i,col in enumerate(lfs.columns) if "Maximum" in col and "Flow" in col][0]]) if False else None
    rows.append(dict(link=lid, slope=s, cap_utils=cap, to_outfall=c.to_node in O))
d = pd.DataFrame(rows).set_index("link")
colmax = [c for c in lfs.columns if "Max" in c and "Full" in c and "Flow" in c][0]
colq = [c for c in lfs.columns if ("Maximum" in c or "Max" in c) and ("|Flow|" in c or "Flow" in c) and "Full" not in c and "Time" not in c][0]
d["qmax"] = lfs[colq].astype(float); d["ratio"] = lfs[colmax].astype(float)
d["cap_swmm"] = d.qmax / d.ratio
m = d[(d.ratio > 0.05) & d.cap_utils.notna()]
print(f"conductos comparados: {len(m)}; cap_utils/cap_swmm: min {np.min(m.cap_utils/m.cap_swmm):.4f} mediana {np.median(m.cap_utils/m.cap_swmm):.4f} max {np.max(m.cap_utils/m.cap_swmm):.4f}")
print("pendientes: n<0 (adversas):", int((d.slope<0).sum()), " n==0:", int((d.slope==0).sum()), " min %.5f max %.5f" % (d.slope.min(), d.slope.max()))
print("conductos hacia outfall:", d[d.to_outfall].index.tolist())
st = compute_topology_features(extract_static_features(INP), INP)
print("NaN en features estaticas (n nodos):", st[["diam_max_in","pendiente_max_in","pendiente_out","dist_outfall_m","upstream_capacity_lps"]].isna().sum().to_dict())
print("nodo(s) con pendiente_out NaN y n_tuberias_out>0:", st[(st.pendiente_out.isna()) & (st.n_tuberias_out>0)].node_id.tolist())
```

### `d14_parity.py`

```python
"""Paridad train/serve de features dinamicas: DB (entrenamiento) vs ruta de inferencia
(write_shape_validation_csv -> load_scenario -> mismas lineas que ScenarioPredictor.predict_timed).
Tambien mide el efecto del redondeo a minutos (H:MM) sobre volumen y tiempo al pico. Sin simular."""
import sys, glob, os, sqlite3, tempfile, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo))
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.extraction.topology import compute_topology_features, build_network_graph
from swmm_resilience.extraction.dynamic_features import compute_scenario_dynamic_features
from swmm_resilience.simulation.hydrograph_shapes import load_all_shapes
from swmm_resilience.simulation.swmm_api_io import load_inp
from swmm_resilience.validation.hydrograph_csv import write_shape_validation_csv, load_scenario
from swmm_resilience.ml.scenario_predict import _peak_inflow_lps, _time_to_peak_h
INP = repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
full = compute_topology_features(extract_static_features(INP), INP); G, _ = build_network_graph(load_inp(INP))
base = dict(zip(full.node_id.astype(str), full.base_inflow_lps)); exp = {n for n,v in base.items() if v > 0}
c = sqlite3.connect(f"file:{repo/'outputs/training_v17.sqlite3'}?mode=ro", uri=True)
tmp = Path(tempfile.mkdtemp(dir="."))
f = 2.0; worst = {}
for root in ("data/hydrograph_shapes", "data/hydrograph_validation"):
    for sid, shape in load_all_shapes(repo / root).items():
        sc = load_scenario(write_shape_validation_csv(sid, shape, base, f, tmp / sid), exp)
        pk = {n: _peak_inflow_lps(s) for n, s in sc.node_series.items()}
        rep = next((s for s in sc.node_series.values() if s), [])
        dyn = compute_scenario_dynamic_features(full, pk, G, sc.last_time_hours, _time_to_peak_h(rep)).set_index("node_id")
        # volumen con tiempos redondeados vs originales
        t0 = np.array([t for t,_ in shape]); q0 = np.array([q for _,q in shape])
        t1 = np.array(sc.time_grid_hours)
        dv = np.trapz(q0, t1)/np.trapz(q0, t0) - 1
        msg = f"{sid:38s} redondeo H:MM: max|dt|={60*np.abs(t1-t0).max():.2f} min, dVol={100*dv:+.2f}%"
        if root.endswith("shapes"):
            db = pd.read_sql("select node_id, q_pico_nodo, q_pico_acum_escalado, duracion_horas, tiempo_al_pico_h from training_samples_v17 where shape_id=? and abs(factor_mult-?)<1e-9", c, params=(sid, f)).set_index("node_id")
            j = db.join(dyn, rsuffix="_inf")
            d = {k: float((j[k]-j[k+"_inf"]).abs().max()) for k in ("q_pico_nodo","q_pico_acum_escalado","duracion_horas","tiempo_al_pico_h")}
            msg += "  | max|train-inferencia|: " + ", ".join(f"{k}={v:.4g}" for k,v in d.items())
        print(msg)
```

### `d15_identity.py`

```python
"""Control de identidad: .inp original (copia) vs pipeline factor 1.0; Qx2/Qx3 originales vs Qx1 x2/x3 por pipeline."""
import sys, shutil
sys.argv = [sys.argv[0], sys.argv[1]]
exec(open("d05_experiments.py", encoding="utf-8").read().split("rows = []")[0])
N = repo / "data/networks"
cases = [("Qx1 original (copia)", N/"chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp", None),
         ("Qx1 pipeline f=1.0", None, 1.0),
         ("Qx2 original (copia)", N/"chico_hydro-qx2/SWMM - Chico (PVC) Prueba 5 - Qx2.00.inp", None),
         ("Qx1 pipeline f=2.0", None, 2.0),
         ("Qx3 original (copia)", N/"chico_hydro-qx3/SWMM - Chico (PVC) Prueba 5 - Qx3.00.inp", None),
         ("Qx1 pipeline f=3.0", None, 3.0)]
labs = {}
for name, src, f in cases:
    if src is not None:
        p = OUT / ("id_" + name.split()[0] + ".inp"); shutil.copy(src, p)
    else:
        p = OUT / f"id_f{f}.inp"; write_scaled_inp(str(INP), f, None, p, scenario_mode="timeseries")
    m = run(p); labs[name] = extract_labels(p.with_suffix(".rpt"), nodes, 1.0).set_index("node_id")
    print(name, {k: m[k] for k in ("ext_in_m3","flood_loss_m3","vol_labels_m3","n_inunda","cont_err_pct")})
for a, b in (("Qx2 original (copia)","Qx1 pipeline f=2.0"), ("Qx3 original (copia)","Qx1 pipeline f=3.0"), ("Qx1 original (copia)","Qx1 pipeline f=1.0")):
    d = labs[a].join(labs[b], rsuffix="_b")
    print(f"{a} vs {b}: etiquetas distintas={int((d.inunda!=d.inunda_b).sum())}, max|dVol| nodo={float((d.vol_inundacion_m3-d.vol_inundacion_m3_b).abs().max()):.0f} m3")
```

### `d16_capacity_index.py`

```python
"""Poder discriminante (AUC) de un indice hidraulico q_pico_acum_escalado / capacidad Manning del conducto
de SALIDA vs. features existentes. Lectura mode=ro de la BD; capacidad con utils.circular_full_flow_lps."""
import sys, sqlite3, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score
repo = Path(sys.argv[1]); sys.path.insert(0, str(repo))
from swmm_api import read_inp_file
from swmm_resilience.utils import circular_full_flow_lps
inp = read_inp_file(str(repo / "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"), encoding="utf-8")
J, O = inp["JUNCTIONS"], inp["OUTFALLS"]; el = lambda n: J[n].elevation if n in J else O[n].elevation
cap_out = {}
for lid, c in inp["CONDUITS"].items():
    s = (el(c.from_node) + (c.offset_upstream or 0) - el(c.to_node) - (c.offset_downstream or 0)) / c.length
    cap_out[str(c.from_node)] = circular_full_flow_lps(inp["XSECTIONS"][lid].height, abs(s), c.roughness)
db = sqlite3.connect(f"file:{repo/'outputs/training_v17.sqlite3'}?mode=ro", uri=True)
df = pd.read_sql("select node_id, q_pico_acum_escalado, q_pico_nodo, upstream_capacity_lps, inunda from training_samples_v17", db)
df["cap_out_lps"] = df.node_id.map(cap_out)
df["indice_q_cap_out"] = df.q_pico_acum_escalado / df.cap_out_lps
for col in ("q_pico_acum_escalado", "q_pico_nodo", "indice_q_cap_out"):
    m = df[col].notna(); print(f"AUC({col}) = {roc_auc_score(df.inunda[m], df[col][m]):.3f}  (n={m.sum()})")
m = df.upstream_capacity_lps.notna(); print(f"AUC(q_pico_acum_escalado/upstream_capacity_lps) = {roc_auc_score(df.inunda[m], (df.q_pico_acum_escalado/df.upstream_capacity_lps)[m]):.3f} (n={m.sum()}, excluye {(~m).sum()} filas de cabecera)")
```

### `d17_tpico_inferencia.py`

```python
"""Muestra que ScenarioPredictor toma tiempo_al_pico_h de la PRIMERA serie del CSV (aunque sea nula)."""
import sys; sys.path.insert(0, sys.argv[1])
from swmm_resilience.ml.scenario_predict import _time_to_peak_h
from swmm_resilience.validation.hydrograph_csv import HydrographScenario
sc = HydrographScenario('x', {'3C': [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], '1C': [(0.0, 0.0), (1.0, 12.0), (2.0, 0.0)]}, [0, 1, 2], 2.0)
rep = next((s for s in sc.node_series.values() if s), [])   # misma linea que scenario_predict.py:118
print('t_pico usado por ScenarioPredictor:', _time_to_peak_h(rep), '(real del evento: 1.0 h)')
```

### `d18_db_checks.py`

```python
"""Consultas (mode=ro): totales por escenario para cruzar con las re-simulaciones (V-03) y umbral fisico (V-12)."""
import sqlite3, sys
import pandas as pd
c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
for s, f in [('base',3.0),('base',5.0),('long_moderate',3.0),('long_moderate',5.0),('shape_ultra_short_impulse',5.0),('shape_2h',3.0),('shape_micro_burst_45min',5.0)]:
    print(s, f, c.execute('select sum(vol_inundacion_m3), sum(inunda) from training_samples_v17 where shape_id=? and abs(factor_mult-?)<1e-9', (s, f)).fetchone())
d = pd.read_sql('select shape_id, factor_mult, sum(inunda) n from training_samples_v17 group by 1,2', c)
print('corridas sin ningun nodo inundado:', int((d.n == 0).sum()), 'de', len(d), '; corridas con factor<=1.0:', int((d.factor_mult <= 1.0 + 1e-9).sum()))
first = d[d.n > 0].groupby('shape_id').factor_mult.min()
print('factor minimo con inundacion: min', first.min(), 'max', first.max())
```
