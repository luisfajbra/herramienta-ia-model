# SWMM + XGBoost Hydraulic Failure Predictor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a hydraulic failure prediction pipeline for the Chico Sur urban drainage network that auto-generates a training dataset via SWMM batch simulation and trains classifier + regressor XGBoost models.

**Architecture:** New package layout under existing `swmm_resilience/` with new subfolders `extraction/`, `dataset/`, `visualization/`. Existing `swmm_api_io.py` and `utils.py` are reused unchanged. Old `config.py` and `simulation/runner.py` are replaced with simpler versions.

**Tech Stack:** pyswmm, swmm-api, networkx, pandas, numpy, xgboost, scikit-learn, matplotlib, pyyaml, joblib

---

## Key Design Decisions

- **`config.py`** exports new `Config` dataclass (from `config.yaml`) AND backward-compat constants `SCENARIO_MODE_TIMESERIES`/`SCENARIO_MODE_STEADY` (required by `swmm_api_io.py` which we keep unchanged).
- **`simulation/runner.py`** is replaced with a simple function: `run_simulation(inp_path, factor, run_dir) → rpt_path`.
- **`base_inflow_lps`** = max value from the node's timeseries (not baseline/steady), since Chico Sur uses timeseries inflows.
- **Volume units**: `read_node_flooding_summary` in `swmm_api_io.py` already multiplies by 1000 (10⁶ L → m³). Verified in line 279 of that file.
- **NaN propagation**: headwater nodes have `diam_max_in=NaN`, `pendiente_max_in=NaN` → propagated to CSV → imputed by `SimpleImputer` in ML pipeline.
- **Outfall** (`109C`): included in networkx graph for topology, excluded from dataset rows.

## File Map

| File | Action |
|---|---|
| `config.yaml` | Create (root) |
| `swmm_resilience/config.py` | Replace |
| `swmm_resilience/simulation/runner.py` | Replace (simpler) |
| `swmm_resilience/simulation/batch.py` | Create |
| `swmm_resilience/extraction/__init__.py` | Create |
| `swmm_resilience/extraction/static_features.py` | Create |
| `swmm_resilience/extraction/topology.py` | Create |
| `swmm_resilience/extraction/dynamic_features.py` | Create |
| `swmm_resilience/extraction/labels.py` | Create |
| `swmm_resilience/dataset/__init__.py` | Create |
| `swmm_resilience/dataset/assembler.py` | Create |
| `swmm_resilience/dataset/validator.py` | Create |
| `swmm_resilience/ml/trainer.py` | Create |
| `swmm_resilience/ml/evaluator.py` | Create |
| `swmm_resilience/ml/feature_importance.py` | Create |
| `swmm_resilience/ml/predict.py` | Create |
| `swmm_resilience/visualization/__init__.py` | Create |
| `swmm_resilience/visualization/flood_map.py` | Create |
| `main.py` | Replace (root) |

---

## Task 1: config.yaml + config.py

**Files:**
- Create: `config.yaml`
- Replace: `swmm_resilience/config.py`

- [ ] **Step 1: Write config.yaml**

```yaml
# config.yaml (root of herramienta/)
network:
  inp_path: "data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"
  name: "Chico Sur"

simulation:
  factor_min: 0.2
  factor_max: 5.0
  factor_step: 0.2

dataset:
  output_path: "data/training/dataset_final.csv"
  flood_threshold_m3: 0.0

ml:
  classifier:
    algorithm: "xgboost"
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8
    scale_pos_weight: "auto"
  regressor:
    algorithm: "xgboost"
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8
  use_scaler: false

evaluation:
  methods:
    - "LOSO"
    - "GroupKFold5"
  stratify_by_factor: true

visualization:
  factors_to_plot: [1.4, 1.6, 2.0, 3.0, 5.0]
  colormap: "RdYlBu_r"
  output_path: "outputs/maps/"
  show_labels_top_n: 5
```

- [ ] **Step 2: Write swmm_resilience/config.py**

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import yaml

# Backward-compat: swmm_api_io.py imports these
SCENARIO_MODE_TIMESERIES = "timeseries"
SCENARIO_MODE_STEADY = "steady"


@dataclass
class NetworkConfig:
    inp_path: Path
    name: str


@dataclass
class SimulationConfig:
    factor_min: float
    factor_max: float
    factor_step: float


@dataclass
class DatasetConfig:
    output_path: Path
    flood_threshold_m3: float


@dataclass
class ClassifierConfig:
    algorithm: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    scale_pos_weight: Union[str, float]


@dataclass
class RegressorConfig:
    algorithm: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float


@dataclass
class MLConfig:
    classifier: ClassifierConfig
    regressor: RegressorConfig
    use_scaler: bool


@dataclass
class EvaluationConfig:
    methods: list
    stratify_by_factor: bool


@dataclass
class VisualizationConfig:
    factors_to_plot: list
    colormap: str
    output_path: Path
    show_labels_top_n: int


@dataclass
class Config:
    network: NetworkConfig
    simulation: SimulationConfig
    dataset: DatasetConfig
    ml: MLConfig
    evaluation: EvaluationConfig
    visualization: VisualizationConfig

    def factors(self) -> list:
        values = []
        current = self.simulation.factor_min
        while current <= self.simulation.factor_max + 1e-9:
            values.append(round(current, 6))
            current = round(current + self.simulation.factor_step, 6)
        return values


def load_config(config_path: str = "config.yaml") -> Config:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    base_dir = config_path.resolve().parent

    net = raw["network"]
    sim = raw["simulation"]
    ds = raw["dataset"]
    ml = raw["ml"]
    ev = raw["evaluation"]
    viz = raw["visualization"]

    inp_path = base_dir / net["inp_path"]
    if not inp_path.exists():
        raise FileNotFoundError(f"El archivo .inp no existe: {inp_path}")
    if sim["factor_min"] >= sim["factor_max"]:
        raise ValueError("factor_min debe ser menor que factor_max")
    if sim["factor_step"] <= 0:
        raise ValueError("factor_step debe ser mayor que 0")

    clf = ml["classifier"]
    reg = ml["regressor"]
    return Config(
        network=NetworkConfig(inp_path=inp_path, name=net["name"]),
        simulation=SimulationConfig(
            factor_min=float(sim["factor_min"]),
            factor_max=float(sim["factor_max"]),
            factor_step=float(sim["factor_step"]),
        ),
        dataset=DatasetConfig(
            output_path=base_dir / ds["output_path"],
            flood_threshold_m3=float(ds["flood_threshold_m3"]),
        ),
        ml=MLConfig(
            classifier=ClassifierConfig(
                algorithm=clf["algorithm"],
                n_estimators=int(clf["n_estimators"]),
                max_depth=int(clf["max_depth"]),
                learning_rate=float(clf["learning_rate"]),
                subsample=float(clf["subsample"]),
                scale_pos_weight=clf["scale_pos_weight"],
            ),
            regressor=RegressorConfig(
                algorithm=reg["algorithm"],
                n_estimators=int(reg["n_estimators"]),
                max_depth=int(reg["max_depth"]),
                learning_rate=float(reg["learning_rate"]),
                subsample=float(reg["subsample"]),
            ),
            use_scaler=bool(ml["use_scaler"]),
        ),
        evaluation=EvaluationConfig(
            methods=ev["methods"],
            stratify_by_factor=bool(ev["stratify_by_factor"]),
        ),
        visualization=VisualizationConfig(
            factors_to_plot=[float(f) for f in viz["factors_to_plot"]],
            colormap=viz["colormap"],
            output_path=base_dir / viz["output_path"],
            show_labels_top_n=int(viz["show_labels_top_n"]),
        ),
    )
```

- [ ] **Step 3: Quick test**

```bash
cd herramienta
python -c "from swmm_resilience.config import load_config; c = load_config('config.yaml'); print(c.network.name, c.factors()[:3])"
```
Expected output: `Chico Sur [0.2, 0.4, 0.6]`

---

## Task 2: extraction/static_features.py

**Files:**
- Create: `swmm_resilience/extraction/__init__.py`
- Create: `swmm_resilience/extraction/static_features.py`

- [ ] **Step 1: Create empty __init__.py**

```python
# swmm_resilience/extraction/__init__.py
```

- [ ] **Step 2: Write static_features.py**

```python
# swmm_resilience/extraction/static_features.py
import pandas as pd
from pathlib import Path
from ..simulation.swmm_api_io import load_inp


def _get_peak_inflows(inp) -> dict:
    """Return {node_id: peak_flow_lps} — max value from the node's timeseries."""
    if "INFLOWS" not in inp:
        return {}
    ts_map = dict(inp["TIMESERIES"]) if "TIMESERIES" in inp else {}
    result = {}
    for (node, constituent), inflow in inp["INFLOWS"].items():
        if str(constituent).upper() != "FLOW":
            continue
        node = str(node)
        ts_name = inflow.time_series
        if ts_name and str(ts_name).strip() not in ("", '""', "''") and ts_name in ts_map:
            values = [v for _, v in ts_map[ts_name].data]
            result[node] = max(values) if values else 0.0
        elif inflow.base_value:
            result[node] = float(inflow.base_value)
        else:
            result[node] = 0.0
    return result


def extract_static_features(inp_path: Path) -> pd.DataFrame:
    """Extract per-junction static features. Returns 1 row per junction (outfalls excluded).

    Columns: node_id, elev_fondo, prof_max, n_tuberias_in, n_tuberias_out,
             diam_max_in, diam_max_out, pendiente_max_in, pendiente_out,
             base_inflow_lps, coord_x, coord_y
    """
    inp = load_inp(inp_path)

    junctions = {}
    if "JUNCTIONS" in inp:
        for nid, j in inp["JUNCTIONS"].items():
            junctions[str(nid)] = {
                "elev_fondo": float(j.elevation),
                "prof_max": float(j.max_depth),
            }

    coords = {}
    if "COORDINATES" in inp:
        for nid, c in inp["COORDINATES"].items():
            coords[str(nid)] = (float(c.x), float(c.y))

    xsections = {}
    if "XSECTIONS" in inp:
        for lid, x in inp["XSECTIONS"].items():
            xsections[str(lid)] = float(x.height) if x.height is not None else None

    conduits = []
    if "CONDUITS" in inp:
        for lid, c in inp["CONDUITS"].items():
            conduits.append({
                "link_id": str(lid),
                "from_node": str(c.from_node),
                "to_node": str(c.to_node),
                "length": float(c.length) if c.length else 1.0,
                "inlet_offset": float(c.offset_upstream) if c.offset_upstream else 0.0,
                "outlet_offset": float(c.offset_downstream) if c.offset_downstream else 0.0,
            })

    peak_inflows = _get_peak_inflows(inp)

    in_conduits = {nid: [] for nid in junctions}
    out_conduits = {nid: [] for nid in junctions}
    for c in conduits:
        if c["to_node"] in in_conduits:
            in_conduits[c["to_node"]].append(c)
        if c["from_node"] in out_conduits:
            out_conduits[c["from_node"]].append(c)

    def compute_slope(c):
        fe = junctions.get(c["from_node"], {}).get("elev_fondo")
        te = junctions.get(c["to_node"], {}).get("elev_fondo")
        if fe is None or te is None or c["length"] <= 0:
            return None
        return (fe + c["inlet_offset"] - te - c["outlet_offset"]) / c["length"]

    rows = []
    for nid, nd in junctions.items():
        ins = in_conduits[nid]
        outs = out_conduits[nid]

        d_in = [xsections[c["link_id"]] for c in ins if xsections.get(c["link_id"]) is not None]
        d_out = [xsections[c["link_id"]] for c in outs if xsections.get(c["link_id"]) is not None]
        s_in = [s for c in ins if (s := compute_slope(c)) is not None]
        s_out = [s for c in outs if (s := compute_slope(c)) is not None]

        cx, cy = coords.get(nid, (None, None))
        rows.append({
            "node_id": nid,
            "elev_fondo": nd["elev_fondo"],
            "prof_max": nd["prof_max"],
            "n_tuberias_in": len(ins),
            "n_tuberias_out": len(outs),
            "diam_max_in": max(d_in) if d_in else None,
            "diam_max_out": max(d_out) if d_out else None,
            "pendiente_max_in": max(s_in) if s_in else None,
            "pendiente_out": s_out[0] if s_out else None,
            "base_inflow_lps": peak_inflows.get(nid, 0.0),
            "coord_x": cx,
            "coord_y": cy,
        })

    return pd.DataFrame(rows)
```

- [ ] **Step 3: Quick test — verify swmm-api reads Chico Sur .inp**

```python
from swmm_resilience.extraction.static_features import extract_static_features
from pathlib import Path
df = extract_static_features(Path("data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"))
print(df.shape, df.columns.tolist())
print(df[["node_id","base_inflow_lps","diam_max_in"]].head())
```
Expected: shape ~(108, 12), no crash.

---

## Task 3: extraction/topology.py

**Files:**
- Create: `swmm_resilience/extraction/topology.py`

- [ ] **Step 1: Write topology.py**

```python
# swmm_resilience/extraction/topology.py
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from ..simulation.swmm_api_io import load_inp
from ..utils import circular_full_flow_lps


def _build_graph(inp) -> tuple:
    """Build directed graph and return (G, outfalls_set)."""
    G = nx.DiGraph()
    outfalls = set()
    if "JUNCTIONS" in inp:
        for nid in inp["JUNCTIONS"]:
            G.add_node(str(nid))
    if "OUTFALLS" in inp:
        for nid in inp["OUTFALLS"]:
            G.add_node(str(nid))
            outfalls.add(str(nid))
    if "CONDUITS" in inp:
        for lid, c in inp["CONDUITS"].items():
            G.add_edge(str(c.from_node), str(c.to_node),
                       length=float(c.length) if c.length else 0.0,
                       link_id=str(lid))
    return G, outfalls


def compute_topology_features(static_df: pd.DataFrame, inp_path: Path) -> pd.DataFrame:
    """Add topology columns to static_df in-place and return the augmented DataFrame.

    Adds: dist_outfall_m, n_nodos_aguas_arriba, q_pico_acum_base, upstream_capacity_lps
    """
    inp = load_inp(inp_path)
    G, outfalls = _build_graph(inp)

    xsections = {}
    if "XSECTIONS" in inp:
        for lid, x in inp["XSECTIONS"].items():
            xsections[str(lid)] = float(x.height) if x.height is not None else None

    conduit_meta = {}
    if "CONDUITS" in inp:
        for lid, c in inp["CONDUITS"].items():
            conduit_meta[str(lid)] = {
                "from_node": str(c.from_node),
                "to_node": str(c.to_node),
                "length": float(c.length) if c.length else 1.0,
                "roughness": float(c.roughness) if c.roughness else None,
                "inlet_offset": float(c.offset_upstream) if c.offset_upstream else 0.0,
                "outlet_offset": float(c.offset_downstream) if c.offset_downstream else 0.0,
            }

    junction_elev = dict(zip(static_df["node_id"], static_df["elev_fondo"]))
    base_inflows = dict(zip(static_df["node_id"], static_df["base_inflow_lps"]))

    rows = []
    for _, row in static_df.iterrows():
        nid = row["node_id"]

        # dist_outfall_m: shortest weighted path to any outfall
        dist = None
        for outfall in outfalls:
            if nid == outfall:
                dist = 0.0
                break
            if G.has_node(nid) and G.has_node(outfall):
                try:
                    d = nx.shortest_path_length(G, nid, outfall, weight="length")
                    dist = d if dist is None else min(dist, d)
                except nx.NetworkXNoPath:
                    pass

        # n_nodos_aguas_arriba
        ancestors = nx.ancestors(G, nid) if G.has_node(nid) else set()
        n_upstream = len(ancestors)

        # q_pico_acum_base: own + all upstream
        upstream_nodes = ancestors | {nid}
        q_acum = sum(base_inflows.get(n, 0.0) for n in upstream_nodes)

        # upstream_capacity_lps: Manning full-flow for immediate upstream conduits
        cap_total = 0.0
        for pred in G.predecessors(nid):
            lid = G[pred][nid].get("link_id")
            if lid is None:
                continue
            diam = xsections.get(lid)
            meta = conduit_meta.get(lid, {})
            roughness = meta.get("roughness")
            fn_elev = junction_elev.get(meta.get("from_node"))
            tn_elev = junction_elev.get(meta.get("to_node"))
            length = meta.get("length", 1.0)
            if fn_elev is not None and tn_elev is not None and length > 0:
                slope = abs(
                    (fn_elev + meta.get("inlet_offset", 0.0)
                     - tn_elev - meta.get("outlet_offset", 0.0)) / length
                )
            else:
                slope = None
            cap = circular_full_flow_lps(diam, slope, roughness)
            if cap:
                cap_total += cap

        rows.append({
            "node_id": nid,
            "dist_outfall_m": dist,
            "n_nodos_aguas_arriba": n_upstream,
            "q_pico_acum_base": q_acum,
            "upstream_capacity_lps": cap_total if cap_total > 0 else None,
        })

    topo_df = pd.DataFrame(rows)
    return static_df.merge(topo_df, on="node_id", how="left")
```

- [ ] **Step 2: Quick test**

```python
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.extraction.topology import compute_topology_features
from pathlib import Path
inp = Path("data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp")
df = compute_topology_features(extract_static_features(inp), inp)
print(df[["node_id","dist_outfall_m","n_nodos_aguas_arriba","q_pico_acum_base"]].head(10))
print("NaN dist_outfall:", df["dist_outfall_m"].isna().sum())
```
Expected: at least some non-NaN values in `dist_outfall_m`.

---

## Task 4: simulation/runner.py (new)

**Files:**
- Replace: `swmm_resilience/simulation/runner.py`

- [ ] **Step 1: Write new runner.py**

```python
# swmm_resilience/simulation/runner.py
"""Runs a single SWMM simulation for a given factor. Returns .rpt path."""
from contextlib import suppress
from pathlib import Path
from pyswmm import Simulation
from .swmm_api_io import write_scaled_inp


def run_simulation(inp_path: Path, factor: float, run_dir: Path) -> Path:
    """Scale inflows by factor, run SWMM, return path to generated .rpt.

    The original .inp is never modified. A temporary scaled copy is written
    to run_dir and deleted after SWMM finishes; the .rpt persists.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = f"factor_{factor:.4f}"
    tmp_inp = run_dir / f"{stem}.inp"

    write_scaled_inp(str(inp_path), factor, None, str(tmp_inp), scenario_mode="timeseries")

    with Simulation(str(tmp_inp)) as sim:
        for _ in sim:
            pass

    with suppress(OSError):
        tmp_inp.unlink()

    return tmp_inp.with_suffix(".rpt")
```

- [ ] **Step 2: Quick test — run one simulation at factor=1.0**

```python
from swmm_resilience.simulation.runner import run_simulation
from pathlib import Path
import tempfile, os
run_dir = Path(tempfile.mkdtemp(prefix="swmm_test_"))
inp = Path("data/networks/chico_hydro-qx1/SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp")
rpt = run_simulation(inp, 1.0, run_dir)
print("RPT exists:", rpt.exists(), "size:", rpt.stat().st_size if rpt.exists() else "N/A")
```
Expected: RPT exists: True, size > 0.

---

## Task 5: simulation/batch.py

**Files:**
- Create: `swmm_resilience/simulation/batch.py`

- [ ] **Step 1: Write batch.py**

```python
# swmm_resilience/simulation/batch.py
from pathlib import Path
from .runner import run_simulation
from ..config import Config


def run_batch(config: Config, run_dir: Path) -> list:
    """Run SWMM for all factors in config. Returns list of (factor, rpt_path)."""
    factors = config.factors()
    results = []
    for i, factor in enumerate(factors, 1):
        print(f"  [{i:>2}/{len(factors)}] factor={factor:.2f}", end=" ... ", flush=True)
        rpt_path = run_simulation(config.network.inp_path, factor, run_dir)
        print("OK")
        results.append((factor, rpt_path))
    return results
```

- [ ] **Step 2: Quick sanity test (no full run — just import)**

```python
from swmm_resilience.simulation.batch import run_batch
print("batch.py imported OK")
```

---

## Task 6: extraction/labels.py

**Files:**
- Create: `swmm_resilience/extraction/labels.py`

- [ ] **Step 1: Write labels.py**

```python
# swmm_resilience/extraction/labels.py
import pandas as pd
from pathlib import Path
from ..simulation.swmm_api_io import read_node_flooding_summary


def extract_labels(rpt_path: Path, all_node_ids: list, threshold_m3: float = 0.0) -> pd.DataFrame:
    """Parse Node Flooding Summary from .rpt.

    Nodes absent from the .rpt (no flooding) get vol=0, inunda=0.
    read_node_flooding_summary already converts 10^6 L → m³ (×1000).

    Returns DataFrame: node_id, vol_inundacion_m3, inunda
    """
    result = pd.DataFrame({"node_id": [str(n) for n in all_node_ids]})
    result["vol_inundacion_m3"] = 0.0

    df_rpt = read_node_flooding_summary(rpt_path)
    if df_rpt is not None and not df_rpt.empty and "flooding_volume_m3" in df_rpt.columns:
        df_rpt["node_id"] = df_rpt["node_id"].astype(str)
        flood_map = dict(zip(df_rpt["node_id"], df_rpt["flooding_volume_m3"].fillna(0.0)))
        result["vol_inundacion_m3"] = result["node_id"].map(flood_map).fillna(0.0)

    result["inunda"] = (result["vol_inundacion_m3"] > threshold_m3).astype(int)
    return result
```

- [ ] **Step 2: Quick test using the .rpt from Task 4**

```python
from swmm_resilience.extraction.labels import extract_labels
from pathlib import Path
rpt = Path("...")   # path from Task 4 test
df = extract_labels(rpt, ["1A", "2C", "3B"])
print(df)
# Verify: all rows present, inunda matches vol > 0
```
Also check: print the raw .rpt volume column to confirm units (look for "10^6" header in the .rpt text).

---

## Task 7: extraction/dynamic_features.py

**Files:**
- Create: `swmm_resilience/extraction/dynamic_features.py`

- [ ] **Step 1: Write dynamic_features.py**

```python
# swmm_resilience/extraction/dynamic_features.py
import pandas as pd


def compute_dynamic_features(static_topo_df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Compute simulation-level dynamic features for a given factor.

    Returns DataFrame with columns: node_id, factor_mult, q_pico_nodo, q_pico_acum_escalado
    """
    df = static_topo_df[["node_id", "base_inflow_lps", "q_pico_acum_base"]].copy()
    df["factor_mult"] = round(factor, 6)
    df["q_pico_nodo"] = df["base_inflow_lps"] * factor
    df["q_pico_acum_escalado"] = df["q_pico_acum_base"] * factor
    return df[["node_id", "factor_mult", "q_pico_nodo", "q_pico_acum_escalado"]]
```

- [ ] **Step 2: Quick test**

```python
import pandas as pd
from swmm_resilience.extraction.dynamic_features import compute_dynamic_features
dummy = pd.DataFrame({"node_id": ["A","B"], "base_inflow_lps": [10.0, 5.0], "q_pico_acum_base": [15.0, 5.0]})
print(compute_dynamic_features(dummy, 2.0))
# Expected: q_pico_nodo=[20, 10], q_pico_acum_escalado=[30, 10]
```

---

## Task 8: dataset/assembler.py

**Files:**
- Create: `swmm_resilience/dataset/__init__.py`
- Create: `swmm_resilience/dataset/assembler.py`

- [ ] **Step 1: Create empty __init__.py**

```python
# swmm_resilience/dataset/__init__.py
```

- [ ] **Step 2: Write assembler.py**

```python
# swmm_resilience/dataset/assembler.py
import pandas as pd
from pathlib import Path

# Columns that stay constant across simulations (not replicated per factor)
_STATIC_COLS = [
    "node_id", "elev_fondo", "prof_max", "n_tuberias_in", "n_tuberias_out",
    "diam_max_in", "diam_max_out", "pendiente_max_in", "pendiente_out",
    "base_inflow_lps", "dist_outfall_m", "n_nodos_aguas_arriba",
    "q_pico_acum_base", "upstream_capacity_lps", "coord_x", "coord_y",
]


def assemble_dataset(
    static_topo_df: pd.DataFrame,
    simulation_results: list,
    output_path: Path,
) -> pd.DataFrame:
    """Join static+topo features with dynamic features and labels for every factor.

    simulation_results: list of (factor, dynamic_df, labels_df)
    Returns the full dataset and writes it to output_path.
    """
    static_base = static_topo_df[[c for c in _STATIC_COLS if c in static_topo_df.columns]]
    all_rows = []
    for _, dynamic_df, labels_df in simulation_results:
        merged = static_base.merge(dynamic_df, on="node_id", how="left")
        merged = merged.merge(labels_df, on="node_id", how="left")
        all_rows.append(merged)

    dataset = pd.concat(all_rows, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    print(f"Dataset guardado: {output_path}  ({dataset.shape[0]} filas × {dataset.shape[1]} cols)")
    return dataset
```

- [ ] **Step 3: Quick test (no simulation needed)**

```python
import pandas as pd
from swmm_resilience.dataset.assembler import assemble_dataset
from swmm_resilience.extraction.dynamic_features import compute_dynamic_features
from swmm_resilience.extraction.labels import extract_labels
# Use 2 dummy nodes and 2 factors
static = pd.DataFrame({"node_id":["A","B"],"elev_fondo":[10.,5.],"prof_max":[2.,2.],
  "n_tuberias_in":[0,1],"n_tuberias_out":[1,1],"diam_max_in":[None,0.4],
  "diam_max_out":[0.4,0.4],"pendiente_max_in":[None,0.01],"pendiente_out":[0.01,0.01],
  "base_inflow_lps":[5.,3.],"dist_outfall_m":[100.,50.],"n_nodos_aguas_arriba":[0,1],
  "q_pico_acum_base":[5.,8.],"upstream_capacity_lps":[None,30.],"coord_x":[0.,10.],"coord_y":[0.,0.]})
from pathlib import Path; import tempfile
results = []
for f in [1.0, 2.0]:
    dyn = compute_dynamic_features(static, f)
    lab = pd.DataFrame({"node_id":["A","B"],"vol_inundacion_m3":[0.,10.*f],"inunda":[0,1]})
    results.append((f, dyn, lab))
out = Path(tempfile.mktemp(suffix=".csv"))
df = assemble_dataset(static, results, out)
print(df.shape, df.columns.tolist())
assert df.shape == (4, len(df.columns))
```

---

## Task 9: dataset/validator.py

**Files:**
- Create: `swmm_resilience/dataset/validator.py`

- [ ] **Step 1: Write validator.py**

```python
# swmm_resilience/dataset/validator.py
import pandas as pd


def validate_dataset(df: pd.DataFrame, n_nodes: int, n_factors: int) -> None:
    """Validate training dataset before fitting. Raises ValueError on fatal issues."""
    for col in ("inunda", "vol_inundacion_m3"):
        if df[col].isna().any():
            raise ValueError(f"NaN en columna de etiqueta '{col}' — revisa el .rpt")

    if (df["vol_inundacion_m3"] < 0).any():
        raise ValueError("vol_inundacion_m3 contiene valores negativos")

    invalid = ~df["inunda"].isin([0, 1])
    if invalid.any():
        raise ValueError("inunda contiene valores fuera de {0, 1}")

    expected = n_nodes * n_factors
    if len(df) != expected:
        raise ValueError(
            f"Filas esperadas: {n_nodes} nodos × {n_factors} factores = {expected}, "
            f"encontradas: {len(df)}"
        )

    if df["inunda"].sum() == 0:
        raise ValueError("Ningún nodo inunda — verifica que SWMM produce flooding con el rango de factores configurado")

    ratio = df["inunda"].mean()
    if ratio < 0.05:
        print(f"ADVERTENCIA: solo {ratio:.1%} de filas inundan (ratio muy bajo)")
```

- [ ] **Step 2: Quick test**

```python
import pandas as pd
from swmm_resilience.dataset.validator import validate_dataset
df = pd.DataFrame({"inunda":[0,1,0,1],"vol_inundacion_m3":[0.,5.,0.,10.]})
validate_dataset(df, 2, 2)
print("Validator OK")
```

---

## Task 10: ml/trainer.py

**Files:**
- Create: `swmm_resilience/ml/trainer.py`

- [ ] **Step 1: Write trainer.py**

```python
# swmm_resilience/ml/trainer.py
import hashlib
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ..config import Config

FEATURE_COLS = [
    "elev_fondo", "prof_max", "n_tuberias_in", "n_tuberias_out",
    "diam_max_in", "diam_max_out", "pendiente_max_in", "pendiente_out",
    "base_inflow_lps", "dist_outfall_m", "n_nodos_aguas_arriba",
    "q_pico_acum_base", "upstream_capacity_lps",
    "factor_mult", "q_pico_nodo", "q_pico_acum_escalado",
]


def make_classifier(config: Config, scale_pos_weight: float) -> Pipeline:
    clf_cfg = config.ml.classifier
    spw = scale_pos_weight if clf_cfg.scale_pos_weight == "auto" else float(clf_cfg.scale_pos_weight)
    if clf_cfg.algorithm == "xgboost":
        model = XGBClassifier(
            n_estimators=clf_cfg.n_estimators, max_depth=clf_cfg.max_depth,
            learning_rate=clf_cfg.learning_rate, subsample=clf_cfg.subsample,
            scale_pos_weight=spw, eval_metric="logloss", random_state=42,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=clf_cfg.n_estimators, max_depth=clf_cfg.max_depth, random_state=42,
        )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def make_regressor(config: Config) -> Pipeline:
    reg_cfg = config.ml.regressor
    if reg_cfg.algorithm == "xgboost":
        model = XGBRegressor(
            n_estimators=reg_cfg.n_estimators, max_depth=reg_cfg.max_depth,
            learning_rate=reg_cfg.learning_rate, subsample=reg_cfg.subsample, random_state=42,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=reg_cfg.n_estimators, max_depth=reg_cfg.max_depth, random_state=42,
        )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_models(df: pd.DataFrame, config: Config, output_dir: Path) -> tuple:
    """Train classifier and regressor on full dataset. Returns (clf_pipeline, reg_pipeline)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df[FEATURE_COLS]
    y_clf = df["inunda"]
    n_neg, n_pos = (y_clf == 0).sum(), (y_clf == 1).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    clf = make_classifier(config, spw)
    clf.fit(X, y_clf)

    df_flooded = df[df["inunda"] == 1]
    reg = make_regressor(config)
    reg.fit(df_flooded[FEATURE_COLS], df_flooded["vol_inundacion_m3"])

    joblib.dump(clf, output_dir / "classifier.joblib")
    joblib.dump(reg, output_dir / "regressor.joblib")
    (output_dir / "training_inp_hash.txt").write_text(_md5(config.network.inp_path))

    print(f"Modelos guardados en {output_dir}")
    return clf, reg
```

- [ ] **Step 2: Quick test (no real data needed)**

```python
from swmm_resilience.ml.trainer import FEATURE_COLS
print(f"{len(FEATURE_COLS)} features:", FEATURE_COLS)
```

---

## Task 11: ml/evaluator.py

**Files:**
- Create: `swmm_resilience/ml/evaluator.py`

- [ ] **Step 1: Write evaluator.py**

```python
# swmm_resilience/ml/evaluator.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from ..config import Config
from .trainer import FEATURE_COLS, make_classifier, make_regressor


def _nse(y_true, y_pred) -> float:
    """Nash-Sutcliffe Efficiency."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _mean(lst: list, key: str) -> float:
    vals = [d[key] for d in lst if not np.isnan(d.get(key, float("nan")))]
    return float(np.mean(vals)) if vals else float("nan")


def _avg_metrics(lst: list) -> dict:
    if not lst:
        return {}
    return {k: _mean(lst, k) for k in lst[0]}


def _run_cv(df: pd.DataFrame, config: Config, cv) -> dict:
    X = df[FEATURE_COLS].values
    y_clf = df["inunda"].values
    y_reg = df["vol_inundacion_m3"].values
    groups = df["factor_mult"].values

    clf_m, reg_m, e2e_m = [], [], []
    by_factor: dict = {}

    for train_idx, test_idx in cv.split(X, y_clf, groups):
        X_tr, X_te = X[train_idx], X[test_idx]
        yc_tr, yc_te = y_clf[train_idx], y_clf[test_idx]
        yr_tr, yr_te = y_reg[train_idx], y_reg[test_idx]
        g_te = groups[test_idx]

        n_neg, n_pos = (yc_tr == 0).sum(), (yc_tr == 1).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        clf = make_classifier(config, spw)
        clf.fit(X_tr, yc_tr)

        reg = make_regressor(config)
        flooded_tr = yc_tr == 1
        if flooded_tr.sum() > 0:
            reg.fit(X_tr[flooded_tr], yr_tr[flooded_tr])

        # Level 1
        yc_pred = clf.predict(X_te)
        yc_prob = clf.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(yc_te, yc_prob) if yc_te.sum() > 0 and (1 - yc_te).sum() > 0 else float("nan")
        clf_m.append({
            "precision": precision_score(yc_te, yc_pred, zero_division=0),
            "recall": recall_score(yc_te, yc_pred, zero_division=0),
            "f1": f1_score(yc_te, yc_pred, zero_division=0),
            "auc_roc": auc,
        })

        # Level 2 (oracle)
        flooded_te = yc_te == 1
        if flooded_te.sum() > 0:
            yr_pred_oracle = reg.predict(X_te[flooded_te])
            yr_true_oracle = yr_te[flooded_te]
            reg_m.append({
                "nse": _nse(yr_true_oracle, yr_pred_oracle),
                "rmse": float(np.sqrt(mean_squared_error(yr_true_oracle, yr_pred_oracle))),
                "mae": float(mean_absolute_error(yr_true_oracle, yr_pred_oracle)),
                "r2": float(r2_score(yr_true_oracle, yr_pred_oracle)),
            })

        # Level 3 (end-to-end)
        yr_pred_e2e = np.zeros(len(X_te))
        clf_flood_mask = yc_pred == 1
        if clf_flood_mask.sum() > 0:
            yr_pred_e2e[clf_flood_mask] = reg.predict(X_te[clf_flood_mask])
        e2e_m.append({
            "pct_nodos_correctos": float((yc_pred == yc_te).mean()),
            "rmse_vol_todos_nodos": float(np.sqrt(mean_squared_error(yr_te, yr_pred_e2e))),
            "vol_total_pred_m3": float(yr_pred_e2e.sum()),
            "vol_total_real_m3": float(yr_te.sum()),
        })

        # Stratify by factor
        if config.evaluation.stratify_by_factor:
            for fv in np.unique(g_te):
                fmask = g_te == fv
                fkey = f"{fv:.2f}"
                if fkey not in by_factor:
                    by_factor[fkey] = []
                by_factor[fkey].append({
                    "f1": float(f1_score(yc_te[fmask], yc_pred[fmask], zero_division=0)),
                    "rmse_vol": float(np.sqrt(mean_squared_error(yr_te[fmask], yr_pred_e2e[fmask]))),
                })

    result = {
        "classifier": _avg_metrics(clf_m),
        "regressor_oracle": _avg_metrics(reg_m),
        "end_to_end": _avg_metrics(e2e_m),
    }
    if config.evaluation.stratify_by_factor:
        result["by_factor"] = {k: _avg_metrics(v) for k, v in by_factor.items()}
    return result


def evaluate_models(df: pd.DataFrame, config: Config, output_dir: Path) -> dict:
    """Run LOSO and GroupKFold5 evaluation at all 3 levels. Saves 4 JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for method in config.evaluation.methods:
        print(f"  Evaluando {method}...")
        if method == "LOSO":
            cv = LeaveOneGroupOut()
        elif method == "GroupKFold5":
            cv = GroupKFold(n_splits=5)
        else:
            raise ValueError(f"Método de evaluación no soportado: {method}")
        all_results[method] = _run_cv(df, config, cv)

    def _save(data, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    primary = all_results.get("LOSO", next(iter(all_results.values())))
    _save(primary.get("classifier", {}), output_dir / "metrics_classifier.json")
    _save(primary.get("regressor_oracle", {}), output_dir / "metrics_regressor.json")
    _save(primary.get("end_to_end", {}), output_dir / "metrics_endtoend.json")
    _save({m: v.get("by_factor", {}) for m, v in all_results.items()},
          output_dir / "metrics_by_factor.json")

    return all_results
```

- [ ] **Step 2: Quick test (import only)**

```python
from swmm_resilience.ml.evaluator import evaluate_models, _nse
import numpy as np
print("NSE perfect:", _nse(np.array([1.,2.,3.]), np.array([1.,2.,3.])))
print("NSE bad:", _nse(np.array([1.,2.,3.]), np.array([3.,2.,1.])))
```
Expected: `NSE perfect: 1.0`, `NSE bad: -3.0`.

---

## Task 12: ml/feature_importance.py

**Files:**
- Create: `swmm_resilience/ml/feature_importance.py`

- [ ] **Step 1: Write feature_importance.py**

```python
# swmm_resilience/ml/feature_importance.py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from .trainer import FEATURE_COLS


def _plot_importance(pipeline, title: str, output_path: Path):
    model = pipeline.named_steps["model"]
    importances = model.feature_importances_
    df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances})
    df = df.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(FEATURE_COLS))))
    ax.barh(df["feature"], df["importance"], color="steelblue")
    ax.set_xlabel("Importancia")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_feature_importance_plots(clf_pipeline, reg_pipeline, output_dir: Path):
    """Save feature importance bar charts for classifier and regressor."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_importance(clf_pipeline, "Importancia de Variables — Clasificador",
                     output_dir / "feature_importance_classifier.png")
    _plot_importance(reg_pipeline, "Importancia de Variables — Regresor",
                     output_dir / "feature_importance_regressor.png")
    print(f"Gráficos de importancia guardados en {output_dir}")
```

- [ ] **Step 2: Quick import test**

```python
from swmm_resilience.ml.feature_importance import generate_feature_importance_plots
print("feature_importance.py OK")
```

---

## Task 13: ml/predict.py

**Files:**
- Create: `swmm_resilience/ml/predict.py`

- [ ] **Step 1: Write predict.py**

```python
# swmm_resilience/ml/predict.py
import hashlib
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from ..config import Config
from ..extraction.static_features import extract_static_features
from ..extraction.topology import compute_topology_features
from ..extraction.dynamic_features import compute_dynamic_features
from .trainer import FEATURE_COLS


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict_network(factor: float, config: Config, models_dir: Path) -> pd.DataFrame:
    """Predict flooding for all junction nodes at a given factor without running SWMM.

    Returns DataFrame: node_id, inunda_pred, vol_pred_m3, coord_x, coord_y
    """
    clf = joblib.load(models_dir / "classifier.joblib")
    reg = joblib.load(models_dir / "regressor.joblib")

    stored_hash = (models_dir / "training_inp_hash.txt").read_text().strip()
    current_hash = _md5(config.network.inp_path)
    if stored_hash != current_hash:
        raise ValueError(
            f"El .inp en '{config.network.inp_path}' ha cambiado desde el entrenamiento. "
            "Re-entrena el modelo o usa el .inp original."
        )

    if not (config.simulation.factor_min <= factor <= config.simulation.factor_max):
        print(
            f"ADVERTENCIA: factor={factor} fuera del rango de entrenamiento "
            f"[{config.simulation.factor_min}, {config.simulation.factor_max}] — extrapolación no validada"
        )

    static_df = extract_static_features(config.network.inp_path)
    full_df = compute_topology_features(static_df, config.network.inp_path)
    dynamic_df = compute_dynamic_features(full_df, factor)
    merged = full_df.merge(dynamic_df, on="node_id", how="left")

    X = merged[FEATURE_COLS]
    inunda_pred = clf.predict(X)
    vol_pred = np.zeros(len(X))
    flood_mask = inunda_pred == 1
    if flood_mask.sum() > 0:
        vol_pred[flood_mask] = reg.predict(X.values[flood_mask])

    merged["inunda_pred"] = inunda_pred
    merged["vol_pred_m3"] = vol_pred
    return merged[["node_id", "inunda_pred", "vol_pred_m3", "coord_x", "coord_y"]]
```

- [ ] **Step 2: Quick import test**

```python
from swmm_resilience.ml.predict import predict_network
print("predict.py OK")
```

---

## Task 14: visualization/flood_map.py

**Files:**
- Create: `swmm_resilience/visualization/__init__.py`
- Create: `swmm_resilience/visualization/flood_map.py`

- [ ] **Step 1: Create empty __init__.py**

```python
# swmm_resilience/visualization/__init__.py
```

- [ ] **Step 2: Write flood_map.py**

```python
# swmm_resilience/visualization/flood_map.py
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from pathlib import Path
from ..simulation.swmm_api_io import load_inp


def generate_flood_map(
    inp_path: Path,
    vol_data: pd.DataFrame,
    factor: float,
    output_path: Path,
    network_name: str = "Red",
    colormap: str = "RdYlBu_r",
    show_labels_top_n: int = 5,
):
    """Render network with flood gradient. vol_data must have node_id + vol column."""
    inp = load_inp(inp_path)

    coords = {}
    if "COORDINATES" in inp:
        for nid, c in inp["COORDINATES"].items():
            coords[str(nid)] = [float(c.x), float(c.y)]

    conduits = []
    if "CONDUITS" in inp:
        for lid, c in inp["CONDUITS"].items():
            fn, tn = str(c.from_node), str(c.to_node)
            if fn in coords and tn in coords:
                conduits.append((fn, tn))

    # Detect duplicate coordinates and apply ±2m offset
    from collections import defaultdict
    coord_groups: dict = defaultdict(list)
    for nid, (x, y) in coords.items():
        coord_groups[(x, y)].append(nid)
    for (x, y), nodes in coord_groups.items():
        if len(nodes) == 2:
            coords[nodes[0]] = [x - 2.0, y]
            coords[nodes[1]] = [x + 2.0, y]

    vol_col = "vol_inundacion_m3" if "vol_inundacion_m3" in vol_data.columns else "vol_pred_m3"
    node_vol = dict(zip(vol_data["node_id"].astype(str), vol_data[vol_col].fillna(0.0)))

    fig, ax = plt.subplots(figsize=(14, 10))

    for fn, tn in conduits:
        xs = [coords[fn][0], coords[tn][0]]
        ys = [coords[fn][1], coords[tn][1]]
        ax.plot(xs, ys, color="#888888", linewidth=0.7, zorder=1)

    vols = np.array([max(0.0, node_vol.get(nid, 0.0)) for nid in coords])
    max_vol = vols.max() if vols.max() > 0 else 1.0
    norm = mcolors.Normalize(vmin=0, vmax=max_vol)
    cmap = plt.get_cmap(colormap)

    for nid, (x, y) in coords.items():
        vol = max(0.0, node_vol.get(nid, 0.0))
        if vol > 0:
            ax.scatter(x, y, c=[cmap(norm(vol))], s=40 + (vol / max_vol) * 180,
                       zorder=3, edgecolors="none")
        else:
            ax.scatter(x, y, color="#aec6e8", s=15, zorder=2)

    sorted_nids = sorted(coords, key=lambda n: node_vol.get(n, 0.0), reverse=True)
    for nid in sorted_nids[:show_labels_top_n]:
        vol = node_vol.get(nid, 0.0)
        if vol > 0:
            x, y = coords[nid]
            ax.annotate(f"{nid}\n{vol:.1f} m³", (x, y),
                        textcoords="offset points", xytext=(6, 4), fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Volumen de inundación (m³)", shrink=0.7)
    ax.set_title(f"{network_name} — Factor de escala: {factor:.2f}", fontsize=13)
    ax.set_aspect("equal")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Mapa guardado: {output_path}")
```

- [ ] **Step 3: Quick import test**

```python
from swmm_resilience.visualization.flood_map import generate_flood_map
print("flood_map.py OK")
```

---

## Task 15: main.py (root)

**Files:**
- Replace: `main.py` (root of herramienta/)

- [ ] **Step 1: Write main.py**

```python
# main.py
import argparse
import tempfile
from pathlib import Path
import pandas as pd

from swmm_resilience.config import load_config
from swmm_resilience.extraction.static_features import extract_static_features
from swmm_resilience.extraction.topology import compute_topology_features
from swmm_resilience.extraction.dynamic_features import compute_dynamic_features
from swmm_resilience.extraction.labels import extract_labels
from swmm_resilience.dataset.assembler import assemble_dataset
from swmm_resilience.dataset.validator import validate_dataset
from swmm_resilience.ml.trainer import train_models
from swmm_resilience.ml.evaluator import evaluate_models
from swmm_resilience.ml.feature_importance import generate_feature_importance_plots
from swmm_resilience.ml.predict import predict_network
from swmm_resilience.visualization.flood_map import generate_flood_map

MODELS_DIR = Path("outputs/models")
METRICS_DIR = Path("outputs/metrics")


def main():
    parser = argparse.ArgumentParser(description="Pipeline de predicción hidráulica — Chico Sur")
    parser.add_argument("--skip-simulation", action="store_true",
                        help="Saltar simulaciones SWMM (el dataset ya existe)")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Saltar extracción (el dataset CSV ya existe)")
    parser.add_argument("--only-ml", action="store_true",
                        help="Solo entrenar y evaluar (lee dataset CSV existente)")
    parser.add_argument("--only-maps", action="store_true",
                        help="Solo generar mapas de inundación desde el CSV existente")
    parser.add_argument("--predict", action="store_true",
                        help="Inferencia sin SWMM para el factor dado")
    parser.add_argument("--factor", type=float, help="Factor para --predict")
    args = parser.parse_args()

    config = load_config("config.yaml")

    if args.predict:
        if args.factor is None:
            parser.error("--predict requiere --factor VALUE")
        print(f"Prediciendo factor={args.factor}...")
        result = predict_network(args.factor, config, MODELS_DIR)
        map_out = config.visualization.output_path / f"flood_map_pred_{args.factor:.2f}.png"
        generate_flood_map(config.network.inp_path, result.rename(
            columns={"vol_pred_m3": "vol_inundacion_m3"}),
            args.factor, map_out, config.network.name,
            config.visualization.colormap, config.visualization.show_labels_top_n)
        print(result.to_string())
        return

    if args.only_maps:
        df = pd.read_csv(config.dataset.output_path)
        for factor in config.visualization.factors_to_plot:
            df_f = df[abs(df["factor_mult"] - factor) < 1e-6]
            if df_f.empty:
                print(f"Factor {factor} no encontrado en dataset, saltando.")
                continue
            out = config.visualization.output_path / f"flood_map_factor_{factor:.2f}.png"
            generate_flood_map(config.network.inp_path, df_f, factor, out,
                               config.network.name, config.visualization.colormap,
                               config.visualization.show_labels_top_n)
        return

    use_existing_dataset = args.skip_extraction or args.only_ml

    # Steps 1-3: Static + topology features (always needed for node list)
    print("Extrayendo features estáticas...")
    static_df = extract_static_features(config.network.inp_path)
    print(f"  {len(static_df)} nodos junction encontrados")
    print("Calculando features topológicas...")
    static_topo_df = compute_topology_features(static_df, config.network.inp_path)

    all_node_ids = static_topo_df["node_id"].tolist()
    n_nodes = len(all_node_ids)
    factors = config.factors()
    n_factors = len(factors)
    print(f"  Red: {n_nodes} nodos, {n_factors} factores ({factors[0]:.2f}–{factors[-1]:.2f})")

    if not use_existing_dataset:
        # Step 4: Batch simulations
        run_dir = Path(tempfile.mkdtemp(prefix="swmm_runs_"))
        if not args.skip_simulation:
            from swmm_resilience.simulation.batch import run_batch
            print(f"Corriendo {n_factors} simulaciones SWMM...")
            sim_results = run_batch(config, run_dir)
        else:
            # Reconstruct rpt paths from existing files in run_dir (not supported without re-run)
            raise RuntimeError("--skip-simulation sin --skip-extraction requiere los .rpt originales.")

        # Steps 5-7: Dynamic features + labels
        simulation_results = []
        for factor, rpt_path in sim_results:
            dynamic_df = compute_dynamic_features(static_topo_df, factor)
            labels_df = extract_labels(rpt_path, all_node_ids, config.dataset.flood_threshold_m3)
            simulation_results.append((factor, dynamic_df, labels_df))

        # Step 8-9: Assemble + validate
        print("Ensamblando dataset...")
        df = assemble_dataset(static_topo_df, simulation_results, config.dataset.output_path)
        validate_dataset(df, n_nodes, n_factors)
        print(f"Dataset validado: {df.shape}")
    else:
        print(f"Leyendo dataset desde {config.dataset.output_path}...")
        df = pd.read_csv(config.dataset.output_path)

    # Step 10: Train
    print("Entrenando modelos finales...")
    clf, reg = train_models(df, config, MODELS_DIR)

    # Step 11: Evaluate
    print("Evaluando (LOSO + GroupKFold5)...")
    results = evaluate_models(df, config, METRICS_DIR)

    # Step 12: Feature importance
    print("Generando gráficos de importancia...")
    generate_feature_importance_plots(clf, reg, METRICS_DIR)

    # Step 14: Maps
    print("Generando mapas de inundación...")
    for factor in config.visualization.factors_to_plot:
        df_f = df[abs(df["factor_mult"] - factor) < 1e-6]
        if df_f.empty:
            continue
        out = config.visualization.output_path / f"flood_map_factor_{factor:.2f}.png"
        generate_flood_map(config.network.inp_path, df_f, factor, out,
                           config.network.name, config.visualization.colormap,
                           config.visualization.show_labels_top_n)

    # Step 13: Summary
    print("\n=== RESUMEN DE MÉTRICAS (LOSO) ===")
    loso = results.get("LOSO", {})
    for level, label in [("classifier", "Clasificador"), ("regressor_oracle", "Regresor (oracle)"),
                          ("end_to_end", "End-to-end")]:
        print(f"\n{label}:")
        for k, v in loso.get(level, {}).items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

```bash
python main.py --help
```
Expected: argument help displayed with all flags.

---

## Self-Review Checklist

- [x] All 15 modules from spec section 8 covered
- [x] `SCENARIO_MODE_TIMESERIES` / `SCENARIO_MODE_STEADY` kept in new config.py (swmm_api_io.py needs them)
- [x] `read_node_flooding_summary` already handles ×1000 unit conversion (verified in swmm_api_io.py:279)
- [x] NaN propagation from headwater nodes: NOT imputed in extraction, propagated to CSV, imputed by `SimpleImputer` in ML pipeline
- [x] Outfall excluded from junctions (static_features only reads `[JUNCTIONS]` section)
- [x] `.inp` original never modified: runner.py writes to `run_dir/factor_X.inp`, deletes it after sim
- [x] MD5 hash saved at training, validated at inference
- [x] FEATURE_COLS consistent between trainer.py, evaluator.py, predict.py (all import from trainer.py)
- [x] Oracle evaluation uses true labels (not predicted) to filter flooded test rows
- [x] `q_pico_acum_base` is in both static_topo_df AND as a feature column
- [x] `base_inflow_lps` = max of timeseries (not baseline/steady)
