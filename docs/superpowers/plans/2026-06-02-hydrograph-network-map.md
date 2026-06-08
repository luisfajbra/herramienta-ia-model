# Hydrograph & Network Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--hydrograph` (design storm shape) and `--network-map` (topology with pipe classification) as standalone CLI commands that each produce a PNG.

**Architecture:** Two new modules in `swmm_resilience/visualization/`, each with a single public function. Both are driven directly from `main.py` as standalone modes — no dataset or trained models required. Tests monkeypatch `load_inp` / `get_node_inflow_profiles` exactly as `test_visualization.py` does for `flood_map`.

**Tech Stack:** Python 3.10+, matplotlib, swmm-api (via existing `swmm_api_io.py`), pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `swmm_resilience/visualization/hydrograph.py` | Create | `plot_hydrograph(inp_path, output_path) → Path` |
| `swmm_resilience/visualization/network_map.py` | Create | `generate_network_map(inp_path, output_path, network_name) → Path` |
| `tests/test_hydrograph.py` | Create | Monkeypatched unit test for hydrograph |
| `tests/test_network_map.py` | Create | Monkeypatched unit test for network map |
| `main.py` | Modify | Add `--hydrograph` and `--network-map` flags + handlers |

---

## Task 1: Hydrograph Visualization

**Files:**
- Create: `swmm_resilience/visualization/hydrograph.py`
- Create: `tests/test_hydrograph.py`
- Modify: `main.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_hydrograph.py`:

```python
from swmm_resilience.visualization import hydrograph


def test_plot_hydrograph_selects_peak_node_and_writes_png(monkeypatch, tmp_path):
    fake_profiles = {
        "J1": {"timeseries": "J1", "points": [(0.0, 1.0), (30.0, 5.0), (60.0, 2.0)], "mfactor": 1.0, "baseline": 0.0},
        "J2": {"timeseries": "J2", "points": [(0.0, 2.0), (30.0, 10.0), (60.0, 3.0)], "mfactor": 1.0, "baseline": 0.0},
    }
    monkeypatch.setattr(hydrograph, "get_node_inflow_profiles", lambda inp: fake_profiles)
    monkeypatch.setattr(hydrograph, "load_inp", lambda path: {})

    output = hydrograph.plot_hydrograph(tmp_path / "network.inp", tmp_path / "hydro.png")

    assert output == tmp_path / "hydro.png"
    assert (tmp_path / "hydro.png").exists()
    assert (tmp_path / "hydro.png").stat().st_size > 0
```

- [ ] **Step 2: Run failing test**

```bash
python -m pytest tests/test_hydrograph.py -v
```

Expected: FAIL — `ModuleNotFoundError` or `ImportError` because the module does not exist yet.

- [ ] **Step 3: Create the hydrograph module**

Create `swmm_resilience/visualization/hydrograph.py`:

```python
import matplotlib.pyplot as plt
from pathlib import Path

from ..simulation.swmm_api_io import load_inp, get_node_inflow_profiles


def plot_hydrograph(inp_path: Path, output_path: Path) -> Path:
    """Plot the design-storm hydrograph for the node with the highest peak inflow.

    Reads all node inflow profiles from the .inp, selects the node whose
    timeseries has the maximum single-point flow value, and saves a PNG.
    """
    inp = load_inp(inp_path)
    profiles = get_node_inflow_profiles(inp)

    peak_node, peak_profile = max(
        profiles.items(),
        key=lambda kv: max((q for _, q in kv[1]["points"]), default=0.0),
    )

    times = [t for t, _ in peak_profile["points"]]
    flows = [q for _, q in peak_profile["points"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, flows, color="#2176ae", linewidth=2)
    ax.fill_between(times, flows, alpha=0.15, color="#2176ae")
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("Caudal (L/s)")
    ax.set_title(f"Hidrograma de entrada — Nodo {peak_node} (Qx1)")
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Hidrograma guardado: {output_path}")
    return output_path
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_hydrograph.py -v
```

Expected: PASS.

- [ ] **Step 5: Add CLI flag to main.py**

In `main.py`, add this import at the top with the other visualization imports:

```python
from swmm_resilience.visualization.hydrograph import plot_hydrograph
```

Add the argument to the parser block (after the existing `--predict` argument):

```python
    parser.add_argument("--hydrograph", action="store_true",
                        help="Graficar hidrograma del nodo con mayor caudal pico")
```

Add the handler after the `--predict` block (after line `return` inside `if args.predict:`) and before `if args.only_maps:`:

```python
    # ── Modo: hidrograma ──────────────────────────────────────────────────────
    if args.hydrograph:
        out = config.visualization.output_path / "hydrograph_Qx1.png"
        plot_hydrograph(config.network.inp_path, out)
        return
```

- [ ] **Step 6: Compile check**

```bash
python -m compileall main.py swmm_resilience/visualization/hydrograph.py
```

Expected: no errors.

- [ ] **Step 7: Smoke test**

```bash
python main.py --hydrograph
```

Expected output ends with:
```
Hidrograma guardado: outputs/maps/hydrograph_Qx1.png
```

Verify the file exists and is non-empty:
```bash
python -c "from pathlib import Path; p = Path('outputs/maps/hydrograph_Qx1.png'); assert p.exists() and p.stat().st_size > 0; print('ok')"
```

- [ ] **Step 8: Commit**

```bash
git add swmm_resilience/visualization/hydrograph.py tests/test_hydrograph.py main.py
git commit -m "feat: add --hydrograph command to plot design storm shape"
```

---

## Task 2: Network Map

**Files:**
- Create: `swmm_resilience/visualization/network_map.py`
- Create: `tests/test_network_map.py`
- Modify: `main.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_network_map.py`:

```python
from types import SimpleNamespace

from swmm_resilience.visualization import network_map


def test_generate_network_map_writes_png(monkeypatch, tmp_path):
    fake_inp = {
        "COORDINATES": {
            "J1":  SimpleNamespace(x=0.0,  y=10.0),
            "J2":  SimpleNamespace(x=5.0,  y=5.0),
            "J3":  SimpleNamespace(x=10.0, y=0.0),
            "OUT1": SimpleNamespace(x=15.0, y=0.0),
        },
        "CONDUITS": {
            "C1": SimpleNamespace(from_node="J1",  to_node="J2"),
            "C2": SimpleNamespace(from_node="J2",  to_node="J3"),
            "C3": SimpleNamespace(from_node="J3",  to_node="OUT1"),
        },
        "OUTFALLS": {
            "OUT1": SimpleNamespace(),
        },
        "JUNCTIONS": {
            "J1": SimpleNamespace(),
            "J2": SimpleNamespace(),
            "J3": SimpleNamespace(),
        },
    }
    monkeypatch.setattr(network_map, "load_inp", lambda path: fake_inp)

    output = network_map.generate_network_map(
        tmp_path / "network.inp",
        tmp_path / "network_map.png",
        "Test Network",
    )

    assert output == tmp_path / "network_map.png"
    assert (tmp_path / "network_map.png").exists()
    assert (tmp_path / "network_map.png").stat().st_size > 0
```

- [ ] **Step 2: Run failing test**

```bash
python -m pytest tests/test_network_map.py -v
```

Expected: FAIL — `ModuleNotFoundError` because the module does not exist yet.

- [ ] **Step 3: Create the network map module**

Create `swmm_resilience/visualization/network_map.py`:

```python
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from pathlib import Path

from ..simulation.swmm_api_io import load_inp

_INITIAL_COLOR = "#2176ae"    # blue — pipe whose from_node has no upstream input
_CONTINUOUS_COLOR = "#888888"  # gray — pipe whose from_node receives upstream flow


def generate_network_map(
    inp_path: Path,
    output_path: Path,
    network_name: str = "Red",
) -> Path:
    """Render network topology with pipe-type coloring and flow-direction arrows.

    Pipe classification:
      INITIAL   — from_node does NOT appear as any conduit's to_node (headwater)
      CONTINUOUS — from_node DOES appear as some conduit's to_node
    Outfall nodes (OUTFALLS section) are drawn as downward triangles.
    All junction nodes are drawn as small circles with their ID as label.
    Arrow at the midpoint of each conduit indicates flow direction.
    """
    inp = load_inp(inp_path)

    coords: dict[str, tuple[float, float]] = {}
    if "COORDINATES" in inp:
        for nid, c in inp["COORDINATES"].items():
            coords[str(nid)] = (float(c.x), float(c.y))

    conduits: dict[str, tuple[str, str]] = {}
    if "CONDUITS" in inp:
        for lid, c in inp["CONDUITS"].items():
            conduits[str(lid)] = (str(c.from_node), str(c.to_node))

    outfalls: set[str] = set()
    if "OUTFALLS" in inp:
        for nid in inp["OUTFALLS"]:
            outfalls.add(str(nid))

    to_nodes = {tn for _, (_, tn) in conduits.items()}

    fig, ax = plt.subplots(figsize=(14, 12))

    for fn, tn in conduits.values():
        if fn not in coords or tn not in coords:
            continue
        x0, y0 = coords[fn]
        x1, y1 = coords[tn]
        is_initial = fn not in to_nodes
        color = _INITIAL_COLOR if is_initial else _CONTINUOUS_COLOR
        lw = 1.5 if is_initial else 1.0

        ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw, zorder=1)

        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        dx, dy = x1 - x0, y1 - y0
        step = 0.01
        ax.annotate(
            "",
            xy=(mx + dx * step, my + dy * step),
            xytext=(mx - dx * step, my - dy * step),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.0, mutation_scale=8),
            zorder=2,
        )

    for nid, (x, y) in coords.items():
        if nid in outfalls:
            ax.scatter(x, y, marker="v", color="black", s=60, zorder=4)
        else:
            ax.scatter(x, y, color="#cccccc", s=8, zorder=3,
                       edgecolors="#888888", linewidths=0.5)
        ax.text(x, y, nid, fontsize=5, ha="center", va="bottom", zorder=5)

    legend_elements = [
        mpatches.Patch(facecolor=_INITIAL_COLOR,    label="Tubería inicial"),
        mpatches.Patch(facecolor=_CONTINUOUS_COLOR, label="Tubería continua"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="black",
                   markersize=8, label="Nodo de salida"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax.set_title(network_name, fontsize=13)
    ax.set_aspect("equal")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Mapa de red guardado: {output_path}")
    return output_path
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_network_map.py -v
```

Expected: PASS.

- [ ] **Step 5: Add CLI flag to main.py**

Add this import at the top of `main.py` with the other visualization imports:

```python
from swmm_resilience.visualization.network_map import generate_network_map
```

Add the argument to the parser block (after `--hydrograph`):

```python
    parser.add_argument("--network-map", action="store_true",
                        help="Generar mapa de topología de la red con clasificación de tuberías")
```

Add the handler after the `--hydrograph` block:

```python
    # ── Modo: mapa de red ────────────────────────────────────────────────────
    if args.network_map:
        out = config.visualization.output_path / "network_map.png"
        generate_network_map(config.network.inp_path, out, config.network.name)
        return
```

- [ ] **Step 6: Compile check**

```bash
python -m compileall main.py swmm_resilience/visualization/network_map.py
```

Expected: no errors.

- [ ] **Step 7: Smoke test**

```bash
python main.py --network-map
```

Expected output ends with:
```
Mapa de red guardado: outputs/maps/network_map.png
```

Verify:
```bash
python -c "from pathlib import Path; p = Path('outputs/maps/network_map.png'); assert p.exists() and p.stat().st_size > 0; print('ok')"
```

- [ ] **Step 8: Commit**

```bash
git add swmm_resilience/visualization/network_map.py tests/test_network_map.py main.py
git commit -m "feat: add --network-map command with pipe classification and flow arrows"
```

---

## Task 3: Full Verification

**Files:** No changes expected.

- [ ] **Step 1: Run full test suite**

```bash
python -m pytest tests -v
```

Expected: all tests PASS (was 19 before this feature; now 21).

- [ ] **Step 2: Compile all modules**

```bash
python -m compileall main.py swmm_resilience
```

Expected: no errors.

- [ ] **Step 3: Commit if any fixes were needed**

If any fixes were required in steps 1–2:

```bash
git add main.py swmm_resilience tests
git commit -m "fix: stabilize hydrograph and network map features"
```

If no fixes were needed, skip this commit.
