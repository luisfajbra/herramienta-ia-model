# Obsidian Graph Generator — Design Spec

## Goal

Generate a folder of Obsidian-compatible markdown files (`obsidian/`) that represent the
`swmm_resilience` project as a knowledge graph: architecture (modules + key files) and
data flow (artifacts). A single script regenerates everything when the project structure
changes.

## Approach

**Structure-fixed + artifacts-dynamic.** Module relationships are defined in the script
as a Python dictionary (the architecture is stable). Data artifacts (CSV, parquet, `.pt`
weights, SQLite DB) are discovered at runtime by scanning `data/`. Re-run the script
any time to refresh.

## Node Types

### 1. Module nodes (one per logical subsystem)

| File | Represents |
|---|---|
| `Simulation.md` | `swmm_resilience/simulation/` — SWMM runner, io adapter |
| `ML Tabular.md` | `swmm_resilience/ml/` — XGBoost/SVC pipeline, train, predict |
| `ML Temporal.md` | `swmm_resilience/ml/temporal/` — CNN/LSTM surrogate |
| `Database.md` | `swmm_resilience/database/` — schema, queries, repository |
| `Visualization.md` | `swmm_resilience/visualization/` — flood maps, network map, inp parser |
| `Desktop.md` | `swmm_resilience/desktop/` — Tkinter app |
| `Config.md` | `swmm_resilience/config.py` — paths, ML params |

### 2. Key file sub-nodes (linked from their parent module)

Each module note lists its most important files. Those files also get their own `.md`
with a back-link to the module. Key files per module:

- **Simulation**: `runner.py`, `swmm_api_io.py`
- **ML Tabular**: `train.py`, `predict_tabular.py`, `predict_from_inp.py`, `preprocessing.py`
- **ML Temporal**: `dataset.py`, `train_surrogate.py`, `predict.py`, `compare_surrogate.py`, `surrogate_cnn.py`, `surrogate_lstm.py`
- **Database**: `schema.py`, `queries.py`, `repository.py`
- **Visualization**: `flood_map.py`, `network_map.py`, `runner.py`, `_inp_parser.py`
- **Desktop**: `app.py`

### 3. Data artifact nodes (auto-detected from filesystem)

Scanned paths and their node names:

| Scan path | Node name |
|---|---|
| `data/networks/**/results/dataset_ml.csv` | `Dataset ML CSV` |
| `data/training/swmm_resilience.db` | `SQLite DB` |
| `data/**/results/temporal/node_timeseries/*.parquet` | `Parquets Timeseries` |
| `data/**/results/model_artifacts/*.joblib` + `*.pt` | `Tabular Model Artifacts` |
| `data/**/results/temporal/model_artifacts/*.pt` | `Surrogate Weights` |
| `data/**/results/temporal/maps/*.png` | `Surrogate Maps` |

Each artifact node reports its actual file count and paths found at generation time.

### 4. Index node

`00 - Index.md` — map of content. Lists all module nodes and all artifact nodes with
`[[links]]`. This is the entry point of the graph in Obsidian.

## Note Structure

Each module note follows this template:

```markdown
# <Module Name>

> <One-line description>

## Archivos clave
- [[file_a]] — propósito
- [[file_b]] — propósito

## Recibe datos de
[[Node A]] · [[Node B]]

## Produce
[[Node C]] · [[Node D]]

## Depende de
[[Node E]]
```

Each key-file note:

```markdown
# <filename.py>

> <One-line description>

Módulo: [[Parent Module]]

## Recibe
[[...]]

## Produce
[[...]]
```

Each artifact node:

```markdown
# <Artifact Name>

> Artefacto de datos detectado automáticamente.

**Archivos encontrados:** N
**Rutas:**
- `path/to/file`

## Producido por
[[Module or File]]

## Consumido por
[[Module or File]]
```

## Script

**File:** `generate_obsidian_graph.py` at project root.

**Behavior:**
- Reads the hardcoded module/file graph definition
- Scans `data/` for actual artifact files (using `pathlib.glob`)
- Writes all `.md` files to `obsidian/` (creates dir if missing, overwrites existing)
- Prints a summary: nodes written, artifacts found

**No external dependencies** — only Python stdlib + `pathlib`.

**Usage:**
```bash
python generate_obsidian_graph.py
# → obsidian/ folder ready to open in Obsidian
```

## Data Flow Captured

```
[.inp file] → Simulation → [SQLite DB] → [Parquets Timeseries]
                                              ↓
                              ML Temporal (train_surrogate.py)
                                              ↓
                                    [Surrogate Weights]
                                              ↓
                              ML Temporal (predict.py)
                                              ↓
                                    [Surrogate Maps]

[Dataset ML CSV] → ML Tabular (train.py) → [Tabular Model Artifacts]
                                                     ↓
                              ML Tabular (predict_tabular.py / predict_from_inp.py)

Desktop → ML Tabular + ML Temporal + Visualization → results shown in UI
```

## Non-Goals

- No live sync or file watching
- No Dataview plugin support
- No frontmatter tags or properties (plain links only)
- No parsing of actual Python imports (architecture is hardcoded)
