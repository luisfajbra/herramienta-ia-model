"""Generate an Obsidian knowledge graph for the swmm_resilience project.

Usage:
    python generate_obsidian_graph.py
    # → writes obsidian/ folder ready to open in Obsidian
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ── Graph definition ──────────────────────────────────────────────────────────

MODULES: dict[str, dict] = {
    "Simulation": {
        "description": "Runs SWMM simulations through PySWMM and extracts node results",
        "path": "swmm_resilience/simulation/",
        "files": {
            "runner.py": "Orchestrates simulations and extracts static topology and time series",
            "swmm_api_io.py": "IO adapter for the native SWMM API",
        },
        "receives": ["Config", "INP File"],
        "produces": ["SQLite DB", "Parquets Timeseries"],
        "depends_on": ["Config"],
    },
    "ML Tabular": {
        "description": "XGBoost/SVC pipeline for tabular flood prediction without SWMM",
        "path": "swmm_resilience/ml/",
        "files": {
            "train.py": "Trains and saves tabular models (classification and regression)",
            "predict_tabular.py": "Prediction from CSV with saved tabular models",
            "predict_from_inp.py": "Prediction from .inp without a prior simulation",
            "preprocessing.py": "Feature selection, cleaning and normalization",
        },
        "receives": ["Dataset ML CSV", "Config"],
        "produces": ["Tabular Model Artifacts"],
        "depends_on": ["Config"],
    },
    "ML Temporal": {
        "description": "CNN/LSTM surrogate for temporal flood prediction per multiplier",
        "path": "swmm_resilience/ml/temporal/",
        "files": {
            "dataset.py": "Builds temporal windows and the surrogate dataset from parquets",
            "train_surrogate.py": "Trains the CNN/LSTM surrogate with GroupKFold and saves artifacts",
            "predict.py": "Surrogate inference per flow multiplier",
            "compare_surrogate.py": "Compares XGBoost vs CNN vs LSTM on identical splits",
            "surrogate_cnn.py": "Dual-branch CNN architecture for flood prediction",
            "surrogate_lstm.py": "Dual-branch LSTM architecture for flood prediction",
        },
        "receives": ["Parquets Timeseries", "SQLite DB", "Config"],
        "produces": ["Surrogate Weights", "Surrogate Maps"],
        "depends_on": ["Database", "Config"],
    },
    "Database": {
        "description": "SQLite schema, queries and repository for simulation artifacts",
        "path": "swmm_resilience/database/",
        "files": {
            "schema.py": "Defines and migrates the SQLite schema (runs, network_nodes, temporal_artifacts)",
            "queries.py": "Parameterized SQL queries for reading data",
            "repository.py": "Repository layer that abstracts DB access",
        },
        "receives": ["Config"],
        "produces": ["SQLite DB"],
        "depends_on": ["Config"],
    },
    "Visualization": {
        "description": "Flood maps, hydraulic network and .inp file parser",
        "path": "swmm_resilience/visualization/",
        "files": {
            "flood_map.py": "Generates a flood map colored by volume or probability",
            "network_map.py": "Visualizes the hydraulic network with nodes and conduits",
            "_inp_parser.py": "Parses coordinates and conduits from SWMM .inp files",
            "runner.py": "Orchestrates the generation of ML and SWMM maps",
        },
        "receives": ["Surrogate Weights", "Tabular Model Artifacts", "SQLite DB", "INP File"],
        "produces": ["Surrogate Maps"],
        "depends_on": ["Config", "ML Temporal", "ML Tabular"],
    },
    "Desktop": {
        "description": "Local Tkinter application to run simulations and ML workflows",
        "path": "swmm_resilience/desktop/",
        "files": {
            "app.py": "Main UI: simulation, ML, results and visualization tabs",
        },
        "receives": ["Dataset ML CSV", "SQLite DB", "INP File"],
        "produces": ["Surrogate Maps"],
        "depends_on": ["Simulation", "ML Tabular", "ML Temporal", "Visualization", "Config"],
    },
    "Config": {
        "description": "Central configuration: data paths, ML parameters, scenario modes",
        "path": "swmm_resilience/config.py",
        "files": {},
        "receives": [],
        "produces": [],
        "depends_on": [],
    },
}

ARTIFACT_GLOBS: dict[str, list[str]] = {
    "Dataset ML CSV": ["data/**/results/dataset_ml.csv"],
    "SQLite DB": ["data/training/swmm_resilience.db"],
    "Parquets Timeseries": ["data/**/results/temporal/node_timeseries/*.parquet"],
    "Tabular Model Artifacts": [
        "data/**/results/model_artifacts/*.joblib",
        "data/**/results/model_artifacts/*.pkl",
    ],
    "Surrogate Weights": ["data/**/results/temporal/model_artifacts/*.pt"],
    "Surrogate Maps": ["data/**/results/temporal/maps/*.png"],
    "INP File": ["data/**/*.inp"],
}

ARTIFACT_PRODUCERS: dict[str, list[str]] = {
    "Dataset ML CSV": ["Simulation"],
    "SQLite DB": ["Database", "Simulation"],
    "Parquets Timeseries": ["Simulation"],
    "Tabular Model Artifacts": ["ML Tabular"],
    "Surrogate Weights": ["ML Temporal"],
    "Surrogate Maps": ["ML Temporal", "Visualization"],
    "INP File": [],
}

ARTIFACT_CONSUMERS: dict[str, list[str]] = {
    "Dataset ML CSV": ["ML Tabular", "Desktop"],
    "SQLite DB": ["ML Temporal", "Visualization", "Desktop"],
    "Parquets Timeseries": ["ML Temporal"],
    "Tabular Model Artifacts": ["ML Tabular", "Visualization", "Desktop"],
    "Surrogate Weights": ["ML Temporal", "Visualization"],
    "Surrogate Maps": ["Desktop"],
    "INP File": ["Simulation", "Visualization", "Desktop"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _link(name: str) -> str:
    return f"[[{name}]]"


def _links(names: list[str]) -> str:
    if not names:
        return "_none_"
    return " · ".join(_link(n) for n in names)


# ── Note generators ───────────────────────────────────────────────────────────

def generate_module_note(name: str, data: dict) -> str:
    lines = [
        f"# {name}",
        "",
        f"> {data['description']}",
        "",
        f"**Path:** `{data['path']}`",
        "",
    ]
    if data["files"]:
        lines += ["## Key files", ""]
        for fname, desc in data["files"].items():
            lines.append(f"- {_link(fname)} — {desc}")
        lines.append("")
    lines += [
        "## Receives data from",
        _links(data["receives"]),
        "",
        "## Produce",
        _links(data["produces"]),
        "",
        "## Depends on",
        _links(data["depends_on"]),
        "",
    ]
    return "\n".join(lines)


def generate_file_note(
    filename: str,
    description: str,
    parent_module: str,
    receives: list[str],
    produces: list[str],
) -> str:
    lines = [
        f"# {filename}",
        "",
        f"> {description}",
        "",
        f"**Module:** {_link(parent_module)}",
        "",
        "## Receives",
        _links(receives),
        "",
        "## Produce",
        _links(produces),
        "",
    ]
    return "\n".join(lines)


def generate_artifact_note(
    name: str,
    paths: list[str],
    producers: list[str],
    consumers: list[str],
) -> str:
    lines = [
        f"# {name}",
        "",
        "> Data artifact detected automatically.",
        "",
        f"**Files found:** {len(paths)}",
    ]
    if paths:
        lines += ["", "**Paths:**", ""]
        for p in sorted(paths):
            lines.append(f"- `{p}`")
    lines += [
        "",
        "## Produced by",
        _links(producers),
        "",
        "## Consumed by",
        _links(consumers),
        "",
    ]
    return "\n".join(lines)


def generate_index_note(modules: list[str], artifacts: list[str]) -> str:
    lines = [
        "# resiliencIA — Graph index",
        "",
        "> Content map generated automatically.",
        "> Re-run `python generate_obsidian_graph.py` to update.",
        "",
        "## Modules",
        "",
    ]
    for m in sorted(modules):
        lines.append(f"- {_link(m)}")
    lines += ["", "## Data artifacts", ""]
    for a in sorted(artifacts):
        lines.append(f"- {_link(a)}")
    lines.append("")
    return "\n".join(lines)


# ── Scanner ───────────────────────────────────────────────────────────────────

def scan_artifacts(project_root: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for name, globs in ARTIFACT_GLOBS.items():
        paths: list[str] = []
        for pattern in globs:
            for p in project_root.glob(pattern):
                paths.append(str(p.relative_to(project_root)))
        found[name] = sorted(paths)
    return found


# ── Writer ────────────────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    return name.replace("/", "-").replace("\\", "-")


def write_obsidian(output_dir: Path, notes: dict[str, str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in notes.items():
        (output_dir / f"{_safe_filename(name)}.md").write_text(content, encoding="utf-8")


# ── Orchestration ─────────────────────────────────────────────────────────────

def build_notes(project_root: Path) -> dict[str, str]:
    artifact_paths = scan_artifacts(project_root)
    notes: dict[str, str] = {}

    for mod_name, mod_data in MODULES.items():
        notes[mod_name] = generate_module_note(mod_name, mod_data)

    for mod_name, mod_data in MODULES.items():
        for fname, fdesc in mod_data["files"].items():
            notes[fname] = generate_file_note(
                filename=fname,
                description=fdesc,
                parent_module=mod_name,
                receives=[mod_name],
                produces=[],
            )

    for art_name in ARTIFACT_GLOBS:
        notes[art_name] = generate_artifact_note(
            name=art_name,
            paths=artifact_paths.get(art_name, []),
            producers=ARTIFACT_PRODUCERS.get(art_name, []),
            consumers=ARTIFACT_CONSUMERS.get(art_name, []),
        )

    notes["00 - Index"] = generate_index_note(
        modules=list(MODULES.keys()),
        artifacts=list(ARTIFACT_GLOBS.keys()),
    )
    return notes


def main() -> None:
    output_dir = PROJECT_ROOT / "obsidian"
    print("Generating Obsidian graph...")
    notes = build_notes(PROJECT_ROOT)
    write_obsidian(output_dir, notes)
    total = len(notes)
    print(f"✓ {total} notes written to {output_dir}/")
    for name in sorted(notes):
        print(f"  {_safe_filename(name)}.md")


if __name__ == "__main__":
    main()
