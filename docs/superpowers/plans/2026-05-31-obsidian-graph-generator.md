# Obsidian Graph Generator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate a folder `obsidian/` with interlinked markdown files representing the `swmm_resilience` project architecture and data flow, readable as a knowledge graph in Obsidian.

**Architecture:** A single script `generate_obsidian_graph.py` at the project root defines the module graph as a Python dict and auto-detects data artifacts via `pathlib.glob`. It writes one `.md` file per node (modules, key files, artifacts, index) to `obsidian/`. No external dependencies.

**Tech Stack:** Python 3.9+, stdlib only (`pathlib`, `sys`). Tests with `pytest`.

---

## File Structure

| File | Role |
|---|---|
| `generate_obsidian_graph.py` | Script: graph definition + note generators + scanner + writer + `main()` |
| `tests/test_obsidian_graph.py` | Unit tests for all functions |
| `obsidian/` | Generated output (not tracked in git — add to `.gitignore`) |

---

## Task 1: Test file scaffold + note generators

**Files:**
- Create: `tests/test_obsidian_graph.py`
- Create: `generate_obsidian_graph.py` (partial — data + generators only)

- [ ] **Step 1: Write failing tests for note generators**

Create `tests/test_obsidian_graph.py`:

```python
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import generate_obsidian_graph as G


def test_generate_module_note_contains_title():
    note = G.generate_module_note("ML Temporal", {
        "description": "Surrogate CNN/LSTM",
        "path": "swmm_resilience/ml/temporal/",
        "files": {"dataset.py": "Builds windows"},
        "receives": ["Parquets Timeseries"],
        "produces": ["Surrogate Weights"],
        "depends_on": ["Config"],
    })
    assert "# ML Temporal" in note
    assert "Surrogate CNN/LSTM" in note
    assert "[[dataset.py]]" in note
    assert "[[Parquets Timeseries]]" in note
    assert "[[Surrogate Weights]]" in note
    assert "[[Config]]" in note


def test_generate_module_note_empty_links():
    note = G.generate_module_note("Config", {
        "description": "Config central",
        "path": "swmm_resilience/config.py",
        "files": {},
        "receives": [],
        "produces": [],
        "depends_on": [],
    })
    assert "# Config" in note
    assert "_ninguno_" in note


def test_generate_file_note():
    note = G.generate_file_note(
        filename="predict.py",
        description="Inferencia surrogate",
        parent_module="ML Temporal",
        receives=["ML Temporal"],
        produces=[],
    )
    assert "# predict.py" in note
    assert "[[ML Temporal]]" in note
    assert "_ninguno_" in note


def test_generate_artifact_note_with_files():
    note = G.generate_artifact_note(
        name="SQLite DB",
        paths=["data/training/swmm_resilience.db"],
        producers=["Database"],
        consumers=["ML Temporal"],
    )
    assert "# SQLite DB" in note
    assert "**Archivos encontrados:** 1" in note
    assert "data/training/swmm_resilience.db" in note
    assert "[[Database]]" in note
    assert "[[ML Temporal]]" in note


def test_generate_artifact_note_empty():
    note = G.generate_artifact_note(
        name="Surrogate Maps",
        paths=[],
        producers=["ML Temporal"],
        consumers=["Desktop"],
    )
    assert "**Archivos encontrados:** 0" in note


def test_generate_index_note():
    note = G.generate_index_note(
        modules=["Simulation", "ML Tabular"],
        artifacts=["SQLite DB"],
    )
    assert "# resiliencIA" in note
    assert "[[Simulation]]" in note
    assert "[[ML Tabular]]" in note
    assert "[[SQLite DB]]" in note
```

- [ ] **Step 2: Run tests — confirm they fail**

```bash
pytest tests/test_obsidian_graph.py -v
```

Expected: `ModuleNotFoundError: No module named 'generate_obsidian_graph'`

- [ ] **Step 3: Create `generate_obsidian_graph.py` with graph data + generators**

Create `generate_obsidian_graph.py` at project root:

```python
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
        "description": "Ejecuta simulaciones SWMM vía PySWMM y extrae resultados de nodos",
        "path": "swmm_resilience/simulation/",
        "files": {
            "runner.py": "Orquesta simulaciones y extrae topología estática y series temporales",
            "swmm_api_io.py": "Adaptador de IO para la API nativa de SWMM",
        },
        "receives": ["Config", "INP File"],
        "produces": ["SQLite DB", "Parquets Timeseries"],
        "depends_on": ["Config"],
    },
    "ML Tabular": {
        "description": "Pipeline XGBoost/SVC para predicción tabular de inundación sin SWMM",
        "path": "swmm_resilience/ml/",
        "files": {
            "train.py": "Entrena y guarda modelos tabulares (clasificación y regresión)",
            "predict_tabular.py": "Predicción desde CSV con modelos tabulares guardados",
            "predict_from_inp.py": "Predicción desde .inp sin simulación previa",
            "preprocessing.py": "Selección de features, limpieza y normalización",
        },
        "receives": ["Dataset ML CSV", "Config"],
        "produces": ["Tabular Model Artifacts"],
        "depends_on": ["Config"],
    },
    "ML Temporal": {
        "description": "Surrogate CNN/LSTM para predicción temporal de inundación por multiplicador",
        "path": "swmm_resilience/ml/temporal/",
        "files": {
            "dataset.py": "Construye ventanas temporales y dataset surrogate desde parquets",
            "train_surrogate.py": "Entrena CNN/LSTM surrogate con GroupKFold y guarda artefactos",
            "predict.py": "Inferencia surrogate por multiplicador de caudal",
            "compare_surrogate.py": "Compara XGBoost vs CNN vs LSTM en splits idénticos",
            "surrogate_cnn.py": "Arquitectura CNN dual-branch para predicción de inundación",
            "surrogate_lstm.py": "Arquitectura LSTM dual-branch para predicción de inundación",
        },
        "receives": ["Parquets Timeseries", "SQLite DB", "Config"],
        "produces": ["Surrogate Weights", "Surrogate Maps"],
        "depends_on": ["Database", "Config"],
    },
    "Database": {
        "description": "Esquema SQLite, consultas y repositorio para artefactos de simulación",
        "path": "swmm_resilience/database/",
        "files": {
            "schema.py": "Define y migra el esquema SQLite (runs, network_nodes, temporal_artifacts)",
            "queries.py": "Consultas SQL parametrizadas para lectura de datos",
            "repository.py": "Capa de repositorio que abstrae acceso a la DB",
        },
        "receives": ["Config"],
        "produces": ["SQLite DB"],
        "depends_on": ["Config"],
    },
    "Visualization": {
        "description": "Mapas de inundación, red hidráulica y parser de archivos .inp",
        "path": "swmm_resilience/visualization/",
        "files": {
            "flood_map.py": "Genera mapa de inundación coloreado por volumen o probabilidad",
            "network_map.py": "Visualiza la red hidráulica con nodos y conductos",
            "_inp_parser.py": "Parsea coordenadas y conductos de archivos .inp de SWMM",
            "runner.py": "Orquesta generación de mapas ML y SWMM",
        },
        "receives": ["Surrogate Weights", "Tabular Model Artifacts", "SQLite DB", "INP File"],
        "produces": ["Surrogate Maps"],
        "depends_on": ["Config", "ML Temporal", "ML Tabular"],
    },
    "Desktop": {
        "description": "Aplicación Tkinter local para correr simulaciones y workflows ML",
        "path": "swmm_resilience/desktop/",
        "files": {
            "app.py": "UI principal: tabs de simulación, ML, resultados y visualización",
        },
        "receives": ["Dataset ML CSV", "SQLite DB", "INP File"],
        "produces": ["Surrogate Maps"],
        "depends_on": ["Simulation", "ML Tabular", "ML Temporal", "Visualization", "Config"],
    },
    "Config": {
        "description": "Configuración central: rutas de datos, parámetros ML, modos de escenario",
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
        return "_ninguno_"
    return " · ".join(_link(n) for n in names)


# ── Note generators ───────────────────────────────────────────────────────────

def generate_module_note(name: str, data: dict) -> str:
    lines = [
        f"# {name}",
        "",
        f"> {data['description']}",
        "",
        f"**Ruta:** `{data['path']}`",
        "",
    ]
    if data["files"]:
        lines += ["## Archivos clave", ""]
        for fname, desc in data["files"].items():
            lines.append(f"- {_link(fname)} — {desc}")
        lines.append("")
    lines += [
        "## Recibe datos de",
        _links(data["receives"]),
        "",
        "## Produce",
        _links(data["produces"]),
        "",
        "## Depende de",
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
        f"**Módulo:** {_link(parent_module)}",
        "",
        "## Recibe",
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
        "> Artefacto de datos detectado automáticamente.",
        "",
        f"**Archivos encontrados:** {len(paths)}",
    ]
    if paths:
        lines += ["", "**Rutas:**", ""]
        for p in sorted(paths):
            lines.append(f"- `{p}`")
    lines += [
        "",
        "## Producido por",
        _links(producers),
        "",
        "## Consumido por",
        _links(consumers),
        "",
    ]
    return "\n".join(lines)


def generate_index_note(modules: list[str], artifacts: list[str]) -> str:
    lines = [
        "# resiliencIA — Índice del grafo",
        "",
        "> Mapa de contenido generado automáticamente.",
        "> Re-ejecuta `python generate_obsidian_graph.py` para actualizar.",
        "",
        "## Módulos",
        "",
    ]
    for m in sorted(modules):
        lines.append(f"- {_link(m)}")
    lines += ["", "## Artefactos de datos", ""]
    for a in sorted(artifacts):
        lines.append(f"- {_link(a)}")
    lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests — confirm they pass**

```bash
pytest tests/test_obsidian_graph.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add generate_obsidian_graph.py tests/test_obsidian_graph.py
git commit -m "feat: obsidian graph note generators + tests"
```

---

## Task 2: Artifact scanner + writer

**Files:**
- Modify: `tests/test_obsidian_graph.py` — add scanner and writer tests
- Modify: `generate_obsidian_graph.py` — add `scan_artifacts`, `_safe_filename`, `write_obsidian`

- [ ] **Step 1: Add failing tests for scanner and writer**

Append to `tests/test_obsidian_graph.py`:

```python
def test_scan_artifacts_finds_db(tmp_path):
    (tmp_path / "data" / "training").mkdir(parents=True)
    (tmp_path / "data" / "training" / "swmm_resilience.db").touch()
    result = G.scan_artifacts(tmp_path)
    assert any("swmm_resilience.db" in p for p in result["SQLite DB"])


def test_scan_artifacts_empty_dir(tmp_path):
    result = G.scan_artifacts(tmp_path)
    assert result["SQLite DB"] == []
    assert result["Parquets Timeseries"] == []


def test_scan_artifacts_finds_parquet(tmp_path):
    p = tmp_path / "data" / "networks" / "net1" / "results" / "temporal" / "node_timeseries"
    p.mkdir(parents=True)
    (p / "run_abc.parquet").touch()
    result = G.scan_artifacts(tmp_path)
    assert any("run_abc.parquet" in x for x in result["Parquets Timeseries"])


def test_write_obsidian_creates_files(tmp_path):
    notes = {"Simulation": "# Simulation\n", "00 - Index": "# Index\n"}
    G.write_obsidian(tmp_path / "obsidian", notes)
    assert (tmp_path / "obsidian" / "Simulation.md").exists()
    assert (tmp_path / "obsidian" / "00 - Index.md").read_text() == "# Index\n"


def test_write_obsidian_creates_dir(tmp_path):
    G.write_obsidian(tmp_path / "new" / "obsidian", {"A": "# A\n"})
    assert (tmp_path / "new" / "obsidian" / "A.md").exists()


def test_safe_filename_no_slashes():
    assert "/" not in G._safe_filename("ML Tabular")
    assert G._safe_filename("ML/Tabular") == "ML-Tabular"
```

- [ ] **Step 2: Run tests — confirm new ones fail**

```bash
pytest tests/test_obsidian_graph.py -v
```

Expected: 6 old tests PASS, 6 new tests FAIL with `AttributeError`.

- [ ] **Step 3: Add scanner, filename helper, and writer to script**

Append to `generate_obsidian_graph.py` (after the note generators):

```python
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
```

- [ ] **Step 4: Run tests — confirm all pass**

```bash
pytest tests/test_obsidian_graph.py -v
```

Expected: 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add generate_obsidian_graph.py tests/test_obsidian_graph.py
git commit -m "feat: obsidian artifact scanner and file writer"
```

---

## Task 3: `build_notes` + `main()` + integration + run

**Files:**
- Modify: `tests/test_obsidian_graph.py` — add integration tests
- Modify: `generate_obsidian_graph.py` — add `build_notes` and `main()`
- Modify: `.gitignore` — exclude `obsidian/`

- [ ] **Step 1: Add integration tests**

Append to `tests/test_obsidian_graph.py`:

```python
def test_build_notes_has_all_modules(tmp_path):
    notes = G.build_notes(tmp_path)
    for mod in G.MODULES:
        assert mod in notes, f"Missing module note: {mod}"


def test_build_notes_has_all_artifacts(tmp_path):
    notes = G.build_notes(tmp_path)
    for art in G.ARTIFACT_GLOBS:
        assert art in notes, f"Missing artifact note: {art}"


def test_build_notes_has_index(tmp_path):
    notes = G.build_notes(tmp_path)
    assert "00 - Index" in notes


def test_build_notes_has_key_files(tmp_path):
    notes = G.build_notes(tmp_path)
    assert "dataset.py" in notes
    assert "train_surrogate.py" in notes
    assert "predict.py" in notes
    assert "app.py" in notes


def test_build_notes_module_links_to_files(tmp_path):
    notes = G.build_notes(tmp_path)
    assert "[[dataset.py]]" in notes["ML Temporal"]
    assert "[[train_surrogate.py]]" in notes["ML Temporal"]


def test_build_notes_file_links_to_parent(tmp_path):
    notes = G.build_notes(tmp_path)
    assert "[[ML Temporal]]" in notes["dataset.py"]


def test_build_notes_index_links_all_modules(tmp_path):
    notes = G.build_notes(tmp_path)
    for mod in G.MODULES:
        assert f"[[{mod}]]" in notes["00 - Index"]
```

- [ ] **Step 2: Run tests — confirm new ones fail**

```bash
pytest tests/test_obsidian_graph.py -v
```

Expected: 12 old tests PASS, 7 new tests FAIL with `AttributeError: module has no attribute 'build_notes'`.

- [ ] **Step 3: Add `build_notes` and `main()` to script**

Append to `generate_obsidian_graph.py`:

```python
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
    print("Generando grafo Obsidian...")
    notes = build_notes(PROJECT_ROOT)
    write_obsidian(output_dir, notes)
    total = len(notes)
    print(f"✓ {total} notas escritas en {output_dir}/")
    for name in sorted(notes):
        print(f"  {_safe_filename(name)}.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests — confirm all pass**

```bash
pytest tests/test_obsidian_graph.py -v
```

Expected: 19 tests PASS.

- [ ] **Step 5: Add `obsidian/` to `.gitignore`**

Add this line to `.gitignore` (create the file if it doesn't exist):

```
obsidian/
```

- [ ] **Step 6: Run the script and verify output**

```bash
python generate_obsidian_graph.py
```

Expected output (example):

```
Generando grafo Obsidian...
✓ 34 notas escritas en /Users/luis/herramienta-ia-model/obsidian/
  00 - Index.md
  Config.md
  Database.md
  Desktop.md
  ML Tabular.md
  ML Temporal.md
  Simulation.md
  Visualization.md
  app.py.md
  ...
```

Verify a note manually:

```bash
cat obsidian/ML\ Temporal.md
```

Expected: contains `# ML Temporal`, `[[dataset.py]]`, `[[Surrogate Weights]]`, `[[Config]]`.

- [ ] **Step 7: Commit**

```bash
git add generate_obsidian_graph.py tests/test_obsidian_graph.py .gitignore
git commit -m "feat: obsidian graph generator complete — build_notes + main"
```

---

## Opening in Obsidian

After running the script:
1. Open Obsidian → "Open folder as vault"
2. Select `herramienta-ia-model/obsidian/`
3. Click the graph icon (top-left) to see the network view
