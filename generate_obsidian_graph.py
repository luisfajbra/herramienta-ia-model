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
    print("Generando grafo Obsidian...")
    notes = build_notes(PROJECT_ROOT)
    write_obsidian(output_dir, notes)
    total = len(notes)
    print(f"✓ {total} notas escritas en {output_dir}/")
    for name in sorted(notes):
        print(f"  {_safe_filename(name)}.md")


if __name__ == "__main__":
    main()
