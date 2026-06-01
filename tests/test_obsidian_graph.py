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
