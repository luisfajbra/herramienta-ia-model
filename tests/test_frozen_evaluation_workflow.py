"""Safety checks for the opt-in frozen evaluation workflow."""
import json
import hashlib
from pathlib import Path
import shutil
from uuid import uuid4

import pytest

from swmm_resilience.evaluation import frozen_workflow
from scripts import run_chico_norte_novel_hydrograph_eval as backend
from scripts import build_chico_norte_presentation_figures as figures


@pytest.fixture
def output():
    root = Path("outputs") / ("test_frozen_workflow_" + uuid4().hex)
    yield root
    if root.exists():
        assert root.resolve().is_relative_to((Path.cwd() / "outputs").resolve())
        shutil.rmtree(root)


def test_preflight_does_not_create_output_or_run_swmm(monkeypatch, output):
    monkeypatch.setattr(backend, "_preflight", lambda: {"scenario_count": 10})
    monkeypatch.setattr(backend, "run_evaluation", lambda *a: pytest.fail("unexpected SWMM"))
    frozen_workflow.run(execute=False, output=output)
    assert not output.exists()


def test_execution_uses_new_isolated_directory_and_restores_figure_settings(monkeypatch, output):
    calls = []
    previous = (figures.RUN, figures.OUTPUT, figures.BASE_FIGURES)
    monkeypatch.setattr(backend, "_preflight", lambda: {"scenario_count": 10})
    monkeypatch.setattr(backend, "run_evaluation", lambda run, base: calls.append((run, base)) or {"eligible_scenarios": 10})
    monkeypatch.setattr(frozen_workflow, "_verify_figure_sources", lambda _: None)
    monkeypatch.setattr(figures, "build_figures", lambda: calls.append((figures.RUN, figures.OUTPUT, figures.BASE_FIGURES)))
    frozen_workflow.run(execute=True, output=output)
    assert calls[0] == (output.resolve() / "evaluation", output.resolve() / "figuras_base")
    assert calls[1] == (output.resolve() / "evaluation", output.resolve() / "figuras", output.resolve() / "figuras_base")
    assert (figures.RUN, figures.OUTPUT, figures.BASE_FIGURES) == previous
    assert json.loads((output / "estado_integracion.json").read_text(encoding="utf-8"))["status"] == "complete"


def test_existing_output_never_overwritten(monkeypatch, output):
    output.mkdir(parents=True)
    marker = output / "keep.txt"
    marker.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(backend, "run_evaluation", lambda *a: pytest.fail("unexpected SWMM"))
    with pytest.raises(FileExistsError):
        frozen_workflow.run(execute=True, output=output)
    assert marker.read_text(encoding="utf-8") == "existing"


def test_failure_is_recorded_and_figure_settings_are_restored(monkeypatch, output):
    previous = (figures.RUN, figures.OUTPUT, figures.BASE_FIGURES)
    monkeypatch.setattr(backend, "_preflight", lambda: {"scenario_count": 10})
    monkeypatch.setattr(backend, "run_evaluation", lambda *a: {"eligible_scenarios": 10})
    monkeypatch.setattr(frozen_workflow, "_verify_figure_sources", lambda _: None)
    monkeypatch.setattr(figures, "build_figures", lambda: (_ for _ in ()).throw(ValueError("figure error")))
    with pytest.raises(ValueError, match="figure error"):
        frozen_workflow.run(execute=True, output=output)
    assert (figures.RUN, figures.OUTPUT, figures.BASE_FIGURES) == previous
    assert json.loads((output / "estado_integracion.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_figure_inputs_reject_a_changed_hydrograph(output):
    output.mkdir(parents=True)
    shape = output / "shape.csv"
    shape.write_text("original", encoding="utf-8")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    design = output / "evaluation_design.json"
    design.write_text(json.dumps({"hydrograph_files": {"shape": {"path": str(shape.resolve()), "sha256": sha(shape)}}}), encoding="utf-8")
    results = output / "evaluation_results.json"
    results.write_text(json.dumps({"design_sha256": sha(design)}), encoding="utf-8")
    (output / "evaluation_completion.json").write_text(json.dumps({"status": "complete", "results_sha256": sha(results)}), encoding="utf-8")
    frozen_workflow._verify_figure_sources(output)
    shape.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="hydrograph"):
        frozen_workflow._verify_figure_sources(output)
