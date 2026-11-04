"""Opt-in interface to the documented Chico Norte frozen evaluation.

This first integration intentionally supports the existing ten-scenario study,
not arbitrary networks or event inputs. It executes the verified research
pipeline inside a new directory and retains its QA and prediction seals.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path


def _verify_figure_sources(run_root: Path) -> None:
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    completion = json.loads((run_root / "evaluation_completion.json").read_text(encoding="utf-8"))
    results_path = run_root / "evaluation_results.json"
    if completion.get("status") != "complete" or digest(results_path) != completion.get("results_sha256"):
        raise ValueError("evaluation completion seal mismatch")
    results = json.loads(results_path.read_text(encoding="utf-8"))
    design_path = run_root / "evaluation_design.json"
    if digest(design_path) != results.get("design_sha256"):
        raise ValueError("evaluation design hash mismatch")
    design = json.loads(design_path.read_text(encoding="utf-8"))
    for name, source in design["hydrograph_files"].items():
        if digest(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"hydrograph source changed: {name}")


def run(*, execute: bool, output: Path | None) -> None:
    from scripts import run_chico_norte_novel_hydrograph_eval as backend

    if execute and output is None:
        raise ValueError("execution requires a new output directory")
    root = Path(output).resolve() if output is not None else None
    if execute and root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {root}")
    preflight = backend._preflight()
    if not execute:
        print(json.dumps({"mode": "inspection", "executes_swmm": False,
                          "scope": "documented ten-scenario Chico Norte evaluation",
                          "preflight": preflight}, ensure_ascii=False, indent=2))
        return

    from scripts import build_chico_norte_presentation_figures as figures

    root.mkdir(parents=True, exist_ok=False)
    status = root / "estado_integracion.json"
    def record(payload: dict) -> None:
        status.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    record({"status": "running", "scope": "ten frozen Chico Norte scenarios"})
    old = (figures.RUN, figures.OUTPUT, figures.BASE_FIGURES)
    try:
        result = backend.run_evaluation(root / "evaluation", root / "figuras_base")
        _verify_figure_sources(root / "evaluation")
        figures.RUN, figures.OUTPUT, figures.BASE_FIGURES = root / "evaluation", root / "figuras", root / "figuras_base"
        figure_manifest = figures.build_figures()
        _verify_figure_sources(root / "evaluation")
        payload = {"status": "complete", "experiment_root": str(root),
                   "eligible_scenarios": result["eligible_scenarios"],
                   "evaluation_results": str(root / "evaluation/evaluation_results.json"),
                   "figures_manifest": str(root / "figuras/presentation_figures_manifest.json")}
        if isinstance(figure_manifest, dict):
            payload["figures_count"] = len(figure_manifest.get("figure_sha256", {}))
        record(payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    except Exception as exc:
        record({"status": "failed", "error_type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        figures.RUN, figures.OUTPUT, figures.BASE_FIGURES = old
