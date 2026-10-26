"""Part 2: fair XGBoost / MLP / approximate SVM comparison on ``tabular_v4_21``.

Spec: ``docs/superpowers/specs/2026-10-12-parte2-svm-comparacion-justa-design.md``.

Usage (from the repository root; one process per family, they can run at the same time)::

    venv/Scripts/python.exe scripts/ejecutar_parte2_v4_21.py --family mlp
    venv/Scripts/python.exe scripts/ejecutar_parte2_v4_21.py --family xgboost
    venv/Scripts/python.exe scripts/ejecutar_parte2_v4_21.py --family svm
    venv/Scripts/python.exe scripts/ejecutar_parte2_v4_21.py --build-tables

Each family runs three stages, in this order, and leaves each one in
``outputs/bench/parte2/<prep_id>/<family>/<stage>/``:

1. ``default``: ``evaluate_candidate`` with the ``config.yaml`` parameters
   (plus the fixed parameters of the comparison) in LeaveOneShapeOut.
2. ``tune``: Optuna study (``N_TRIALS`` trials, inner GroupKFold(5) by
   shape, the 5 folds of each trial in parallel).
3. ``tuned``: ``evaluate_candidate`` with the best parameters of the study.

A stage with a ``done`` marker is skipped: after a crash it is enough to
relaunch the same command. ``--build-tables`` joins the already persisted
baseline and all the stage OOFs into the report tables
(``docs/auditoria/correcciones/datos_parte2/``).

Parallelism: the folds (40 of LeaveOneShapeOut; 5 inner ones per trial) run in
separate processes with one thread each. XGBoost is pinned to ``n_jobs=1``
because inside those processes ``OMP_NUM_THREADS=1`` already reduces it to one
thread even if more are requested (measured: ``n_jobs=4`` in a child gives
exactly the predictions of ``n_jobs=1`` and not those of ``n_jobs=4`` in the
main process); pinning 1 makes tuning and evaluation reproducible and
consistent with each other.

It never writes to ``outputs/models``, ``outputs/bench/reports``,
``outputs/bench/prepared*``, ``outputs/bench/oof`` (read-only), the SQLite
database, ``data/``, ``config.yaml`` or ``docs/auditoria/correcciones/datos/``
(hashed tables): ``ensure_parte2_output`` verifies this before every write.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version as _package_version
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.ejecutar_evaluacion_v4_21 import ensure_allowed_output  # noqa: E402
from scripts.informe_evaluacion import (  # noqa: E402
    TABLES_PROVENANCE,
    _sha256,
    build_tables,
    git_state,
    grid_counts,
    portable_path,
    write_tables,
)
from swmm_resilience.config import ML_RANDOM_STATE  # noqa: E402

# --- constantes de la comparación -----------------------------------------------------

PREP_ID = "ca706e27654e0cf5"
CONTRACT_ID = "tabular_v4_21"
N_ROWS = 154080

N_TRIALS = 20
INNER_SPLITS = 5
SEED = ML_RANDOM_STATE
EVAL_PROTOCOL = "LeaveOneShapeOut"
SMOKE_FOLDS = 6
SMOKE_TRIALS = 2
FAMILIES = ("svm", "xgboost", "mlp")
STAGES = ("default", "tune", "tuned")

# Ajustes que NO se buscan: se imponen igual en la afinación y en las dos evaluaciones.
FIXED_PARAMS = {
    "xgboost": {"classifier": {"n_jobs": 1}, "regressor": {"n_jobs": 1}},
    "mlp": {
        "classifier": {"epochs": 200, "batch_size": 1024},
        "regressor": {"epochs": 200, "batch_size": 1024},
    },
    "svm": {"classifier": {}, "regressor": {}},
}

PREPARED_DIR = Path("outputs/bench/prepared_v4_21")
BASELINE_OOF = Path("outputs/bench/oof") / PREP_ID / "baseline_forma_mas_cercana.parquet"
DEFAULT_OUT_ROOT = Path("outputs/bench/parte2") / PREP_ID
DEFAULT_TABLES_DIR = Path("docs/auditoria/correcciones/datos_parte2")

_EXTRA_PROTECTED = (
    Path("outputs/bench/oof"),
    Path("docs/auditoria/correcciones/datos"),
    Path("config.yaml"),
)


# --- guardas --------------------------------------------------------------------------

def check_manifest(manifest: dict) -> None:
    """Falla si el PreparedDataset no es exactamente el de la comparación."""
    expected = {"prep_id": PREP_ID, "contract_id": CONTRACT_ID, "n_rows": N_ROWS}
    for field, want in expected.items():
        have = manifest.get(field)
        if have != want:
            raise ValueError(
                f"The manifest does not match on {field}: expected {want!r}, has {have!r}. "
                "This comparison is only valid on the dataset fixed in the spec."
            )


def ensure_parte2_output(path: Path | str, repo_root: Path | str = REPO_ROOT) -> None:
    """Falla si ``path`` cae en una ruta que la parte 2 no puede escribir.

    Además de las rutas protegidas del script de evaluación (modelos, reportes,
    datos, base), protege los OOF de lectura, ``outputs/bench/prepared*``
    (``ensure_allowed_output`` solo cubre el directorio exacto), ``config.yaml``
    y las tablas con hash de ``docs/auditoria/correcciones/datos/``.
    """
    ensure_allowed_output(path, repo_root)
    root = Path(repo_root).resolve()
    target = Path(path)
    target = (target if target.is_absolute() else root / target).resolve()
    try:
        relative = target.relative_to(root)
    except ValueError:
        return
    parts = relative.parts
    if len(parts) >= 3 and parts[:2] == ("outputs", "bench") and parts[2].startswith("prepared"):
        raise ValueError(f"Protected path, not written here: {relative.as_posix()}")
    for protected in _EXTRA_PROTECTED:
        if relative == protected or relative.is_relative_to(protected):
            raise ValueError(f"Protected path, not written here: {relative.as_posix()}")


# --- parámetros -----------------------------------------------------------------------

def merge_fixed(params: dict, fixed: dict) -> dict:
    """``{"classifier", "regressor"}`` con ``fixed`` pisando a ``params``; no muta ninguno."""
    return {
        task: {**(params.get(task) or {}), **((fixed or {}).get(task) or {})}
        for task in ("classifier", "regressor")
    }


def config_params(config, family: str) -> dict:
    """Parámetros de ``config.bench.families[familia]`` como ``{"classifier", "regressor"}``."""
    spec = config.bench.families[family]
    return {"classifier": dict(spec.classifier), "regressor": dict(spec.regressor)}


def model_name(family: str, stage: str) -> str:
    """Nombre del modelo en las tablas: ``<familia>_default`` o ``<familia>_tuned``."""
    if stage not in ("default", "tuned"):
        raise ValueError(f"The stage {stage!r} does not produce an evaluated model (default|tuned).")
    return f"{family}_{stage}"


def effective_n_trials(args) -> int:
    return SMOKE_TRIALS if args.smoke else args.n_trials


# --- etapas resumibles ----------------------------------------------------------------

def stage_path(out_root: Path | str, family: str, stage: str) -> Path:
    return Path(out_root) / family / stage


def is_done(path: Path | str) -> bool:
    return (Path(path) / "done").exists()


def library_versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in (
        "numpy", "pandas", "scikit-learn", "xgboost", "torch", "optuna", "joblib", "pyarrow"
    ):
        try:
            versions[name] = _package_version(name)
        except Exception:
            versions[name] = "unknown"
    return versions


def build_stage_record(
    header: dict, result: dict, started: datetime, finished: datetime
) -> dict:
    """Contenido de ``stage.json``: cabecera de la etapa + lo que devolvió el ejecutor."""
    return {
        **header,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_s": round((finished - started).total_seconds(), 3),
        **result,
    }


def _log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def run_stage_once(path: Path | str, runner, *, header: dict, log=_log) -> bool:
    """Ejecuta ``runner(path)`` si la etapa no tiene marca ``done``.

    Devuelve True si corrió y False si se saltó. ``stage.json`` y ``done`` se
    escriben solo si el ejecutor termina: una etapa caída queda sin marca y se
    repite entera al relanzar.
    """
    directory = Path(path)
    label = f"{header.get('family')}/{header.get('stage')}"
    if is_done(directory):
        log(f"{label}: already done ({directory}), skipping")
        return False
    directory.mkdir(parents=True, exist_ok=True)
    log(f"{label}: start")
    started = datetime.now(timezone.utc)
    result = runner(directory) or {}
    finished = datetime.now(timezone.utc)
    record = build_stage_record(header, result, started, finished)
    (directory / "stage.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8"
    )
    (directory / "done").write_text(finished.isoformat() + "\n", encoding="utf-8")
    log(f"{label}: finished in {record['duration_s']} s")
    return True


def find_study_json(tune_dir: Path | str, family: str) -> Path:
    """``study.json`` de la etapa ``tune`` (el más reciente si hubo reintentos)."""
    candidates = sorted(Path(tune_dir).glob(f"{family}/*/study.json"))
    if not candidates:
        raise FileNotFoundError(
            f"There is no study.json in {tune_dir}: the tune stage of {family!r} is missing."
        )
    return candidates[-1]


def best_params_from_study(study_json: Path | str, fixed: dict) -> dict:
    """Mejores parámetros del estudio, con los fijos de la comparación encima."""
    payload = json.loads(Path(study_json).read_text(encoding="utf-8"))
    return merge_fixed(
        {
            "classifier": payload["best_classifier_params"],
            "regressor": payload["best_regressor_params"],
        },
        fixed,
    )


# --- subconjunto de folds (humo) ------------------------------------------------------

def _first_fold_ids(folds: pd.DataFrame, protocol: str, max_folds: int) -> list:
    ids = sorted(folds.loc[folds["protocol"] == protocol, "fold_id"].unique())
    return ids[:max_folds]


def restrict_prepared(prepared, protocol: str, max_folds: int):
    """Copia del PreparedDataset con solo los primeros ``max_folds`` folds de ``protocol``."""
    keep = _first_fold_ids(prepared.folds, protocol, max_folds)
    folds = prepared.folds
    folds = folds[(folds["protocol"] == protocol) & folds["fold_id"].isin(keep)]
    # dataclasses.replace() revalidaría que cada muestra esté en test una vez por
    # protocolo, que a propósito deja de cumplirse en el subconjunto de humo.
    subset = copy.copy(prepared)
    object.__setattr__(subset, "folds", folds.reset_index(drop=True))
    return subset


def restrict_oof(oof: pd.DataFrame, protocol: str, max_folds: int) -> pd.DataFrame:
    keep = _first_fold_ids(oof, protocol, max_folds)
    subset = oof[(oof["protocol"] == protocol) & oof["fold_id"].isin(keep)]
    return subset.reset_index(drop=True)


# --- armado de tablas -----------------------------------------------------------------

def _done_eval_stages(out_root: Path | str) -> list[tuple[str, str, Path]]:
    """``(familia, etapa, carpeta)`` de cada evaluación terminada, en orden fijo."""
    found = []
    for family in FAMILIES:
        for stage in ("default", "tuned"):
            directory = stage_path(out_root, family, stage)
            if is_done(directory) and (directory / "oof.parquet").exists():
                found.append((family, stage, directory))
    return found


def _tuning_tables(out_root: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary, convergence = [], []
    for family in FAMILIES:
        directory = stage_path(out_root, family, "tune")
        if not is_done(directory):
            continue
        payload = json.loads(find_study_json(directory, family).read_text(encoding="utf-8"))
        summary.append(
            {
                "family": family,
                "n_trials": payload["n_trials"],
                "best_value": payload["best_value"],
                "duration_s": payload["duration_s"],
                "best_classifier_params": json.dumps(
                    payload["best_classifier_params"], ensure_ascii=False, sort_keys=True
                ),
                "best_regressor_params": json.dumps(
                    payload["best_regressor_params"], ensure_ascii=False, sort_keys=True
                ),
                "fixed_params": json.dumps(
                    payload.get("fixed_params", {}), ensure_ascii=False, sort_keys=True
                ),
                "fold_n_jobs": payload.get("fold_n_jobs", 1),
                "status": payload["status"],
            }
        )
        values = pd.Series(
            [trial["value"] for trial in payload["trials"]], dtype="float64"
        )
        best_so_far = values.cummin().ffill()
        for trial, best in zip(payload["trials"], best_so_far):
            convergence.append(
                {
                    "family": family,
                    "trial": trial["number"],
                    "value": trial["value"],
                    "duration_s": trial["duration_s"],
                    "best_so_far": best,
                }
            )
    return (
        pd.DataFrame(
            summary,
            columns=[
                "family", "n_trials", "best_value", "duration_s", "best_classifier_params",
                "best_regressor_params", "fixed_params", "fold_n_jobs", "status",
            ],
        ),
        pd.DataFrame(
            convergence, columns=["family", "trial", "value", "duration_s", "best_so_far"]
        ),
    )


def assemble_tables(
    prepared,
    out_root: Path | str,
    baseline_oof_path: Path | str,
    flood_threshold_m3: float,
    *,
    max_folds: int | None = None,
) -> dict:
    """Las cuatro tablas del informe + resumen y convergencia de la afinación.

    Modelos: ``<familia>_default`` / ``<familia>_tuned`` por cada evaluación
    terminada, más la referencia sin aprendizaje restringida al protocolo de
    la comparación. ``max_folds`` (humo) limita todo a los primeros folds.
    """
    stages = _done_eval_stages(out_root)
    if not stages:
        raise FileNotFoundError(
            f"There is no finished evaluation stage (with a done marker) in {out_root}."
        )
    baseline_path = Path(baseline_oof_path)
    baseline = pd.read_parquet(baseline_path)
    baseline = baseline[baseline["protocol"] == EVAL_PROTOCOL].reset_index(drop=True)

    if max_folds is not None:
        prepared = restrict_prepared(prepared, EVAL_PROTOCOL, max_folds)
        baseline = restrict_oof(baseline, EVAL_PROTOCOL, max_folds)

    oof_by_name = {baseline_path.stem: baseline}
    for family, stage, directory in stages:
        oof_by_name[model_name(family, stage)] = pd.read_parquet(directory / "oof.parquet")

    tables = build_tables(prepared, oof_by_name, flood_threshold_m3)
    summary, convergence = _tuning_tables(out_root)
    tables["resumen_afinacion"] = summary
    tables["convergencia_afinacion"] = convergence
    return tables


def write_parte2_tables(
    tables: dict,
    tables_dir: Path | str,
    *,
    prepared,
    out_root: Path | str,
    baseline_path: Path | str,
    flood_threshold_m3: float,
    command: str,
    git: dict | None = None,
) -> Path:
    """Escribe los CSV y ``tables_provenance.json`` (comando, git, SHA-256 de entradas y salidas)."""
    directory = Path(tables_dir)
    git = git if git is not None else git_state()
    written = write_tables(tables, directory)

    root = Path(out_root)
    inputs = {Path(baseline_path).name: _sha256(Path(baseline_path))}
    for family, stage, stage_dir in _done_eval_stages(root):
        inputs[(stage_dir / "oof.parquet").relative_to(root).as_posix()] = _sha256(
            stage_dir / "oof.parquet"
        )
    for family in FAMILIES:
        tune_dir = stage_path(root, family, "tune")
        if is_done(tune_dir):
            study = find_study_json(tune_dir, family)
            inputs[study.relative_to(root).as_posix()] = _sha256(study)

    record = {
        "prep_id": prepared.prep_id,
        "feature_contract_id": prepared.manifest.get("contract_id"),
        "n_rows": int(len(prepared.keys)),
        **grid_counts(prepared.keys),
        "flood_threshold_m3": float(flood_threshold_m3),
        "max_continuity_error_pct": prepared.manifest.get("max_continuity_error_pct"),
        "evaluation_protocol": EVAL_PROTOCOL,
        "models": sorted(set(tables["comparacion_protocolos"]["modelo"])),
        "command": command,
        **git,
        "git_dirty_definition": "git status --porcelain --untracked-files=no non-empty at start",
        "out_root": portable_path(root),
        "inputs_sha256": inputs,
        "outputs_sha256": {path.name: _sha256(path) for path in sorted(written.values())},
        "versions": library_versions(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = directory / TABLES_PROVENANCE
    path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return path


# --- CLI ------------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--family", choices=FAMILIES, help="family whose stages are run")
    mode.add_argument(
        "--build-tables", action="store_true",
        help="build the report tables from whatever has finished",
    )
    parser.add_argument(
        "--eval-jobs", type=int, default=40,
        help="processes for the evaluation folds (default 40 = one per shape)",
    )
    parser.add_argument(
        "--fold-jobs", type=int, default=INNER_SPLITS,
        help="processes for the inner folds of each trial (default 5)",
    )
    parser.add_argument(
        "--stages", default=",".join(STAGES), help="stages to run, comma-separated"
    )
    # Opciones de apoyo para pruebas; no forman parte de la corrida real.
    parser.add_argument("--n-trials", type=int, default=N_TRIALS, help=argparse.SUPPRESS)
    parser.add_argument(
        "--smoke", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--out-root", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--tables-dir", type=Path, default=None, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def load_prepared(repo_root: Path = REPO_ROOT):
    """Carga el PreparedDataset fijado y verifica su manifiesto."""
    from swmm_resilience.ml.bench.preprocess import resolve_prepared

    prepared = resolve_prepared(repo_root / PREPARED_DIR, PREP_ID)
    check_manifest(prepared.manifest)
    if prepared.prep_id != PREP_ID or len(prepared.keys) != N_ROWS:
        raise ValueError(
            f"The loaded PreparedDataset is not the expected one: {prepared.prep_id}, "
            f"{len(prepared.keys)} rows."
        )
    return prepared


def _headline(metrics: dict) -> str:
    block = metrics[EVAL_PROTOCOL]
    return (
        f"rmse_vol_agregado={block['end_to_end_pooled'].get('rmse_vol_todos_nodos'):.3f} m3 "
        f"f1={block['classifier'].get('f1'):.4f} auc={block['classifier'].get('auc_roc'):.4f}"
    )


def _persist_evaluation(directory: Path, oof: pd.DataFrame, metrics: dict, prep_id: str) -> None:
    from swmm_resilience.ml.bench.reports import write_metrics_files

    oof.to_parquet(directory / "oof.parquet", index=False)
    write_metrics_files({"metrics": metrics}, directory, prefix="", prep_id=prep_id)


def run_family(args) -> int:
    from swmm_resilience.config import load_config
    from swmm_resilience.ml.bench.evaluate import evaluate_candidate
    from swmm_resilience.ml.bench.tune import SEARCH_SPACES, run_study, save_tune_result

    family = args.family
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise SystemExit(f"Unknown stages: {unknown}. Valid: {list(STAGES)}")

    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / DEFAULT_OUT_ROOT
    ensure_parte2_output(out_root)
    prepared = load_prepared()
    eval_prepared = (
        restrict_prepared(prepared, EVAL_PROTOCOL, SMOKE_FOLDS) if args.smoke else prepared
    )
    config = load_config(str(REPO_ROOT / "config.yaml"))
    fixed = FIXED_PARAMS[family]
    n_trials = effective_n_trials(args)
    git = git_state()
    versions = library_versions()
    n_folds = eval_prepared.folds.loc[
        eval_prepared.folds["protocol"] == EVAL_PROTOCOL, "fold_id"
    ].nunique()
    _log(
        f"family={family} prep_id={prepared.prep_id} rows={len(prepared.keys)} "
        f"protocol={EVAL_PROTOCOL} folds={n_folds} smoke={args.smoke} out_root={out_root}"
    )

    def evaluate(directory: Path, params: dict) -> dict:
        oof, metrics = evaluate_candidate(
            eval_prepared, family, params["classifier"], params["regressor"],
            protocols=[EVAL_PROTOCOL], n_jobs=args.eval_jobs,
        )
        _persist_evaluation(directory, oof, metrics, prepared.prep_id)
        _log(f"{family}: {_headline(metrics)}")
        return {"params": params, "headline": _headline(metrics)}

    def run_default(directory: Path) -> dict:
        return evaluate(directory, merge_fixed(config_params(config, family), fixed))

    def run_tuned(directory: Path) -> dict:
        study = find_study_json(stage_path(out_root, family, "tune"), family)
        result = evaluate(directory, best_params_from_study(study, fixed))
        result["study_json"] = study.relative_to(out_root).as_posix()
        return result

    def run_tune(directory: Path) -> dict:
        result = run_study(
            prepared, family, n_trials=n_trials, n_splits=INNER_SPLITS, seed=SEED,
            fold_n_jobs=args.fold_jobs, fixed_params=fixed,
        )
        path = save_tune_result(result, directory)
        _log(f"{family}: best value {result.best_value:.3f} m3 in {result.duration_s:.0f} s ({path})")
        return {
            "params": {
                "n_trials": result.n_trials,
                "inner_splits": INNER_SPLITS,
                "seed": SEED,
                "best_value": result.best_value,
                "best_classifier_params": result.best_classifier_params,
                "best_regressor_params": result.best_regressor_params,
                "search_space": SEARCH_SPACES[family],
            },
            "study_json": path.relative_to(out_root).as_posix(),
        }

    runners = {"default": run_default, "tune": run_tune, "tuned": run_tuned}
    for stage in STAGES:
        if stage not in stages:
            continue
        directory = stage_path(out_root, family, stage)
        ensure_parte2_output(directory)
        header = {
            "family": family,
            "stage": stage,
            "prep_id": prepared.prep_id,
            "contract_id": CONTRACT_ID,
            "protocol": EVAL_PROTOCOL,
            "smoke": bool(args.smoke),
            "fixed_params": fixed,
            "n_jobs": args.fold_jobs if stage == "tune" else args.eval_jobs,
            "seed": SEED,
            "git": git,
            "versions": versions,
            "script": "scripts/ejecutar_parte2_v4_21.py",
            "command": " ".join(["python", "scripts/ejecutar_parte2_v4_21.py", *sys.argv[1:]]),
        }
        run_stage_once(directory, runners[stage], header=header)
    _log(f"{family}: finished")
    return 0


def run_build_tables(args) -> int:
    from swmm_resilience.config import load_config

    out_root = Path(args.out_root) if args.out_root else REPO_ROOT / DEFAULT_OUT_ROOT
    if args.tables_dir:
        tables_dir = Path(args.tables_dir)
    elif args.smoke:
        tables_dir = out_root / "tables"
    else:
        tables_dir = REPO_ROOT / DEFAULT_TABLES_DIR
    ensure_parte2_output(tables_dir)
    prepared = load_prepared()
    config = load_config(str(REPO_ROOT / "config.yaml"))
    threshold = config.dataset.flood_threshold_m3
    baseline = REPO_ROOT / BASELINE_OOF
    max_folds = SMOKE_FOLDS if args.smoke else None

    git = git_state()
    tables = assemble_tables(prepared, out_root, baseline, threshold, max_folds=max_folds)
    confusion = tables["matriz_confusion"]
    n_expected = len(prepared.keys) if max_folds is None else None
    totals = confusion[["tn", "fp", "fn", "tp"]].sum(axis=1)
    if n_expected is not None and not (totals == n_expected).all():
        raise AssertionError(f"The confusion matrix does not add up to n_rows:\n{confusion}")
    # En humo la provenance describe el subconjunto de folds usado.
    described = restrict_prepared(prepared, EVAL_PROTOCOL, max_folds) if max_folds else prepared
    write_parte2_tables(
        tables, tables_dir, prepared=described, out_root=out_root, baseline_path=baseline,
        flood_threshold_m3=threshold,
        command=" ".join(["python", "scripts/ejecutar_parte2_v4_21.py", *sys.argv[1:]]),
        git=git,
    )
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(
            tables["comparacion_protocolos"].drop(columns=["que_mide"]).to_string(index=False)
        )
        print(tables["resumen_afinacion"].drop(
            columns=["best_classifier_params", "best_regressor_params", "status"]
        ).to_string(index=False))
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    return run_build_tables(args) if args.build_tables else run_family(args)


if __name__ == "__main__":
    raise SystemExit(main())
