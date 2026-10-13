"""Tablas del informe de evaluación a partir de OOF persistidos.

Uso:
    python scripts/informe_evaluacion.py --prep-id <ID>
        [--prepared-dir outputs/bench/prepared_v4_21] [--oof-root outputs/bench/oof]
        [--out docs/auditoria/correcciones/datos] [--flood-threshold 1.0]

Lee el PreparedDataset ``<prepared-dir>/<prep-id>/`` y cada
``<oof-root>/<prep-id>/<nombre>.parquet`` (un modelo o referencia por
archivo) y escribe cuatro CSV: ``comparacion_protocolos``,
``matriz_confusion``, ``banda_limitrofe`` y ``por_forma`` (fin de línea LF),
más ``tables_provenance.json``: comando, commit y estado del árbol de git,
SHA-256 de cada OOF leído y de cada CSV escrito. Solo lee datos ya
calculados: no entrena ni toca la base.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from swmm_resilience.ml.bench.evaluate import score_predictions  # noqa: E402
from swmm_resilience.ml.bench.preprocess import resolve_prepared  # noqa: E402
from swmm_resilience.ml.bench.schemas import PROTOCOL_INTERPRETATION  # noqa: E402

BAND_MAX_M3 = 2.0
DEFAULT_PREPARED_DIR = Path("outputs/bench/prepared_v4_21")
DEFAULT_OOF_ROOT = Path("outputs/bench/oof")
DEFAULT_OUT = Path("docs/auditoria/correcciones/datos")
TABLES_PROVENANCE = "tables_provenance.json"


def portable_path(path: Path | str) -> str:
    """Ruta relativa al repositorio (posix); fuera de él, solo ``<externo>/<nombre>``.

    Evita dejar rutas absolutas de la máquina en archivos versionados.
    """
    target = Path(path)
    if not target.is_absolute():
        return target.as_posix()
    try:
        return target.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return f"<externo>/{target.name}"


def git_state() -> dict:
    """Commit de HEAD y si hay cambios sin commitear en archivos versionados.

    Los archivos sin seguimiento no cuentan (``--untracked-files=no``): no
    alteran el código que produce las tablas.
    """

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout

    return {
        "git_commit": git("rev-parse", "HEAD").strip(),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no").strip()),
    }


def grid_counts(keys: pd.DataFrame) -> dict:
    """Runs presentes frente a la malla completa (forma x factor)."""
    n_full = int(keys["shape_id"].nunique() * keys["factor_mult"].nunique())
    n_cells = int(len(keys[["shape_id", "factor_mult"]].drop_duplicates()))
    return {
        "n_runs": int(keys["run_id"].nunique()),
        "n_runs_full_grid": n_full,
        "n_missing_cells": n_full - n_cells,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scored(prepared, oof):
    protocols = tuple(sorted(oof["protocol"].unique()))
    return score_predictions(prepared, oof, {"prep_id": prepared.prep_id}, protocols=protocols)


def variant_of(name: str) -> str:
    """Variante de un modelo según su nombre: default | tuned | referencia.

    ``*_tuned`` es el modelo afinado, ``baseline_*`` una referencia sin
    aprendizaje y cualquier otro nombre (``xgboost``, ``*_default``) se
    considera con los parámetros por defecto de ``config.yaml``.
    """
    if name.startswith("baseline_"):
        return "referencia"
    if name.endswith("_tuned"):
        return "tuned"
    return "default"


def build_tables(prepared, oof_by_name: dict, flood_threshold_m3: float) -> dict:
    """Las cuatro tablas del informe, una fila por (modelo, protocolo[, forma]).

    ``banda_limitrofe`` separa los falsos negativos por volumen real: entre el
    umbral y ``BAND_MAX_M3`` (casi en el límite, error de poca consecuencia) y
    fuera de esa banda.

    ``flood_threshold_m3`` debe coincidir con el del manifiesto del
    PreparedDataset (el de ``inunda``); si no, la banda estaría mal rotulada.
    """
    declared = prepared.manifest.get("flood_threshold_m3")
    if declared is None or float(declared) != float(flood_threshold_m3):
        raise ValueError(
            f"flood_threshold_m3={flood_threshold_m3} no coincide con el del manifiesto "
            f"del PreparedDataset {prepared.prep_id} ({declared})."
        )
    y_true = prepared.y_clf.to_numpy()
    volume = prepared.y_reg.to_numpy()
    comparison, confusion, band, by_shape = [], [], [], []

    # Orden fijo (modelo, protocolo, forma) para que los CSV no dependan del
    # orden en que llegan los OOF (dict del script vs. glob del disco).
    for name, oof in sorted(oof_by_name.items()):
        scored = _scored(prepared, oof)
        for protocol, block in sorted(scored.items()):
            comparison.append(
                {
                    "modelo": name,
                    "variante": variant_of(name),
                    "protocolo": protocol,
                    "que_mide": PROTOCOL_INTERPRETATION.get(protocol, ""),
                    "f1_agregado": block["classifier"].get("f1"),
                    "recall_agregado": block["classifier"].get("recall"),
                    "precision_agregada": block["classifier"].get("precision"),
                    # AUC agregado: no depende del umbral ni de class_weight, así que
                    # compara clasificadores calibrados de forma distinta (el SVM con Platt).
                    "auc_roc_agregado": block["classifier"].get("auc_roc"),
                    "f1_media_por_fold_con_ceros": block["classifier_legacy_mean_zero_fill"].get("f1"),
                    "folds_f1_indefinido": block["classifier_per_fold"]["f1"]["n_undefined"],
                    "rmse_vol_agregado_m3": block["end_to_end_pooled"].get("rmse_vol_todos_nodos"),
                    "rmse_vol_media_por_fold_m3": block["end_to_end"].get("rmse_vol_todos_nodos"),
                    "error_vol_total_pct": block["end_to_end_pooled"].get("vol_total_error_pct"),
                    "nse_regresor_oracle": block["regressor_oracle"].get("nse"),
                }
            )
            confusion.append({"modelo": name, "protocolo": protocol, **block["confusion_matrix"]})
            for shape, entry in sorted(block["by_shape"].items()):
                by_shape.append({"modelo": name, "protocolo": protocol, "forma": shape, **entry})

            subset = oof[oof["protocol"] == protocol]
            idx = subset["sample_idx"].to_numpy()
            false_neg = (y_true[idx] == 1) & (subset["y_pred_clf"].to_numpy() == 0)
            in_band = false_neg & (volume[idx] < BAND_MAX_M3)
            band.append(
                {
                    "modelo": name,
                    "protocolo": protocol,
                    "fn_total": int(false_neg.sum()),
                    "fn_en_banda_1_2m3": int(in_band.sum()),
                    "fn_fuera_de_banda": int((false_neg & ~in_band).sum()),
                    "fn_con_vol_ge_100m3": int((false_neg & (volume[idx] >= 100.0)).sum()),
                }
            )

    return {
        "comparacion_protocolos": pd.DataFrame(comparison),
        "matriz_confusion": pd.DataFrame(confusion),
        "banda_limitrofe": pd.DataFrame(band),
        "por_forma": pd.DataFrame(by_shape),
    }


def write_tables(tables: dict, out_dir: Path | str) -> dict[str, Path]:
    """Escribe cada tabla como ``<out_dir>/<nombre>.csv`` y devuelve las rutas."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, table in tables.items():
        path = directory / f"{name}.csv"
        # LF explícito: mismos bytes en Windows y en Linux (SHA-256 comparables).
        table.to_csv(path, index=False, lineterminator="\n")
        written[name] = path
        print(f"{name}: {len(table)} filas -> {path}")
    return written


def _command(args) -> str:
    """Línea de comando efectiva (con los valores por defecto), sin rutas absolutas."""
    parts = [
        "python scripts/informe_evaluacion.py",
        "--prep-id", args.prep_id,
        "--prepared-dir", portable_path(args.prepared_dir),
        "--oof-root", portable_path(args.oof_root),
        "--out", portable_path(args.out),
        "--flood-threshold", repr(float(args.flood_threshold)),
    ]
    return " ".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prep-id", required=True)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--oof-root", type=Path, default=DEFAULT_OOF_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--flood-threshold", type=float, default=1.0)
    args = parser.parse_args(argv)
    # Antes de escribir nada: el estado de git que produce las tablas.
    git = git_state()

    # Sin ``protocols``: el informe cubre los que traiga el PreparedDataset; la
    # cobertura de cada OOF se verifica después en score_predictions.
    prepared = resolve_prepared(args.prepared_dir, args.prep_id)
    oof_dir = args.oof_root / args.prep_id
    oof_paths = sorted(oof_dir.glob("*.parquet"))
    if not oof_paths:
        raise SystemExit(f"No hay archivos OOF (*.parquet) en {oof_dir}.")
    inputs_sha256 = {f"{args.prep_id}/{path.name}": _sha256(path) for path in oof_paths}
    oof_by_name = {path.stem: pd.read_parquet(path) for path in oof_paths}
    tables = build_tables(prepared, oof_by_name, args.flood_threshold)
    written = write_tables(tables, args.out)

    record = {
        "prep_id": prepared.prep_id,
        "feature_contract_id": prepared.manifest.get("contract_id"),
        "n_rows": int(len(prepared.keys)),
        **grid_counts(prepared.keys),
        "flood_threshold_m3": float(args.flood_threshold),
        "max_continuity_error_pct": prepared.manifest.get("max_continuity_error_pct"),
        "command": _command(args),
        **git,
        "git_dirty_definition": "git status --porcelain --untracked-files=no no vacío al empezar",
        "oof_root": portable_path(args.oof_root),
        "inputs_sha256": inputs_sha256,
        "outputs_sha256": {path.name: _sha256(path) for path in sorted(written.values())},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (Path(args.out) / TABLES_PROVENANCE).write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
