"""Tablas del informe de evaluación a partir de OOF persistidos.

Uso:
    python scripts/informe_evaluacion.py --prep-id <ID>
        [--prepared-dir outputs/bench/prepared_v4_21] [--oof-root outputs/bench/oof]
        [--out docs/auditoria/correcciones/datos] [--flood-threshold 1.0]

Lee el PreparedDataset ``<prepared-dir>/<prep-id>/`` y cada
``<oof-root>/<prep-id>/<nombre>.parquet`` (un modelo o referencia por
archivo) y escribe cuatro CSV: ``comparacion_protocolos``,
``matriz_confusion``, ``banda_limitrofe`` y ``por_forma``. Solo lee datos ya
calculados: no entrena ni toca la base.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swmm_resilience.ml.bench.evaluate import score_predictions  # noqa: E402
from swmm_resilience.ml.bench.preprocess import resolve_prepared  # noqa: E402
from swmm_resilience.ml.bench.schemas import PROTOCOL_INTERPRETATION  # noqa: E402

BAND_MAX_M3 = 2.0
DEFAULT_PREPARED_DIR = Path("outputs/bench/prepared_v4_21")
DEFAULT_OOF_ROOT = Path("outputs/bench/oof")
DEFAULT_OUT = Path("docs/auditoria/correcciones/datos")


def _scored(prepared, oof):
    protocols = tuple(sorted(oof["protocol"].unique()))
    return score_predictions(prepared, oof, {"prep_id": prepared.prep_id}, protocols=protocols)


def build_tables(prepared, oof_by_name: dict, flood_threshold_m3: float) -> dict:
    """Las cuatro tablas del informe, una fila por (modelo, protocolo[, forma]).

    ``banda_limitrofe`` separa los falsos negativos por volumen real: entre el
    umbral y ``BAND_MAX_M3`` (casi en el límite, error de poca consecuencia) y
    fuera de esa banda.
    """
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
                    "protocolo": protocol,
                    "que_mide": PROTOCOL_INTERPRETATION.get(protocol, ""),
                    "f1_agregado": block["classifier"].get("f1"),
                    "recall_agregado": block["classifier"].get("recall"),
                    "precision_agregada": block["classifier"].get("precision"),
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
        table.to_csv(path, index=False)
        written[name] = path
        print(f"{name}: {len(table)} filas -> {path}")
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prep-id", required=True)
    parser.add_argument("--prepared-dir", type=Path, default=DEFAULT_PREPARED_DIR)
    parser.add_argument("--oof-root", type=Path, default=DEFAULT_OOF_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--flood-threshold", type=float, default=1.0)
    args = parser.parse_args(argv)

    # Sin ``protocols``: el informe cubre los que traiga el PreparedDataset; la
    # cobertura de cada OOF se verifica después en score_predictions.
    prepared = resolve_prepared(args.prepared_dir, args.prep_id)
    oof_dir = args.oof_root / args.prep_id
    oof_by_name = {path.stem: pd.read_parquet(path) for path in sorted(oof_dir.glob("*.parquet"))}
    if not oof_by_name:
        raise SystemExit(f"No hay archivos OOF (*.parquet) en {oof_dir}.")
    tables = build_tables(prepared, oof_by_name, args.flood_threshold)
    write_tables(tables, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
