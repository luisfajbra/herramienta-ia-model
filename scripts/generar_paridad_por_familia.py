"""Genera graficos de paridad (SWMM vs. prediccion) para cada familia del banco.

Usa las mismas predicciones out-of-fold (protocolo LeaveOneShapeOut) que el
ranking de `--bench-evaluate`, asi que los graficos son consistentes con los
numeros ya reportados -- no es un ajuste optimista sobre el 100% de los datos.

Uso:
    venv/Scripts/python.exe scripts/generar_paridad_por_familia.py \
        --models xgboost,random_forest,linear,mlp \
        --prep-id 2db0578f875526cb \
        --out-dir outputs/bench/reports/paridad
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from swmm_resilience.config import load_config
from swmm_resilience.ml.bench.evaluate import evaluate_candidate
from swmm_resilience.ml.bench.preprocess import resolve_prepared
from swmm_resilience.visualization.model_comparison import (
    plot_node_profiles,
    plot_parity_aggregated,
    plot_parity_nodes,
)

PROTOCOL = "LeaveOneShapeOut"


def _comparison_df(prepared, oof_protocol: pd.DataFrame) -> pd.DataFrame:
    """Construye el DataFrame vol_swmm_m3/vol_pred_m3/node_id esperado por
    las funciones de ploteo, a partir de las filas OOF de un protocolo."""
    idx = oof_protocol["sample_idx"].to_numpy()
    yc_pred = oof_protocol["y_pred_clf"].to_numpy()
    yr_pred = oof_protocol["y_pred_reg"].to_numpy()
    routed = np.where(yc_pred == 1, yr_pred, 0.0)

    keys = prepared.keys.iloc[idx]
    y_true = prepared.y_reg.iloc[idx].to_numpy()

    return pd.DataFrame(
        {
            "node_id": keys["node_id"].to_numpy(),
            "shape_id": keys["shape_id"].to_numpy(),
            "factor_mult": keys["factor_mult"].to_numpy(),
            "vol_swmm_m3": y_true,
            "vol_pred_m3": routed,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="Familias separadas por coma")
    parser.add_argument("--prep-id", required=True, help="PreparedDataset a usar")
    parser.add_argument(
        "--out-dir", default="outputs/bench/reports/paridad",
        help="Directorio de salida (una subcarpeta por familia)",
    )
    args = parser.parse_args()

    config = load_config("config.yaml")
    families = [name.strip() for name in args.models.split(",")]
    prepared = resolve_prepared(
        Path("outputs/bench/prepared"), args.prep_id, tuple(config.bench.protocols)
    )
    out_root = Path(args.out_dir)

    for family in families:
        print(f"Evaluando {family} (OOF, protocolo {PROTOCOL})...")
        spec = config.bench.families[family]
        oof, metrics = evaluate_candidate(prepared, family, spec.classifier, spec.regressor)

        oof_protocol = oof[oof["protocol"] == PROTOCOL]
        comp_df = _comparison_df(prepared, oof_protocol)

        out_dir = out_root / family
        label = family.replace("_", " ").title().replace("Mlp", "MLP").replace("Xgboost", "XGBoost")

        agg = comp_df.groupby("node_id", as_index=False)[["vol_swmm_m3", "vol_pred_m3"]].sum()
        plot_parity_nodes(agg, out_dir, "todas_las_corridas", model_label=label)
        plot_parity_aggregated(comp_df, out_dir, "todas_las_corridas", model_label=label)
        plot_node_profiles(agg, out_dir, "todas_las_corridas", model_label=label)

        rmse = metrics[PROTOCOL]["end_to_end"]["rmse_vol_todos_nodos"]
        print(f"  {family}: RMSE OOF = {rmse:.2f}  ->  {out_dir}")

    print("Listo.")


if __name__ == "__main__":
    main()
