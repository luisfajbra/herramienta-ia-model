"""Experimento rapido: ¿las 6 features tributarias (contrato tabular_v4_21,
Fase B2) mejoran el RMSE de XGBoost frente a las 17 features actuales
(tabular_v3_17)?

No migra el contrato ni persiste nada -- recalcula las features tributarias
para los escenarios ya generados (dataset_final.csv), entrena XGBoost con
LeaveOneGroupOut por shape_id sobre el MISMO conjunto de filas con y sin las
6 columnas nuevas, y compara el RMSE end-to-end. Si no hay mejora real, no
vale la pena migrar el contrato completo todavia.

La forma base ("base") se excluye: su node_series no viene de
load_all_shapes/apply_shape sino de la hidrografia embebida en el .inp, y
replicar ese camino no es necesario para esta validacion exploratoria.

Uso:
    venv/Scripts/python.exe scripts/validar_features_tributarias.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from swmm_resilience.config import ML_RANDOM_STATE, load_config
from swmm_resilience.extraction.dynamic_features import compute_tributary_dynamic_features
from swmm_resilience.extraction.topology import build_network_graph
from swmm_resilience.ml.contracts import FEATURE_COLUMNS_V17, FEATURE_COLUMNS_V21
from swmm_resilience.simulation.hydrograph_shapes import apply_shape, load_all_shapes
from swmm_resilience.simulation.inflow_closure import apply_closure_policy
from swmm_resilience.simulation.swmm_api_io import load_inp
from swmm_resilience.validation.hydrograph_csv import HydrographScenario

TRIBUTARY_COLUMNS = (
    "volumen_local_integrado_m3",
    "volumen_tributario_integrado_m3",
    "duracion_efectiva_tributario_h",
    "tiempo_primer_pico_tributario_h",
    "tiempo_sobre_capacidad_h",
    "integral_exceso_capacidad_m3",
)

_XGB_PARAMS = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8)


def _build_tributary_columns(df: pd.DataFrame, config) -> pd.DataFrame:
    """Recalcula las 6 features tributarias por (shape_id, factor_mult) y
    las devuelve como un DataFrame indexado igual que `df` (mismo orden)."""
    graph, _ = build_network_graph(load_inp(config.network.inp_path))
    shapes = load_all_shapes(config.simulation.hydrograph_shapes_dir)

    base_inflows = df.drop_duplicates("node_id").set_index("node_id")["base_inflow_lps"].to_dict()
    capacity_by_node = df.drop_duplicates("node_id").set_index("node_id")["upstream_capacity_lps"].to_dict()

    out_frames = []
    n_groups = df[["shape_id", "factor_mult"]].drop_duplicates().shape[0]
    done = 0
    for (shape_id, factor), group in df.groupby(["shape_id", "factor_mult"]):
        shape = shapes[shape_id]
        node_series = apply_shape(shape, base_inflows, factor)
        time_grid = [t for t, _ in shape]
        effective_scenario, _ = apply_closure_policy(
            HydrographScenario(
                scenario_id=f"{shape_id}_f{factor:.3f}",
                node_series=node_series,
                time_grid_hours=time_grid,
                last_time_hours=shape[-1][0],
            ),
            config.validation.inflow_closure_tolerance_lps,
        )
        trib_df = compute_tributary_dynamic_features(
            group["node_id"].tolist(), effective_scenario.node_series, graph, capacity_by_node,
        )
        trib_df = trib_df.set_index("node_id").loc[group["node_id"]].reset_index(drop=True)
        trib_df.index = group.index
        out_frames.append(trib_df)
        done += 1
        if done % 200 == 0 or done == n_groups:
            print(f"  features tributarias: {done}/{n_groups} escenarios")

    return pd.concat(out_frames).sort_index()


def _routed_rmse(X: pd.DataFrame, y_clf: np.ndarray, y_reg: np.ndarray, groups: np.ndarray) -> float:
    """LeaveOneGroupOut por shape_id, XGBoost con los hiperparametros de
    config.yaml, RMSE end-to-end promediado por fold (mismo criterio que el
    ranking del banco)."""
    logo = LeaveOneGroupOut()
    fold_rmse = []
    for train_idx, test_idx in logo.split(X, y_clf, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        yc_train, yc_test = y_clf[train_idx], y_clf[test_idx]
        yr_train, yr_test = y_reg[train_idx], y_reg[test_idx]

        n_neg, n_pos = int((yc_train == 0).sum()), int((yc_train == 1).sum())
        scale_pos_weight = n_neg / n_pos if n_pos else 1.0

        clf = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(**_XGB_PARAMS, scale_pos_weight=scale_pos_weight,
                                     eval_metric="logloss", random_state=ML_RANDOM_STATE)),
        ])
        clf.fit(X_train, yc_train)
        yc_pred = clf.predict(X_test)

        flooded_train = yc_train == 1
        if flooded_train.sum():
            reg = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", XGBRegressor(**_XGB_PARAMS, random_state=ML_RANDOM_STATE)),
            ])
            reg.fit(X_train.iloc[flooded_train], np.log1p(yr_train[flooded_train]))
            yr_pred = np.clip(np.expm1(reg.predict(X_test)), a_min=0.0, a_max=None)
        else:
            yr_pred = np.zeros(len(test_idx), dtype=float)

        routed = np.where(yc_pred == 1, yr_pred, 0.0)
        fold_rmse.append(float(np.sqrt(mean_squared_error(yr_test, routed))))

    return float(np.mean(fold_rmse))


def main() -> None:
    config = load_config("config.yaml")
    df = pd.read_csv("data/training/dataset_final.csv")
    df = df[df["shape_id"] != "base"].reset_index(drop=True)
    if config.validation.max_continuity_error_pct is not None:
        df = df[
            df["continuity_error_pct"].isna()
            | (df["continuity_error_pct"].abs() <= config.validation.max_continuity_error_pct)
        ].reset_index(drop=True)
    print(f"Filas tras excluir 'base' y continuidad: {len(df)} ({df['shape_id'].nunique()} formas)")

    print("Calculando las 6 features tributarias para cada escenario...")
    trib = _build_tributary_columns(df, config)
    df_v21 = pd.concat([df.reset_index(drop=True), trib.reset_index(drop=True)], axis=1)

    y_clf = df["inunda"].astype(int).to_numpy()
    y_reg = df["vol_inundacion_m3"].astype(float).to_numpy()
    groups = df["shape_id"].to_numpy()

    print("\nEvaluando XGBoost con 17 features (tabular_v3_17)...")
    rmse_v17 = _routed_rmse(df[list(FEATURE_COLUMNS_V17)], y_clf, y_reg, groups)
    print(f"  RMSE end-to-end (v3_17, 17 features): {rmse_v17:.4f}")

    print("\nEvaluando XGBoost con 17+6 features (ad-hoc, union simple)...")
    cols_union = list(FEATURE_COLUMNS_V17) + list(TRIBUTARY_COLUMNS)
    rmse_union = _routed_rmse(df_v21[cols_union], y_clf, y_reg, groups)
    print(f"  RMSE end-to-end (union, 23 features): {rmse_union:.4f}")

    print("\nEvaluando XGBoost con el contrato EXACTO tabular_v4_21 (21 features, "
          "sin duracion_horas ni tiempo_al_pico_h)...")
    rmse_v21 = _routed_rmse(df_v21[list(FEATURE_COLUMNS_V21)], y_clf, y_reg, groups)
    print(f"  RMSE end-to-end (tabular_v4_21, 21 features): {rmse_v21:.4f}")

    for label, rmse_alt in [("union 23f", rmse_union), ("tabular_v4_21 21f", rmse_v21)]:
        delta = rmse_v17 - rmse_alt
        pct = 100 * delta / rmse_v17 if rmse_v17 else float("nan")
        print(f"\nDelta vs. v3_17 ({label}): {delta:+.4f} ({pct:+.2f}%) "
              f"{'mejora' if delta > 0 else 'empeora o igual'}")


if __name__ == "__main__":
    main()
