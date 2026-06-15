import hashlib
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..config import Config

# Feature contract v2: factor_mult is intentionally NOT a model input. It is a
# global scenario attribute with no valid definition for arbitrary hydrographs;
# it remains in the dataset CSV only as metadata (LOSO grouping, stratified
# evaluation). The per-node dynamic signal enters via q_pico_nodo and
# q_pico_acum_escalado.
FEATURE_COLS = [
    "elev_fondo", "prof_max", "n_tuberias_in", "n_tuberias_out",
    "diam_max_in", "diam_max_out", "pendiente_max_in", "pendiente_out",
    "base_inflow_lps", "dist_outfall_m", "n_nodos_aguas_arriba",
    "q_pico_acum_base", "upstream_capacity_lps",
    "q_pico_nodo", "q_pico_acum_escalado",
]


def make_classifier(config: Config, scale_pos_weight: float) -> Pipeline:
    clf_cfg = config.ml.classifier
    spw = scale_pos_weight if clf_cfg.scale_pos_weight == "auto" else float(clf_cfg.scale_pos_weight)
    if clf_cfg.algorithm == "xgboost":
        model = XGBClassifier(
            n_estimators=clf_cfg.n_estimators, max_depth=clf_cfg.max_depth,
            learning_rate=clf_cfg.learning_rate, subsample=clf_cfg.subsample,
            scale_pos_weight=spw, eval_metric="logloss", random_state=42,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=clf_cfg.n_estimators, max_depth=clf_cfg.max_depth, random_state=42,
        )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def make_regressor(config: Config) -> Pipeline:
    reg_cfg = config.ml.regressor
    if reg_cfg.algorithm == "xgboost":
        model = XGBRegressor(
            n_estimators=reg_cfg.n_estimators, max_depth=reg_cfg.max_depth,
            learning_rate=reg_cfg.learning_rate, subsample=reg_cfg.subsample,
            random_state=42,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=reg_cfg.n_estimators, max_depth=reg_cfg.max_depth, random_state=42,
        )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_models(df: pd.DataFrame, config: Config, output_dir: Path) -> tuple:
    """Train classifier and regressor on the full dataset. Returns (clf_pipeline, reg_pipeline).

    Saves classifier.joblib, regressor.joblib, and training_inp_hash.txt to output_dir.
    Regressor is trained only on rows where inunda=1 (oracle training set).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df[FEATURE_COLS]
    y_clf = df["inunda"]
    n_neg, n_pos = (y_clf == 0).sum(), (y_clf == 1).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    clf = make_classifier(config, spw)
    clf.fit(X, y_clf)

    df_flooded = df[df["inunda"] == 1]
    if df_flooded.empty:
        raise ValueError("No hay filas inundadas para entrenar el regresor.")
    reg = make_regressor(config)
    reg.fit(df_flooded[FEATURE_COLS], np.log1p(df_flooded["vol_inundacion_m3"]))

    joblib.dump(clf, output_dir / "classifier.joblib")
    joblib.dump(reg, output_dir / "regressor.joblib")
    (output_dir / "training_inp_hash.txt").write_text(_md5(config.network.inp_path))

    print(f"Modelos guardados en {output_dir}  (spw={spw:.2f}, flooded_train={n_pos})")
    return clf, reg
