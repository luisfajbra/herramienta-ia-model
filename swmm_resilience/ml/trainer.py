import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from ..config import Config, ML_RANDOM_STATE
from .contracts import FEATURE_COLUMNS_V17, FEATURE_COLUMNS_V21, TABULAR_V3_17

FEATURE_COLS = list(FEATURE_COLUMNS_V21)


def make_classifier(config: Config, scale_pos_weight: float) -> Pipeline:
    clf_cfg = config.ml.classifier
    spw = scale_pos_weight if clf_cfg.scale_pos_weight == "auto" else float(clf_cfg.scale_pos_weight)
    if clf_cfg.algorithm == "xgboost":
        model = XGBClassifier(
            n_estimators=clf_cfg.n_estimators, max_depth=clf_cfg.max_depth,
            learning_rate=clf_cfg.learning_rate, subsample=clf_cfg.subsample,
            scale_pos_weight=spw, eval_metric="logloss", random_state=ML_RANDOM_STATE,
        )
    else:
        model = RandomForestClassifier(
            n_estimators=clf_cfg.n_estimators, max_depth=clf_cfg.max_depth, random_state=ML_RANDOM_STATE,
        )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def make_regressor(config: Config) -> Pipeline:
    reg_cfg = config.ml.regressor
    if reg_cfg.algorithm == "xgboost":
        model = XGBRegressor(
            n_estimators=reg_cfg.n_estimators, max_depth=reg_cfg.max_depth,
            learning_rate=reg_cfg.learning_rate, subsample=reg_cfg.subsample,
            random_state=ML_RANDOM_STATE,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=reg_cfg.n_estimators, max_depth=reg_cfg.max_depth, random_state=ML_RANDOM_STATE,
        )
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
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

    promotion_path = output_dir / "promotion.json"
    if promotion_path.exists():
        existing = json.loads(promotion_path.read_text(encoding="utf-8"))
        existing_contract = existing.get("feature_contract_id", TABULAR_V3_17.contract_id)
        if existing_contract != TABULAR_V3_17.contract_id:
            raise RuntimeError(
                f"outputs/models/ ya tiene un modelo promovido bajo el contrato "
                f"'{existing_contract}' (no '{TABULAR_V3_17.contract_id}'). "
                "train_models() (pipeline legacy / --only-ml) no puede sobreescribirlo "
                "silenciosamente. Si de verdad quieres reemplazar el modelo activo por "
                "el legacy de 17 features, borra outputs/models/promotion.json primero."
            )

    X = TABULAR_V3_17.validate_frame(df.loc[:, FEATURE_COLUMNS_V17])
    y_clf = df["inunda"]
    n_neg, n_pos = (y_clf == 0).sum(), (y_clf == 1).sum()
    spw = n_neg / n_pos if n_pos > 0 else 1.0

    clf = make_classifier(config, spw)
    clf.fit(X, y_clf)

    df_flooded = df[df["inunda"] == 1]
    if df_flooded.empty:
        raise ValueError("No hay filas inundadas para entrenar el regresor.")
    reg = make_regressor(config)
    reg.fit(X.loc[df_flooded.index], np.log1p(df_flooded["vol_inundacion_m3"]))

    joblib.dump(clf, output_dir / "classifier.joblib")
    joblib.dump(reg, output_dir / "regressor.joblib")
    (output_dir / "training_inp_hash.txt").write_text(_md5(config.network.inp_path))

    promotion_path.write_text(
        json.dumps(
            {
                "family": "xgboost_legacy" if config.ml.classifier.algorithm == "xgboost" else "random_forest_legacy",
                "feature_contract_id": TABULAR_V3_17.contract_id,
                "ordered_features": list(FEATURE_COLUMNS_V17),
                "classifier_sha256": _sha256_of(output_dir / "classifier.joblib"),
                "regressor_sha256": _sha256_of(output_dir / "regressor.joblib"),
                "source": "train_models (legacy, --only-ml / default pipeline)",
                "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Modelos guardados en {output_dir}  (spw={spw:.2f}, flooded_train={n_pos})")
    return clf, reg
