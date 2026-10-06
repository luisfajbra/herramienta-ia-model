"""Compara XGBoost sobre tabular_v3_17 (legado) vs. tabular_v4_21 (con
features tributarias), con el protocolo OFICIAL del banco
(LeaveOneShapeOut) y los datos ya persistidos en SQL -- la confirmacion
definitiva de la mejora exploratoria encontrada antes de este plan
(+27.9% RMSE con un experimento ad-hoc y LeaveOneGroupOut).

Uso:
    venv/Scripts/python.exe scripts/comparar_contratos_xgboost.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swmm_resilience.config import load_config
from swmm_resilience.ml.bench.evaluate import evaluate_candidate
from swmm_resilience.ml.bench.preprocess import prepare_dataset
from swmm_resilience.ml.contracts import (
    FEATURE_COLUMNS_V17, FEATURE_COLUMNS_V21, TABULAR_V3_17, TABULAR_V4_21,
)

PROTOCOL = "LeaveOneShapeOut"


def main() -> None:
    config = load_config("config.yaml")
    xgboost_spec = config.bench.families["xgboost"]

    print("Preparando tabular_v3_17 (legado)...")
    prepared_v17 = prepare_dataset(
        config.dataset.db_path,
        protocols=(PROTOCOL,),
        flood_threshold_m3=config.dataset.flood_threshold_m3,
        output_dir=Path("outputs/bench/prepared_comparacion_v17"),
        max_continuity_error_pct=config.validation.max_continuity_error_pct,
    )
    _, metrics_v17 = evaluate_candidate(
        prepared_v17, "xgboost", xgboost_spec.classifier, xgboost_spec.regressor,
    )
    rmse_v17 = metrics_v17[PROTOCOL]["end_to_end"]["rmse_vol_todos_nodos"]
    print(f"  RMSE ({PROTOCOL}, tabular_v3_17, {len(FEATURE_COLUMNS_V17)} features): {rmse_v17:.4f}")

    print("\nPreparando tabular_v4_21 (con features tributarias)...")
    prepared_v21 = prepare_dataset(
        config.dataset.db_path,
        protocols=(PROTOCOL,),
        flood_threshold_m3=config.dataset.flood_threshold_m3,
        output_dir=Path("outputs/bench/prepared_comparacion_v4_21"),
        max_continuity_error_pct=config.validation.max_continuity_error_pct,
        feature_columns=FEATURE_COLUMNS_V21,
        feature_contract=TABULAR_V4_21,
        view_name="training_samples_v4_21",
        node_features_table="node_features_v4_21",
    )
    _, metrics_v21 = evaluate_candidate(
        prepared_v21, "xgboost", xgboost_spec.classifier, xgboost_spec.regressor,
    )
    rmse_v21 = metrics_v21[PROTOCOL]["end_to_end"]["rmse_vol_todos_nodos"]
    print(f"  RMSE ({PROTOCOL}, tabular_v4_21, {len(FEATURE_COLUMNS_V21)} features): {rmse_v21:.4f}")

    delta = rmse_v17 - rmse_v21
    pct = 100 * delta / rmse_v17 if rmse_v17 else float("nan")
    print(f"\nDelta oficial (protocolo {PROTOCOL}): {delta:+.4f} ({pct:+.2f}%) "
          f"{'mejora' if delta > 0 else 'empeora o igual'}")

    if delta > 0:
        import joblib

        from swmm_resilience.ml.bench.models.xgboost_family import build_classifier, build_regressor

        print("\nMejora confirmada -- entrenando el modelo final sobre el 100% de tabular_v4_21...")
        X = prepared_v21.X
        y_clf = prepared_v21.y_clf.to_numpy()
        y_reg = prepared_v21.y_reg.to_numpy()
        n_neg, n_pos = int((y_clf == 0).sum()), int((y_clf == 1).sum())
        scale_pos_weight = n_neg / n_pos if n_pos else 1.0

        clf = build_classifier(dict(xgboost_spec.classifier), scale_pos_weight)
        clf.fit(X, y_clf)
        flooded = y_clf == 1
        import numpy as np
        reg = build_regressor(dict(xgboost_spec.regressor))
        reg.fit(X.iloc[flooded], np.log1p(y_reg[flooded]))

        models_dir = Path("outputs/models")
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, models_dir / "classifier.joblib")
        joblib.dump(reg, models_dir / "regressor.joblib")

        import hashlib
        from datetime import datetime, timezone

        def _sha256(path):
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(8192), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        (models_dir / "promotion.json").write_text(
            json.dumps(
                {
                    "family": "xgboost",
                    "feature_contract_id": TABULAR_V4_21.contract_id,
                    "ordered_features": list(FEATURE_COLUMNS_V21),
                    "classifier_sha256": _sha256(models_dir / "classifier.joblib"),
                    "regressor_sha256": _sha256(models_dir / "regressor.joblib"),
                    "source": "scripts/comparar_contratos_xgboost.py (migracion tabular_v4_21, delta confirmado)",
                    "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        def _md5(path):
            digest = hashlib.md5()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(8192), b""):
                    digest.update(chunk)
            return digest.hexdigest()

        (models_dir / "training_inp_hash.txt").write_text(_md5(config.network.inp_path))
        print(f"  Modelo tabular_v4_21 promovido a {models_dir}")


if __name__ == "__main__":
    main()
