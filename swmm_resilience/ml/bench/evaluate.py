"""Etapa 3 del banco, en dos capas.

score_predictions() es la capa núcleo: NO sabe de modelos. Recibe
predicciones out-of-fold ya calculadas, verifica su proveniencia contra el
PreparedDataset, y devuelve las métricas de los tres niveles. Cualquier
modelo —tabular, temporal, escrito en otro script o dentro de seis meses—
entra por aquí y su número cae en la misma tabla. Es el mecanismo que
garantiza "mismas entradas, mismos benchmarks" (spec §13.1).

evaluate_candidate() es la capa de conveniencia que produce esas
predicciones para las familias tabulares del banco (Task 9).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import (
    classifier_metrics,
    end_to_end_metrics,
    mean_metrics,
    pooled_regressor_metrics,
)
from .schemas import OOF_COLUMNS, PreparedDataset


class ProvenanceMismatchError(ValueError):
    """Las predicciones no corresponden a este PreparedDataset."""


def _validate_oof(prepared: PreparedDataset, oof: pd.DataFrame) -> None:
    missing = [column for column in OOF_COLUMNS if column not in oof.columns]
    if missing:
        raise ValueError(f"Al frame OOF le faltan columnas: {missing}")

    duplicated = oof.duplicated(subset=["protocol", "fold_id", "sample_idx"]).sum()
    if duplicated:
        raise ValueError(
            f"El frame OOF tiene {duplicated} predicción(es) duplicadas para la misma "
            "(protocol, fold_id, sample_idx)."
        )

    expected = prepared.folds[prepared.folds["split"] == "test"]
    expected_keys = set(
        map(tuple, expected[["protocol", "fold_id", "sample_idx"]].itertuples(index=False))
    )
    got_keys = set(
        map(tuple, oof[["protocol", "fold_id", "sample_idx"]].itertuples(index=False))
    )
    if expected_keys != got_keys:
        raise ValueError(
            "La cobertura del frame OOF no coincide con las filas de test de los folds: "
            f"faltan {len(expected_keys - got_keys)}, sobran {len(got_keys - expected_keys)}."
        )


def score_predictions(
    prepared: PreparedDataset,
    oof: pd.DataFrame,
    provenance: dict,
) -> dict:
    """Puntúa predicciones out-of-fold contra ``prepared``.

    ``provenance`` debe traer el ``prep_id`` del dataset con el que se
    generaron. Sin esa verificación se podrían comparar números de datasets
    distintos sin notarlo, que es el modo de fallo silencioso que este
    diseño existe para cerrar.
    """
    if provenance.get("prep_id") != prepared.prep_id:
        raise ProvenanceMismatchError(
            f"prep_id de las predicciones ({provenance.get('prep_id')!r}) no coincide "
            f"con el del dataset ({prepared.prep_id!r}). No son comparables."
        )
    _validate_oof(prepared, oof)

    y_clf = prepared.y_clf.to_numpy()
    y_reg = prepared.y_reg.to_numpy()
    factors = prepared.keys["factor_mult"].to_numpy()

    results: dict = {}
    for protocol, protocol_oof in oof.groupby("protocol"):
        classifier_folds, e2e_folds = [], []
        oracle_true, oracle_pred = [], []
        by_factor: dict[str, list] = {}

        for _, fold_oof in protocol_oof.groupby("fold_id"):
            idx = fold_oof["sample_idx"].to_numpy()
            yc_true, yr_true = y_clf[idx], y_reg[idx]
            yc_pred = fold_oof["y_pred_clf"].to_numpy()
            yc_prob = fold_oof["y_prob_clf"].to_numpy()
            yr_pred = fold_oof["y_pred_reg"].to_numpy()

            classifier_folds.append(classifier_metrics(yc_true, yc_pred, yc_prob))

            flooded = yc_true == 1
            if flooded.sum():
                oracle_true.append(yr_true[flooded])
                oracle_pred.append(yr_pred[flooded])

            routed = np.where(yc_pred == 1, yr_pred, 0.0)
            e2e_folds.append(end_to_end_metrics(yr_true, routed, yc_true, yc_pred))

            for factor in np.unique(factors[idx]):
                mask = factors[idx] == factor
                key = f"{factor:.2f}"
                by_factor.setdefault(key, []).append(
                    {
                        "f1": classifier_metrics(
                            yc_true[mask], yc_pred[mask], yc_prob[mask]
                        )["f1"],
                        "rmse_vol": end_to_end_metrics(
                            yr_true[mask], routed[mask], yc_true[mask], yc_pred[mask]
                        )["rmse_vol_todos_nodos"],
                    }
                )

        results[protocol] = {
            "classifier": mean_metrics(classifier_folds),
            "regressor_oracle": pooled_regressor_metrics(oracle_true, oracle_pred),
            "end_to_end": mean_metrics(e2e_folds),
            "by_factor": {key: mean_metrics(value) for key, value in by_factor.items()},
        }
    return results


def evaluate_candidate(
    prepared: PreparedDataset,
    family_name: str,
    classifier_params: dict,
    regressor_params: dict,
) -> tuple[pd.DataFrame, dict]:
    """Entrena y predice fold a fold, y puntúa el resultado.

    Es una función pura de los hiperparámetros: exactamente la forma que
    necesita un ``objective`` de Optuna (spec §13.2).

    El Pipeline de la familia lleva dentro la imputación y el escalado, así
    que ``fit`` sobre el train del fold los ajusta SOLO con esas filas. Ahí
    está la garantía anti-fuga en ejecución.
    """
    from .registry import get_family

    family = get_family(family_name)
    X = prepared.X
    y_clf = prepared.y_clf.to_numpy()
    y_reg = prepared.y_reg.to_numpy()

    records: list[pd.DataFrame] = []
    for (protocol, fold_id), chunk in prepared.folds.groupby(["protocol", "fold_id"]):
        train_idx = chunk.loc[chunk["split"] == "train", "sample_idx"].to_numpy()
        test_idx = chunk.loc[chunk["split"] == "test", "sample_idx"].to_numpy()

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        yc_train = y_clf[train_idx]
        yr_train = y_reg[train_idx]

        n_negative, n_positive = int((yc_train == 0).sum()), int((yc_train == 1).sum())
        scale_pos_weight = n_negative / n_positive if n_positive else 1.0

        classifier = family.build_classifier(classifier_params, scale_pos_weight)
        classifier.fit(X_train, yc_train)
        yc_pred = classifier.predict(X_test)
        yc_prob = classifier.predict_proba(X_test)[:, 1]

        flooded_train = yc_train == 1
        if flooded_train.sum():
            regressor = family.build_regressor(regressor_params)
            regressor.fit(X_train.iloc[flooded_train], np.log1p(yr_train[flooded_train]))
            yr_pred = np.clip(np.expm1(regressor.predict(X_test)), a_min=0.0, a_max=None)
        else:
            # Divergencia deliberada de _run_cv: allí, si el fold no tiene filas
            # inundadas de entrenamiento, el regresor nunca se ajusta pero el
            # nivel end-to-end igual puede invocar reg.predict(...), lo que
            # lanzaría NotFittedError (bug latente no manejado en el código
            # legado). Aquí predecimos volumen cero explícitamente en su
            # lugar. Con LOSO sobre los ~25 factores reales cada fold de
            # entrenamiento tiene filas inundadas, así que esta rama no
            # afecta al test de paridad.
            yr_pred = np.zeros(len(test_idx), dtype=float)

        records.append(
            pd.DataFrame(
                {
                    "sample_idx": test_idx,
                    "protocol": protocol,
                    "fold_id": fold_id,
                    "y_pred_clf": yc_pred.astype(int),
                    "y_prob_clf": yc_prob.astype(float),
                    "y_pred_reg": yr_pred.astype(float),
                },
                columns=list(OOF_COLUMNS),
            )
        )

    oof = pd.concat(records, ignore_index=True)
    metrics = score_predictions(prepared, oof, {"prep_id": prepared.prep_id})
    return oof, metrics
