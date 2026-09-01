"""Backfill the SQLite v17 schema from an already-assembled dataset_final.csv.

This exists because the active CLI pipeline (main.py) produces
dataset_final.csv + classifier.joblib/regressor.joblib directly and never
touches the SQLite v17 database. This module reads that CSV (one row per
node x factor x hydrograph shape) and writes the equivalent networks /
nodes / scenarios / runs / node_features / node_results rows, then trains
a fresh GroupKFold(5)-by-factor_mult model pair and persists the full
training_runs / model_evaluations / oof_predictions / trained_models
evidence required by migration 005's mandatory PENDING->RUNNING->COMPLETE
lifecycle.

Deliberately out of scope (see project decision — minimal path, not the
full evidence chain): model_candidates, model_rankings, model_promotions,
model_selections. The two trained_models rows this writes are valid,
COMPLETE, queryable artifacts, but are "historical storage only" per the
provenance-hardening spec (section 3.4.1) — they do not become an active
promotion/selection. That machinery can be added later without touching
what this module writes.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import GroupKFold

from ..config import Config, ML_RANDOM_STATE
from ..ml.contracts import FEATURE_COLUMNS_V17, TABULAR_V3_17
from ..ml.trainer import make_classifier, make_regressor

FOLD_COUNT = 5


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_json(obj) -> str:
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def backfill_networks_and_runs(
    conn: sqlite3.Connection,
    dataset: pd.DataFrame,
    inp_path: Path,
    network_name: str,
) -> dict:
    """Insert networks/nodes/scenarios/runs/node_features/node_results.

    Returns {"network_id": int, "node_pk_by_id": {node_id: node_pk},
    "run_id_by_key": {(shape_id, factor_mult): run_id}}.

    Idempotent on network identity (re-running with the same .inp reuses
    the existing network row instead of failing), but NOT idempotent on
    scenarios/runs — call this once per dataset snapshot.
    """
    inp_bytes = Path(inp_path).read_bytes()
    network_sha256 = _sha256_bytes(inp_bytes)

    existing = conn.execute(
        "SELECT network_id FROM networks WHERE network_sha256 = ?", (network_sha256,)
    ).fetchone()
    if existing:
        network_id = existing[0]
    else:
        cur = conn.execute(
            """
            INSERT INTO networks (
                network_sha256, name, source_filename, inp_bytes, flow_units, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (network_sha256, network_name, Path(inp_path).name, inp_bytes, "LPS", _now()),
        )
        network_id = cur.lastrowid

    node_pk_by_id: dict[str, int] = {}
    node_static = dataset.drop_duplicates("node_id")[
        ["node_id", "coord_x", "coord_y", "elev_fondo", "prof_max", "base_inflow_lps"]
    ]
    for row in node_static.itertuples(index=False):
        existing_node = conn.execute(
            "SELECT node_pk FROM nodes WHERE network_id = ? AND node_id = ?",
            (network_id, row.node_id),
        ).fetchone()
        if existing_node:
            node_pk_by_id[row.node_id] = existing_node[0]
            continue
        cur = conn.execute(
            """
            INSERT INTO nodes (
                network_id, node_id, node_type, coord_x, coord_y,
                invert_elevation_m, max_depth_m, base_inflow_lps
            ) VALUES (?, ?, 'junction', ?, ?, ?, ?, ?)
            """,
            (network_id, row.node_id, row.coord_x, row.coord_y,
             row.elev_fondo, row.prof_max, row.base_inflow_lps),
        )
        node_pk_by_id[row.node_id] = cur.lastrowid

    run_id_by_key: dict[tuple, int] = {}
    scenario_cols = ["shape_id", "factor_mult", "duracion_horas", "tiempo_al_pico_h"]
    scenarios = dataset.drop_duplicates(subset=["shape_id", "factor_mult"])[scenario_cols]

    for srow in scenarios.itertuples(index=False):
        scenario_key = f"{srow.shape_id}__f{srow.factor_mult:.2f}"
        config_payload = {
            "shape_id": srow.shape_id,
            "factor_mult": float(srow.factor_mult),
            "duracion_horas": float(srow.duracion_horas),
            "tiempo_al_pico_h": float(srow.tiempo_al_pico_h),
        }
        config_sha256 = _sha256_json(config_payload)

        cur = conn.execute(
            """
            INSERT INTO scenarios (
                network_id, scenario_key, scenario_kind, factor_mult, shape_id,
                duracion_horas, tiempo_al_pico_h, config_json, config_sha256
            ) VALUES (?, ?, 'swmm_batch', ?, ?, ?, ?, ?, ?)
            """,
            (network_id, scenario_key, float(srow.factor_mult), srow.shape_id,
             float(srow.duracion_horas), float(srow.tiempo_al_pico_h),
             json.dumps(config_payload, sort_keys=True), config_sha256),
        )
        scenario_id = cur.lastrowid

        subset = dataset[
            (dataset["shape_id"] == srow.shape_id)
            & (abs(dataset["factor_mult"] - srow.factor_mult) < 1e-9)
        ]
        cur = conn.execute(
            """
            INSERT INTO runs (
                scenario_id, network_id, status, started_at_utc, completed_at_utc,
                config_sha256, node_count
            ) VALUES (?, ?, 'COMPLETE', ?, ?, ?, ?)
            """,
            (scenario_id, network_id, _now(), _now(), config_sha256, len(subset)),
        )
        run_id = cur.lastrowid
        run_id_by_key[(srow.shape_id, float(srow.factor_mult))] = run_id

        feature_rows = []
        result_rows = []
        for r in subset.itertuples(index=False):
            node_pk = node_pk_by_id[r.node_id]
            feature_rows.append((
                run_id, network_id, node_pk,
                r.elev_fondo, r.prof_max, int(r.n_tuberias_in), int(r.n_tuberias_out),
                r.diam_max_in if pd.notna(r.diam_max_in) else None,
                r.diam_max_out if pd.notna(r.diam_max_out) else None,
                r.pendiente_max_in if pd.notna(r.pendiente_max_in) else None,
                r.pendiente_out if pd.notna(r.pendiente_out) else None,
                r.base_inflow_lps,
                r.dist_outfall_m if pd.notna(r.dist_outfall_m) else None,
                int(r.n_nodos_aguas_arriba), r.q_pico_acum_base,
                r.upstream_capacity_lps if pd.notna(r.upstream_capacity_lps) else None,
                r.q_pico_nodo, r.q_pico_acum_escalado,
                r.duracion_horas, r.tiempo_al_pico_h, "tabular_v3_17",
            ))
            result_rows.append((
                run_id, network_id, node_pk, int(r.inunda), float(r.vol_inundacion_m3),
            ))

        conn.executemany(
            """
            INSERT INTO node_features (
                run_id, network_id, node_pk, elev_fondo, prof_max,
                n_tuberias_in, n_tuberias_out, diam_max_in, diam_max_out,
                pendiente_max_in, pendiente_out, base_inflow_lps, dist_outfall_m,
                n_nodos_aguas_arriba, q_pico_acum_base, upstream_capacity_lps,
                q_pico_nodo, q_pico_acum_escalado, duracion_horas, tiempo_al_pico_h,
                feature_contract_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            feature_rows,
        )
        conn.executemany(
            """
            INSERT INTO node_results (run_id, network_id, node_pk, inunda, vol_inundacion_m3)
            VALUES (?,?,?,?,?)
            """,
            result_rows,
        )

    conn.commit()
    return {
        "network_id": network_id,
        "node_pk_by_id": node_pk_by_id,
        "run_id_by_key": run_id_by_key,
    }


def persist_training_run(
    conn: sqlite3.Connection,
    dataset: pd.DataFrame,
    run_id_by_key: dict,
    node_pk_by_id: dict,
    config: Config,
) -> int:
    """Train fresh GroupKFold(5)-by-factor_mult classifier+regressor and
    persist the full training_runs -> ... -> trained_models evidence chain
    (minimal path: no candidates/rankings/promotions/selections).
    """
    if (dataset["inunda"] == 1).sum() == 0:
        raise ValueError("No hay filas inundadas para entrenar el regresor.")

    dataset = dataset.copy()
    dataset["run_id"] = [
        run_id_by_key[(row.shape_id, float(row.factor_mult))]
        for row in dataset.itertuples(index=False)
    ]
    all_run_ids = sorted(int(v) for v in dataset["run_id"].unique())

    query_sql = "SELECT * FROM training_samples_v17"
    training_query_contract_sha256 = _sha256_bytes(query_sql.encode("utf-8"))

    stamp = _now()
    conn.execute(
        """
        INSERT INTO training_runs (
            training_run_id, target, feature_contract_id, feature_contract_sha256,
            query_sql, query_params_json, included_run_ids_json, grouping_strategy,
            fold_count, random_seed, primary_metric, tie_breakers_json,
            python_version, library_versions_json, status,
            training_query_contract_id, training_query_contract_sha256
        ) VALUES (
            (SELECT COALESCE(MAX(training_run_id), 0) + 1 FROM training_runs),
            'system', 'tabular_v3_17', ?, ?, '{}', ?, 'group_kfold', ?, ?,
            'f1', '[]', ?, '{}', 'PENDING', 'training_samples_v17', ?
        )
        """,
        (
            TABULAR_V3_17.descriptor_sha256, query_sql, json.dumps(all_run_ids),
            FOLD_COUNT, ML_RANDOM_STATE, sys.version.split()[0],
            training_query_contract_sha256,
        ),
    )
    training_run_id = conn.execute(
        "SELECT MAX(training_run_id) FROM training_runs"
    ).fetchone()[0]

    conn.executemany(
        "INSERT INTO training_run_inputs (training_run_id, run_id) VALUES (?, ?)",
        [(training_run_id, run_id) for run_id in all_run_ids],
    )
    conn.execute(
        "UPDATE training_runs SET status = 'RUNNING', started_at_utc = ? WHERE training_run_id = ?",
        (stamp, training_run_id),
    )

    X_full = TABULAR_V3_17.validate_frame(dataset.loc[:, FEATURE_COLUMNS_V17]).values
    y_clf_full = dataset["inunda"].to_numpy()
    y_reg_full = dataset["vol_inundacion_m3"].to_numpy()
    groups = dataset["factor_mult"].to_numpy()
    run_ids = dataset["run_id"].to_numpy()
    node_pks = np.array([node_pk_by_id[n] for n in dataset["node_id"]])

    cv = GroupKFold(n_splits=FOLD_COUNT)
    evaluation_id_clf: dict[int, int] = {}
    evaluation_id_reg: dict[int, int] = {}
    pooled_yc_true, pooled_yc_pred = [], []
    pooled_yr_true, pooled_yr_pred = [], []

    for fold_id, (train_idx, val_idx) in enumerate(cv.split(X_full, y_clf_full, groups)):
        train_run_ids = sorted(int(v) for v in np.unique(run_ids[train_idx]))
        val_run_ids = sorted(int(v) for v in np.unique(run_ids[val_idx]))

        for task, algorithm in (("classification", "xgboost"), ("regression", "xgboost")):
            conn.execute(
                """
                INSERT INTO model_evaluations (
                    training_run_id, task, algorithm, hyperparameters_json, fold_id,
                    train_run_ids_json, validation_run_ids_json, status,
                    fit_seconds, predict_seconds
                ) VALUES (?, ?, ?, '{}', ?, ?, ?, 'PENDING', 0, 0)
                """,
                (training_run_id, task, algorithm, fold_id,
                 json.dumps(train_run_ids), json.dumps(val_run_ids)),
            )
            evaluation_id = conn.execute(
                "SELECT MAX(evaluation_id) FROM model_evaluations"
            ).fetchone()[0]
            if task == "classification":
                evaluation_id_clf[fold_id] = evaluation_id
            else:
                evaluation_id_reg[fold_id] = evaluation_id

            role = "train"
            for rid in train_run_ids:
                conn.execute(
                    "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (?, ?, ?)",
                    (evaluation_id, role, rid),
                )
            for rid in val_run_ids:
                conn.execute(
                    "INSERT INTO model_evaluation_runs (evaluation_id, role, run_id) VALUES (?, 'validation', ?)",
                    (evaluation_id, rid),
                )
            conn.execute(
                "UPDATE model_evaluations SET status = 'RUNNING' WHERE evaluation_id = ?",
                (evaluation_id,),
            )

        # --- fit both models for this fold ---
        X_tr, X_val = X_full[train_idx], X_full[val_idx]
        yc_tr, yc_val = y_clf_full[train_idx], y_clf_full[val_idx]
        yr_tr = y_reg_full[train_idx]
        n_neg, n_pos = (yc_tr == 0).sum(), (yc_tr == 1).sum()
        spw = n_neg / n_pos if n_pos > 0 else 1.0

        clf = make_classifier(config, spw)
        clf.fit(X_tr, yc_tr)
        yc_pred = clf.predict(X_val)
        yc_prob = clf.predict_proba(X_val)[:, 1]

        reg = make_regressor(config)
        flooded_tr = yc_tr == 1
        if flooded_tr.sum() > 0:
            reg.fit(X_tr[flooded_tr], np.log1p(yr_tr[flooded_tr]))
        yr_pred = np.zeros(len(X_val))
        if flooded_tr.sum() > 0:
            yr_pred = np.expm1(reg.predict(X_val))
            yr_pred = np.clip(yr_pred, a_min=0.0, a_max=None)

        oof_rows_clf = [
            (evaluation_id_clf[fold_id], int(run_ids[val_idx][i]), int(node_pks[val_idx][i]),
             "inunda", float(yc_val[i]), float(yc_pred[i]), float(yc_prob[i]), fold_id)
            for i in range(len(val_idx))
        ]
        conn.executemany(
            """
            INSERT INTO oof_predictions (
                evaluation_id, run_id, node_pk, target, observed, predicted, probability, fold_id
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            oof_rows_clf,
        )

        oof_rows_reg = [
            (evaluation_id_reg[fold_id], int(run_ids[val_idx][i]), int(node_pks[val_idx][i]),
             "vol_inundacion_m3", float(y_reg_full[val_idx][i]), float(yr_pred[i]), None, fold_id)
            for i in range(len(val_idx))
        ]
        conn.executemany(
            """
            INSERT INTO oof_predictions (
                evaluation_id, run_id, node_pk, target, observed, predicted, probability, fold_id
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            oof_rows_reg,
        )

        for evaluation_id in (evaluation_id_clf[fold_id], evaluation_id_reg[fold_id]):
            conn.execute(
                "UPDATE model_evaluations SET status = 'COMPLETE' WHERE evaluation_id = ?",
                (evaluation_id,),
            )

        pooled_yc_true.append(yc_val)
        pooled_yc_pred.append(yc_pred)
        flooded_val = yc_val == 1
        if flooded_val.sum() > 0:
            pooled_yr_true.append(y_reg_full[val_idx][flooded_val])
            pooled_yr_pred.append(yr_pred[flooded_val])

    conn.commit()

    pooled_f1 = float(f1_score(
        np.concatenate(pooled_yc_true), np.concatenate(pooled_yc_pred), zero_division=0
    ))
    pooled_rmse = (
        float(np.sqrt(mean_squared_error(np.concatenate(pooled_yr_true), np.concatenate(pooled_yr_pred))))
        if pooled_yr_true else float("nan")
    )

    # --- final artifacts: fit on the full dataset, same as the CLI joblib path ---
    X_all = TABULAR_V3_17.validate_frame(dataset.loc[:, FEATURE_COLUMNS_V17])
    y_clf_all = dataset["inunda"]
    n_neg, n_pos = (y_clf_all == 0).sum(), (y_clf_all == 1).sum()
    spw_all = n_neg / n_pos if n_pos > 0 else 1.0

    clf_final = make_classifier(config, spw_all)
    clf_final.fit(X_all, y_clf_all)
    reg_final = make_regressor(config)
    flooded = dataset["inunda"] == 1
    reg_final.fit(dataset.loc[flooded, FEATURE_COLUMNS_V17], np.log1p(dataset.loc[flooded, "vol_inundacion_m3"]))

    for target, model, metric_name, metric_value in (
        ("inunda", clf_final, "f1", pooled_f1),
        ("vol_inundacion_m3", reg_final, "rmse", pooled_rmse),
    ):
        buf = io.BytesIO()
        joblib.dump(model, buf)
        model_blob = buf.getvalue()
        conn.execute(
            """
            INSERT INTO trained_models (
                training_run_id, target, algorithm, hyperparameters_json, preprocessing_json,
                feature_contract_id, feature_contract_sha256, ordered_features_json,
                target_transform_json, query_params_json, included_run_ids_json,
                random_seed, grouping_strategy, python_version, library_versions_json,
                model_sha256, model_blob, created_at_utc
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                training_run_id, target, "xgboost", "{}", "{}",
                "tabular_v3_17", TABULAR_V3_17.descriptor_sha256,
                json.dumps(list(FEATURE_COLUMNS_V17)),
                json.dumps({"log1p": target == "vol_inundacion_m3"}), "{}",
                json.dumps(all_run_ids), ML_RANDOM_STATE, "group_kfold",
                sys.version.split()[0], "{}",
                _sha256_bytes(model_blob), model_blob, _now(),
            ),
        )
        model_id = conn.execute("SELECT MAX(model_id) FROM trained_models").fetchone()[0]
        is_valid = not (isinstance(metric_value, float) and np.isnan(metric_value))
        conn.execute(
            """
            INSERT INTO model_metrics (model_id, scope, metric_name, value, valid, reason)
            VALUES (?, 'pooled_oof_group_kfold5', ?, ?, ?, ?)
            """,
            (model_id, metric_name, metric_value if is_valid else None,
             1 if is_valid else 0, None if is_valid else "no flooded validation rows"),
        )

    conn.execute(
        "UPDATE training_runs SET status = 'COMPLETE', completed_at_utc = ? WHERE training_run_id = ?",
        (_now(), training_run_id),
    )
    conn.commit()
    return training_run_id
