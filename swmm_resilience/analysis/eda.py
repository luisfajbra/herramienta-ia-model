"""
Dataset review utilities for post-export quality checks.

This module covers the first three review steps:
1. Group columns by analytical role
2. Generate a base dataset quality summary
3. Compute feature correlations for model-oriented inspection

The follow-up plan for later iterations keeps ydata-profiling and PCA as
complementary tools to justify feature decisions after the initial findings.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any
import zipfile

import numpy as np
import pandas as pd

from ..config import (
    DEFAULT_DATASET_REVIEW_DIR,
    DEFAULT_OUTPUT_CSV,
    ML_DROP_COLUMNS,
    ML_TARGET_CLASSIFICATION,
    ML_TARGET_REGRESSION,
)
from ..ml.preprocessing import dataset_info, get_feature_columns


KNOWN_IDENTIFIER_COLUMNS = {"run_id", "node_id", "network_hash", "network_file"}
KNOWN_SCENARIO_COLUMNS = {"scenario_type", "spatial_pattern"}
KNOWN_INPUT_COLUMNS = {"inflow_multiplier"}
KNOWN_RESULT_COLUMNS = {
    "max_depth_m",
    "max_depth_ratio",
    "time_to_peak_min",
    "depth_rate_m_per_min",
    "flooding_duration_min",
}


def _json_default(value: Any):
    """Convert numpy and pandas values into JSON-safe Python objects."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Unsupported type for JSON serialization: {type(value)!r}")


def _write_json(data: dict[str, Any], output_path: Path):
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=True, default=_json_default),
        encoding="utf-8",
    )


def _write_text(content: str, output_path: Path):
    output_path.write_text(content, encoding="utf-8")


def _available_excel_engine() -> str | None:
    """Return the first available pandas Excel writer engine for .xlsx output."""
    if importlib.util.find_spec("openpyxl") is not None:
        return "openpyxl"
    if importlib.util.find_spec("xlsxwriter") is not None:
        return "xlsxwriter"
    return None


def _write_workbook(sheets: dict[str, pd.DataFrame], output_path: Path) -> dict[str, str]:
    """Write tabular artifacts into a single file, preferring Excel when available."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    engine = _available_excel_engine()
    if engine is not None:
        with pd.ExcelWriter(output_path, engine=engine) as writer:
            for sheet_name, dataframe in sheets.items():
                safe_sheet_name = sheet_name[:31]
                dataframe.to_excel(writer, sheet_name=safe_sheet_name, index=False)
        return {"format": "xlsx", "path": str(output_path), "engine": engine}

    zip_path = output_path.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for sheet_name, dataframe in sheets.items():
            csv_content = dataframe.to_csv(index=False)
            archive.writestr(f"{sheet_name}.csv", csv_content)
    return {"format": "zip_csv_bundle", "path": str(zip_path), "engine": "none"}


def _components_needed(cumulative_variance: np.ndarray, threshold: float) -> int:
    indices = np.where(cumulative_variance >= threshold)[0]
    if len(indices) == 0:
        return int(len(cumulative_variance))
    return int(indices[0] + 1)


def group_dataset_columns(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Classify dataset columns by their analytical role."""
    ordered_columns = df.columns.tolist()

    feature_candidates = get_feature_columns(df, target=ML_TARGET_REGRESSION)
    static_topology_features = [
        column
        for column in feature_candidates
        if column not in KNOWN_INPUT_COLUMNS and column not in KNOWN_IDENTIFIER_COLUMNS
    ]

    groups: dict[str, dict[str, Any]] = {
        "identifiers": {
            "description": "Technical identifiers used for joins and tracing rows.",
            "columns": [column for column in ordered_columns if column in KNOWN_IDENTIFIER_COLUMNS],
        },
        "scenario_metadata": {
            "description": "Labels that describe how each simulation scenario was generated.",
            "columns": [column for column in ordered_columns if column in KNOWN_SCENARIO_COLUMNS],
        },
        "dynamic_input_features": {
            "description": "Input values that can change from one scenario to another.",
            "columns": [column for column in ordered_columns if column in KNOWN_INPUT_COLUMNS],
        },
        "static_topology_features": {
            "description": "Candidate static predictors derived from network topology.",
            "columns": [column for column in ordered_columns if column in static_topology_features],
        },
        "targets": {
            "description": "Primary ML targets currently used by regression and classification.",
            "columns": [
                column
                for column in ordered_columns
                if column in {ML_TARGET_REGRESSION, ML_TARGET_CLASSIFICATION}
            ],
        },
        "derived_result_columns": {
            "description": "Hydraulic outputs that are useful for diagnostics but should not be model inputs.",
            "columns": [column for column in ordered_columns if column in KNOWN_RESULT_COLUMNS],
        },
        "excluded_from_modeling": {
            "description": "Columns excluded by the current ML preprocessing rules.",
            "columns": [column for column in ordered_columns if column in ML_DROP_COLUMNS],
        },
        "model_feature_candidates": {
            "description": "Numeric columns that remain available for model training after preprocessing.",
            "columns": feature_candidates,
        },
    }

    assigned_columns = {
        column for group in groups.values() for column in group["columns"]
    }
    groups["unclassified_columns"] = {
        "description": "Columns not matched by the current review rules.",
        "columns": [column for column in ordered_columns if column not in assigned_columns],
    }

    return groups


def build_dataset_summary(
    df: pd.DataFrame,
) -> tuple[
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build the base quality summary and supporting tables."""
    info = dataset_info(df)
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_columns = [column for column in df.columns if column not in numeric_columns]

    summary: dict[str, Any] = {
        "total_rows": int(len(df)),
        "total_columns": int(len(df.columns)),
        "numeric_columns": int(len(numeric_columns)),
        "non_numeric_columns": int(len(non_numeric_columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_total": int(info["missing_total"]),
        "missing_columns": int((df.isna().sum() > 0).sum()),
    }

    if "run_id" in df.columns:
        summary["run_count"] = int(df["run_id"].nunique(dropna=True))
    if "node_id" in df.columns:
        summary["node_count"] = int(df["node_id"].nunique(dropna=True))
    if "network_hash" in df.columns:
        summary["network_count"] = int(df["network_hash"].nunique(dropna=True))
    if "scenario_type" in df.columns:
        summary["scenario_type_count"] = int(df["scenario_type"].nunique(dropna=True))
    if {"run_id", "node_id"}.issubset(df.columns):
        summary["duplicate_run_node_pairs"] = int(df.duplicated(subset=["run_id", "node_id"]).sum())

    rows_per_run = pd.DataFrame(columns=["run_id", "rows"])
    if "run_id" in df.columns:
        rows_per_run = (
            df.groupby("run_id", dropna=False)
            .size()
            .rename("rows")
            .reset_index()
            .sort_values(["rows", "run_id"], ascending=[False, True])
        )
        summary["rows_per_run_min"] = int(rows_per_run["rows"].min())
        summary["rows_per_run_max"] = int(rows_per_run["rows"].max())
        summary["rows_per_run_mean"] = float(rows_per_run["rows"].mean())

    rows_per_node = pd.DataFrame(columns=["node_id", "rows"])
    if "node_id" in df.columns:
        rows_per_node = (
            df.groupby("node_id", dropna=False)
            .size()
            .rename("rows")
            .reset_index()
            .sort_values(["rows", "node_id"], ascending=[False, True])
        )
        summary["rows_per_node_min"] = int(rows_per_node["rows"].min())
        summary["rows_per_node_max"] = int(rows_per_node["rows"].max())
        summary["rows_per_node_mean"] = float(rows_per_node["rows"].mean())

    rows_per_scenario = pd.DataFrame(columns=["scenario_type", "rows", "run_count", "node_count"])
    if "scenario_type" in df.columns:
        rows_per_scenario = (
            df.groupby("scenario_type", dropna=False)
            .agg(
                rows=("scenario_type", "size"),
                run_count=("run_id", pd.Series.nunique) if "run_id" in df.columns else ("scenario_type", "size"),
                node_count=("node_id", pd.Series.nunique) if "node_id" in df.columns else ("scenario_type", "size"),
            )
            .reset_index()
            .sort_values(["rows", "scenario_type"], ascending=[False, True])
        )
        summary["scenario_type_rows"] = {
            str(row["scenario_type"]): int(row["rows"])
            for row in rows_per_scenario.to_dict(orient="records")
        }

    rows_per_network = pd.DataFrame(columns=["network_hash", "rows", "run_count", "node_count"])
    if "network_hash" in df.columns:
        rows_per_network = (
            df.groupby("network_hash", dropna=False)
            .agg(
                rows=("network_hash", "size"),
                run_count=("run_id", pd.Series.nunique) if "run_id" in df.columns else ("network_hash", "size"),
                node_count=("node_id", pd.Series.nunique) if "node_id" in df.columns else ("network_hash", "size"),
            )
            .reset_index()
            .sort_values(["rows", "network_hash"], ascending=[False, True])
        )
        summary["network_rows"] = {
            str(row["network_hash"]): int(row["rows"])
            for row in rows_per_network.to_dict(orient="records")
        }

    missing_df = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().astype(int).values,
            "missing_pct": (df.isna().mean() * 100.0).round(4).values,
        }
    ).sort_values(["missing_count", "column"], ascending=[False, True])

    dtype_df = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": df.dtypes.astype(str).values,
            "non_null_count": df.notna().sum().astype(int).values,
            "unique_count": [int(df[column].nunique(dropna=True)) for column in df.columns],
        }
    ).sort_values("column")

    target_rows: list[dict[str, Any]] = []
    for target in [ML_TARGET_CLASSIFICATION, ML_TARGET_REGRESSION]:
        if target not in df.columns:
            continue

        series = df[target]
        row: dict[str, Any] = {
            "target": target,
            "non_null_count": int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series):
            numeric_series = series.dropna().astype(float)
            if not numeric_series.empty:
                row.update(
                    {
                        "mean": float(numeric_series.mean()),
                        "std": float(numeric_series.std(ddof=1)) if len(numeric_series) > 1 else 0.0,
                        "min": float(numeric_series.min()),
                        "q25": float(numeric_series.quantile(0.25)),
                        "median": float(numeric_series.median()),
                        "q75": float(numeric_series.quantile(0.75)),
                        "max": float(numeric_series.max()),
                    }
                )
                if target == ML_TARGET_CLASSIFICATION:
                    row["positive_rate"] = float(numeric_series.mean())
        target_rows.append(row)

    target_summary_df = pd.DataFrame(target_rows)
    return (
        summary,
        missing_df,
        dtype_df,
        target_summary_df,
        rows_per_run,
        rows_per_node,
        rows_per_scenario,
        rows_per_network,
    )


def build_correlation_artifacts(
    df: pd.DataFrame,
    threshold: float = 0.9,
    method: str = "pearson",
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute model-oriented feature correlations and target associations."""
    feature_columns = get_feature_columns(df, target=ML_TARGET_REGRESSION)
    feature_df = df[feature_columns].copy()

    constant_columns = [
        column for column in feature_columns if feature_df[column].nunique(dropna=True) <= 1
    ]
    usable_columns = [column for column in feature_columns if column not in constant_columns]

    if usable_columns:
        correlation_matrix = feature_df[usable_columns].corr(method=method).round(6)
    else:
        correlation_matrix = pd.DataFrame(index=pd.Index([], name="feature"))

    high_corr_rows: list[dict[str, Any]] = []
    for left_index, left_column in enumerate(usable_columns):
        for right_column in usable_columns[left_index + 1 :]:
            value = correlation_matrix.loc[left_column, right_column]
            if pd.notna(value) and abs(float(value)) >= threshold:
                high_corr_rows.append(
                    {
                        "feature_a": left_column,
                        "feature_b": right_column,
                        "correlation": float(value),
                        "abs_correlation": abs(float(value)),
                    }
                )

    high_corr_df = pd.DataFrame(
        high_corr_rows,
        columns=["feature_a", "feature_b", "correlation", "abs_correlation"],
    )
    if not high_corr_df.empty:
        high_corr_df = high_corr_df.sort_values(
            ["abs_correlation", "feature_a", "feature_b"],
            ascending=[False, True, True],
        )

    target_corr_rows: list[dict[str, Any]] = []
    for target in [ML_TARGET_CLASSIFICATION, ML_TARGET_REGRESSION]:
        if target not in df.columns:
            continue
        target_series = pd.to_numeric(df[target], errors="coerce")
        if target_series.notna().sum() == 0:
            continue
        correlations = feature_df[usable_columns].corrwith(target_series, method=method)
        for feature_name, value in correlations.dropna().items():
            target_corr_rows.append(
                {
                    "target": target,
                    "feature": feature_name,
                    "correlation": float(value),
                    "abs_correlation": abs(float(value)),
                }
            )

    target_corr_df = pd.DataFrame(
        target_corr_rows,
        columns=["target", "feature", "correlation", "abs_correlation"],
    )
    if not target_corr_df.empty:
        target_corr_df = target_corr_df.sort_values(
            ["target", "abs_correlation", "feature"],
            ascending=[True, False, True],
        )

    summary = {
        "correlation_method": method,
        "correlation_threshold": float(threshold),
        "candidate_feature_count": int(len(feature_columns)),
        "usable_feature_count": int(len(usable_columns)),
        "constant_feature_count": int(len(constant_columns)),
        "constant_features": constant_columns,
        "high_correlation_pair_count": int(len(high_corr_df)),
    }
    return summary, correlation_matrix, high_corr_df, target_corr_df


def build_follow_up_plan() -> dict[str, Any]:
    """Track what is done now and what remains for later iterations."""
    return {
        "completed_steps": [
            {
                "step": "Clasificar columnas del dataset por rol analitico.",
                "status": "completed",
            },
            {
                "step": "Generar un chequeo base de calidad del dataset.",
                "status": "completed",
            },
            {
                "step": "Calcular correlacion entre features candidatas y targets.",
                "status": "completed",
            },
            {
                "step": "Agregar PCA diagnostico para interpretar redundancia y estructura del dataset.",
                "status": "completed",
            },
            {
                "step": "Preparar la generacion de ydata-profiling como auditoria visual complementaria.",
                "status": "completed",
            },
        ],
        "pending_steps": [
            {
                "step": "Revisar leakage de forma explicita y ajustar ML_DROP_COLUMNS si hace falta.",
                "status": "pending",
            },
            {
                "step": "Traducir hallazgos a una recomendacion formal de keep/drop para el pipeline de entrenamiento.",
                "status": "pending",
            },
        ],
    }


def build_pca_artifacts(
    df: pd.DataFrame,
    projection_components: int = 5,
    top_loadings_per_component: int = 5,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute PCA diagnostics on the candidate feature space."""
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    feature_columns = get_feature_columns(df, target=ML_TARGET_REGRESSION)
    numeric_features = df[feature_columns].apply(pd.to_numeric, errors="coerce")

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    imputed = imputer.fit_transform(numeric_features)
    scaled = scaler.fit_transform(imputed)

    pca = PCA()
    transformed = pca.fit_transform(scaled)

    component_names = [f"PC{i}" for i in range(1, len(feature_columns) + 1)]
    explained_variance_df = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance_ratio": np.cumsum(pca.explained_variance_ratio_),
            "eigenvalue": pca.explained_variance_,
        }
    )

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=feature_columns,
        columns=component_names,
    ).reset_index(names="feature")

    top_loading_rows: list[dict[str, Any]] = []
    for component_name in component_names:
        ordered = loadings_df[["feature", component_name]].copy()
        ordered["abs_loading"] = ordered[component_name].abs()
        ordered = ordered.sort_values(["abs_loading", "feature"], ascending=[False, True]).head(
            top_loadings_per_component
        )
        for _, row in ordered.iterrows():
            top_loading_rows.append(
                {
                    "component": component_name,
                    "feature": row["feature"],
                    "loading": float(row[component_name]),
                    "abs_loading": float(row["abs_loading"]),
                }
            )
    top_loadings_df = pd.DataFrame(top_loading_rows)

    projection_count = min(projection_components, len(component_names))
    projection_df = pd.DataFrame(
        transformed[:, :projection_count],
        columns=component_names[:projection_count],
    )
    for column in [
        "run_id",
        "node_id",
        "inflow_multiplier",
        ML_TARGET_CLASSIFICATION,
        ML_TARGET_REGRESSION,
    ]:
        if column in df.columns:
            projection_df.insert(len(projection_df.columns), column, df[column].values)

    cumulative = explained_variance_df["cumulative_explained_variance_ratio"].to_numpy()
    summary = {
        "feature_count": int(len(feature_columns)),
        "sample_count": int(len(df)),
        "projection_components_saved": int(projection_count),
        "components_for_80pct_variance": _components_needed(cumulative, 0.80),
        "components_for_90pct_variance": _components_needed(cumulative, 0.90),
        "components_for_95pct_variance": _components_needed(cumulative, 0.95),
        "pc1_variance_ratio": float(explained_variance_df.iloc[0]["explained_variance_ratio"]),
        "pc2_cumulative_variance_ratio": float(
            explained_variance_df.iloc[min(1, len(explained_variance_df) - 1)][
                "cumulative_explained_variance_ratio"
            ]
        ),
        "pc3_cumulative_variance_ratio": float(
            explained_variance_df.iloc[min(2, len(explained_variance_df) - 1)][
                "cumulative_explained_variance_ratio"
            ]
        ),
    }
    return summary, explained_variance_df, loadings_df, top_loadings_df, projection_df


def build_pca_interpretation(summary: dict[str, Any], top_loadings_df: pd.DataFrame) -> str:
    """Generate a compact text interpretation for the PCA artifacts."""
    lines = [
        "PCA diagnostic summary",
        f"- Features analizadas: {summary['feature_count']}",
        f"- Muestras: {summary['sample_count']}",
        f"- Varianza explicada por PC1: {summary['pc1_variance_ratio']:.4f}",
        f"- Varianza acumulada PC1-PC2: {summary['pc2_cumulative_variance_ratio']:.4f}",
        f"- Varianza acumulada PC1-PC3: {summary['pc3_cumulative_variance_ratio']:.4f}",
        f"- Componentes para 80%: {summary['components_for_80pct_variance']}",
        f"- Componentes para 90%: {summary['components_for_90pct_variance']}",
        f"- Componentes para 95%: {summary['components_for_95pct_variance']}",
        "",
        "Top loadings por componente guardado:",
    ]

    for component in sorted(top_loadings_df["component"].unique()):
        component_rows = top_loadings_df[top_loadings_df["component"] == component]
        features_text = ", ".join(
            f"{row.feature} ({row.loading:+.3f})"
            for row in component_rows.itertuples(index=False)
        )
        lines.append(f"- {component}: {features_text}")

    return "\n".join(lines) + "\n"


def generate_ydata_profile(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
) -> dict[str, Any]:
    """Generate a ydata-profiling HTML report when the dependency is available."""
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        return {
            "status": "skipped_missing_dependency",
            "message": "No se pudo generar ydata-profiling porque la dependencia no esta instalada.",
            "expected_output": str(output_path),
        }

    try:
        profile_df = df.copy()
        # ydata-profiling triggers a scipy chi-square path that is incompatible
        # with the scipy build available in this environment. Disabling only
        # that metric keeps the HTML report useful without weakening the rest
        # of the dataset audit.
        profile_kwargs = {
            "title": title,
            "explorative": True,
            "progress_bar": False,
            "minimal": False,
            "vars": {
                "num": {"chi_squared_threshold": 0.0},
                "cat": {"chi_squared_threshold": 0.0},
            },
        }
        profile = ProfileReport(
            profile_df,
            **profile_kwargs,
        )
        profile.to_file(output_path)
        return {
            "status": "generated",
            "message": "Reporte ydata-profiling generado correctamente.",
            "output_file": str(output_path),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "message": f"Fallo la generacion del reporte ydata-profiling: {exc}",
            "expected_output": str(output_path),
        }


def build_model_input_profile_frame(
    df: pd.DataFrame,
    target: str = ML_TARGET_REGRESSION,
) -> tuple[pd.DataFrame, list[str]]:
    """Return the exact feature frame that would be used as model input."""
    feature_columns = get_feature_columns(df, target=target)
    if not feature_columns:
        raise ValueError("No se encontraron columnas de entrada para el profiling del modelo.")
    return df[feature_columns].copy(), feature_columns


def run_dataset_review(
    csv_path: Path | str = DEFAULT_OUTPUT_CSV,
    output_dir: Path | str | None = None,
    correlation_threshold: float = 0.9,
    correlation_method: str = "pearson",
    generate_profile: bool = True,
    generate_pca: bool = True,
) -> dict[str, Any]:
    """Run the dataset review and write reproducible artifacts to disk."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontro el dataset: {csv_path}")

    review_dir = Path(output_dir) if output_dir is not None else DEFAULT_DATASET_REVIEW_DIR
    review_dir.mkdir(parents=True, exist_ok=True)
    pca_dir = review_dir / "pca"
    profile_dir = review_dir / "profiling"
    workbook_target_path = review_dir / "dataset_review_tables.xlsx"
    pca_dir.mkdir(parents=True, exist_ok=True)
    profile_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    column_groups = group_dataset_columns(df)
    (
        summary,
        missing_df,
        dtype_df,
        target_summary_df,
        rows_per_run,
        rows_per_node,
        rows_per_scenario,
        rows_per_network,
    ) = build_dataset_summary(df)
    corr_summary, correlation_matrix, high_corr_df, target_corr_df = build_correlation_artifacts(
        df,
        threshold=correlation_threshold,
        method=correlation_method,
    )
    plan = build_follow_up_plan()
    workbook_sheets: dict[str, pd.DataFrame] = {
        "missing_values": missing_df,
        "column_dtypes": dtype_df,
        "target_summary": target_summary_df,
        "rows_per_run": rows_per_run,
        "rows_per_node": rows_per_node,
        "rows_per_scenario": rows_per_scenario,
        "rows_per_network": rows_per_network,
        "correlation_matrix": correlation_matrix.reset_index(),
        "high_corr_pairs": high_corr_df,
        "target_correlations": target_corr_df,
    }

    pca_status: dict[str, Any] = {"status": "skipped"}
    if generate_pca:
        try:
            (
                pca_summary,
                explained_variance_df,
                loadings_df,
                top_loadings_df,
                projection_df,
            ) = build_pca_artifacts(df)
            _write_json(
                {
                    "dataset_csv": csv_path,
                    "review_dir": review_dir,
                    **pca_summary,
                },
                pca_dir / "pca_summary.json",
            )
            workbook_sheets["pca_explained_variance"] = explained_variance_df
            workbook_sheets["pca_loadings"] = loadings_df
            workbook_sheets["pca_top_loadings"] = top_loadings_df
            workbook_sheets["pca_projection"] = projection_df
            _write_text(
                build_pca_interpretation(pca_summary, top_loadings_df),
                pca_dir / "pca_interpretation.txt",
            )
            pca_status = {
                "status": "generated",
                "output_dir": str(pca_dir),
                "tables_workbook_target": str(workbook_target_path),
                **pca_summary,
            }
        except ImportError:
            pca_status = {
                "status": "skipped_missing_dependency",
                "message": "No se pudo ejecutar PCA porque scikit-learn no esta disponible en este interprete.",
                "expected_output_dir": str(pca_dir),
            }
        except Exception as exc:
            pca_status = {
                "status": "failed",
                "message": f"Fallo la generacion del PCA diagnostico: {exc}",
                "expected_output_dir": str(pca_dir),
            }

    profile_status: dict[str, Any] = {
        "full_dataset": {"status": "skipped"},
        "model_inputs": {"status": "skipped"},
    }
    if generate_profile:
        full_profile_status = generate_ydata_profile(
            df,
            profile_dir / "dataset_profile.html",
            title="SWMM Resilience - General Training Dataset Profile",
        )
        full_profile_status.update(
            {
                "profile_scope": "full_dataset",
                "profile_column_count": int(df.shape[1]),
                "profile_row_count": int(df.shape[0]),
            }
        )

        try:
            model_input_profile_df, model_input_columns = build_model_input_profile_frame(df)
            model_input_profile_status = generate_ydata_profile(
                model_input_profile_df,
                profile_dir / "dataset_profile_model_inputs.html",
                title="SWMM Resilience - Model Input Dataset Profile",
            )
            model_input_profile_status.update(
                {
                    "profile_scope": "model_inputs_after_drop_columns",
                    "profile_column_count": len(model_input_columns),
                    "profile_row_count": int(model_input_profile_df.shape[0]),
                    "profile_columns": model_input_columns,
                    "excluded_columns": [
                        column for column in df.columns.tolist() if column not in model_input_columns
                    ],
                }
            )
        except Exception as exc:
            model_input_profile_status = {
                "status": "failed",
                "message": f"Fallo la generacion del reporte ydata-profiling para inputs del modelo: {exc}",
                "expected_output": str(profile_dir / "dataset_profile_model_inputs.html"),
                "profile_scope": "model_inputs_after_drop_columns",
            }

        profile_status = {
            "full_dataset": full_profile_status,
            "model_inputs": model_input_profile_status,
        }
    _write_json(profile_status, profile_dir / "ydata_profile_status.json")

    _write_json(
        {
            "dataset_csv": csv_path,
            "review_dir": review_dir,
            "groups": column_groups,
        },
        review_dir / "column_groups.json",
    )
    _write_json(
        {
            "dataset_csv": csv_path,
            "review_dir": review_dir,
            **summary,
        },
        review_dir / "dataset_summary.json",
    )
    _write_json(
        {
            "dataset_csv": csv_path,
            "review_dir": review_dir,
            **corr_summary,
        },
        review_dir / "correlation_summary.json",
    )
    _write_json(
        {
            "dataset_csv": csv_path,
            "review_dir": review_dir,
            "pca": pca_status,
            "ydata_profiling": profile_status,
        },
        review_dir / "complementary_analysis_status.json",
    )
    _write_json(plan, review_dir / "analysis_plan.json")
    workbook_artifact = _write_workbook(workbook_sheets, workbook_target_path)

    print(f"\nDataset review generado en: {review_dir}")
    print(f"  - Resumen base         : {review_dir / 'dataset_summary.json'}")
    print(f"  - Grupos de columnas   : {review_dir / 'column_groups.json'}")
    print(f"  - Tablas consolidadas  : {workbook_artifact['path']} ({workbook_artifact['format']})")
    print(f"  - PCA diagnostico      : {pca_dir}")
    print(f"  - ydata-profiling full : {profile_status['full_dataset'].get('status')}")
    print(f"  - ydata-profiling X    : {profile_status['model_inputs'].get('status')}")
    print(f"  - Plan de seguimiento  : {review_dir / 'analysis_plan.json'}")

    return {
        "review_dir": review_dir,
        "workbook_artifact": workbook_artifact,
        "summary": summary,
        "correlation_summary": corr_summary,
        "pca_status": pca_status,
        "profile_status": profile_status,
        "plan": plan,
    }


def main():
    parser = argparse.ArgumentParser(description="Run post-export dataset review artifacts.")
    parser.add_argument("--csv-path", default=str(DEFAULT_OUTPUT_CSV), help="Path to dataset_ml.csv")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for review artifacts. Defaults to data/training/dataset_review",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.9,
        help="Absolute correlation threshold used to flag feature pairs.",
    )
    parser.add_argument(
        "--correlation-method",
        default="pearson",
        choices=["pearson", "spearman", "kendall"],
        help="Correlation method for feature diagnostics.",
    )
    parser.add_argument(
        "--skip-profile",
        action="store_true",
        help="Skip ydata-profiling generation.",
    )
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Skip PCA diagnostics.",
    )
    args = parser.parse_args()
    run_dataset_review(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        correlation_threshold=args.correlation_threshold,
        correlation_method=args.correlation_method,
        generate_profile=not args.skip_profile,
        generate_pca=not args.skip_pca,
    )


if __name__ == "__main__":
    main()
