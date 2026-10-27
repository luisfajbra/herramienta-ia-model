"""Scenario-level admission gate for datasets built directly from SWMM runs."""

from __future__ import annotations

import math

import pandas as pd

from swmm_resilience.database.hydraulic_quality import build_hydraulic_quality_decision


def _complete_finite_coverage(
    dynamic_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    expected_node_ids,
) -> bool:
    expected = [str(node_id) for node_id in expected_node_ids]
    if not expected or len(set(expected)) != len(expected):
        return False
    if "node_id" not in dynamic_df or "node_id" not in labels_df:
        return False
    dynamic_ids = dynamic_df["node_id"].astype(str).tolist()
    label_ids = labels_df["node_id"].astype(str).tolist()
    if (len(dynamic_ids) != len(expected) or len(label_ids) != len(expected)
            or len(set(dynamic_ids)) != len(expected) or len(set(label_ids)) != len(expected)
            or set(dynamic_ids) != set(expected) or set(label_ids) != set(expected)):
        return False
    required_labels = {"vol_inundacion_m3", "inunda"}
    if not required_labels.issubset(labels_df.columns):
        return False
    try:
        volumes = pd.to_numeric(labels_df["vol_inundacion_m3"], errors="coerce")
        labels = pd.to_numeric(labels_df["inunda"], errors="coerce")
        if not volumes.map(lambda value: math.isfinite(float(value)) and value >= 0).all():
            return False
        if not labels.map(lambda value: math.isfinite(float(value)) and value in (0, 1)).all():
            return False
        # All computed numeric inputs must be finite; identifiers and categorical
        # fields are excluded. This prevents NaN/Inf from entering either model.
        numeric = dynamic_df.select_dtypes(include="number")
        if not numeric.empty and not numeric.map(lambda value: math.isfinite(float(value))).all().all():
            return False
    except (TypeError, ValueError, OverflowError):
        return False
    return True


def assess_scenario_quality(
    dynamic_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    expected_node_ids,
    continuity_error_pct: float | None,
    max_continuity_error_pct: float,
    continuity_limit_rationale: str,
    drainage_manifest: dict | None,
    initial_condition_manifest: dict | None,
) -> dict:
    """Apply the shared hydraulic admission policy to one extracted scenario."""
    drainage_state = (
        drainage_manifest.get("state")
        if isinstance(drainage_manifest, dict)
        else "unverified"
    )
    if drainage_state not in {"drained", "not_drained", "unverified"}:
        drainage_state = "unverified"
    return build_hydraulic_quality_decision(
        continuity_error_pct=continuity_error_pct,
        max_continuity_error_pct=max_continuity_error_pct,
        continuity_limit_rationale=continuity_limit_rationale,
        finite_coverage_ok=_complete_finite_coverage(
            dynamic_df, labels_df, expected_node_ids
        ),
        drainage_state=drainage_state,
        initial_condition=(initial_condition_manifest
                           if isinstance(initial_condition_manifest, dict) else {}),
        flooding_report_manifest=labels_df.attrs.get("flooding_report_manifest"),
    )


def filter_simulation_results_by_quality(simulation_results, decisions):
    """Return eligible result tuples and an ordered list of excluded decisions.

    Any result without a matching decision fails closed and is included in the
    exclusion report rather than silently admitted.
    """
    def key(shape_id, factor):
        return str(shape_id), round(float(factor), 6)

    decision_map = {
        key(item.get("shape_id", "base"), item["factor_mult"]): item
        for item in decisions
        if "factor_mult" in item
    }
    kept = []
    excluded = []
    for result in simulation_results:
        factor, dynamic_df, _labels_df = result
        shape_id = (str(dynamic_df["shape_id"].iloc[0])
                    if "shape_id" in dynamic_df and not dynamic_df.empty else "base")
        scenario_key = key(shape_id, factor)
        decision = decision_map.get(scenario_key)
        if decision is None:
            excluded.append({
                "shape_id": shape_id,
                "factor_mult": float(factor),
                "eligible": False,
                "reason_codes": ["quality_assessment_missing"],
            })
        elif decision.get("eligible") is True:
            kept.append(result)
        else:
            excluded.append(decision)
    return kept, excluded
