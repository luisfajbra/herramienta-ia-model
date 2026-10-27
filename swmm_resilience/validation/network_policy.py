"""Explicit, fail-closed hydraulic-domain checks for an INP profile."""

from __future__ import annotations

from typing import Any

from ..ml.contracts import FeatureContractError, resolve_feature_contract


_PERSISTENT_FLOW_SECTIONS = {"DWF", "RDII", "GROUNDWATER"}


_REQUIRED_POLICY = {
    "policy_version": "non-empty string",
    "allowed_flow_units": "list of strings",
    "allowed_length_units": "list of strings",
    "supported_node_types": "list of strings",
    "supported_link_types": "list of strings",
    "supported_cross_section_shapes": "list of strings",
    "supported_inflow_constituents": "list of strings",
    "allow_parallel_links": "explicit boolean value",
    "supported_outfall_boundary_types": "list of strings",
    "supported_structure_sections": "list of strings",
    "supported_additional_flow_input_sections": "list of strings",
    "allowed_offset_modes": "list of strings",
    "allow_persistent_inflows": "explicit boolean value",
    "allow_negative_flow_samples": "explicit boolean value",
    "allow_non_finite_flow_samples": "explicit boolean value",
    "allow_inflow_patterns": "explicit boolean value",
    "allow_rainfall_model": "explicit boolean value",
    "allow_missing_geometry": "explicit boolean value",
    "feature_contract_version": "registered feature contract",
}


def _issue(code: str, path: str, observed: Any, permitted: Any, object_id: str | None = None) -> dict[str, Any]:
    result = {
        "code": code,
        "path": path,
        "observed": observed,
        "permitted": permitted,
        "object_id": object_id,
    }
    return result


def _string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _policy_shape_issues(policy: dict[str, Any]) -> list[dict[str, Any]]:
    issues = []
    for key, expected in _REQUIRED_POLICY.items():
        if key not in policy:
            issues.append(_issue("missing_policy_value", key, None, expected))
            continue
        value = policy[key]
        valid = (
            isinstance(value, str) and bool(value.strip())
            if key == "policy_version"
            else isinstance(value, bool)
            if key in {
                "allow_parallel_links",
                "allow_persistent_inflows",
                "allow_negative_flow_samples",
                "allow_non_finite_flow_samples",
                "allow_inflow_patterns",
                "allow_rainfall_model",
                "allow_missing_geometry",
            }
            else _string_list(value)
            if key in {
                "allowed_flow_units",
                "allowed_length_units",
                "supported_node_types",
                "supported_link_types",
                "supported_cross_section_shapes",
                "supported_inflow_constituents",
                "supported_outfall_boundary_types",
                "supported_structure_sections",
                "supported_additional_flow_input_sections",
                "allowed_offset_modes",
            }
            else isinstance(value, str) and bool(value.strip())
        )
        if not valid:
            issues.append(_issue("invalid_policy_value", key, value, expected))
    return issues


def _allowed(values: list[str], observed: Any) -> bool:
    return observed is not None and str(observed).casefold() in {
        value.casefold() for value in values
    }


def validate_network_profile(
    profile: dict[str, Any], policy: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return stable issues when an explicit hydraulic policy is missing or mismatched.

    This applies declared policy only. It does not declare the network hydraulically
    valid or decide engineering values such as flood and continuity thresholds.
    """
    if not isinstance(policy, dict):
        return [_issue("missing_policy", "policy", policy, "explicit policy object")]

    issues = _policy_shape_issues(policy)
    if issues:
        return sorted(issues, key=lambda item: (item["code"], item["path"]))

    for section in profile.get("unparsed_sections", []):
        issues.append(
            _issue(
                "unparsed_input_section",
                "unparsed_sections",
                section,
                "every present section parsed by swmm-api",
                str(section),
            )
        )

    supported_additional_inputs = {
        value.upper() for value in policy["supported_additional_flow_input_sections"]
    }
    for section, object_ids in profile.get("additional_flow_input_sections", {}).items():
        if object_ids and section.upper() not in supported_additional_inputs:
            issues.append(
                _issue(
                    "unsupported_additional_flow_input_section",
                    f"additional_flow_input_sections.{section}",
                    object_ids,
                    policy["supported_additional_flow_input_sections"],
                    ",".join(object_ids),
                )
            )

    try:
        resolve_feature_contract(policy["feature_contract_version"])
    except FeatureContractError:
        issues.append(
            _issue(
                "unknown_feature_contract",
                "feature_contract_version",
                policy["feature_contract_version"],
                "registered feature contract",
            )
        )

    flow_units = profile.get("units", {}).get("flow_units")
    if not _allowed(policy["allowed_flow_units"], flow_units):
        issues.append(_issue("unsupported_flow_units", "units.flow_units", flow_units, policy["allowed_flow_units"]))
    has_flow_inflows = any(
        str(inflow.get("constituent", inflow.get("kind", ""))).casefold()
        == "flow"
        for inflow in profile.get("inflows", {}).get("records", [])
    )
    if has_flow_inflows and str(flow_units or "").upper() != "LPS":
        issues.append(
            _issue(
                "effective_flow_units_unsupported",
                "units.flow_units",
                flow_units,
                "LPS for the implemented effective-flow evaluator",
            )
        )
    length_units = profile.get("units", {}).get("length_units_from_flow_units")
    if not _allowed(policy["allowed_length_units"], length_units):
        issues.append(
            _issue(
                "unsupported_length_units",
                "units.length_units_from_flow_units",
                length_units,
                policy["allowed_length_units"],
            )
        )

    supported_nodes = {value.casefold() for value in policy["supported_node_types"]}
    for node in profile.get("nodes", []):
        node_type = str(node.get("node_type", "")).casefold()
        if node_type not in supported_nodes:
            issues.append(
                _issue(
                    "unsupported_node_type",
                    "nodes.node_type",
                    node.get("node_type"),
                    policy["supported_node_types"],
                    str(node.get("node_id", "")),
                )
            )

    supported_links = {value.casefold() for value in policy["supported_link_types"]}
    for link in profile.get("links", []):
        link_type = str(link.get("link_type", "")).casefold()
        if link_type not in supported_links:
            issues.append(
                _issue(
                    "unsupported_link_type",
                    "links.link_type",
                    link.get("link_type"),
                    policy["supported_link_types"],
                    str(link.get("link_id", "")),
                )
            )

    supported_shapes = {value.casefold() for value in policy["supported_cross_section_shapes"]}
    for shape, count in profile.get("cross_sections", {}).get("shape_counts", {}).items():
        if count and shape.casefold() not in supported_shapes:
            issues.append(
                _issue(
                    "unsupported_cross_section_shape",
                    "cross_sections.shape_counts",
                    {shape: count},
                    policy["supported_cross_section_shapes"],
                )
            )

    supported_constituents = {
        value.casefold() for value in policy["supported_inflow_constituents"]
    }
    for constituent, count in profile.get("inflows", {}).get("constituent_counts", {}).items():
        if count and constituent.casefold() not in supported_constituents:
            issues.append(
                _issue(
                    "unsupported_inflow_constituent",
                    "inflows.constituent_counts",
                    {constituent: count},
                    policy["supported_inflow_constituents"],
                )
            )

    if profile.get("hydrology", {}).get("has_rainfall_model") and not policy["allow_rainfall_model"]:
        issues.append(
            _issue(
                "rainfall_model_not_supported",
                "hydrology.has_rainfall_model",
                True,
                False,
            )
        )

    if profile.get("topology", {}).get("parallel_link_groups") and not policy["allow_parallel_links"]:
        for group in profile["topology"]["parallel_link_groups"]:
            issues.append(
                _issue(
                    "parallel_links_not_supported",
                    "topology.parallel_link_groups",
                    group.get("link_ids"),
                    "no parallel link groups",
                    ",".join(group.get("link_ids", [])),
                )
            )

    supported_boundaries = policy["supported_outfall_boundary_types"]
    for outfall in profile.get("outfalls", []):
        boundary_type = outfall.get("boundary_type")
        if not _allowed(supported_boundaries, boundary_type):
            issues.append(
                _issue(
                    "unsupported_outfall_boundary_type",
                    "outfalls.boundary_type",
                    boundary_type,
                    supported_boundaries,
                    str(outfall.get("node_id", "")),
                )
            )

    supported_structures = {value.upper() for value in policy["supported_structure_sections"]}
    for section, object_ids in profile.get("structures", {}).items():
        if object_ids and section.upper() not in supported_structures:
            issues.append(
                _issue(
                    "unsupported_structure_section",
                    f"structures.{section}",
                    object_ids,
                    policy["supported_structure_sections"],
                    ",".join(object_ids),
                )
            )

    offset_mode = profile.get("link_offset_mode")
    if not _allowed(policy["allowed_offset_modes"], offset_mode):
        issues.append(_issue("unsupported_offset_mode", "link_offset_mode", offset_mode, policy["allowed_offset_modes"]))

    series_by_name = {
        str(record.get("name", "")).casefold(): record
        for record in profile.get("time_series", {}).get("records", [])
    }
    flow_nodes_by_series: dict[str, list[str]] = {}
    for inflow in profile.get("inflows", {}).get("records", []):
        series_name = str(inflow.get("time_series", "")).casefold()
        if series_name:
            flow_nodes_by_series.setdefault(series_name, []).append(
                str(inflow.get("node_id", ""))
            )
    inflow_records = profile.get("inflows", {}).get("records", [])
    for inflow in inflow_records:
        series_name = str(inflow.get("time_series", "")).casefold()
        series = series_by_name.get(series_name)
        node_id = str(inflow.get("node_id", ""))
        if series and series.get("non_finite_sample_count", 0):
            policy_code = (
                "non_finite_flow_evaluator_unavailable"
                if policy["allow_non_finite_flow_samples"]
                else "non_finite_flow_samples_not_supported"
            )
            issues.append(
                _issue(
                    policy_code,
                    f"time_series.{series.get('name')}.non_finite_sample_count",
                    series["non_finite_sample_count"],
                    "finite samples for the implemented effective-flow evaluator",
                    node_id,
                )
            )
        if not policy["allow_negative_flow_samples"]:
            baseline = inflow.get("baseline_defaulted_before_pattern")
            baseline = 0.0 if baseline is None else float(baseline)
            scale = inflow.get("scale_factor")
            scale = 1.0 if scale is None else float(scale)
            effective_values = [baseline]
            if series:
                minimum = series.get("minimum_value")
                maximum = series.get("maximum_value")
                if minimum is not None:
                    effective_values.append(baseline + scale * float(minimum))
                if maximum is not None:
                    effective_values.append(baseline + scale * float(maximum))
            if min(effective_values) < 0:
                issues.append(
                    _issue(
                        "negative_flow_samples_not_supported",
                        f"inflows.records.{node_id}",
                        {"minimum_effective_flow_lps": min(effective_values)},
                        "non-negative effective FLOW values",
                        node_id,
                    )
                )
    persistent = []
    for inflow in profile.get("inflows", {}).get("records", []):
        series = series_by_name.get(str(inflow.get("time_series", "")).casefold())
        end_flow = series.get("final_value") if series else None
        baseline = inflow.get("baseline_defaulted_before_pattern")
        has_nonzero_baseline = baseline not in (None, 0)
        if (end_flow is not None and end_flow != 0) or has_nonzero_baseline:
            persistent.append(str(inflow.get("node_id", "")))
    additional_flow_inputs = profile.get("additional_flow_input_sections", {})
    for section, object_ids in additional_flow_inputs.items():
        if section.upper() in _PERSISTENT_FLOW_SECTIONS:
            persistent.extend(f"{section}:{object_id}" for object_id in object_ids)
    if persistent and not policy["allow_persistent_inflows"]:
        issues.append(
            _issue(
                "persistent_inflows_not_supported",
                "inflows.records and additional_flow_input_sections",
                sorted(set(persistent)),
                "no nonzero end flow, baseline, or active DWF/RDII/GROUNDWATER input",
                ",".join(sorted(set(persistent))),
            )
        )

    patterned_inflows = [
        str(inflow.get("node_id", ""))
        for inflow in profile.get("inflows", {}).get("records", [])
        if inflow.get("pattern") is not None
    ]
    if patterned_inflows and not policy["allow_inflow_patterns"]:
        issues.append(
            _issue(
                "inflow_patterns_not_supported",
                "inflows.records.pattern",
                sorted(set(patterned_inflows)),
                "no inflow patterns",
            )
        )
    elif patterned_inflows:
        issues.append(
            _issue(
                "flow_pattern_evaluator_unavailable",
                "inflows.records.pattern",
                sorted(set(patterned_inflows)),
                "an implemented and verified effective-flow pattern evaluator",
                ",".join(sorted(set(patterned_inflows))),
            )
        )

    if not policy["allow_missing_geometry"]:
        missing_geometry = [
            item
            for item in profile.get("unresolved_references", [])
            if item.get("code") in {"missing_cross_section", "orphan_cross_section"}
        ]
        missing_coordinates = profile.get("coordinates", {}).get("nodes_without_coordinates", [])
        if missing_geometry or missing_coordinates:
            issues.append(
                _issue(
                    "missing_geometry_not_supported",
                    "geometry",
                    {
                        "unresolved_cross_sections": missing_geometry,
                        "nodes_without_coordinates": missing_coordinates,
                    },
                    "complete link geometry and node coordinates",
                )
            )

    for reference in profile.get("unresolved_references", []):
        if reference.get("code") not in {"missing_cross_section", "orphan_cross_section"}:
            issues.append(
                _issue(
                    "unresolved_reference",
                    f"unresolved_references.{reference.get('code')}",
                    reference.get("reference"),
                    "resolved reference",
                    reference.get("object_id"),
                )
            )

    return sorted(
        issues,
        key=lambda item: (
            item["code"],
            item["path"],
            item.get("object_id") or "",
        ),
    )
