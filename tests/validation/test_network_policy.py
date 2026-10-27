from __future__ import annotations

from copy import deepcopy

from swmm_resilience.validation.network_policy import validate_network_profile


PROFILE = {
    "units": {"flow_units": "LPS", "length_units_from_flow_units": "m"},
    "unparsed_sections": [],
    "nodes": [
        {"node_id": "J1", "node_type": "junction"},
        {"node_id": "S1", "node_type": "storage"},
        {"node_id": "OUT1", "node_type": "outfall"},
    ],
    "link_type_counts": {"CONDUITS": 2, "PUMPS": 0},
    "hydrology": {"has_rainfall_model": False},
    "topology": {
        "parallel_link_groups": [],
    },
    "outfalls": [{"node_id": "OUT1", "boundary_type": "FIXED"}],
    "structures": {"STORAGE": ["S1"], "PUMPS": [], "CONTROLS": []},
    "link_offset_mode": "DEPTH",
    "links": [{"link_id": "C1", "link_type": "conduit"}],
    "inflows": {
        "constituent_counts": {"FLOW": 1},
        "records": [
            {
                "node_id": "J1",
                "constituent": "FLOW",
                "time_series": "H1",
                "baseline_defaulted_before_pattern": 0.0,
            }
        ]
    },
    "time_series": {
        "records": [{"name": "H1", "final_value": 0.0}],
    },
    "cross_sections": {"count": 2, "shape_counts": {"CIRCULAR": 2}},
    "coordinates": {"nodes_without_coordinates": []},
    "unresolved_references": [],
}


POLICY = {
    "policy_version": "network_domain_v1",
    "allowed_flow_units": ["LPS"],
    "allowed_length_units": ["m"],
    "supported_node_types": ["junction", "storage", "outfall"],
    "supported_link_types": ["conduit"],
    "supported_cross_section_shapes": ["CIRCULAR"],
    "supported_inflow_constituents": ["FLOW"],
    "allow_parallel_links": False,
    "supported_outfall_boundary_types": ["FIXED"],
    "supported_structure_sections": ["STORAGE"],
    "supported_additional_flow_input_sections": [],
    "allowed_offset_modes": ["DEPTH"],
    "allow_persistent_inflows": False,
    "allow_negative_flow_samples": False,
    "allow_non_finite_flow_samples": False,
    "allow_inflow_patterns": False,
    "allow_rainfall_model": False,
    "allow_missing_geometry": False,
    "feature_contract_version": "tabular_v4_21",
}


def test_valid_explicit_policy_has_no_issues():
    assert validate_network_profile(PROFILE, POLICY) == []


def test_missing_policy_values_are_reported():
    incomplete = {key: value for key, value in POLICY.items() if key != "allow_parallel_links"}

    issues = validate_network_profile(PROFILE, incomplete)

    assert issues == [
        {
            "code": "missing_policy_value",
            "path": "allow_parallel_links",
            "observed": None,
            "permitted": "explicit boolean value",
            "object_id": None,
        }
    ]


def test_unsupported_observed_properties_are_reported_by_stable_code():
    profile = deepcopy(PROFILE)
    profile["units"]["flow_units"] = "CFS"
    profile["unparsed_sections"] = ["TRANSECTS"]
    profile["units"]["length_units_from_flow_units"] = "ft"
    profile["nodes"].append({"node_id": "D1", "node_type": "divider"})
    profile["link_type_counts"]["WEIRS"] = 1
    profile["links"].append({"link_id": "W1", "link_type": "weir"})
    profile["cross_sections"]["shape_counts"]["IRREGULAR"] = 1
    profile["inflows"]["constituent_counts"]["POLLUTANT"] = 1
    profile["hydrology"]["has_rainfall_model"] = True
    profile["topology"]["parallel_link_groups"] = [
        {"from_node": "J1", "to_node": "OUT1", "link_ids": ["C1", "C2"]}
    ]
    profile["outfalls"][0]["boundary_type"] = "FREE"
    profile["structures"]["PUMPS"] = ["P1"]
    profile["link_offset_mode"] = "ELEVATION"
    profile["time_series"]["records"][0]["final_value"] = 1.0
    profile["inflows"]["records"][0]["pattern"] = "WEEKLY"
    profile["coordinates"]["nodes_without_coordinates"] = ["D1"]
    profile["unresolved_references"] = [
        {
            "code": "missing_cross_section",
            "object_id": "W1",
            "field": "cross_section",
            "reference": "W1",
        }
    ]

    issues = validate_network_profile(profile, POLICY)

    assert {issue["code"] for issue in issues} == {
        "unsupported_flow_units",
        "effective_flow_units_unsupported",
        "unparsed_input_section",
        "unsupported_length_units",
        "unsupported_node_type",
        "unsupported_link_type",
        "unsupported_cross_section_shape",
        "unsupported_inflow_constituent",
        "rainfall_model_not_supported",
        "parallel_links_not_supported",
        "unsupported_outfall_boundary_type",
        "unsupported_structure_section",
        "unsupported_offset_mode",
        "persistent_inflows_not_supported",
        "inflow_patterns_not_supported",
        "missing_geometry_not_supported",
    }


def test_feature_contract_must_be_known():
    policy = {**POLICY, "feature_contract_version": "unknown_contract"}

    issues = validate_network_profile(PROFILE, policy)

    assert issues == [
        {
            "code": "unknown_feature_contract",
            "path": "feature_contract_version",
            "observed": "unknown_contract",
            "permitted": "registered feature contract",
            "object_id": None,
        }
    ]


def test_dwf_is_rejected_when_additional_flow_inputs_are_not_supported():
    profile = deepcopy(PROFILE)
    profile["additional_flow_input_sections"] = {"DWF": ["J1"]}

    issues = validate_network_profile(profile, POLICY)

    assert any(issue["code"] == "unsupported_additional_flow_input_section" for issue in issues)


def test_dwf_is_rejected_when_persistent_inflows_are_disallowed_even_if_section_is_supported():
    profile = deepcopy(PROFILE)
    profile["additional_flow_input_sections"] = {"DWF": ["J1:FLOW"]}
    policy = {**POLICY, "supported_additional_flow_input_sections": ["DWF"]}

    issues = validate_network_profile(profile, policy)

    persistent_issue = next(
        issue for issue in issues if issue["code"] == "persistent_inflows_not_supported"
    )
    assert persistent_issue["object_id"] == "DWF:J1:FLOW"


def test_nonzero_baseline_with_pattern_is_persistent_even_when_patterns_are_allowed():
    profile = deepcopy(PROFILE)
    profile["inflows"]["records"][0].update(
        baseline_defaulted_before_pattern=2.0, pattern="DAILY"
    )
    policy = {**POLICY, "allow_inflow_patterns": True}

    issues = validate_network_profile(profile, policy)

    assert any(issue["code"] == "persistent_inflows_not_supported" for issue in issues)


def test_every_policy_issue_has_an_object_id_field():
    incomplete = {key: value for key, value in POLICY.items() if key != "allow_parallel_links"}

    issues = validate_network_profile(PROFILE, incomplete)

    assert "object_id" in issues[0]


def test_null_and_object_specific_issue_ids_have_a_stable_sort_order():
    profile = deepcopy(PROFILE)
    profile["unresolved_references"] = [
        {"code": "missing_pump_curve", "object_id": "P1", "reference": "CURVE_1"},
        {"code": "missing_pump_curve", "object_id": None, "reference": "CURVE_2"},
    ]

    issues = validate_network_profile(profile, POLICY)

    assert [issue["object_id"] for issue in issues] == [None, "P1"]


def test_negative_and_non_finite_samples_in_referenced_flow_series_are_rejected():
    profile = deepcopy(PROFILE)
    profile["time_series"]["records"][0].update(
        minimum_value=-1.0,
        maximum_value=1.0,
        negative_sample_count=2,
        non_finite_sample_count=1,
    )

    issues = validate_network_profile(profile, POLICY)

    assert {issue["code"] for issue in issues} >= {
        "negative_flow_samples_not_supported",
        "non_finite_flow_samples_not_supported",
    }


def test_pattern_is_rejected_even_if_policy_claims_pattern_support():
    profile = deepcopy(PROFILE)
    profile["inflows"]["records"][0]["pattern"] = "DAILY"
    policy = {**POLICY, "allow_inflow_patterns": True}

    issues = validate_network_profile(profile, policy)

    assert any(issue["code"] == "flow_pattern_evaluator_unavailable" for issue in issues)


def test_negative_effective_flow_from_scale_or_baseline_is_rejected():
    profile = deepcopy(PROFILE)
    profile["inflows"]["records"][0].update(
        baseline_defaulted_before_pattern=0.0, scale_factor=-1.0
    )
    profile["time_series"]["records"][0].update(
        minimum_value=1.0, maximum_value=1.0, negative_sample_count=0
    )

    issues = validate_network_profile(profile, POLICY)

    assert any(issue["code"] == "negative_flow_samples_not_supported" for issue in issues)


def test_policy_cannot_admit_units_the_effective_flow_evaluator_does_not_support():
    profile = deepcopy(PROFILE)
    profile["units"] = {"flow_units": "CFS", "length_units_from_flow_units": "ft"}
    policy = {**POLICY, "allowed_flow_units": ["CFS"], "allowed_length_units": ["ft"]}

    issues = validate_network_profile(profile, policy)

    assert any(issue["code"] == "effective_flow_units_unsupported" for issue in issues)


def test_non_flow_inflows_do_not_require_effective_flow_units():
    profile = deepcopy(PROFILE)
    profile["units"] = {"flow_units": "CFS", "length_units_from_flow_units": "ft"}
    profile["inflows"]["records"][0]["constituent"] = "POLLUTANT"
    policy = {
        **POLICY,
        "allowed_flow_units": ["CFS"],
        "allowed_length_units": ["ft"],
        "supported_inflow_constituents": ["FLOW", "POLLUTANT"],
    }

    issues = validate_network_profile(profile, policy)

    assert not any(issue["code"] == "effective_flow_units_unsupported" for issue in issues)


def test_non_finite_flow_samples_cannot_be_admitted_by_policy_override():
    profile = deepcopy(PROFILE)
    profile["time_series"]["records"][0].update(
        minimum_value=None, maximum_value=None, non_finite_sample_count=1
    )
    policy = {**POLICY, "allow_non_finite_flow_samples": True}

    issues = validate_network_profile(profile, policy)

    assert any(issue["code"] == "non_finite_flow_evaluator_unavailable" for issue in issues)
