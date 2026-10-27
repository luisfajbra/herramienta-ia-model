"""Read-only structural inventory for a SWMM INP file.

The profile records facts needed for engineering review. It does not decide
whether the network belongs to the supported hydraulic domain.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date, datetime, time
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import networkx as nx

from ..simulation.swmm_api_io import load_inp


_NODE_SECTIONS = ("JUNCTIONS", "OUTFALLS", "STORAGE", "DIVIDERS")
_LINK_SECTIONS = ("CONDUITS", "PUMPS", "ORIFICES", "WEIRS", "OUTLETS")
_NODE_TYPE_NAMES = {
    "JUNCTIONS": "junction",
    "OUTFALLS": "outfall",
    "STORAGE": "storage",
    "DIVIDERS": "divider",
}


def _section(inp, name: str):
    try:
        value = inp[name]
    except KeyError:
        value = {}
    return value if callable(getattr(value, "items", None)) else {}


def _section_count(inp, name: str) -> int:
    try:
        value = inp[name]
    except KeyError:
        return 0
    return len(value) if callable(getattr(value, "items", None)) else 1


def _json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _json_value(value.value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (ValueError, TypeError):
            pass
    return str(value)


def _read(obj: object, attribute: str, default: Any = None) -> Any:
    return _json_value(getattr(obj, attribute, default))


def _numeric_if_possible(value: Any) -> Any:
    if isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return value
        return number if math.isfinite(number) else None
    return _json_value(value)


def _reference_name(value: Any) -> str | None:
    normalized = _numeric_if_possible(value)
    if normalized is None:
        return None
    text = str(normalized).strip()
    return None if not text or text.casefold() == "nan" else text


def _mean_step(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    differences = [b[0] - a[0] for a, b in zip(points, points[1:])]
    if all(math.isclose(value, differences[0], rel_tol=1e-9, abs_tol=1e-12) for value in differences):
        return differences[0]
    return None


def _timeseries_profile(inp) -> dict[str, Any]:
    records = []
    section = _section(inp, "TIMESERIES")
    for name, series in sorted(section.items(), key=lambda item: str(item[0])):
        points = list(getattr(series, "data", ()) or ())
        times = [float(time_value) for time_value, _ in points]
        values = [float(value) for _, value in points]
        finite_points = [
            (time_value, value)
            for time_value, value in zip(times, values)
            if math.isfinite(time_value) and math.isfinite(value)
        ]
        finite_times = [time_value for time_value, _ in finite_points]
        finite_values = [value for _, value in finite_points]
        records.append(
            {
                "name": str(name),
                "point_count": len(points),
                "start_hour": min(finite_times) if finite_times else None,
                "end_hour": max(finite_times) if finite_times else None,
                "minimum_value": min(finite_values) if finite_values else None,
                "maximum_value": max(finite_values) if finite_values else None,
                "initial_value": _json_value(values[0]) if values else None,
                "final_value": _json_value(values[-1]) if values else None,
                "nonzero_sample_count": sum(value != 0 for value in finite_values),
                "negative_sample_count": sum(value < 0 for value in finite_values),
                "non_finite_sample_count": len(points) - len(finite_points),
                "uniform_step_hours": _mean_step(
                    finite_points
                ),
            }
        )
    return {"count": len(section), "records": records}


def _link_records(inp) -> list[dict[str, Any]]:
    records = []
    for section_name in _LINK_SECTIONS:
        for link_id, link in sorted(
            _section(inp, section_name).items(), key=lambda item: str(item[0])
        ):
            from_node = getattr(link, "from_node", None)
            to_node = getattr(link, "to_node", None)
            if from_node is None or to_node is None:
                continue
            records.append(
                {
                    "link_id": str(link_id),
                    "link_type": section_name[:-1].lower(),
                    "from_node": str(from_node),
                    "to_node": str(to_node),
                    "length": _read(link, "length"),
                    "roughness": _read(link, "roughness"),
                    "offset_upstream": _read(link, "offset_upstream"),
                    "offset_downstream": _read(link, "offset_downstream"),
                    "initial_flow": _read(link, "flow_initial"),
                    "flow_max": _read(link, "flow_max"),
                }
            )
    return records


def _initial_values(section: dict, attribute: str) -> dict[str, Any]:
    records = [
        {"object_id": str(object_id), "value": _read(obj, attribute)}
        for object_id, obj in sorted(section.items(), key=lambda item: str(item[0]))
    ]
    values = [record["value"] for record in records if record["value"] is not None]
    return {
        "count": len(records),
        "nonzero_count": sum(value != 0 for value in values),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "records": records,
    }


def _initial_conditions_profile(inp) -> dict[str, Any]:
    """Inventory the state declared by the source INP and hash its vectors."""
    categories = {
        "junction_depths": _initial_values(_section(inp, "JUNCTIONS"), "depth_init"),
        "storage_depths": _initial_values(_section(inp, "STORAGE"), "depth_init"),
        "conduit_flows": _initial_values(_section(inp, "CONDUITS"), "flow_initial"),
    }
    encoded = json.dumps(
        categories, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "policy_id": "as_declared_in_source_inp",
        "state_sha256": hashlib.sha256(encoded).hexdigest(),
        **categories,
    }


def _outfall_records(inp) -> list[dict[str, Any]]:
    section = _section(inp, "OUTFALLS")
    return [
        {
            "node_id": str(node_id),
            "elevation": _read(outfall, "elevation"),
            "boundary_type": _read(outfall, "kind"),
            "stage_data": _numeric_if_possible(getattr(outfall, "data", None)),
            "has_flap_gate": _read(outfall, "has_flap_gate"),
            "route_to": _read(outfall, "route_to"),
        }
        for node_id, outfall in sorted(section.items(), key=lambda item: str(item[0]))
    ]


def _inflow_records(inp) -> list[dict[str, Any]]:
    records = []
    for key, inflow in sorted(
        _section(inp, "INFLOWS").items(),
        key=lambda item: (str(item[0][0]), str(item[0][1])),
    ):
        node_id, constituent = key
        records.append(
            {
                "node_id": str(node_id),
                "constituent": str(constituent),
                "time_series": _numeric_if_possible(
                    getattr(inflow, "time_series", None)
                ),
                "kind": _read(inflow, "kind"),
                "mass_unit_factor": _read(inflow, "mass_unit_factor"),
                "scale_factor": _read(inflow, "scale_factor"),
                "baseline": _numeric_if_possible(getattr(inflow, "base_value", None)),
                "baseline_defaulted_before_pattern": _numeric_if_possible(
                    getattr(inflow, "base_value", None)
                ) or 0.0,
                "pattern": _numeric_if_possible(getattr(inflow, "pattern", None)),
            }
        )
    return records


def _unresolved_references(
    inp, node_ids: set[str], link_records: list[dict[str, Any]]
) -> list[dict[str, str]]:
    unresolved = []
    link_id_by_normalized = {
        record["link_id"].casefold(): record["link_id"] for record in link_records
    }
    link_ids = set(link_id_by_normalized)
    normalized_node_ids = {node_id.casefold() for node_id in node_ids}

    for record in link_records:
        for field in ("from_node", "to_node"):
            reference = record[field]
            if reference.casefold() not in normalized_node_ids:
                unresolved.append(
                    {
                        "code": "missing_link_endpoint",
                        "object_id": record["link_id"],
                        "field": field,
                        "reference": reference,
                    }
                )

    for (node_id, constituent), inflow in sorted(
        _section(inp, "INFLOWS").items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        inflow_id = f"{node_id}:{constituent}"
        if str(node_id).casefold() not in normalized_node_ids:
            unresolved.append(
                {
                    "code": "missing_inflow_node",
                    "object_id": inflow_id,
                    "field": "node_id",
                    "reference": str(node_id),
                }
            )
        series_name = _reference_name(getattr(inflow, "time_series", None))
        if series_name is not None and series_name.casefold() not in {
            str(name).casefold() for name in _section(inp, "TIMESERIES")
        }:
            unresolved.append(
                {
                    "code": "missing_inflow_timeseries",
                    "object_id": inflow_id,
                    "field": "time_series",
                    "reference": series_name,
                }
            )

    node_lookup = {node_id.casefold() for node_id in node_ids}
    subcatchments = _section(inp, "SUBCATCHMENTS")
    subcatchment_lookup = {str(name).casefold() for name in subcatchments}
    raingages = _section(inp, "RAINGAGES")
    raingage_lookup = {str(name).casefold() for name in raingages}
    timeseries_lookup = {
        str(name).casefold() for name in _section(inp, "TIMESERIES")
    }
    for raingage_id, raingage in sorted(raingages.items(), key=lambda item: str(item[0])):
        if str(getattr(raingage, "source", "")).upper() == "TIMESERIES":
            series_name = _reference_name(getattr(raingage, "timeseries", None))
            if series_name is not None and series_name.casefold() not in timeseries_lookup:
                unresolved.append(
                    {
                        "code": "missing_raingage_timeseries",
                        "object_id": str(raingage_id),
                        "field": "timeseries",
                        "reference": series_name,
                    }
                )
    for subcatchment_id, subcatchment in sorted(
        subcatchments.items(), key=lambda item: str(item[0])
    ):
        rain_gage = _reference_name(getattr(subcatchment, "rain_gage", None))
        if rain_gage is not None and rain_gage.casefold() not in raingage_lookup:
            unresolved.append(
                {
                    "code": "missing_subcatchment_raingage",
                    "object_id": str(subcatchment_id),
                    "field": "rain_gage",
                    "reference": rain_gage,
                }
            )
        outlet = _reference_name(getattr(subcatchment, "outlet", None))
        if outlet is not None and outlet.casefold() not in node_lookup | subcatchment_lookup:
            unresolved.append(
                {
                    "code": "missing_subcatchment_outlet",
                    "object_id": str(subcatchment_id),
                    "field": "outlet",
                    "reference": outlet,
                }
            )

    curves = _section(inp, "CURVES")
    for storage_id, storage in sorted(_section(inp, "STORAGE").items(), key=lambda item: str(item[0])):
        curve_name = _reference_name(getattr(storage, "curve_name", None))
        if curve_name is not None and curve_name.casefold() not in {
            str(name).casefold() for name in curves
        }:
            unresolved.append(
                {
                    "code": "missing_storage_curve",
                    "object_id": str(storage_id),
                    "field": "curve_name",
                    "reference": curve_name,
                }
            )

    for node_id, outfall in sorted(_section(inp, "OUTFALLS").items(), key=lambda item: str(item[0])):
        route_to = _reference_name(getattr(outfall, "route_to", None))
        if route_to is not None and route_to.casefold() not in subcatchment_lookup:
            unresolved.append(
                {
                    "code": "missing_outfall_route_subcatchment",
                    "object_id": str(node_id),
                    "field": "route_to",
                    "reference": route_to,
                }
            )
        boundary_type = str(_read(outfall, "kind") or "").upper()
        stage_name = _reference_name(getattr(outfall, "data", None))
        if boundary_type == "TIDAL" and stage_name is not None and stage_name.casefold() not in {
            str(name).casefold() for name in curves
        }:
            unresolved.append(
                {
                    "code": "missing_outfall_tidal_curve",
                    "object_id": str(node_id),
                    "field": "stage_data",
                    "reference": stage_name,
                }
            )
        elif boundary_type == "TIMESERIES" and stage_name is not None and stage_name.casefold() not in timeseries_lookup:
            unresolved.append(
                {
                    "code": "missing_outfall_stage_timeseries",
                    "object_id": str(node_id),
                    "field": "stage_data",
                    "reference": stage_name,
                }
            )

    cross_sections = _section(inp, "XSECTIONS")
    cross_section_id_by_normalized = {
        str(value).casefold(): str(value) for value in cross_sections
    }
    normalized_cross_section_ids = set(cross_section_id_by_normalized)
    conduit_ids = {
        record["link_id"].casefold()
        for record in link_records
        if record["link_type"] == "conduit"
    }
    for normalized_id in sorted(link_ids - normalized_cross_section_ids):
        if normalized_id in conduit_ids:
            link_id = link_id_by_normalized[normalized_id]
            unresolved.append(
                {
                    "code": "missing_cross_section",
                    "object_id": link_id,
                    "field": "cross_section",
                    "reference": link_id,
                }
            )
    for normalized_id in sorted(normalized_cross_section_ids - link_ids):
        link_id = cross_section_id_by_normalized[normalized_id]
        unresolved.append(
            {
                "code": "orphan_cross_section",
                "object_id": link_id,
                "field": "link_id",
                "reference": link_id,
            }
        )

    curve_ids = {str(name).casefold() for name in _section(inp, "CURVES")}
    transect_ids = {str(name).casefold() for name in _section(inp, "TRANSECTS")}
    street_ids = {str(name).casefold() for name in _section(inp, "STREETS")}
    for link_id, cross_section in sorted(cross_sections.items(), key=lambda item: str(item[0])):
        shape = str(getattr(cross_section, "shape", "")).upper()
        reference_fields = {
            "CUSTOM": ("curve_name", "missing_custom_shape_curve", curve_ids),
            "IRREGULAR": ("transect", "missing_irregular_transect", transect_ids),
            "STREET": ("street", "missing_street_geometry", street_ids),
        }
        if shape not in reference_fields:
            continue
        field, code, known_ids = reference_fields[shape]
        reference = _reference_name(getattr(cross_section, field, None))
        if reference is not None and reference.casefold() not in known_ids:
            unresolved.append(
                {
                    "code": code,
                    "object_id": str(link_id),
                    "field": "geometry_reference",
                    "reference": reference,
                }
            )

    for pump_id, pump in sorted(_section(inp, "PUMPS").items(), key=lambda item: str(item[0])):
        curve_name = _reference_name(getattr(pump, "pump_curve", None))
        if curve_name is not None and curve_name.casefold() not in {
            str(name).casefold() for name in curves
        }:
            unresolved.append(
                {
                    "code": "missing_pump_curve",
                    "object_id": str(pump_id),
                    "field": "pump_curve",
                    "reference": curve_name,
                }
            )
    patterns = _section(inp, "PATTERNS")
    for (node_id, constituent), inflow in sorted(
        _section(inp, "INFLOWS").items(), key=lambda item: (str(item[0][0]), str(item[0][1]))
    ):
        pattern_name = _reference_name(getattr(inflow, "pattern", None))
        if pattern_name is not None and pattern_name.casefold() not in {
            str(name).casefold() for name in patterns
        }:
            unresolved.append(
                {
                    "code": "missing_inflow_pattern",
                    "object_id": f"{node_id}:{constituent}",
                    "field": "pattern",
                    "reference": pattern_name,
                }
            )
    return sorted(
        unresolved,
        key=lambda item: (item["code"], item["object_id"], item["field"], item["reference"]),
    )


def inspect_network_inp(inp_path: Path | str) -> dict[str, Any]:
    """Return a deterministic, JSON-safe structural profile without writing files."""
    path = Path(inp_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"SWMM INP file does not exist: {path}")
    raw = path.read_bytes()
    inp = load_inp(path)

    section_names = sorted(str(section).upper() for section in inp.keys())
    sections = {section_name: _section_count(inp, section_name) for section_name in section_names}
    unparsed_sections = []
    for section_name in section_names:
        try:
            value = inp[section_name]
        except KeyError:
            continue
        if isinstance(value, str) and section_name != "TITLE":
            unparsed_sections.append(section_name)
    options = {
        str(key).upper(): _json_value(value)
        for key, value in sorted(
            _section(inp, "OPTIONS").items(), key=lambda item: str(item[0]).upper()
        )
    }
    flow_units = str(options.get("FLOW_UNITS") or "").upper() or None
    length_units = (
        "m" if flow_units in {"CMS", "LPS", "MLD"}
        else "ft" if flow_units in {"CFS", "GPM", "MGD"}
        else None
    )

    node_records = []
    node_ids: set[str] = set()
    for section_name in _NODE_SECTIONS:
        for node_id, node in sorted(
            _section(inp, section_name).items(), key=lambda item: str(item[0])
        ):
            normalized_id = str(node_id)
            node_ids.add(normalized_id)
            node_records.append(
                {
                    "node_id": normalized_id,
                    "node_type": _NODE_TYPE_NAMES[section_name],
                    "elevation": _read(node, "elevation"),
                }
            )
    node_records.sort(key=lambda item: (item["node_type"], item["node_id"]))

    link_records = _link_records(inp)
    link_graph = nx.MultiDiGraph()
    link_graph.add_nodes_from(node_ids)
    endpoint_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for record in link_records:
        link_graph.add_edge(
            record["from_node"], record["to_node"], key=record["link_id"]
        )
        endpoint_groups[(record["from_node"], record["to_node"])].append(
            record["link_id"]
        )
    parallel_groups = [
        {
            "from_node": from_node,
            "to_node": to_node,
            "link_ids": sorted(link_ids),
        }
        for (from_node, to_node), link_ids in sorted(endpoint_groups.items())
        if len(link_ids) > 1
    ]

    cross_sections = _section(inp, "XSECTIONS")
    shape_counts = Counter(
        str(getattr(section, "shape", "UNKNOWN")).upper()
        for section in cross_sections.values()
    )
    barrel_counts = Counter(
        str(_read(section, "n_barrels"))
        for section in cross_sections.values()
    )
    time_series = _timeseries_profile(inp)
    rain_gages = _section(inp, "RAINGAGES")
    subcatchments = _section(inp, "SUBCATCHMENTS")
    coordinates = _section(inp, "COORDINATES")

    def item_ids(section_name: str) -> list[str]:
        return sorted(str(value) for value in _section(inp, section_name))

    node_type_counts = {
        section_name: _section_count(inp, section_name)
        for section_name in _NODE_SECTIONS
    }
    link_type_counts = {
        section_name: _section_count(inp, section_name)
        for section_name in _LINK_SECTIONS
    }
    additional_flow_sections = ("DWF", "RDII", "GROUNDWATER", "SNOWPACKS")
    additional_flow_input_sections = {}
    for section_name in additional_flow_sections:
        section = _section(inp, section_name)
        if section:
            object_ids = []
            for key, item in section.items():
                if isinstance(key, tuple):
                    object_ids.append(":".join(str(value) for value in key))
                else:
                    object_ids.append(str(key))
            additional_flow_input_sections[section_name] = sorted(object_ids)
    try:
        parser_version = version("swmm-api")
    except PackageNotFoundError:
        parser_version = None
    try:
        from swmm.toolkit.solver import swmm_get_version, swmm_version_info

        engine_runtime_version = str(swmm_version_info())
        engine_runtime_build = swmm_get_version()
    except (ImportError, OSError):
        engine_runtime_version = None
        engine_runtime_build = None

    return {
        "schema_version": "network_admission_profile_v1",
        "source": {
            "path": str(path),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw),
        },
        "swmm": {
            "engine_version_declared": options.get("SWMM_VERSION"),
            "engine_runtime_version": engine_runtime_version,
            "engine_runtime_build": engine_runtime_build,
            "parser": "swmm-api",
            "parser_version": parser_version,
        },
        "units": {"flow_units": flow_units, "length_units_from_flow_units": length_units},
        "options": options,
        "sections": sections,
        "unparsed_sections": unparsed_sections,
        "node_type_counts": node_type_counts,
        "link_type_counts": link_type_counts,
        "additional_flow_input_sections": additional_flow_input_sections,
        "hydrology": {
            "rain_gage_count": len(rain_gages),
            "subcatchment_count": len(subcatchments),
            "has_rainfall_model": bool(rain_gages or subcatchments),
        },
        "link_offset_mode": options.get("LINK_OFFSETS"),
        "nodes": node_records,
        "links": link_records,
        "outfalls": _outfall_records(inp),
        "structures": {
            section_name: item_ids(section_name)
            for section_name in ("STORAGE", "DIVIDERS", "PUMPS", "ORIFICES", "WEIRS", "OUTLETS", "CONTROLS")
        },
        "topology": {
            "node_count": len(node_ids),
            "link_count": len(link_records),
            "weakly_connected_component_count": nx.number_weakly_connected_components(link_graph),
            "is_directed_acyclic": nx.is_directed_acyclic_graph(link_graph),
            "parallel_link_groups": parallel_groups,
        },
        "cross_sections": {
            "count": len(cross_sections),
            "shape_counts": dict(sorted(shape_counts.items())),
            "barrel_counts": dict(sorted(barrel_counts.items())),
        },
        "initial_conditions": _initial_conditions_profile(inp),
        "inflows": {
            "count": len(_section(inp, "INFLOWS")),
            "constituent_counts": dict(
                sorted(Counter(str(key[1]).upper() for key in _section(inp, "INFLOWS")).items())
            ),
            "records": _inflow_records(inp),
        },
        "time_series": time_series,
        "coordinates": {
            "count": len(coordinates),
            "nodes_without_coordinates": sorted(node_ids - {str(value) for value in coordinates}),
        },
        "unresolved_references": _unresolved_references(inp, node_ids, link_records),
        "reference_check_scope": [
            "link endpoints",
            "inflow node, time series and pattern",
            "conduit cross sections",
            "pump curves",
            "storage rating curves",
            "outfall routed nodes",
            "outfall tidal curves and stage time series",
            "subcatchment rain gages and outlets",
            "control and rule expressions are inventoried but not parsed for references",
        ],
    }
