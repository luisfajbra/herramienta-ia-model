"""Display-only labels shared by chart generators."""

from __future__ import annotations

import re


FEATURE_DISPLAY_NAMES = {
    "elev_fondo": "Invert Elevation",
    "prof_max": "Maximum Depth in Manhole shaft",
    "n_tuberias_in": "Inlet Pipe Count",
    "n_tuberias_out": "Outlet Pipe Count",
    "diam_max_in": "Maximum Inlet Diameter",
    "diam_max_out": "Maximum Outlet Diameter",
    "pendiente_max_in": "Maximum Inlet Slope",
    "pendiente_out": "Outlet Slope",
    "base_inflow_lps": "Base Inflow",
    "dist_outfall_m": "Distance to Outfall",
    "n_nodos_aguas_arriba": "Upstream Node Count",
    "q_pico_acum_base": "Base Accumulated Peak Flow",
    "upstream_capacity_lps": "Upstream Capacity",
    "factor_mult": "Flow Multiplier",
    "q_pico_nodo": "Node Peak Inflow",
    "q_pico_acum_escalado": "Scaled Accumulated Peak Flow",
    "duracion_horas": "Event Duration (h)",
    "tiempo_al_pico_h": "Time to Peak (h)",
}

_NUMERIC_WITH_ALPHA_SUFFIX = re.compile(r"^(\d+)[A-Z]+$", re.IGNORECASE)


def format_node_label(node_id: object) -> str:
    """Return a shorter display label without changing the underlying node ID."""
    label = str(node_id)
    match = _NUMERIC_WITH_ALPHA_SUFFIX.fullmatch(label)
    return match.group(1) if match else label


def feature_display_name(feature: object) -> str:
    """Return an explanatory English label for a model feature."""
    name = str(feature)
    return FEATURE_DISPLAY_NAMES.get(name, name.replace("_", " ").title())
