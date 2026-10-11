"""Display-only labels shared by chart generators."""

from __future__ import annotations

import re


IN_SAMPLE_NOTE = (
    "In-sample: the model was trained on these scenarios; this is not an independent validation."
)
IN_SAMPLE_BANNER = (
    "[EN MUESTRA] Estos escenarios (formas y factores) se usaron para entrenar el modelo; "
    "las métricas no son una validación independiente."
)


def add_in_sample_note(fig) -> None:
    """Pie de figura que avisa que el modelo ya vio estos escenarios."""
    fig.text(
        0.5, 0.005, IN_SAMPLE_NOTE,
        ha="center", va="bottom", fontsize=8, color="#7a7a7a", style="italic",
    )


def print_in_sample_banner() -> None:
    """Imprime el aviso de que las métricas siguientes son en muestra."""
    print(f"\n{IN_SAMPLE_BANNER}")


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
