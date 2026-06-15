import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from ..simulation.swmm_api_io import load_inp, get_node_inflow_profiles
from ..validation.hydrograph_csv import HydrographScenario
from .labels import format_node_label


def plot_hydrograph(inp_path: Path, output_path: Path) -> Path:
    """Plot the design-storm hydrograph for the node with the highest peak inflow.

    Reads all node inflow profiles from the .inp, selects the node whose
    timeseries has the maximum single-point flow value, and saves a PNG.
    """
    inp = load_inp(inp_path)
    profiles = get_node_inflow_profiles(inp)

    if not profiles:
        raise ValueError(
            f"No inflow profiles were found in {inp_path}. "
            "Check the [INFLOWS] section of the .inp file."
        )

    peak_node, peak_profile = max(
        profiles.items(),
        key=lambda kv: max((q for _, q in kv[1]["points"]), default=0.0),
    )
    times = [t for t, _ in peak_profile["points"]]
    flows = [q for _, q in peak_profile["points"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, flows, color="#2176ae", linewidth=2)
    ax.fill_between(times, flows, alpha=0.15, color="#2176ae")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Flow (L/s)")
    ax.set_title(f"Inflow Hydrograph - Node {format_node_label(peak_node)} (Qx1)")
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Hydrograph saved: {output_path}")
    return output_path


def plot_scenario_hydrograph(
    scenario: HydrographScenario,
    output_path: Path,
) -> Path:
    """Plot the input hydrograph for the scenario node with the highest peak."""
    if not scenario.node_series:
        raise ValueError(f"Scenario '{scenario.scenario_id}' has no node series.")

    peak_node, peak_series = max(
        scenario.node_series.items(),
        key=lambda item: max((flow for _, flow in item[1]), default=0.0),
    )
    times_min = [hours * 60.0 for hours, _ in peak_series]
    flows = [flow for _, flow in peak_series]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times_min, flows, color="#2176ae", linewidth=2)
    ax.fill_between(times_min, flows, alpha=0.15, color="#2176ae")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Flow (L/s)")
    ax.set_title(
        f"Inflow Hydrograph - Node {format_node_label(peak_node)} "
        f"({scenario.scenario_id})"
    )
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Hydrograph saved: {output_path}")
    return output_path
