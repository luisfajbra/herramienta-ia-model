import matplotlib.pyplot as plt
from pathlib import Path

from ..simulation.swmm_api_io import load_inp, get_node_inflow_profiles


def plot_hydrograph(inp_path: Path, output_path: Path) -> Path:
    """Plot the design-storm hydrograph for the node with the highest peak inflow.

    Reads all node inflow profiles from the .inp, selects the node whose
    timeseries has the maximum single-point flow value, and saves a PNG.
    """
    inp = load_inp(inp_path)
    profiles = get_node_inflow_profiles(inp)

    peak_node, peak_profile = max(
        profiles.items(),
        key=lambda kv: max((q for _, q in kv[1]["points"]), default=0.0),
    )

    times = [t for t, _ in peak_profile["points"]]
    flows = [q for _, q in peak_profile["points"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(times, flows, color="#2176ae", linewidth=2)
    ax.fill_between(times, flows, alpha=0.15, color="#2176ae")
    ax.set_xlabel("Tiempo (min)")
    ax.set_ylabel("Caudal (L/s)")
    ax.set_title(f"Hidrograma de entrada — Nodo {peak_node} (Qx1)")
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Hidrograma guardado: {output_path}")
    return output_path
