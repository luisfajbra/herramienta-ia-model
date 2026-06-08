from pathlib import Path

from .runner import run_simulation_simple as run_simulation
from ..config import Config


def run_batch(config: Config, run_dir: Path) -> list:
    """Run SWMM for every factor in config. Returns list of (factor, rpt_path)."""
    factors = config.factors()
    results = []
    for i, factor in enumerate(factors, 1):
        print(f"  [{i:>2}/{len(factors)}] factor={factor:.2f}", end=" ... ", flush=True)
        rpt_path = run_simulation(config.network.inp_path, factor, run_dir)
        print("OK")
        results.append((factor, rpt_path))
    return results
