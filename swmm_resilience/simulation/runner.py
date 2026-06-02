"""Runs a single SWMM simulation for a given factor. Returns .rpt path."""
from contextlib import suppress
from pathlib import Path

from pyswmm import Simulation

from .swmm_api_io import write_scaled_inp


def run_simulation(inp_path: Path, factor: float, run_dir: Path) -> Path:
    """Scale inflows by factor, run SWMM, return path to the generated .rpt.

    The original .inp is never modified. A temporary scaled copy is written
    to run_dir and deleted after SWMM finishes; the .rpt persists.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_inp = run_dir / f"factor_{factor:.4f}.inp"

    write_scaled_inp(str(inp_path), factor, None, str(tmp_inp), scenario_mode="timeseries")

    with Simulation(str(tmp_inp)) as sim:
        for _ in sim:
            pass

    rpt_path = tmp_inp.with_suffix(".rpt")

    with suppress(OSError):
        tmp_inp.unlink()

    if not rpt_path.exists():
        raise FileNotFoundError(f"SWMM no genero el archivo .rpt esperado: {rpt_path}")

    return rpt_path
