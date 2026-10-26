"""Divide the [TIMESERIES] values of a SWMM .inp by 2.

Usage (explicit, never on import):
    python halve_timeseries.py --src <input.inp> --dst <output.inp> [--force]

It was used once to derive the Qx1 network from Qx2. The destination is NOT
overwritten without ``--force``: a previous version of this script wrote over
the production base .inp on import (audit finding V-07).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def halve_timeseries(src, dst, force: bool = False) -> int:
    """Escribe en ``dst`` una copia de ``src`` con los valores de [TIMESERIES] a la mitad.

    Devuelve cuántos valores se dividieron.
    """
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("The source and destination are the same file.")
    if dst.exists() and not force:
        raise FileExistsError(f"{dst} already exists; use force=True (--force) to overwrite it.")

    in_timeseries = False
    out_lines: list[str] = []
    halved = 0
    with open(src, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.rstrip("\n")
            if re.match(r"^\[", stripped):
                in_timeseries = stripped.strip() == "[TIMESERIES]"
                out_lines.append(stripped + "\n")
                continue
            if in_timeseries and stripped and not stripped.startswith(";;"):
                parts = stripped.split()
                try:
                    value = float(parts[-1]) / 2
                    last_token = stripped.rsplit(None, 1)[-1]
                    prefix = stripped[: stripped.rfind(last_token)]
                    stripped = prefix + f"{value:.3f}"
                    halved += 1
                except ValueError:
                    pass  # la última columna no es un número: la línea queda igual
            out_lines.append(stripped + "\n")

    with open(dst, "w", encoding="utf-8") as handle:
        handle.writelines(out_lines)
    return halved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--dst", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    count = halve_timeseries(args.src, args.dst, force=args.force)
    print(f"Done: {count} values divided. Written to: {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
