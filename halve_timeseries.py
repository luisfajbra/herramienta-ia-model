"""Divide por 2 los valores de [TIMESERIES] de un .inp de SWMM.

Uso (explícito, nunca al importar):
    python halve_timeseries.py --src <entrada.inp> --dst <salida.inp> [--force]

Se usó una vez para derivar la red Qx1 a partir de Qx2. El destino NO se
sobrescribe sin ``--force``: una versión anterior de este script escribía sobre
el .inp base de producción al importarse (hallazgo V-07 de la auditoría).
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
        raise ValueError("El origen y el destino son el mismo archivo.")
    if dst.exists() and not force:
        raise FileExistsError(f"{dst} ya existe; usa force=True (--force) para sobrescribirlo.")

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
    print(f"Listo: {count} valores divididos. Escrito en: {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
