import re

SRC = r"data\networks\chico_hydro-qx2\SWMM - Chico (PVC) Prueba 5 - Qx2.00.inp"
DST = r"data\networks\chico_hydro-qx1\SWMM - Chico (PVC) Prueba 1 - Qx1.00.inp"

in_timeseries = False
out_lines = []

with open(SRC, "r", encoding="utf-8") as f:
    for line in f:
        stripped = line.rstrip("\n")

        # Detect section headers
        if re.match(r"^\[", stripped):
            in_timeseries = stripped.strip() == "[TIMESERIES]"
            out_lines.append(stripped + "\n")
            continue

        # Inside TIMESERIES: halve the value on data lines
        if in_timeseries and stripped and not stripped.startswith(";;"):
            parts = stripped.split()
            try:
                parts[-1] = f"{float(parts[-1]) / 2:.3f}"
                # Rebuild preserving original whitespace layout:
                # Name(col0) + optional Date + Time + Value
                # Simplest: rejoin with original spacing by replacing last token
                last_token = stripped.rsplit(None, 1)[-1]
                prefix = stripped[: stripped.rfind(last_token)]
                stripped = prefix + parts[-1]
            except ValueError:
                pass  # leave line unchanged if last token is not a number

        out_lines.append(stripped + "\n")

with open(DST, "w", encoding="utf-8") as f:
    f.writelines(out_lines)

print(f"Done. Written to: {DST}")
