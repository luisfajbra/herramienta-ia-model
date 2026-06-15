"""Single source of truth for flood labels derived from SWMM .rpt files.

Both the training pipeline and the hydrograph validation harness must obtain
labels through this module so the rule can never diverge between them.

Label rule: ``inunda = vol_m3 >= threshold_m3`` (with ``threshold_m3 > 0``).
A threshold <= 0 degrades to ``vol_m3 > 0`` so that nodes absent from the
flooding summary (vol = 0) are never marked as flooded.
"""

import warnings

import pandas as pd
from pathlib import Path

from ..simulation.swmm_api_io import read_node_flooding_summary


def flood_label(volumes: pd.Series, threshold_m3: float) -> pd.Series:
    """Apply the canonical flood rule to a volume series. Returns int 0/1."""
    if threshold_m3 > 0:
        return (volumes >= threshold_m3).astype(int)
    return (volumes > 0).astype(int)


def _parse_node_flooding_text(rpt_path: Path) -> pd.DataFrame:
    """Pure-text fallback parser for the 'Node Flooding Summary' section.

    Returns DataFrame: node_id, flooding_volume_m3 (already converted from
    the .rpt unit of 10^6 litres; 1 Megaliter = 1000 m³).

    Data lines start after the dashed separator. Columns are
    whitespace-delimited; the last numeric token is the volume, which is
    robust against varying column widths (but NOT against ALLOW_PONDING,
    which appends a 'Maximum Ponded Volume' column — callers should warn).
    """
    try:
        text = Path(rpt_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return pd.DataFrame(columns=["node_id", "flooding_volume_m3"])

    rows: list[dict] = []
    in_section = False
    past_header = False

    for line in text.splitlines():
        stripped = line.strip()

        if not in_section:
            if "Node Flooding Summary" in line:
                in_section = True
                past_header = False
            continue

        if not stripped:
            continue
        if past_header and stripped.startswith("*"):
            break
        # Section headers ("OUTFALL LOADING SUMMARY") are all-caps words with
        # no digits; data rows always contain numeric columns, and node IDs
        # like "1C" or "J7" would otherwise satisfy isupper().
        if (
            past_header
            and stripped[0].isalpha()
            and stripped.isupper()
            and not any(ch.isdigit() for ch in stripped)
        ):
            break
        if stripped.startswith("---"):
            past_header = True
            continue
        if not past_header:
            continue

        parts = stripped.split()
        if len(parts) < 4:
            continue
        try:
            vol_megalitres = float(parts[-1])
        except ValueError:
            continue

        rows.append(
            {
                "node_id": parts[0],
                "flooding_volume_m3": vol_megalitres * 1000.0,
            }
        )

    return pd.DataFrame(rows, columns=["node_id", "flooding_volume_m3"])


def read_flooding_volumes(rpt_path: Path) -> pd.DataFrame:
    """Read flooded-node volumes from a .rpt file.

    Tries swmm_api first; falls back to the pure-text parser when swmm_api is
    unavailable, fails, or returns an empty table.

    Returns DataFrame: node_id (str), flooding_volume_m3 (float).
    """
    try:
        df_rpt = read_node_flooding_summary(rpt_path)
        if (
            df_rpt is not None
            and not df_rpt.empty
            and "flooding_volume_m3" in df_rpt.columns
        ):
            out = df_rpt[["node_id", "flooding_volume_m3"]].copy()
            out["node_id"] = out["node_id"].astype(str)
            out["flooding_volume_m3"] = pd.to_numeric(
                out["flooding_volume_m3"], errors="coerce"
            ).fillna(0.0)
            return out
    except Exception as exc:
        warnings.warn(
            f"swmm_api RPT read failed ({exc}); falling back to text parser.",
            stacklevel=2,
        )
    return _parse_node_flooding_text(rpt_path)


def extract_labels(rpt_path: Path, all_node_ids: list, threshold_m3: float = 1.0) -> pd.DataFrame:
    """Parse Node Flooding Summary from a SWMM .rpt file.

    Nodes absent from the .rpt (no flooding reported) receive vol=0, inunda=0.
    read_node_flooding_summary already converts 10^6 L → m³ (×1000).

    Returns DataFrame: node_id, vol_inundacion_m3, inunda
    """
    result = pd.DataFrame({"node_id": [str(n) for n in all_node_ids]})
    result["vol_inundacion_m3"] = 0.0

    df_rpt = read_flooding_volumes(rpt_path)
    if not df_rpt.empty:
        flood_map = dict(zip(df_rpt["node_id"], df_rpt["flooding_volume_m3"]))
        result["vol_inundacion_m3"] = result["node_id"].map(flood_map).fillna(0.0)

    result["inunda"] = flood_label(result["vol_inundacion_m3"], threshold_m3)
    return result
