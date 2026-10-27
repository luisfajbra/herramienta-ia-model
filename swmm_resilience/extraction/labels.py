"""Fail-closed extraction of flood labels from SWMM report files."""

from __future__ import annotations

import math
import warnings
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from ..simulation.swmm_api_io import read_node_flooding_summary


class FloodingReportError(ValueError):
    """A flooding report is absent, malformed, or contains invalid values."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def flood_label(volumes: pd.Series, threshold_m3: float) -> pd.Series:
    """Apply the canonical flood rule to a validated volume series."""
    if threshold_m3 > 0:
        return (volumes >= threshold_m3).astype(int)
    return (volumes > 0).astype(int)


def _read_report_text(rpt_path: Path) -> str:
    path = Path(rpt_path)
    if not path.is_file():
        raise FloodingReportError("missing_report", f"SWMM report does not exist: {path}")
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise FloodingReportError("unreadable_report", f"Cannot read SWMM report: {path}") from exc
    if "Node Flooding Summary" not in text:
        raise FloodingReportError(
            "missing_flooding_summary", "SWMM report has no Node Flooding Summary section"
        )
    flooding_start = text.index("Node Flooding Summary")
    if "outfall loading summary" not in text[flooding_start + len("Node Flooding Summary") :].casefold():
        raise FloodingReportError(
            "invalid_flooding_summary",
            "Node Flooding Summary has no following Outfall Loading Summary completion marker",
        )
    return text


def _validate_volume_table(df: pd.DataFrame) -> pd.DataFrame:
    if not {"node_id", "flooding_volume_m3"}.issubset(df.columns):
        raise FloodingReportError(
            "invalid_flooding_summary", "Flooding table lacks node IDs or total flood volume"
        )
    out = df[["node_id", "flooding_volume_m3"]].copy()
    if out["node_id"].isna().any():
        raise FloodingReportError("invalid_flooding_summary", "Flooding table contains an empty node ID")
    out["node_id"] = out["node_id"].astype(str)
    if (out["node_id"].str.strip() == "").any():
        raise FloodingReportError("invalid_flooding_summary", "Flooding table contains an empty node ID")
    if out["node_id"].duplicated().any():
        repeated = sorted(out.loc[out["node_id"].duplicated(keep=False), "node_id"].unique())
        raise FloodingReportError(
            "duplicate_flooding_node", f"Flooding table repeats node IDs: {repeated}"
        )
    out["flooding_volume_m3"] = pd.to_numeric(out["flooding_volume_m3"], errors="coerce")
    values = out["flooding_volume_m3"].tolist()
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in values):
        raise FloodingReportError(
            "invalid_flooding_volume", "Flooding volumes must be finite and non-negative"
        )
    return out.reset_index(drop=True)


def _parse_node_flooding_text(text: str) -> pd.DataFrame:
    """Parse the known SWMM text layout, selecting total volume by column position.

    The summary reports total flooding volume before maximum ponded volume when
    ponding is enabled. Older reports without that last column place total
    flooding volume at the end. The API parser remains primary; this fallback
    accepts only numeric row layouts and rejects ambiguous or malformed rows.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if "Node Flooding Summary" in line)
    except StopIteration as exc:
        raise FloodingReportError("missing_flooding_summary", "Missing Node Flooding Summary") from exc
    section_lines = []
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped and "SUMMARY" in stripped.upper() and "FLOODING" not in stripped.upper():
            break
        section_lines.append(line)
    section = "\n".join(section_lines)
    if "No nodes were flooded" in section:
        return pd.DataFrame(columns=["node_id", "flooding_volume_m3"])

    has_ponding_column = "maximum ponded volume" in section.casefold()
    has_total_flood_volume_column = any(
        label in section.casefold()
        for label in ("total flood volume", "total flooding volume")
    )
    if has_ponding_column and not has_total_flood_volume_column:
        raise FloodingReportError(
            "invalid_flooding_summary",
            "Summary lists maximum ponded volume without total flood volume",
        )

    rows = []
    saw_separator = False
    for line in section_lines:
        stripped = line.strip()
        if stripped.startswith("---") or (stripped and set(stripped) == {"*"}):
            saw_separator = True
            continue
        if not saw_separator or not stripped:
            continue
        parts = stripped.split()
        # SWMM node IDs may include letters, numbers, underscores and dashes.
        if not parts or not any(ch.isalpha() or ch.isdigit() for ch in parts[0]):
            continue
        numeric_tokens = 0
        for token in parts[1:]:
            try:
                float(token)
                numeric_tokens += 1
            except ValueError:
                continue
        if numeric_tokens == 0:
            if len(parts) == 1 and parts[0].casefold() not in {"node", "id"}:
                raise FloodingReportError(
                    "invalid_flooding_summary",
                    f"Node ID without flooding values after data rows began: {parts[0]}",
                )
            continue  # repeated text header or page heading
        if len(parts) < 4:
            raise FloodingReportError(
                "invalid_flooding_summary", f"Truncated flooding row for node {parts[0]}"
            )
        volume_index = -2 if has_ponding_column or len(parts) >= 7 else -1
        try:
            raw_volume = float(parts[volume_index])
        except (ValueError, IndexError) as exc:
            raise FloodingReportError(
                "invalid_flooding_summary", f"Malformed flooding row for node {parts[0]}"
            ) from exc
        rows.append({"node_id": parts[0], "flooding_volume_m3": raw_volume * 1000.0})

    if not rows:
        raise FloodingReportError(
            "invalid_flooding_summary",
            "Flooding summary has neither a parsed table nor an explicit no-flood marker",
        )
    return _validate_volume_table(pd.DataFrame(rows))


def read_flooding_volumes_with_manifest(
    rpt_path: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read flood volumes and return exact parser/report provenance."""
    path = Path(rpt_path)
    text = _read_report_text(path)
    report_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        swmm_api_version = version("swmm-api")
    except PackageNotFoundError:
        swmm_api_version = "unknown"

    try:
        df_rpt = read_node_flooding_summary(path)
        if df_rpt is not None and not df_rpt.empty:
            if "flooding_volume_m3" not in df_rpt.columns and "flooding_volume_m3_raw" in df_rpt.columns:
                df_rpt = df_rpt.rename(columns={"flooding_volume_m3_raw": "flooding_volume_m3"})
                df_rpt["flooding_volume_m3"] = pd.to_numeric(
                    df_rpt["flooding_volume_m3"], errors="coerce"
                ) * 1000.0
            return _validate_volume_table(df_rpt), {
                "schema_version": "flooding_report_v1",
                "parser": "swmm_api.node_flooding_summary",
                "parser_version": swmm_api_version,
                "report_sha256": report_sha256,
            }
    except FloodingReportError:
        raise
    except Exception as exc:
        warnings.warn(
            f"swmm_api RPT read failed ({exc}); validating with the strict text parser.",
            stacklevel=2,
        )
    return _parse_node_flooding_text(text), {
        "schema_version": "flooding_report_v1",
        "parser": "strict_text_v1",
        "parser_version": "strict_text_v1",
        "report_sha256": report_sha256,
    }


def read_flooding_volumes(rpt_path: Path) -> pd.DataFrame:
    """Read and validate total node flooding volumes from one SWMM report."""
    volumes, _manifest = read_flooding_volumes_with_manifest(rpt_path)
    return volumes


def extract_labels(rpt_path: Path, all_node_ids: list, threshold_m3: float = 1.0) -> pd.DataFrame:
    """Build labels only after a valid flooding summary has been parsed.

    Nodes absent from a valid SWMM flooding summary receive zero volume.
    Invalid or missing reports raise ``FloodingReportError`` and never become
    zero-flood labels.
    """
    result = pd.DataFrame({"node_id": [str(n) for n in all_node_ids]})
    result["vol_inundacion_m3"] = 0.0

    df_rpt, manifest = read_flooding_volumes_with_manifest(rpt_path)
    if not df_rpt.empty:
        flood_map = dict(zip(df_rpt["node_id"], df_rpt["flooding_volume_m3"]))
        result["vol_inundacion_m3"] = result["node_id"].map(flood_map).fillna(0.0)

    result["inunda"] = flood_label(result["vol_inundacion_m3"], threshold_m3)
    result.attrs["flooding_report_manifest"] = manifest
    return result
