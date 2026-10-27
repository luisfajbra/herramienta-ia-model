"""Shared interpretation of SWMM external FLOW inflows in project units."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"", "nan", '""', "''"}
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


@dataclass(frozen=True)
class EffectiveInflow:
    """One node's effective FLOW series, with a SWMM-compatible evaluator."""

    source_points: tuple[tuple[float, float], ...]
    points: tuple[tuple[float, float], ...]
    scale_factor: float
    baseline_lps: float
    metadata: dict[str, Any]

    def at(self, time_h: float) -> float:
        """Evaluate baseline + Sfactor * time-series value at elapsed hours.

        SWMM treats the time-series component as zero outside its listed range;
        the baseline remains in effect. Values between points are interpolated
        linearly, as in the SWMM external inflow definition.
        """
        t = _finite_float(time_h, "time_h")
        if not self.source_points:
            return self.baseline_lps
        if t < self.source_points[0][0] or t > self.source_points[-1][0]:
            series_value = 0.0
        elif t == self.source_points[0][0]:
            series_value = self.source_points[0][1]
        elif t == self.source_points[-1][0]:
            series_value = self.source_points[-1][1]
        else:
            series_value = 0.0
            for (t0, q0), (t1, q1) in zip(self.source_points, self.source_points[1:]):
                if t0 <= t <= t1:
                    fraction = (t - t0) / (t1 - t0)
                    series_value = q0 + fraction * (q1 - q0)
                    break
        return self.baseline_lps + self.scale_factor * series_value

    def at_scaled_component(self, time_h: float, multiplier: float) -> float:
        """Evaluate a scenario that multiplies only the time-series component."""
        factor = _finite_float(multiplier, "multiplier")
        return self.baseline_lps + (self.at(time_h) - self.baseline_lps) * factor

    def at_scaled_baseline(self, time_h: float, multiplier: float) -> float:
        """Evaluate a scenario that scales only the declared baseline."""
        factor = _finite_float(multiplier, "multiplier")
        return self.at(time_h) + self.baseline_lps * (factor - 1.0)


def build_effective_inflow(
    points: Iterable[tuple[float, float]],
    *,
    flow_units: str | None,
    constituent: str,
    scale_factor: float | None = 1.0,
    baseline: float | None = 0.0,
    pattern: str | None = None,
    mass_unit_factor: float | None = 1.0,
    timeseries_name: str | None = None,
) -> EffectiveInflow:
    """Validate and interpret one external SWMM FLOW inflow in L/s.

    FLOW inputs are supported only when `FLOW_UNITS` is explicitly `LPS` and
    no unsupported time pattern is attached. For FLOW, SWMM applies Sfactor to
    the time-series values and adds Base; Mfactor applies to pollutant mass
    inflows and is recorded but not applied here.
    """
    if flow_units is None or str(flow_units).strip().upper() != "LPS":
        raise ValueError(
            f"FLOW_UNITS must be explicitly LPS for effective-flow features; got {flow_units!r}"
        )
    normalized_constituent = str(constituent).strip().upper()
    if normalized_constituent != "FLOW":
        raise ValueError(
            f"Unsupported inflow constituent {constituent!r}; only FLOW is supported"
        )
    if not _is_missing(pattern):
        raise ValueError(
            f"Unsupported FLOW inflow pattern {pattern!r}; patterned baselines need an explicit evaluator"
        )

    scale = 1.0 if _is_missing(scale_factor) else _finite_float(scale_factor, "scale_factor")
    base = 0.0 if _is_missing(baseline) else _finite_float(baseline, "baseline")
    mass_factor = 1.0 if _is_missing(mass_unit_factor) else _finite_float(
        mass_unit_factor, "mass_unit_factor"
    )

    normalized_points = []
    previous_time = None
    for index, point in enumerate(points):
        try:
            raw_time, raw_flow = point
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Time-series point {index} must be a (time_h, flow_lps) pair") from exc
        time_h = _finite_float(raw_time, f"time-series point {index} time_h")
        flow_lps = _finite_float(raw_flow, f"time-series point {index} flow_lps")
        if time_h < 0:
            raise ValueError(f"Time-series point {index} time_h must be non-negative")
        if previous_time is not None and time_h <= previous_time:
            raise ValueError("Time-series times must be strictly increasing")
        normalized_points.append((time_h, flow_lps))
        previous_time = time_h

    effective_points = [(time_h, base + scale * value) for time_h, value in normalized_points]
    if normalized_points and normalized_points[0][0] > 0:
        effective_points.insert(0, (0.0, base))
    elif not normalized_points:
        effective_points.append((0.0, base))

    return EffectiveInflow(
        source_points=tuple(normalized_points),
        points=tuple(effective_points),
        scale_factor=scale,
        baseline_lps=base,
        metadata={
            "flow_units": "LPS",
            "constituent": normalized_constituent,
            "source_timeseries": timeseries_name,
            "source_scale_factor": scale,
            "baseline_lps": base,
            "pattern": None,
            "mass_unit_factor": mass_factor,
            "mass_unit_factor_applied": False,
        },
    )
