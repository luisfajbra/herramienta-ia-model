"""Format a compute-time caption for flood-volume map annotations."""

from __future__ import annotations


def format_runtime_text(seconds: float | None) -> str | None:
    """Caption with the compute time (in seconds) for a flood map corner.

    Returns None when ``seconds`` is None (no annotation drawn). Formats the
    value with 2 decimals when ``seconds >= 1`` and 4 decimals otherwise (ML
    times are in the millisecond range).
    """
    if seconds is None:
        return None
    value = f"{seconds:.2f}" if seconds >= 1 else f"{seconds:.4f}"
    return f"Tiempo de cómputo: {value} s"
