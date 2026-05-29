"""
Parse coordinates and conduits from a SWMM .inp file.
Used by both network_map and flood_map; does not touch the DB.
"""

from __future__ import annotations

import re
from pathlib import Path


def parse_coordinates(inp_path: Path | str) -> dict[str, tuple[float, float]]:
    """Return {node_id: (x, y)} from the [COORDINATES] section."""
    coords: dict[str, tuple[float, float]] = {}
    in_section = False
    with open(inp_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("["):
                in_section = stripped.upper() == "[COORDINATES]"
                continue
            if not in_section or not stripped or stripped.startswith(";"):
                continue
            parts = stripped.split()
            if len(parts) >= 3:
                coords[parts[0]] = (float(parts[1]), float(parts[2]))
    return coords


def parse_conduits(inp_path: Path | str) -> list[tuple[str, str, str]]:
    """Return [(link_id, from_node, to_node)] from the [CONDUITS] section."""
    conduits: list[tuple[str, str, str]] = []
    in_section = False
    with open(inp_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("["):
                in_section = stripped.upper() == "[CONDUITS]"
                continue
            if not in_section or not stripped or stripped.startswith(";"):
                continue
            parts = stripped.split()
            if len(parts) >= 3:
                conduits.append((parts[0], parts[1], parts[2]))
    return conduits


def classify_conduits(
    conduits: list[tuple[str, str, str]],
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """
    Split conduits into (initial, continuous).

    Initial: the from_node never appears as a to_node anywhere in the network
             (i.e., it is a leaf / source node — in_degree == 0).
    Continuous: everything else.
    """
    to_nodes = {to for _, _, to in conduits}
    initial = [(lid, frm, to) for lid, frm, to in conduits if frm not in to_nodes]
    continuous = [(lid, frm, to) for lid, frm, to in conduits if frm in to_nodes]
    return initial, continuous
