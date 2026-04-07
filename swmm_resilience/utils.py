"""
Shared helpers used across the package.
"""

import hashlib
import math
import uuid


def file_hash(path: str) -> str:
    """Calculate an MD5 hash for a file."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def new_id() -> str:
    """Return a UUID4 string."""
    return str(uuid.uuid4())


def safe_round(value, decimals=4):
    """Round values safely, preserving None."""
    return round(value, decimals) if value is not None else None


def node_type_str(node) -> str:
    """Return a normalized node type string from a PySWMM node."""
    if node.is_junction():
        return "junction"
    if node.is_outfall():
        return "outfall"
    if node.is_storage():
        return "storage"
    if node.is_divider():
        return "divider"
    return "unknown"


def circular_full_flow_lps(diameter_m: float, slope: float, roughness: float):
    """Compute Manning full-flow capacity for a circular conduit in L/s."""
    if not diameter_m or not slope or not roughness:
        return None
    if diameter_m <= 0 or slope <= 0 or roughness <= 0:
        return None

    area = math.pi * diameter_m**2 / 4.0
    radius = diameter_m / 4.0
    flow_lps = (1.0 / roughness) * area * (radius ** (2.0 / 3.0)) * (slope ** 0.5) * 1000.0
    return round(flow_lps, 6)
