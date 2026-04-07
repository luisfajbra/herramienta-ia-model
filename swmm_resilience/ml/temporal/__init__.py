"""
Scaffolding for future temporal ML models based on hydrographs.

This package intentionally does not implement a CNN yet. It defines the
structure we will use later for time-window datasets, CNN training and
temporal prediction.
"""

from .schemas import TemporalDatasetSpec, TemporalWindowSpec

__all__ = ["TemporalDatasetSpec", "TemporalWindowSpec"]
