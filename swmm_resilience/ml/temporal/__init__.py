"""
Temporal ML models and dataset helpers based on hydrographs.
"""

from .schemas import TemporalDatasetSpec, TemporalWindowSpec

try:
    from .models.cnn import SWMMTemporalCNN
    __all__ = ["TemporalDatasetSpec", "TemporalWindowSpec", "SWMMTemporalCNN"]
except ImportError:
    __all__ = ["TemporalDatasetSpec", "TemporalWindowSpec"]
