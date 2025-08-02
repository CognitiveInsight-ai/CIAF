"""
Dataset and model anchoring system for lazy capsule materialization.
"""

from .dataset_anchor import DatasetAnchor
from .lazy_manager import LazyProvenanceManager
from .simple_lazy_manager import LazyManager

__all__ = [
    'DatasetAnchor',
    'LazyProvenanceManager',
    'LazyManager'
]
