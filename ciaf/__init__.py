"""
Cognitive Insight AI Framework (CIAF)

A modular framework for creating verifiable AI training and inference pipelines
with lazy capsule materialization and cryptographic provenance tracking.
"""

from .core import CryptoUtils, KeyManager, MerkleTree
from .anchoring import DatasetAnchor, LazyManager
from .provenance import ProvenanceCapsule, TrainingSnapshot, ModelAggregationKey
from .simulation import MockLLM, MLFrameworkSimulator
from .inference import InferenceReceipt, ZKEChain
from .wrappers import CIAFModelWrapper
from .api import CIAFFramework

__version__ = "1.0.0"
__all__ = [
    'CryptoUtils', 'KeyManager', 'MerkleTree',
    'DatasetAnchor', 'LazyManager', 
    'ProvenanceCapsule', 'TrainingSnapshot', 'ModelAggregationKey',
    'MockLLM', 'MLFrameworkSimulator',
    'InferenceReceipt', 'ZKEChain',
    'CIAFModelWrapper',
    'CIAFFramework'
]
