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

# Metadata storage and integration
from .metadata_storage import MetadataStorage, get_metadata_storage, save_pipeline_metadata, get_pipeline_trace
from .metadata_config import MetadataConfig, get_metadata_config, load_config_from_file, create_config_template
from .metadata_integration import (
    MetadataCapture, capture_metadata, ModelMetadataManager, ComplianceTracker,
    create_model_manager, create_compliance_tracker, quick_log
)

__version__ = "2.1.0"
__all__ = [
    # Core components
    'CryptoUtils', 'KeyManager', 'MerkleTree',
    'DatasetAnchor', 'LazyManager', 
    'ProvenanceCapsule', 'TrainingSnapshot', 'ModelAggregationKey',
    'MockLLM', 'MLFrameworkSimulator',
    'InferenceReceipt', 'ZKEChain',
    'CIAFModelWrapper',
    'CIAFFramework',
    
    # Metadata storage and management
    'MetadataStorage', 'get_metadata_storage', 'save_pipeline_metadata', 'get_pipeline_trace',
    'MetadataConfig', 'get_metadata_config', 'load_config_from_file', 'create_config_template',
    'MetadataCapture', 'capture_metadata', 'ModelMetadataManager', 'ComplianceTracker',
    'create_model_manager', 'create_compliance_tracker', 'quick_log'
]
