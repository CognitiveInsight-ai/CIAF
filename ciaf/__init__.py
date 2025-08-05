"""
Cognitive Insight AI Framework (CIAF)

A modular framework for creating verifiable AI training and inference pipelines
with lazy capsule materialization and cryptographic provenance tracking.
"""

from .anchoring import DatasetAnchor, LazyManager
from .api import CIAFFramework
from .core import CryptoUtils, KeyManager, MerkleTree
from .inference import InferenceReceipt, ZKEChain
from .metadata_config import (
    MetadataConfig,
    create_config_template,
    get_metadata_config,
    load_config_from_file,
)
from .metadata_integration import (
    ComplianceTracker,
    MetadataCapture,
    ModelMetadataManager,
    capture_metadata,
    create_compliance_tracker,
    create_model_manager,
    quick_log,
)

# Metadata storage and integration
from .metadata_storage import (
    MetadataStorage,
    get_metadata_storage,
    get_pipeline_trace,
    save_pipeline_metadata,
)
from .provenance import ModelAggregationKey, ProvenanceCapsule, TrainingSnapshot
from .simulation import MLFrameworkSimulator, MockLLM
from .wrappers import CIAFModelWrapper

__version__ = "2.1.0"
__all__ = [
    # Core components
    "CryptoUtils",
    "KeyManager",
    "MerkleTree",
    "DatasetAnchor",
    "LazyManager",
    "ProvenanceCapsule",
    "TrainingSnapshot",
    "ModelAggregationKey",
    "MockLLM",
    "MLFrameworkSimulator",
    "InferenceReceipt",
    "ZKEChain",
    "CIAFModelWrapper",
    "CIAFFramework",
    # Metadata storage and management
    "MetadataStorage",
    "get_metadata_storage",
    "save_pipeline_metadata",
    "get_pipeline_trace",
    "MetadataConfig",
    "get_metadata_config",
    "load_config_from_file",
    "create_config_template",
    "MetadataCapture",
    "capture_metadata",
    "ModelMetadataManager",
    "ComplianceTracker",
    "create_model_manager",
    "create_compliance_tracker",
    "quick_log",
]
