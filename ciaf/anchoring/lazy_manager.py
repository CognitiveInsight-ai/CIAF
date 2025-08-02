"""
Lazy capsule management system.

This module orchestrates lazy capsule materialization across multiple datasets
and provides high-level interfaces for dataset management and on-demand capsule materialization.
"""

from datetime import datetime
from typing import Dict, Any

from .dataset_anchor import DatasetAnchor


class LazyProvenanceManager:
    """
    Manager for lazy capsule materialization across multiple datasets.
    
    This class orchestrates the lazy capsule system and provides high-level
    interfaces for dataset management and on-demand capsule materialization.
    """
    
    def __init__(self):
        """Initialize the lazy provenance manager."""
        self.dataset_anchors: Dict[str, DatasetAnchor] = {}
        self.lazy_capsule_registry: Dict[str, Dict[str, Any]] = {}
    
    def create_dataset_anchor(self, dataset_id: str, model_name: str, dataset_metadata: Dict[str, Any]) -> DatasetAnchor:
        """
        Create a new dataset anchor.
        
        Args:
            dataset_id: Unique identifier for the dataset.
            model_name: Name of the model.
            dataset_metadata: Metadata about the dataset.
            
        Returns:
            Created DatasetAnchor instance.
        """
        anchor = DatasetAnchor(dataset_id, model_name, dataset_metadata)
        self.dataset_anchors[dataset_id] = anchor
        return anchor
    
    def register_lazy_capsule(self, dataset_id: str, capsule_id: str, sample_data: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a lazy capsule without full materialization.
        
        Args:
            dataset_id: ID of the dataset anchor.
            capsule_id: Unique identifier for the capsule.
            sample_data: The raw sample data.
            metadata: Additional metadata for the sample.
            
        Returns:
            Lazy capsule metadata.
        """
        if dataset_id not in self.dataset_anchors:
            raise ValueError(f"Dataset anchor '{dataset_id}' not found")
        
        anchor = self.dataset_anchors[dataset_id]
        lazy_metadata = self._create_lazy_capsule_metadata(anchor, capsule_id, sample_data, metadata)
        
        # Register in the lazy capsule registry
        full_capsule_id = f"{dataset_id}:{capsule_id}"
        self.lazy_capsule_registry[full_capsule_id] = {
            'lazy_metadata': lazy_metadata,
            'sample_data': sample_data,
            'materialized_capsule': None
        }
        
        return lazy_metadata
    
    def _create_lazy_capsule_metadata(self, anchor: DatasetAnchor, capsule_id: str, sample_data: Any, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create metadata for a lazy capsule without materializing the full capsule.
        
        Args:
            anchor: The dataset anchor.
            capsule_id: Unique identifier for the capsule.
            sample_data: The raw sample data (string, float, int, or other).
            metadata: Additional metadata for the sample.
            
        Returns:
            Capsule metadata with lazy materialization support.
        """
        from ..core import sha256_hash
        
        # Compute sample hash - handle different data types
        if isinstance(sample_data, (str, bytes)):
            sample_data_str = sample_data if isinstance(sample_data, str) else sample_data.decode('utf-8')
        else:
            # Convert numerical or other data to string representation
            sample_data_str = str(sample_data)
        
        sample_hash = sha256_hash(sample_data_str.encode('utf-8'))
        anchor.add_sample_hash(sample_hash)
        
        # Derive capsule key
        capsule_key = anchor.derive_capsule_key(capsule_id)
        
        # Create lazy capsule metadata
        lazy_metadata = {
            'capsule_id': capsule_id,
            'dataset_anchor_id': anchor.dataset_id,
            'sample_hash': sample_hash,
            'capsule_key_derivation': f"HMAC(dataset_key, {capsule_id})",
            'materialized': False,
            'creation_timestamp': datetime.now().isoformat(),
            'metadata': metadata
        }
        
        return lazy_metadata
    
    def materialize_capsule(self, dataset_id: str, capsule_id: str):
        """
        Materialize a specific capsule on-demand.
        
        Args:
            dataset_id: ID of the dataset anchor.
            capsule_id: Unique identifier for the capsule.
            
        Returns:
            Materialized ProvenanceCapsule.
            
        Raises:
            ValueError: If capsule is not found or dataset anchor doesn't exist.
        """
        full_capsule_id = f"{dataset_id}:{capsule_id}"
        
        if full_capsule_id not in self.lazy_capsule_registry:
            raise ValueError(f"Lazy capsule '{full_capsule_id}' not found")
        
        if dataset_id not in self.dataset_anchors:
            raise ValueError(f"Dataset anchor '{dataset_id}' not found")
        
        registry_entry = self.lazy_capsule_registry[full_capsule_id]
        
        # Check if already materialized
        if registry_entry['materialized_capsule'] is not None:
            return registry_entry['materialized_capsule']
        
        # Materialize the capsule
        anchor = self.dataset_anchors[dataset_id]
        sample_data = registry_entry['sample_data']
        metadata = registry_entry['lazy_metadata']['metadata']
        
        capsule = self._materialize_provenance_capsule(anchor, capsule_id, sample_data, metadata)
        
        # Cache the materialized capsule
        registry_entry['materialized_capsule'] = capsule
        registry_entry['lazy_metadata']['materialized'] = True
        
        return capsule
    
    def _materialize_provenance_capsule(self, anchor: DatasetAnchor, capsule_id: str, sample_data: str, metadata: Dict[str, Any]):
        """
        Materialize a full provenance capsule on-demand.
        
        Args:
            anchor: The dataset anchor.
            capsule_id: Unique identifier for the capsule.
            sample_data: The raw sample data.
            metadata: Additional metadata for the sample.
            
        Returns:
            Fully materialized ProvenanceCapsule.
        """
        from ..provenance import ProvenanceCapsule
        
        # Derive the capsule key
        capsule_key = anchor.derive_capsule_key(capsule_id)
        
        # Create enhanced metadata for the capsule
        enhanced_metadata = metadata.copy()
        enhanced_metadata.update({
            'capsule_id': capsule_id,
            'dataset_anchor_id': anchor.dataset_id,
            'audit_reference': f"provenance_{anchor.model_name}_{capsule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        })
        
        # Create the provenance capsule using the derived key as the data secret
        capsule = ProvenanceCapsule(sample_data, enhanced_metadata, capsule_key)
        
        print(f"Materialized capsule '{capsule_id}' for dataset '{anchor.dataset_id}'")
        return capsule
    
    def get_dataset_summary(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get a summary of a dataset's lazy capsule status.
        
        Args:
            dataset_id: ID of the dataset anchor.
            
        Returns:
            Summary dictionary with statistics.
        """
        if dataset_id not in self.dataset_anchors:
            raise ValueError(f"Dataset anchor '{dataset_id}' not found")
        
        anchor = self.dataset_anchors[dataset_id]
        
        # Count materialized vs lazy capsules
        lazy_count = 0
        materialized_count = 0
        
        for full_id, entry in self.lazy_capsule_registry.items():
            if full_id.startswith(f"{dataset_id}:"):
                if entry['materialized_capsule'] is not None:
                    materialized_count += 1
                else:
                    lazy_count += 1
        
        return {
            'dataset_id': dataset_id,
            'model_name': anchor.model_name,
            'total_samples': anchor.total_samples,
            'lazy_capsules': lazy_count,
            'materialized_capsules': materialized_count,
            'merkle_root': anchor.get_merkle_root(),
            'dataset_hash': anchor.dataset_hash
        }
    
    def audit_capsule_provenance(self, dataset_id: str, capsule_id: str) -> Dict[str, Any]:
        """
        Perform a complete audit of a capsule's provenance.
        
        Args:
            dataset_id: ID of the dataset anchor.
            capsule_id: Unique identifier for the capsule.
            
        Returns:
            Audit results with verification status.
        """
        try:
            # Materialize the capsule for audit
            capsule = self.materialize_capsule(dataset_id, capsule_id)
            
            # Verify capsule integrity
            integrity_valid = capsule.verify_hash_proof()
            
            # Verify dataset anchor consistency
            anchor = self.dataset_anchors[dataset_id]
            anchor_valid = anchor.verify_capsule_integrity(capsule_id, capsule.metadata)
            
            # Generate audit metadata
            audit_metadata = {
                'capsule_id': capsule_id,
                'dataset_anchor_id': dataset_id,
                'audit_timestamp': datetime.now().isoformat(),
                'integrity_verified': integrity_valid,
                'anchor_verified': anchor_valid,
                'audit_passed': integrity_valid and anchor_valid,
                'capsule_hash': capsule.hash_proof,
                'audit_reference': capsule.metadata.get('audit_reference', f"audit_{capsule_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            }
            
            return audit_metadata
            
        except Exception as e:
            return {
                'capsule_id': capsule_id,
                'dataset_anchor_id': dataset_id,
                'audit_timestamp': datetime.now().isoformat(),
                'error': str(e),
                'audit_passed': False
            }
