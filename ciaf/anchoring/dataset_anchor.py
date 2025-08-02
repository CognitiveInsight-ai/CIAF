"""
Dataset-level anchoring for lazy capsule materialization.

This module implements the dataset anchor system that enables lazy capsule
materialization while maintaining cryptographic consistency and audit integrity.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any

from ..core import (
    derive_master_key, 
    derive_dataset_key, 
    derive_capsule_key,
    sha256_hash, 
    secure_random_bytes,
    SALT_LENGTH
)


class DatasetAnchor:
    """
    Represents a dataset anchor for lazy capsule materialization.
    
    This class manages the cryptographic hierarchy:
    Passphrase → Master Key → Dataset Key → Capsule Keys (on demand)
    """
    
    def __init__(self, dataset_id: str, metadata: Dict[str, Any] = None, 
                 model_name: str = None, master_password: str = None, salt: bytes = None):
        """
        Initialize a dataset anchor.
        
        Args:
            dataset_id: Unique identifier for the dataset.
            metadata: Metadata about the dataset (optional, for backwards compatibility)
            model_name: Name of the model (optional, for backwards compatibility)
            master_password: Master password for key derivation (optional)
            salt: Salt for key derivation (optional)
        """
        self.dataset_id = dataset_id
        
        # Handle different initialization patterns for backwards compatibility
        if metadata is not None:
            self.metadata = metadata.copy()
        else:
            self.metadata = {}
            
        # Use master_password if provided, otherwise fall back to model_name
        password = master_password or model_name or "default_password"
        self.model_name = model_name or "default_model"  # Store for backwards compatibility
        
        # Generate or use provided salt
        if salt is not None:
            self.master_key_salt = salt
        else:
            self.master_key_salt = secure_random_bytes(SALT_LENGTH)
        
        # Derive master key and dataset key
        self.master_key = derive_master_key(password, self.master_key_salt)
        
        # Compute dataset hash from metadata and derive dataset key
        self.dataset_hash = self._compute_dataset_hash()
        self.dataset_key = derive_dataset_key(self.master_key, self.dataset_hash)
        
        # Initialize sample tracking
        self.sample_hashes: List[str] = []
        self.total_samples = 0
        
        # Initialize data items tracking for lazy management
        self.data_items: Dict[str, Dict[str, Any]] = {}
        
        # Add anchoring metadata
        self.metadata.update({
            'dataset_anchor_id': self.dataset_id,
            'master_key_salt': self.master_key_salt.hex(),
            'dataset_hash': self.dataset_hash,
            'creation_timestamp': datetime.now().isoformat(),
            'audit_tags': ['lazy_capsules', 'anchored_merkle_root', 'dataset_level_derivation']
        })
        
        print(f"Dataset Anchor '{self.dataset_id}' initialized for model '{self.model_name}'")
    
    def _compute_dataset_hash(self) -> str:
        """
        Compute a hash of the dataset metadata for consistent key derivation.
        
        Returns:
            SHA256 hash of the dataset metadata.
        """
        # Create a deterministic representation of the metadata
        metadata_copy = self.metadata.copy()
        # Remove dynamic fields that shouldn't affect the dataset hash
        metadata_copy.pop('creation_timestamp', None)
        metadata_copy.pop('audit_tags', None)
        
        metadata_str = json.dumps(metadata_copy, sort_keys=True)
        return sha256_hash(metadata_str.encode('utf-8'))
    
    def add_sample_hash(self, sample_hash: str) -> None:
        """
        Add a sample hash to the dataset for Merkle tree construction.
        
        Args:
            sample_hash: SHA256 hash of a data sample.
        """
        if sample_hash not in self.sample_hashes:
            self.sample_hashes.append(sample_hash)
            self.total_samples = len(self.sample_hashes)
    
    def add_data_item(self, item_id: str, content: Any, metadata: Dict[str, Any]) -> None:
        """
        Add a data item to the dataset for lazy capsule management.
        
        Args:
            item_id: Unique identifier for the data item
            content: The data content (string, float, int, or other)
            metadata: Metadata about the data item
        """
        self.data_items[item_id] = {
            'content': content,
            'metadata': metadata.copy()
        }
        
        # Also add to sample hashes for Merkle tree - handle different data types
        if isinstance(content, (str, bytes)):
            content_str = content if isinstance(content, str) else content.decode('utf-8')
        else:
            # Convert numerical or other data to string representation
            content_str = str(content)
            
        item_hash = sha256_hash(f"{item_id}:{content_str}".encode('utf-8'))
        self.add_sample_hash(item_hash)
    
    def derive_item_key(self, item_id: str) -> str:
        """
        Derive a key for a specific data item using the dataset key.
        
        Args:
            item_id: Unique identifier for the data item
            
        Returns:
            Derived key for the data item
        """
        return derive_capsule_key(self.dataset_key, item_id)
    
    def derive_capsule_key(self, capsule_id: str) -> str:
        """
        Derive a capsule key on-demand using HMAC(dataset_key, capsule_id).
        
        Args:
            capsule_id: Unique identifier for the capsule.
            
        Returns:
            Derived capsule key as hexadecimal string.
        """
        return derive_capsule_key(self.dataset_key, capsule_id)
    
    def get_merkle_root(self) -> str:
        """
        Get the Merkle root of all sample hashes in the dataset.
        
        Returns:
            Merkle root hash, or None if no samples.
        """
        if not self.sample_hashes:
            return None
        
        from ..core import MerkleTree
        merkle_tree = MerkleTree(self.sample_hashes)
        return merkle_tree.get_root()
    
    def verify_capsule_integrity(self, capsule_id: str, capsule_metadata: Dict[str, Any]) -> bool:
        """
        Verify that a capsule belongs to this dataset anchor.
        
        Args:
            capsule_id: The capsule ID to verify.
            capsule_metadata: Metadata from the capsule.
            
        Returns:
            True if the capsule is valid for this dataset.
        """
        # Check dataset anchor ID
        if capsule_metadata.get('dataset_anchor_id') != self.dataset_id:
            return False
        
        # Verify capsule key derivation
        expected_key = self.derive_capsule_key(capsule_id)
        if 'capsule_key_derivation' in capsule_metadata:
            # This is a lazy capsule, verify the derivation string
            expected_derivation = f"HMAC(dataset_key, {capsule_id})"
            return capsule_metadata['capsule_key_derivation'] == expected_derivation
        
        return True
    
    def to_json(self) -> Dict[str, Any]:
        """
        Serialize the dataset anchor to JSON.
        
        Returns:
            JSON-serializable dictionary.
        """
        return {
            'dataset_id': self.dataset_id,
            'model_name': self.model_name,
            'metadata': self.metadata,
            'dataset_hash': self.dataset_hash,
            'master_key_salt': self.master_key_salt.hex(),
            'sample_hashes': self.sample_hashes,
            'total_samples': self.total_samples,
            'merkle_root': self.get_merkle_root()
        }
    
    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "DatasetAnchor":
        """
        Reconstruct a dataset anchor from JSON data.
        
        Args:
            json_data: JSON data from to_json().
            
        Returns:
            Reconstructed DatasetAnchor instance.
        """
        # Create new instance
        anchor = cls.__new__(cls)
        anchor.dataset_id = json_data['dataset_id']
        anchor.model_name = json_data['model_name']
        anchor.metadata = json_data['metadata']
        anchor.dataset_hash = json_data['dataset_hash']
        anchor.master_key_salt = bytes.fromhex(json_data['master_key_salt'])
        anchor.sample_hashes = json_data['sample_hashes']
        anchor.total_samples = json_data['total_samples']
        
        # Regenerate derived keys
        anchor.master_key = derive_master_key(anchor.model_name, anchor.master_key_salt, 32)
        anchor.dataset_key = derive_dataset_key(anchor.master_key, anchor.dataset_hash)
        
        return anchor
