"""
CIAF API Framework

High-level API for the Cognitive Insight AI Framework.
"""

import hashlib
from typing import Any, Dict, List, Optional

from ..anchoring import DatasetAnchor, LazyManager
from ..core import CryptoUtils, KeyManager, MerkleTree
from ..provenance import ModelAggregationKey, ProvenanceCapsule, TrainingSnapshot
from ..simulation import MLFrameworkSimulator


class CIAFFramework:
    """
    Main framework class providing high-level API for CIAF operations.
    """

    def __init__(self, framework_name: str = "CIAF"):
        self.framework_name = framework_name
        self.crypto_utils = CryptoUtils()
        self.key_manager = KeyManager()
        self.dataset_anchors: Dict[str, DatasetAnchor] = {}
        self.lazy_managers: Dict[str, LazyManager] = {}
        self.ml_simulators: Dict[str, MLFrameworkSimulator] = {}

    def create_dataset_anchor(
        self, dataset_id: str, dataset_metadata: Dict[str, Any], master_password: str
    ) -> DatasetAnchor:
        """
        Create a new dataset anchor with its own key derivation hierarchy.

        Args:
            dataset_id: Unique identifier for the dataset
            dataset_metadata: Metadata about the dataset
            master_password: Master password for key derivation

        Returns:
            DatasetAnchor instance
        """
        print(f"Creating dataset anchor for: {dataset_id}")

        # Generate dataset-specific salt
        dataset_salt = hashlib.sha256(
            f"{dataset_id}_{self.framework_name}".encode()
        ).digest()

        # Create dataset anchor
        anchor = DatasetAnchor(
            dataset_id=dataset_id,
            metadata=dataset_metadata,
            master_password=master_password,
            salt=dataset_salt,
        )

        self.dataset_anchors[dataset_id] = anchor

        # Create corresponding lazy manager
        lazy_manager = LazyManager(anchor)
        self.lazy_managers[dataset_id] = lazy_manager

        print(f"Dataset anchor created with {len(anchor.data_items)} items")
        return anchor

    def register_ml_simulator(self, model_name: str) -> MLFrameworkSimulator:
        """
        Register a new ML framework simulator.

        Args:
            model_name: Name of the ML model

        Returns:
            MLFrameworkSimulator instance
        """
        simulator = MLFrameworkSimulator(model_name)
        self.ml_simulators[model_name] = simulator
        print(f"Registered ML simulator: {model_name}")
        return simulator

    def create_provenance_capsules(
        self, dataset_id: str, data_items: List[Dict[str, Any]]
    ) -> List[ProvenanceCapsule]:
        """
        Create provenance capsules for a dataset using lazy materialization.

        Args:
            dataset_id: ID of the dataset
            data_items: List of data items with content and metadata

        Returns:
            List of ProvenanceCapsule instances
        """
        if dataset_id not in self.lazy_managers:
            raise ValueError(f"No lazy manager found for dataset: {dataset_id}")

        lazy_manager = self.lazy_managers[dataset_id]
        capsules = []

        print(
            f"Creating {len(data_items)} provenance capsules for dataset: {dataset_id}"
        )

        for item in data_items:
            # Create capsule using lazy manager
            capsule = lazy_manager.create_lazy_capsule(
                item_id=item["metadata"]["id"],
                original_data=item["content"],
                metadata=item["metadata"],
            )
            capsules.append(capsule)

        print(f"Created {len(capsules)} provenance capsules")
        return capsules

    def create_model_aggregation_key(
        self, model_name: str, authorized_datasets: List[str]
    ) -> ModelAggregationKey:
        """
        Create a Model Aggregation Key for training authorization.

        Args:
            model_name: Name of the model
            authorized_datasets: List of dataset IDs authorized for this model

        Returns:
            ModelAggregationKey instance
        """
        print(f"Creating MAK for model: {model_name}")
        print(f"Authorized datasets: {authorized_datasets}")

        # Create MAK with proper constructor
        mak = ModelAggregationKey(
            key_id=f"{model_name}_MAK",
            secret_material=f"secret_for_{model_name}_with_datasets_{'_'.join(authorized_datasets)}",
        )

        print(f"MAK created for model {model_name}")
        return mak

    def train_model(
        self,
        model_name: str,
        capsules: List[ProvenanceCapsule],
        mak: ModelAggregationKey,
        training_params: Dict[str, Any],
        model_version: str,
    ) -> TrainingSnapshot:
        """
        Train a model using the ML framework simulator.

        Args:
            model_name: Name of the model
            capsules: List of provenance capsules
            mak: Model Aggregation Key for authorization
            training_params: Training parameters
            model_version: Version of the model

        Returns:
            TrainingSnapshot instance
        """
        if model_name not in self.ml_simulators:
            self.register_ml_simulator(model_name)

        simulator = self.ml_simulators[model_name]

        print(f"Training model {model_name} version {model_version}")
        print(f"Using {len(capsules)} provenance capsules")

        # Train using the simulator
        snapshot = simulator.train_model(
            training_data_capsules=capsules,
            mak=mak,
            training_params=training_params,
            model_version=model_version,
        )

        return snapshot

    def validate_training_integrity(self, snapshot: TrainingSnapshot) -> bool:
        """
        Validate the integrity of a training snapshot.

        Args:
            snapshot: TrainingSnapshot to validate

        Returns:
            True if valid, False otherwise
        """
        print(f"Validating training snapshot: {snapshot.snapshot_id}")

        # Basic validation - check if snapshot exists and has valid structure
        if not snapshot.snapshot_id or not snapshot.merkle_root_hash:
            print("ERROR: Snapshot missing required fields")
            return False

        # Verify Merkle tree integrity
        if not snapshot.merkle_tree:
            print("ERROR: Snapshot missing Merkle tree")
            return False

        # Verify that the merkle root matches
        computed_root = snapshot.merkle_tree.get_root()
        if computed_root != snapshot.merkle_root_hash:
            print("ERROR: Merkle root mismatch")
            return False

        print("Training snapshot validation successful")
        return True

    def get_performance_metrics(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get performance metrics for a dataset's lazy operations.

        Args:
            dataset_id: ID of the dataset

        Returns:
            Dictionary of performance metrics
        """
        if dataset_id not in self.lazy_managers:
            return {"error": f"No lazy manager found for dataset: {dataset_id}"}

        lazy_manager = self.lazy_managers[dataset_id]

        metrics = {
            "dataset_id": dataset_id,
            "total_items": len(lazy_manager.anchor.data_items),
            "materialized_capsules": len(lazy_manager.materialized_capsules),
            "materialization_rate": (
                len(lazy_manager.materialized_capsules)
                / len(lazy_manager.anchor.data_items)
                if lazy_manager.anchor.data_items
                else 0
            ),
            "dataset_key_derived": lazy_manager.anchor.dataset_key is not None,
            "framework": self.framework_name,
        }

        return metrics
