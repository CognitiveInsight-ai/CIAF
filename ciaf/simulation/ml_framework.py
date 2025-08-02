"""
ML Framework simulator for testing CIAF integration.
"""

from ..provenance import ProvenanceCapsule, TrainingSnapshot, ModelAggregationKey
from .mock_llm import MockLLM


class MLFrameworkSimulator:
    """
    Simulates interactions with an ML framework (e.g., PyTorch, TensorFlow).
    Provides conceptual integration points for CIAF components.
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.current_model_version = "1.0.0"
        self.training_snapshots = {}  # Store TrainingSnapshot objects by ID
        self.llm_model = MockLLM(model_name=f"{model_name}-LLM")

    def prepare_data_for_training(self, raw_data_list: list[dict], data_secrets_map: dict) -> list[ProvenanceCapsule]:
        """
        Simulates preparing data by creating Provenance Capsules.
        
        Args:
            raw_data_list: List of dictionaries, each representing a raw data item.
            data_secrets_map: A dictionary mapping data IDs to their corresponding secrets.
            
        Returns:
            A list of ProvenanceCapsule objects.
        """
        print(f"\\n--- Simulating Data Preparation for {self.model_name} ---")
        provenance_capsules = []
        for i, data_item in enumerate(raw_data_list):
            data_id = data_item['metadata']['id']
            data_secret = data_secrets_map.get(data_id)
            if not data_secret:
                print(f"  WARNING: No secret found for data ID '{data_id}'. Skipping capsule creation.")
                continue

            capsule = ProvenanceCapsule(
                original_data=data_item['content'],
                metadata=data_item['metadata'],
                data_secret=data_secret
            )
            provenance_capsules.append(capsule)
            print(f"  Created Provenance Capsule for data ID: {data_id}")
        return provenance_capsules

    def train_model(self, training_data_capsules: list[ProvenanceCapsule], mak: ModelAggregationKey,
                    training_params: dict, model_version: str) -> TrainingSnapshot:
        """
        Simulates model training, including MAK validation and snapshot generation.
        
        Args:
            training_data_capsules: List of ProvenanceCapsule objects.
            mak: The ModelAggregationKey to use for data authorization.
            training_params: Parameters for this training run.
            model_version: The version tag for this training run.
            
        Returns:
            A TrainingSnapshot object.
        """
        print(f"\\n--- Simulating Model Training for {self.model_name} (Version {model_version}) ---")
        authorized_capsule_hashes = []
        for capsule in training_data_capsules:
            data_hash = capsule.hash_proof
            if capsule.verify_hash_proof():  # Verify internal integrity of capsule first
                authorized_capsule_hashes.append(data_hash)
                print(f"  Data capsule {capsule.metadata['id']} authorized and included.")
            else:
                print(f"  WARNING: Data capsule {capsule.metadata['id']} failed hash proof verification. Skipping.")

        if not authorized_capsule_hashes:
            raise ValueError("No authorized data capsules for training.")

        print(f"  Simulating training with {len(authorized_capsule_hashes)} authorized data items...")
        
        # Conceptual training of the LLM
        self.llm_model.conceptual_train(authorized_capsule_hashes, training_params)
        
        snapshot = TrainingSnapshot(
            model_version=model_version,
            training_parameters=training_params,
            provenance_capsule_hashes=authorized_capsule_hashes
        )
        self.training_snapshots[snapshot.snapshot_id] = snapshot
        self.current_model_version = model_version
        print(f"  Training completed. Snapshot ID: {snapshot.snapshot_id}")
        return snapshot
