"""
Comprehensive test suite for the CIAF Framework.

This test suite validates all components of the Cognitive Insight AI Framework (CIAF)
including cryptographic utilities, provenance capsules, model integrity components,
inference receipts, and simulators.
"""

import unittest
import os
import json
from datetime import datetime, timedelta
import time

# Import CIAF components from the package
from ciaf.core import CryptoUtils, KeyManager, MerkleTree
from ciaf.provenance import ProvenanceCapsule, ModelAggregationKey, TrainingSnapshot
from ciaf.inference import InferenceReceipt, ZKEChain
from ciaf.simulation import MLFrameworkSimulator, MockLLM
from ciaf.anchoring import DatasetAnchor, LazyManager
from ciaf.api import CIAFFramework
from ciaf.wrappers import CIAFModelWrapper


class TestCryptographicUtilities(unittest.TestCase):
    """Test cases for cryptographic utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.crypto = CryptoUtils()
        self.key_manager = KeyManager()

    def test_key_derivation(self):
        """Test key derivation functionality."""
        salt = os.urandom(16)
        password = "test_password"
        key = self.key_manager.derive_key_pbkdf2(password, salt, 32)
        
        self.assertEqual(len(key), 32)
        self.assertIsInstance(key, bytes)
        
        # Test reproducibility
        key2 = self.key_manager.derive_key_pbkdf2(password, salt, 32)
        self.assertEqual(key, key2)

    def test_encryption_decryption(self):
        """Test AES-GCM encryption and decryption."""
        key = os.urandom(32)
        plaintext = b"Hello, CIAF World!"
        
        ciphertext, nonce, tag = self.crypto.encrypt_aes_gcm(key, plaintext)
        decrypted = self.crypto.decrypt_aes_gcm(key, ciphertext, nonce, tag)
        
        self.assertEqual(plaintext, decrypted)

    def test_hashing(self):
        """Test SHA256 hashing."""
        data = b"test data"
        hash1 = self.crypto.sha256_hash(data)
        hash2 = self.crypto.sha256_hash(data)
        
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 64)  # SHA256 hex length

    def test_hmac(self):
        """Test HMAC functionality."""
        key = os.urandom(32)
        data = b"test data"
        
        hmac1 = self.crypto.hmac_sha256(key, data)
        hmac2 = self.crypto.hmac_sha256(key, data)
        
        self.assertEqual(hmac1, hmac2)
        self.assertEqual(len(hmac1), 64)  # HMAC-SHA256 hex length


class TestDatasetAnchor(unittest.TestCase):
    """Test cases for dataset anchoring functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.dataset_metadata = {
            "name": "Test Dataset",
            "version": "1.0",
            "description": "Test dataset for unit tests"
        }

    def test_dataset_anchor_creation(self):
        """Test dataset anchor creation."""
        anchor = DatasetAnchor(
            dataset_id="test_dataset",
            metadata=self.dataset_metadata,
            master_password="test_password"
        )
        
        self.assertEqual(anchor.dataset_id, "test_dataset")
        self.assertIsNotNone(anchor.dataset_key)
        self.assertIsNotNone(anchor.master_key)

    def test_data_item_addition(self):
        """Test adding data items to dataset anchor."""
        anchor = DatasetAnchor(
            dataset_id="test_dataset",
            metadata=self.dataset_metadata,
            master_password="test_password"
        )
        
        # Add a data item
        anchor.add_data_item("item_1", "test content", {"id": "item_1", "type": "test"})
        
        self.assertEqual(len(anchor.data_items), 1)
        self.assertIn("item_1", anchor.data_items)

    def test_lazy_manager(self):
        """Test lazy manager functionality."""
        anchor = DatasetAnchor(
            dataset_id="test_dataset",
            metadata=self.dataset_metadata,
            master_password="test_password"
        )
        
        lazy_manager = LazyManager(anchor)
        
        # Create lazy capsule
        capsule = lazy_manager.create_lazy_capsule(
            item_id="item_1",
            original_data="test content",
            metadata={"id": "item_1", "type": "test"}
        )
        
        self.assertIsInstance(capsule, ProvenanceCapsule)
        self.assertEqual(len(lazy_manager.materialized_capsules), 1)


class TestProvenanceCapsules(unittest.TestCase):
    """Test cases for provenance capsule functionality."""

    def test_capsule_creation(self):
        """Test provenance capsule creation."""
        capsule = ProvenanceCapsule(
            original_data="test data",
            metadata={"id": "test_1", "type": "test"},
            data_secret="test_secret"
        )
        
        self.assertEqual(capsule.original_data, "test data")
        self.assertIsNotNone(capsule.hash_proof)

    def test_capsule_verification(self):
        """Test provenance capsule verification."""
        capsule = ProvenanceCapsule(
            original_data="test data",
            metadata={"id": "test_1", "type": "test"},
            data_secret="test_secret"
        )
        
        # Capsule should verify itself
        self.assertTrue(capsule.verify_hash_proof())


class TestModelAggregationKey(unittest.TestCase):
    """Test cases for Model Aggregation Key functionality."""

    def test_mak_creation(self):
        """Test MAK creation."""
        mak = ModelAggregationKey("test_mak", "test_secret")
        
        self.assertEqual(mak.key_id, "test_mak")
        self.assertIsNotNone(mak.derived_mak_key)

    def test_data_signature(self):
        """Test data signature generation."""
        mak = ModelAggregationKey("test_mak", "test_secret")
        data_hash = "test_hash"
        
        signature = mak.generate_data_signature(data_hash)
        self.assertIsNotNone(signature)
        
        # Verify signature
        self.assertTrue(mak.verify_data_signature(data_hash, signature))


class TestTrainingSnapshot(unittest.TestCase):
    """Test cases for training snapshot functionality."""

    def test_snapshot_creation(self):
        """Test training snapshot creation."""
        capsule_hashes = ["hash1", "hash2", "hash3"]
        
        snapshot = TrainingSnapshot(
            model_version="1.0.0",
            training_parameters={"epochs": 10, "lr": 0.01},
            provenance_capsule_hashes=capsule_hashes
        )
        
        self.assertEqual(snapshot.model_version, "1.0.0")
        self.assertIsNotNone(snapshot.snapshot_id)
        self.assertIsNotNone(snapshot.merkle_root_hash)

    def test_provenance_verification(self):
        """Test provenance verification in snapshot."""
        capsule_hashes = ["hash1", "hash2", "hash3"]
        
        snapshot = TrainingSnapshot(
            model_version="1.0.0",
            training_parameters={"epochs": 10, "lr": 0.01},
            provenance_capsule_hashes=capsule_hashes
        )
        
        # Should verify hashes that were included
        self.assertTrue(snapshot.verify_provenance("hash1"))
        self.assertFalse(snapshot.verify_provenance("hash4"))


class TestInferenceReceipts(unittest.TestCase):
    """Test cases for inference receipt functionality."""

    def test_receipt_creation(self):
        """Test inference receipt creation."""
        receipt = InferenceReceipt(
            query="test query",
            ai_output="test output",
            model_version="1.0.0",
            training_snapshot_id="test_snapshot",
            training_snapshot_merkle_root="test_merkle_root"
        )
        
        self.assertEqual(receipt.query, "test query")
        self.assertEqual(receipt.ai_output, "test output")
        self.assertIsNotNone(receipt.receipt_hash)

    def test_receipt_verification(self):
        """Test receipt verification."""
        receipt = InferenceReceipt(
            query="test query",
            ai_output="test output",
            model_version="1.0.0",
            training_snapshot_id="test_snapshot",
            training_snapshot_merkle_root="test_merkle_root"
        )
        
        self.assertTrue(receipt.verify_integrity())

    def test_receipt_chaining(self):
        """Test receipt chaining functionality."""
        chain = ZKEChain()
        
        # Add multiple receipts
        receipt1 = chain.add_receipt(
            query="query 1",
            ai_output="output 1",
            model_version="1.0.0",
            training_snapshot_id="snapshot",
            training_snapshot_merkle_root="merkle_root"
        )
        
        receipt2 = chain.add_receipt(
            query="query 2",
            ai_output="output 2",
            model_version="1.0.0",
            training_snapshot_id="snapshot",
            training_snapshot_merkle_root="merkle_root"
        )
        
        self.assertEqual(len(chain.receipts), 2)
        self.assertTrue(chain.verify_chain())
        self.assertEqual(receipt2.prev_receipt_hash, receipt1.receipt_hash)


class TestCIAFFramework(unittest.TestCase):
    """Test cases for the main CIAF framework."""

    def setUp(self):
        """Set up test fixtures."""
        self.framework = CIAFFramework("TestFramework")

    def test_framework_initialization(self):
        """Test framework initialization."""
        self.assertEqual(self.framework.framework_name, "TestFramework")
        self.assertIsInstance(self.framework.crypto_utils, CryptoUtils)
        self.assertIsInstance(self.framework.key_manager, KeyManager)

    def test_dataset_anchor_creation(self):
        """Test dataset anchor creation through framework."""
        metadata = {"name": "Test Dataset", "version": "1.0"}
        
        anchor = self.framework.create_dataset_anchor(
            dataset_id="test_dataset",
            dataset_metadata=metadata,
            master_password="test_password"
        )
        
        self.assertIsInstance(anchor, DatasetAnchor)
        self.assertIn("test_dataset", self.framework.dataset_anchors)

    def test_provenance_capsule_creation(self):
        """Test provenance capsule creation through framework."""
        # First create dataset anchor
        metadata = {"name": "Test Dataset", "version": "1.0"}
        anchor = self.framework.create_dataset_anchor(
            dataset_id="test_dataset",
            dataset_metadata=metadata,
            master_password="test_password"
        )
        
        # Create data items
        data_items = [
            {"content": "item 1", "metadata": {"id": "item_1"}},
            {"content": "item 2", "metadata": {"id": "item_2"}}
        ]
        
        capsules = self.framework.create_provenance_capsules("test_dataset", data_items)
        
        self.assertEqual(len(capsules), 2)
        self.assertIsInstance(capsules[0], ProvenanceCapsule)


class TestSimulators(unittest.TestCase):
    """Test cases for ML framework simulators."""

    def test_mock_llm(self):
        """Test MockLLM functionality."""
        llm = MockLLM("TestLLM")
        
        self.assertEqual(llm.model_name, "TestLLM")
        self.assertIsNotNone(llm.model_params)

    def test_ml_framework_simulator(self):
        """Test ML framework simulator."""
        simulator = MLFrameworkSimulator("TestModel")
        
        self.assertEqual(simulator.model_name, "TestModel")
        self.assertIsInstance(simulator.llm_model, MockLLM)


class TestModelWrapper(unittest.TestCase):
    """Test cases for the CIAF model wrapper."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock ML model
        class MockModel:
            def fit(self, X, y):
                self.is_fitted = True
                return self
            
            def predict(self, X):
                return ["prediction"] * len(X)
        
        self.mock_model = MockModel()
        self.wrapper = CIAFModelWrapper(
            model=self.mock_model,
            model_name="TestWrapper"
        )

    def test_wrapper_initialization(self):
        """Test wrapper initialization."""
        self.assertEqual(self.wrapper.model_name, "TestWrapper")
        self.assertIsInstance(self.wrapper.framework, CIAFFramework)

    def test_wrapper_training(self):
        """Test wrapper training functionality."""
        training_data = [
            {"content": "item 1", "metadata": {"id": "item_1", "target": 0}},
            {"content": "item 2", "metadata": {"id": "item_2", "target": 1}}
        ]
        
        snapshot = self.wrapper.train(
            dataset_id="test_training",
            training_data=training_data,
            master_password="test_password",
            model_version="1.0.0"
        )
        
        self.assertIsInstance(snapshot, TrainingSnapshot)
        self.assertEqual(self.wrapper.model_version, "1.0.0")

    def test_wrapper_prediction(self):
        """Test wrapper prediction functionality."""
        # First train the model
        training_data = [
            {"content": "item 1", "metadata": {"id": "item_1", "target": 0}},
            {"content": "item 2", "metadata": {"id": "item_2", "target": 1}}
        ]
        
        self.wrapper.train(
            dataset_id="test_training",
            training_data=training_data,
            master_password="test_password",
            model_version="1.0.0"
        )
        
        # Make prediction
        prediction, receipt = self.wrapper.predict("test query", use_model=False)
        
        self.assertIsNotNone(prediction)
        self.assertIsInstance(receipt, InferenceReceipt)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestCryptographicUtilities,
        TestDatasetAnchor,
        TestProvenanceCapsules,
        TestModelAggregationKey,
        TestTrainingSnapshot,
        TestInferenceReceipts,
        TestCIAFFramework,
        TestSimulators,
        TestModelWrapper
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with error code if tests failed
    if not result.wasSuccessful():
        exit(1)
