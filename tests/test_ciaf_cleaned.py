"""
CIAF Test Suite

Comprehensive tests for the Cognitive Insight AI Framework.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from ciaf.anchoring import DatasetAnchor
from ciaf.core import CryptoUtils, KeyManager
from ciaf.metadata_integration import ModelMetadataManager
from ciaf.metadata_storage import MetadataStorage


class TestCIAFCore(unittest.TestCase):
    """Test core CIAF functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.key_manager = KeyManager()

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_crypto_utils(self):
        """Test cryptographic utilities."""
        # Test hash generation
        test_data = b"test data"
        hash1 = CryptoUtils.sha256_hash(test_data)
        hash2 = CryptoUtils.sha256_hash(test_data)

        self.assertEqual(hash1, hash2, "Hash should be deterministic")
        self.assertEqual(len(hash1), 64, "SHA256 hash should be 64 hex characters")

    def test_key_manager(self):
        """Test key management functionality."""
        # Test key derivation
        from ciaf.core.keys import derive_key

        password = b"test password"
        salt = b"test salt" + b"\x00" * 8  # Pad to 16 bytes
        key = derive_key(salt, password)

        self.assertIsNotNone(key)
        self.assertEqual(len(key), 32, "Key should be 32 bytes")

    def test_dataset_anchor(self):
        """Test dataset anchoring functionality."""
        test_data = [1, 2, 3, 4, 5]
        anchor = DatasetAnchor("test_dataset", {"data": test_data})

        # Test anchor properties
        self.assertEqual(anchor.dataset_id, "test_dataset")
        self.assertIsNotNone(anchor.anchor_hash)
        self.assertIsNotNone(anchor.timestamp)


class TestMetadataStorage(unittest.TestCase):
    """Test metadata storage functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.storage = MetadataStorage(self.temp_dir, backend="json")

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_storage_initialization(self):
        """Test storage initialization."""
        self.assertTrue(Path(self.temp_dir).exists())
        self.assertEqual(self.storage.backend, "json")

    def test_metadata_operations(self):
        """Test basic metadata operations."""
        # Test storing metadata
        test_metadata = {
            "model_id": "test_model",
            "version": "1.0.0",
            "timestamp": "2025-01-01T00:00:00Z",
            "metrics": {"accuracy": 0.95},
        }

        self.storage.save_metadata("test_model", test_metadata)

        # Test retrieving metadata
        retrieved = self.storage.get_metadata("test_model")
        self.assertEqual(retrieved["model_id"], "test_model")
        self.assertEqual(retrieved["metrics"]["accuracy"], 0.95)

    def test_pipeline_trace(self):
        """Test pipeline trace functionality."""
        # Create a simple pipeline trace
        events = [
            {"stage": "data_ingestion", "timestamp": "2025-01-01T00:00:00Z"},
            {"stage": "preprocessing", "timestamp": "2025-01-01T00:01:00Z"},
            {"stage": "training", "timestamp": "2025-01-01T00:02:00Z"},
        ]

        for event in events:
            self.storage.log_event("test_pipeline", event)

        # Retrieve trace
        trace = self.storage.get_pipeline_trace("test_pipeline")
        self.assertEqual(len(trace), 3)
        self.assertEqual(trace[0]["stage"], "data_ingestion")


class TestMetadataIntegration(unittest.TestCase):
    """Test metadata integration functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelMetadataManager("test_model", "1.0.0")

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_manager_initialization(self):
        """Test metadata manager initialization."""
        self.assertEqual(self.manager.model_name, "test_model")
        self.assertEqual(self.manager.version, "1.0.0")

    def test_training_lifecycle(self):
        """Test training lifecycle tracking."""
        # Log training start
        training_config = {
            "algorithm": "RandomForest",
            "parameters": {"n_estimators": 100},
        }
        self.manager.log_training_start(training_config)

        # Log training completion
        training_results = {"accuracy": 0.94, "training_time": 300}
        self.manager.log_training_complete(training_results)

        # Get pipeline trace
        trace = self.manager.get_pipeline_trace()

        # Verify events were logged
        training_start_events = [
            e for e in trace if e.get("event_type") == "training_start"
        ]
        training_complete_events = [
            e for e in trace if e.get("event_type") == "training_complete"
        ]

        self.assertEqual(len(training_start_events), 1)
        self.assertEqual(len(training_complete_events), 1)
        self.assertEqual(
            training_start_events[0]["details"]["algorithm"], "RandomForest"
        )
        self.assertEqual(training_complete_events[0]["details"]["accuracy"], 0.94)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
