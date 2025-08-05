"""
Simple CIAF Integration Test

Tests basic CIAF functionality without complex API dependencies.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from ciaf.core.crypto import CryptoUtils, sha256_hash
from ciaf.metadata_storage import MetadataStorage


class TestCIAFBasic(unittest.TestCase):
    """Test basic CIAF functionality."""

    def test_crypto_operations(self):
        """Test basic cryptographic operations."""
        # Test hash generation
        test_data = b"test data"
        hash1 = sha256_hash(test_data)
        hash2 = sha256_hash(test_data)

        self.assertEqual(hash1, hash2, "Hash should be deterministic")
        self.assertEqual(len(hash1), 64, "SHA256 hash should be 64 hex characters")

        # Test CryptoUtils wrapper
        hash3 = CryptoUtils.sha256_hash(test_data)
        self.assertEqual(hash1, hash3, "CryptoUtils should produce same hash")

    def test_metadata_storage_basic(self):
        """Test basic metadata storage operations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = MetadataStorage(temp_dir, backend="json")

            # Test basic metadata saving and retrieval
            test_metadata = {
                "test_key": "test_value",
                "timestamp": "2025-01-01T00:00:00Z",
            }

            # Use the correct API signature
            metadata_id = storage.save_metadata(
                model_name="test_model",
                stage="testing",
                event_type="test_event",
                metadata=test_metadata,
            )

            retrieved = storage.get_metadata(metadata_id)

            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved["metadata"]["test_key"], "test_value")

    def test_imports(self):
        """Test that all main CIAF modules can be imported."""
        try:
            from ciaf import (
                CIAFFramework,
                CryptoUtils,
                DatasetAnchor,
                KeyManager,
                MerkleTree,
                MetadataStorage,
                ModelMetadataManager,
            )

            # Basic instantiation tests
            crypto = CryptoUtils()  # Static class, should work
            storage = MetadataStorage(".", backend="json")
            manager = ModelMetadataManager("test", "1.0.0")

            self.assertIsNotNone(crypto)
            self.assertIsNotNone(storage)
            self.assertIsNotNone(manager)

        except ImportError as e:
            self.fail(f"Failed to import CIAF modules: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
