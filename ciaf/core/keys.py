"""
Key derivation and management for the Cognitive Insight AI Framework.

This module provides secure key derivation using PBKDF2HMAC and key management
utilities for the lazy capsule materialization system.
"""

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .crypto import SALT_LENGTH, hmac_sha256


def derive_key(salt: bytes, password: bytes, length: int = 32) -> bytes:
    """
    Derives a cryptographic key using PBKDF2HMAC.

    Args:
        salt: A unique salt for key derivation.
        password: The base password/secret for derivation.
        length: Desired length of the derived key in bytes (default: 32).

    Returns:
        The derived key as bytes.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        iterations=100000,  # Industry standard iterations
        backend=default_backend(),
    )
    return kdf.derive(password)


def derive_master_key(passphrase: str, salt: bytes, length: int = 32) -> bytes:
    """
    Derives a master key from a passphrase using PBKDF2HMAC.
    This is the root key for lazy capsule materialization.

    Args:
        passphrase: The model name or secret passphrase.
        salt: A unique salt for key derivation.
        length: Desired length of the derived key in bytes (default: 32).

    Returns:
        The derived master key as bytes.
    """
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        iterations=100000,  # Industry standard iterations
        backend=default_backend(),
    )
    return kdf.derive(passphrase.encode("utf-8"))


def derive_dataset_key(master_key: bytes, dataset_hash: str) -> str:
    """
    Derive a dataset key from master key and dataset hash.

    Args:
        master_key: The master key bytes.
        dataset_hash: Hash of the dataset metadata.

    Returns:
        Dataset key as hexadecimal string.
    """
    return hmac_sha256(master_key, dataset_hash.encode("utf-8"))


def derive_capsule_key(dataset_key: str, capsule_id: str) -> str:
    """
    Derive a capsule key from dataset key and capsule ID.

    Args:
        dataset_key: The dataset key.
        capsule_id: Unique identifier for the capsule.

    Returns:
        Capsule key as hexadecimal string.
    """
    return hmac_sha256(dataset_key.encode("utf-8"), capsule_id.encode("utf-8"))


class KeyManager:
    """
    Utility class for key derivation and management operations.
    """

    @staticmethod
    def derive_key_pbkdf2(
        password: str, salt: bytes, key_length: int, iterations: int = 100000
    ) -> bytes:
        """Derive a key using PBKDF2."""
        # Note: derive_key function expects (salt, password, length)
        return derive_key(salt, password.encode("utf-8"), key_length)

    @staticmethod
    def derive_master_key(password: str, salt: bytes) -> str:
        """Derive master key from password and salt."""
        return derive_master_key(password, salt)

    @staticmethod
    def derive_dataset_key(master_key: str, dataset_id: str) -> str:
        """Derive dataset key from master key and dataset ID."""
        return derive_dataset_key(master_key, dataset_id)

    @staticmethod
    def derive_capsule_key(dataset_key: str, capsule_id: str) -> str:
        """Derive capsule key from dataset key and capsule ID."""
        return derive_capsule_key(dataset_key, capsule_id)
