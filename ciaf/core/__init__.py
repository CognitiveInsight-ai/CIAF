"""
Core cryptographic and foundational components for CIAF.
"""

from .crypto import (
    SALT_LENGTH,
    CryptoUtils,
    decrypt_aes_gcm,
    encrypt_aes_gcm,
    hmac_sha256,
    secure_random_bytes,
    sha256_hash,
)
from .keys import (
    KeyManager,
    derive_capsule_key,
    derive_dataset_key,
    derive_key,
    derive_master_key,
)
from .merkle import MerkleTree

__all__ = [
    # Crypto utilities
    "encrypt_aes_gcm",
    "decrypt_aes_gcm",
    "sha256_hash",
    "hmac_sha256",
    "secure_random_bytes",
    "SALT_LENGTH",
    "CryptoUtils",
    # Key derivation
    "derive_key",
    "derive_master_key",
    "derive_dataset_key",
    "derive_capsule_key",
    "KeyManager",
    # Merkle tree
    "MerkleTree",
]
