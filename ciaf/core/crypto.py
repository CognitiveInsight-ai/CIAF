"""
Core cryptographic utilities for the Cognitive Insight AI Framework.

This module provides essential cryptographic functions including AES-GCM encryption/decryption,
SHA256 hashing, and secure random number generation.
"""

import hashlib
import hmac
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

# --- Configuration ---
SALT_LENGTH = 16


def encrypt_aes_gcm(key: bytes, plaintext: bytes) -> tuple[bytes, bytes, bytes]:
    """
    Encrypts plaintext using AES-256 GCM.
    
    Args:
        key: 32-byte AES key.
        plaintext: Data to encrypt.
        
    Returns:
        A tuple containing (ciphertext, nonce, tag).
    """
    nonce = os.urandom(12)  # 96-bit nonce recommended for GCM
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()
    return ciphertext, nonce, encryptor.tag


def decrypt_aes_gcm(key: bytes, ciphertext: bytes, nonce: bytes, tag: bytes) -> bytes:
    """
    Decrypts ciphertext using AES-256 GCM.
    
    Args:
        key: 32-byte AES key.
        ciphertext: Encrypted data.
        nonce: Nonce used during encryption.
        tag: Authentication tag.
        
    Returns:
        Decrypted plaintext as bytes.
        
    Raises:
        InvalidTag: If the authentication tag is invalid (data tampered).
    """
    cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend())
    decryptor = cipher.decryptor()
    return decryptor.update(ciphertext) + decryptor.finalize()


def sha256_hash(data: bytes) -> str:
    """
    Generates a SHA256 hash of the given bytes data.
    
    Args:
        data: The bytes data to hash.
        
    Returns:
        The hexadecimal representation of the SHA256 hash.
    """
    return hashlib.sha256(data).hexdigest()


def hmac_sha256(key: bytes, data: bytes) -> str:
    """
    Generates an HMAC-SHA256 hash for dataset-level key derivation.
    Used in lazy capsule materialization for deriving dataset keys and capsule keys.
    
    Args:
        key: The HMAC key (e.g., master key or dataset key).
        data: The data to authenticate (e.g., dataset hash or capsule ID).
        
    Returns:
        The hexadecimal representation of the HMAC-SHA256.
    """
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def secure_random_bytes(length: int) -> bytes:
    """
    Generate cryptographically secure random bytes.
    
    Args:
        length: Number of bytes to generate.
        
    Returns:
        Random bytes.
    """
    return os.urandom(length)


class CryptoUtils:
    """
    Utility class providing cryptographic operations for CIAF.
    """
    
    @staticmethod
    def sha256_hash(data: bytes) -> str:
        """Compute SHA256 hash of data."""
        return sha256_hash(data)
    
    @staticmethod
    def hmac_sha256(key: bytes, data: bytes) -> str:
        """Compute HMAC-SHA256 of data."""
        return hmac_sha256(key, data)
    
    @staticmethod
    def encrypt_aes_gcm(key: bytes, plaintext: bytes) -> tuple[bytes, bytes, bytes]:
        """Encrypt data using AES-GCM."""
        return encrypt_aes_gcm(key, plaintext)
    
    @staticmethod
    def decrypt_aes_gcm(key: bytes, ciphertext: bytes, nonce: bytes, tag: bytes) -> bytes:
        """Decrypt data using AES-GCM."""
        return decrypt_aes_gcm(key, ciphertext, nonce, tag)
    
    @staticmethod
    def secure_random_bytes(length: int) -> bytes:
        """Generate secure random bytes."""
        return secure_random_bytes(length)
