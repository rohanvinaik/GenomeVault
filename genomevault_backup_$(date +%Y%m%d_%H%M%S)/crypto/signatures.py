"""
Digital signature utilities using Ed25519.

This module provides high-level functions for creating and verifying
digital signatures using the Ed25519 algorithm.
"""

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from typing import Optional, Tuple
import base64

from genomevault.crypto.types import (
    PrivateKey,
    PublicKey,
    Signature,
    KeyPEM,
)
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


def generate_keypair() -> Tuple[PrivateKey, PublicKey]:
    """
    Generate a new Ed25519 keypair.

    Returns:
        Tuple of (private_key, public_key)
    """
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def sign_data(private_key: PrivateKey, data: bytes) -> Signature:
    """
    Sign data using an Ed25519 private key.

    Args:
        private_key: Ed25519 private key
        data: Data to sign

    Returns:
        Digital signature
    """
    signature = private_key.sign(data)
    return signature


def verify_signature(public_key: PublicKey, signature: Signature, data: bytes) -> bool:
    """
    Verify a signature using an Ed25519 public key.

    Args:
        public_key: Ed25519 public key
        signature: Signature to verify
        data: Original data that was signed

    Returns:
        True if signature is valid, False otherwise
    """
    try:
        public_key.verify(signature, data)
        return True
    except InvalidSignature:
        return False


def export_private_key(private_key: PrivateKey, password: Optional[bytes] = None) -> KeyPEM:
    """
    Export private key to PEM format.

    Args:
        private_key: Private key to export
        password: Optional password for encryption

    Returns:
        PEM-encoded private key
    """
    if password:
        encryption = serialization.BestAvailableEncryption(password)
    else:
        encryption = serialization.NoEncryption()

    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encryption,
    )
    return pem


def export_public_key(public_key: PublicKey) -> KeyPEM:
    """
    Export public key to PEM format.

    Args:
        public_key: Public key to export

    Returns:
        PEM-encoded public key
    """
    pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return pem


def import_private_key(pem_data: KeyPEM, password: Optional[bytes] = None) -> PrivateKey:
    """
    Import private key from PEM format.

    Args:
        pem_data: PEM-encoded private key
        password: Optional password for decryption

    Returns:
        Ed25519 private key
    """
    key = serialization.load_pem_private_key(pem_data, password=password)

    if not isinstance(key, ed25519.Ed25519PrivateKey):
        raise ValueError("Key is not an Ed25519 private key")

    return key


def import_public_key(pem_data: KeyPEM) -> PublicKey:
    """
    Import public key from PEM format.

    Args:
        pem_data: PEM-encoded public key

    Returns:
        Ed25519 public key
    """
    key = serialization.load_pem_public_key(pem_data)

    if not isinstance(key, ed25519.Ed25519PublicKey):
        raise ValueError("Key is not an Ed25519 public key")

    return key


def sign_and_encode(private_key: PrivateKey, data: bytes) -> str:
    """
    Sign data and return base64-encoded signature.

    Args:
        private_key: Private key for signing
        data: Data to sign

    Returns:
        Base64-encoded signature
    """
    signature = sign_data(private_key, data)
    return base64.b64encode(signature).decode("utf-8")


def verify_encoded_signature(public_key: PublicKey, encoded_signature: str, data: bytes) -> bool:
    """
    Verify a base64-encoded signature.

    Args:
        public_key: Public key for verification
        encoded_signature: Base64-encoded signature
        data: Original data that was signed

    Returns:
        True if signature is valid, False otherwise
    """
    try:
        signature = base64.b64decode(encoded_signature)
        return verify_signature(public_key, signature, data)
    except Exception as e:
        logger.warning(f"Failed to verify signature: {e}")
        return False


class SignatureManager:
    """
    High-level manager for signature operations.
    """

    def __init__(self, private_key: Optional[PrivateKey] = None):
        """
        Initialize signature manager.

        Args:
            private_key: Optional private key to use for signing
        """
        if private_key:
            self.private_key = private_key
            self.public_key = private_key.public_key()
        else:
            self.private_key, self.public_key = generate_keypair()

    def sign(self, data: bytes) -> Signature:
        """Sign data with the manager's private key."""
        return sign_data(self.private_key, data)

    def verify(self, signature: Signature, data: bytes) -> bool:
        """Verify signature with the manager's public key."""
        return verify_signature(self.public_key, signature, data)

    def export_keys(self, password: Optional[bytes] = None) -> Tuple[KeyPEM, KeyPEM]:
        """
        Export both keys in PEM format.

        Returns:
            Tuple of (private_pem, public_pem)
        """
        private_pem = export_private_key(self.private_key, password)
        public_pem = export_public_key(self.public_key)
        return private_pem, public_pem
