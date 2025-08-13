"""
Key management utilities for GenomeVault.

This module provides utilities for managing cryptographic keys,
including key generation, storage, rotation, and derivation.
"""

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
import json
import os

from genomevault.crypto.types import (
    PrivateKey,
    PublicKey,
    KeyBytes,
    Salt,
    Ed25519Private,
)
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class KeyMetadata:
    """Metadata for a cryptographic key."""

    key_id: str
    algorithm: str
    created_at: str
    expires_at: Optional[str] = None
    purpose: Optional[str] = None
    rotated_from: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class KeyStore:
    """
    Secure key storage manager.

    Manages cryptographic keys with metadata, rotation, and persistence.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize key store.

        Args:
            storage_dir: Directory for key storage (creates temp if None)
        """
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path.home() / ".genomevault" / "keys"

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._keys: Dict[str, PrivateKey] = {}
        self._metadata: Dict[str, KeyMetadata] = {}
        self._load_metadata()

    def generate_key(
        self, key_id: str, purpose: Optional[str] = None, validity_days: int = 365
        """Generate key."""
    ) -> Ed25519Private:
        """
        Generate and store a new Ed25519 key.

        Args:
            key_id: Unique identifier for the key
            purpose: Description of key purpose
            validity_days: Days until key expires

        Returns:
            Generated private key
        """
        if key_id in self._keys:
            raise ValueError(f"Key {key_id} already exists")

        # Generate new key
        private_key = ed25519.Ed25519PrivateKey.generate()

        # Create metadata
        now = datetime.utcnow()
        expires = now + timedelta(days=validity_days)
        metadata = KeyMetadata(
            key_id=key_id,
            algorithm="Ed25519",
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
            purpose=purpose,
        )

        # Store key and metadata
        self._keys[key_id] = private_key
        self._metadata[key_id] = metadata

        # Persist to disk
        self._save_key(key_id, private_key)
        self._save_metadata()

        logger.info(f"Generated new key: {key_id}")
        return private_key

    def get_key(self, key_id: str) -> Optional[PrivateKey]:
        """
        Retrieve a key by ID.

        Args:
            key_id: Key identifier

        Returns:
            Private key if found, None otherwise
        """
        if key_id not in self._keys:
            # Try to load from disk
            self._load_key(key_id)

        return self._keys.get(key_id)

    def get_public_key(self, key_id: str) -> Optional[PublicKey]:
        """
        Get public key for a stored private key.

        Args:
            key_id: Key identifier

        Returns:
            Public key if found, None otherwise
        """
        private_key = self.get_key(key_id)
        if private_key:
            return private_key.public_key()
        return None

    def rotate_key(
        self, old_key_id: str, new_key_id: str, validity_days: int = 365
        """Rotate key."""
    ) -> Ed25519Private:
        """
        Rotate a key by generating a new one.

        Args:
            old_key_id: ID of key to rotate
            new_key_id: ID for new key
            validity_days: Validity period for new key

        Returns:
            New private key
        """
        if old_key_id not in self._metadata:
            raise ValueError(f"Key {old_key_id} not found")

        old_metadata = self._metadata[old_key_id]

        # Generate new key
        new_key = self.generate_key(
            new_key_id, purpose=old_metadata.purpose, validity_days=validity_days
        )

        # Update metadata to link keys
        self._metadata[new_key_id].rotated_from = old_key_id
        self._save_metadata()

        logger.info(f"Rotated key {old_key_id} to {new_key_id}")
        return new_key

    def delete_key(self, key_id: str) -> bool:
        """
        Delete a key from storage.

        Args:
            key_id: Key to delete

        Returns:
            True if deleted, False if not found
        """
        if key_id not in self._keys and key_id not in self._metadata:
            return False

        # Remove from memory
        self._keys.pop(key_id, None)
        self._metadata.pop(key_id, None)

        # Remove from disk
        key_file = self.storage_dir / f"{key_id}.pem"
        if key_file.exists():
            key_file.unlink()

        self._save_metadata()
        logger.info(f"Deleted key: {key_id}")
        return True

    def list_keys(self) -> List[KeyMetadata]:
        """
        List all stored keys with metadata.

        Returns:
            List of key metadata
        """
        return list(self._metadata.values())

    def _save_key(self, key_id: str, private_key: PrivateKey) -> None:
        """Save key to disk."""
        key_file = self.storage_dir / f"{key_id}.pem"

        # Use password protection (in production, use key management service)
        password = self._derive_password(key_id)

        pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(password),
        )

        key_file.write_bytes(pem)
        # Set restrictive permissions
        key_file.chmod(0o600)

    def _load_key(self, key_id: str) -> None:
        """Load key from disk."""
        key_file = self.storage_dir / f"{key_id}.pem"

        if not key_file.exists():
            return

        password = self._derive_password(key_id)
        pem = key_file.read_bytes()

        try:
            private_key = serialization.load_pem_private_key(pem, password=password)

            if isinstance(private_key, ed25519.Ed25519PrivateKey):
                self._keys[key_id] = private_key
        except Exception as e:
            logger.error(f"Failed to load key {key_id}: {e}")

    def _save_metadata(self) -> None:
        """Save metadata to disk."""
        metadata_file = self.storage_dir / "metadata.json"

        data = {key_id: meta.to_dict() for key_id, meta in self._metadata.items()}

        metadata_file.write_text(json.dumps(data, indent=2))

    def _load_metadata(self) -> None:
        """Load metadata from disk."""
        metadata_file = self.storage_dir / "metadata.json"

        if not metadata_file.exists():
            return

        try:
            data = json.loads(metadata_file.read_text())

            for key_id, meta_dict in data.items():
                self._metadata[key_id] = KeyMetadata(**meta_dict)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")

    def _derive_password(self, key_id: str) -> bytes:
        """
        Derive password for key encryption.

        In production, use a proper KMS or HSM instead.
        """
        # This is a simplified example - use proper key derivation in production
        master_secret = os.environ.get("GENOMEVAULT_MASTER_KEY", "default-insecure-key")

        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=key_id.encode("utf-8"),
            iterations=100000,
        )

        return kdf.derive(master_secret.encode("utf-8"))


def derive_key_from_password(
    password: str, salt: Optional[Salt] = None, length: int = 32
    """Derive key from password."""
) -> KeyBytes:
    """
    Derive a cryptographic key from a password.

    Args:
        password: Password to derive from
        salt: Optional salt (generated if None)
        length: Key length in bytes

    Returns:
        Derived key bytes
    """
    if salt is None:
        salt = os.urandom(16)

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        iterations=100000,
    )

    return kdf.derive(password.encode("utf-8"))
