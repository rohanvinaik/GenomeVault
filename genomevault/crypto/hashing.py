"""Cryptographic hashing utilities"""
from typing import Union, Optional
import hashlib

from genomevault.crypto.types import HashDigest, HashHex, Salt

def secure_hash(data: Union[str, bytes]) -> HashHex:
    """Create a secure SHA-256 hash of the input data.

    Args:
        data: Data to hash (string or bytes)

    Returns:
        Hexadecimal string representation of the hash
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    return hashlib.sha256(data).hexdigest()


def secure_hash_bytes(data: Union[str, bytes]) -> HashDigest:
    """Create a secure SHA-256 hash returning raw bytes.

    Args:
        data: Data to hash (string or bytes)

    Returns:
        Raw hash digest as bytes
    """
    if isinstance(data, str):
        data = data.encode("utf-8")

    return hashlib.sha256(data).digest()


def pbkdf2_hash(
    password: str, salt: Optional[Salt] = None, iterations: int = 100000
) -> tuple[HashHex, Salt]:
    """Create a PBKDF2 hash suitable for password storage.

    Args:
        password: Password to hash
        salt: Optional salt bytes (generated if not provided)
        iterations: Number of iterations for PBKDF2

    Returns:
        Tuple of (hex hash, salt bytes)
    """
    import os

    if salt is None:
        salt = os.urandom(32)

    hash_bytes = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)

    return hash_bytes.hex(), salt


def verify_hash(data: Union[str, bytes], expected_hash: HashHex) -> bool:
    """Verify that data matches an expected hash.

    Args:
        data: Data to verify
        expected_hash: Expected hash in hex format

    Returns:
        True if hash matches, False otherwise
    """
    actual_hash = secure_hash(data)

    # Use constant-time comparison to prevent timing attacks
    import hmac

    return hmac.compare_digest(actual_hash, expected_hash)
