from typing import Any, Dict, List, Optional, Tuple

"""
GenomeVault Encryption Utilities

Provides cryptographic primitives and utilities for secure data handling,
including AES-GCM encryption, homomorphic encryption support, and threshold cryptography.
"""
import time


import base64
import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

import nacl.secret
import nacl.utils
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import constant_time, hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class EncryptionKey:
    """Encryption key with metadata"""

    key_id: str
    key_material: bytes
    algorithm: str
    created_at: float
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = None


@dataclass
class ThresholdShare:
    """Share for threshold cryptography"""

    share_id: int
    share_value: bytes
    threshold: int
    total_shares: int
    checksum: str


class AESGCMCipher:
    """AES-GCM encryption/decryption with 256-bit keys"""

    KEY_SIZE = 32  # 256 bits
    NONCE_SIZE = 12  # 96 bits
    TAG_SIZE = 16  # 128 bits

    @classmethod
    def generate_key(cls) -> bytes:
        """Generate a new AES-256 key"""
        return secrets.token_bytes(cls.KEY_SIZE)

    @classmethod
    def encrypt(
        cls, plaintext: bytes, key: bytes, associated_data: Optional[bytes] = None
    ) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt data using AES-GCM

        Args:
            plaintext: Data to encrypt
            key: 256-bit encryption key
            associated_data: Additional authenticated data

        Returns:
            Tuple of (ciphertext, nonce, tag)
        """
        if len(key) != cls.KEY_SIZE:
            raise ValueError("Key must be {cls.KEY_SIZE} bytes") from e
        nonce = os.urandom(cls.NONCE_SIZE)

        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce), backend=default_backend())
        encryptor = cipher.encryptor()

        if associated_data:
            encryptor.authenticate_additional_data(associated_data)

        ciphertext = encryptor.update(plaintext) + encryptor.finalize()

        logger.debug("Encrypted {len(plaintext)} bytes")
        return ciphertext, nonce, encryptor.tag

    @classmethod
    def decrypt(
        cls,
        ciphertext: bytes,
        key: bytes,
        nonce: bytes,
        tag: bytes,
        associated_data: Optional[bytes] = None,
    ) -> bytes:
        """
        Decrypt data using AES-GCM

        Args:
            ciphertext: Encrypted data
            key: 256-bit decryption key
            nonce: Nonce used for encryption
            tag: Authentication tag
            associated_data: Additional authenticated data

        Returns:
            Decrypted plaintext
        """
        if len(key) != cls.KEY_SIZE:
            raise ValueError("Key must be {cls.KEY_SIZE} bytes") from e
        cipher = Cipher(algorithms.AES(key), modes.GCM(nonce, tag), backend=default_backend())
        decryptor = cipher.decryptor()

        if associated_data:
            decryptor.authenticate_additional_data(associated_data)

        plaintext = decryptor.update(ciphertext) + decryptor.finalize()

        logger.debug("Decrypted {len(ciphertext)} bytes")
        return plaintext

    @classmethod
    def encrypt_file(cls, input_path: Path, output_path: Path, key: bytes):
        """Encrypt a file"""
        with open(input_path, "rb") as f:
            plaintext = f.read()

        ciphertext, nonce, tag = cls.encrypt(plaintext, key)

        # Store nonce and tag with ciphertext
        with open(output_path, "wb") as f:
            f.write(nonce)
            f.write(tag)
            f.write(ciphertext)

        logger.info("Encrypted file {input_path} to {output_path}")

    @classmethod
    def decrypt_file(cls, input_path: Path, output_path: Path, key: bytes):
        """Decrypt a file"""
        with open(input_path, "rb") as f:
            nonce = f.read(cls.NONCE_SIZE)
            tag = f.read(cls.TAG_SIZE)
            ciphertext = f.read()

        plaintext = cls.decrypt(ciphertext, key, nonce, tag)

        with open(output_path, "wb") as f:
            f.write(plaintext)

        logger.info("Decrypted file {input_path} to {output_path}")


class ChaCha20Poly1305:
    """ChaCha20-Poly1305 AEAD encryption"""

    @classmethod
    def generate_key(cls) -> bytes:
        """Generate a new ChaCha20 key"""
        return nacl.utils.random(nacl.secret.SecretBox.KEY_SIZE)

    @classmethod
    def encrypt(cls, plaintext: bytes, key: bytes) -> bytes:
        """Encrypt using ChaCha20-Poly1305"""
        box = nacl.secret.SecretBox(key)
        return box.encrypt(plaintext)

    @classmethod
    def decrypt(cls, ciphertext: bytes, key: bytes) -> bytes:
        """Decrypt using ChaCha20-Poly1305"""
        box = nacl.secret.SecretBox(key)
        return box.decrypt(ciphertext)


class RSAEncryption:
    """RSA encryption for key exchange"""

    @classmethod
    def generate_keypair(cls, key_size: int = 4096) -> Tuple[bytes, bytes]:
        """Generate RSA keypair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537, key_size=key_size, backend=default_backend()
        )

        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        return private_pem, public_pem

    @classmethod
    def encrypt(cls, plaintext: bytes, public_key_pem: bytes) -> bytes:
        """Encrypt using RSA-OAEP"""
        public_key = serialization.load_pem_public_key(public_key_pem, backend=default_backend())

        ciphertext = public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return ciphertext

    @classmethod
    def decrypt(cls, ciphertext: bytes, private_key_pem: bytes) -> bytes:
        """Decrypt using RSA-OAEP"""
        private_key = serialization.load_pem_private_key(
            private_key_pem, password=None, backend=default_backend()
        )

        plaintext = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None,
            ),
        )
        return plaintext


class ThresholdCrypto:
    """Shamir's Secret Sharing for threshold cryptography"""

    PRIME = 2**256 - 189  # Large prime for GF(p)

    @classmethod
    def split_secret(cls, secret: bytes, threshold: int, total_shares: int) -> List[ThresholdShare]:
        """
        Split secret into shares using Shamir's Secret Sharing

        Args:
            secret: Secret to split
            threshold: Minimum shares needed to reconstruct
            total_shares: Total number of shares to generate

        Returns:
            List of threshold shares
        """
        if threshold > total_shares:
            raise ValueError("Threshold cannot exceed total shares") from e
        # Convert secret to integer
        secret_int = int.from_bytes(secret, "big")
        if secret_int >= cls.PRIME:
            raise ValueError("Secret too large for field") from e
        # Generate random coefficients for polynomial
        coefficients = [secret_int]
        for _ in range(threshold - 1):
            coefficients.append(secrets.randbelow(cls.PRIME))

        # Evaluate polynomial at x = 1, 2, ..., total_shares
        shares = []
        for x in range(1, total_shares + 1):
            y = cls._evaluate_polynomial(coefficients, x)

            share_data = {"x": x, "y": y, "threshold": threshold, "prime": cls.PRIME}
            share_bytes = json.dumps(share_data).encode()

            share = ThresholdShare(
                share_id=x,
                share_value=share_bytes,
                threshold=threshold,
                total_shares=total_shares,
                checksum=hashlib.sha256(share_bytes).hexdigest()[:8],
            )
            shares.append(share)

        logger.info("Split secret into {total_shares} shares (threshold={threshold})")
        return shares

    @classmethod
    def reconstruct_secret(cls, shares: List[ThresholdShare]) -> bytes:
        """
        Reconstruct secret from shares

        Args:
            shares: List of shares (at least threshold number)

        Returns:
            Reconstructed secret
        """
        if not shares:
            raise ValueError("No shares provided") from e
        threshold = shares[0].threshold
        if len(shares) < threshold:
            raise ValueError("Need at least {threshold} shares") from e
        # Verify and parse shares
        points = []
        for share in shares[:threshold]:
            # Verify checksum
            expected_checksum = hashlib.sha256(share.share_value).hexdigest()[:8]
            if not constant_time.bytes_eq(share.checksum.encode(), expected_checksum.encode()):
                raise ValueError("Invalid checksum for share {share.share_id}") from e
            # Parse share data
            share_data = json.loads(share.share_value.decode())
            points.append((share_data["x"], share_data["y"]))

        # Reconstruct using Lagrange interpolation
        secret_int = cls._lagrange_interpolation(points, 0)

        # Convert back to bytes
        byte_length = (secret_int.bit_length() + 7) // 8
        secret = secret_int.to_bytes(byte_length, "big")

        logger.info("Reconstructed secret from {len(shares)} shares")
        return secret

    @classmethod
    def _evaluate_polynomial(cls, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at x in GF(p)"""
        result = 0
        for i, coeff in enumerate(coefficients):
            result = (result + coeff * pow(x, i, cls.PRIME)) % cls.PRIME
        return result

    @classmethod
    def _lagrange_interpolation(cls, points: List[Tuple[int, int]], x: int) -> int:
        """Lagrange interpolation in GF(p)"""
        result = 0

        for i, (xi, yi) in enumerate(points):
            numerator = 1
            denominator = 1

            for j, (xj, _) in enumerate(points):
                if i != j:
                    numerator = (numerator * (x - xj)) % cls.PRIME
                    denominator = (denominator * (xi - xj)) % cls.PRIME

            # Modular inverse
            inv_denominator = pow(denominator, cls.PRIME - 2, cls.PRIME)
            term = (yi * numerator * inv_denominator) % cls.PRIME
            result = (result + term) % cls.PRIME

        return result


class KeyDerivation:
    """Key derivation functions"""

    @classmethod
    def derive_key(
        cls, password: str, salt: bytes, key_length: int = 32, iterations: int = 100000
    ) -> bytes:
        """Derive key from password using PBKDF2"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            iterations=iterations,
            backend=default_backend(),
        )
        return kdf.derive(password.encode())

    @classmethod
    def derive_key_hkdf(cls, input_key: bytes, info: bytes, key_length: int = 32) -> bytes:
        """Derive key using HKDF"""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=None,
            info=info,
            backend=default_backend(),
        )
        return hkdf.derive(input_key)


class SecureRandom:
    """Secure random number generation"""

    @classmethod
    def generate_nonce(cls, size: int = 12) -> bytes:
        """Generate cryptographically secure nonce"""
        return os.urandom(size)

    @classmethod
    def generate_salt(cls, size: int = 16) -> bytes:
        """Generate salt for key derivation"""
        return os.urandom(size)

    @classmethod
    def generate_token(cls, size: int = 32) -> str:
        """Generate secure random token"""
        return secrets.token_urlsafe(size)

    @classmethod
    def secure_compare(cls, a: bytes, b: bytes) -> bool:
        """Constant-time comparison"""
        return constant_time.bytes_eq(a, b)


class HomomorphicHelper:
    """Helper for homomorphic encryption operations"""

    @classmethod
    def prepare_for_homomorphic(cls, value: int, bit_length: int = 64) -> bytes:
        """Prepare integer for homomorphic encryption"""
        # Ensure value fits in specified bits
        if value >= 2**bit_length:
            raise ValueError("Value too large for {bit_length} bits") from e
        # Convert to bytes with padding
        return value.to_bytes(bit_length // 8, "little")

    @classmethod
    def extract_from_homomorphic(cls, encrypted_result: bytes) -> int:
        """Extract integer from homomorphic result"""
        # This is a placeholder - actual implementation depends on HE library
        return int.from_bytes(encrypted_result[:8], "little")


class EncryptionManager:
    """Manages encryption keys and operations"""

    def __init__(self, key_store_path: Optional[Path] = None):
        """Initialize encryption manager"""
        self.key_store_path = key_store_path or Path.home() / ".genomevault" / "keys"
        self.key_store_path.mkdir(parents=True, exist_ok=True)
        self._keys: Dict[str, EncryptionKey] = {}
        self._load_keys()

    def generate_key(self, key_id: str, algorithm: str = "AES-GCM") -> EncryptionKey:
        """Generate and store new encryption key"""
        if algorithm == "AES-GCM":
            key_material = AESGCMCipher.generate_key()
        elif algorithm == "ChaCha20-Poly1305":
            key_material = ChaCha20Poly1305.generate_key()
        else:
            raise ValueError("Unsupported algorithm: {algorithm}") from e
        key = EncryptionKey(
            key_id=key_id,
            key_material=key_material,
            algorithm=algorithm,
            created_at=os.time(),
            metadata={},
        )

        self._keys[key_id] = key
        self._save_keys()

        logger.info("Generated new {algorithm} key: {key_id}")
        return key

    def encrypt_data(self, data: bytes, key_id: str) -> Dict[str, Any]:
        """Encrypt data using specified key"""
        if key_id not in self._keys:
            raise ValueError("Key not found: {key_id}") from e
        key = self._keys[key_id]

        if key.algorithm == "AES-GCM":
            ciphertext, nonce, tag = AESGCMCipher.encrypt(data, key.key_material)
            return {
                "ciphertext": base64.b64encode(ciphertext).decode(),
                "nonce": base64.b64encode(nonce).decode(),
                "tag": base64.b64encode(tag).decode(),
                "key_id": key_id,
                "algorithm": key.algorithm,
            }
        elif key.algorithm == "ChaCha20-Poly1305":
            ciphertext = ChaCha20Poly1305.encrypt(data, key.key_material)
            return {
                "ciphertext": base64.b64encode(ciphertext).decode(),
                "key_id": key_id,
                "algorithm": key.algorithm,
            }
        else:
            raise ValueError("Unsupported algorithm: {key.algorithm}") from e
    def decrypt_data(self, encrypted_data: Dict[str, Any]) -> bytes:
        """Decrypt data"""
        key_id = encrypted_data["key_id"]
        if key_id not in self._keys:
            raise ValueError("Key not found: {key_id}") from e
        key = self._keys[key_id]

        if key.algorithm == "AES-GCM":
            ciphertext = base64.b64decode(encrypted_data["ciphertext"])
            nonce = base64.b64decode(encrypted_data["nonce"])
            tag = base64.b64decode(encrypted_data["tag"])
            return AESGCMCipher.decrypt(ciphertext, key.key_material, nonce, tag)
        if key.algorithm == "ChaCha20-Poly1305":
            ciphertext = base64.b64decode(encrypted_data["ciphertext"])
            return ChaCha20Poly1305.decrypt(ciphertext, key.key_material)
        else:
            raise ValueError("Unsupported algorithm: {key.algorithm}") from e
    def _load_keys(self):
        """Load keys from storage"""
        # In production, keys should be stored in HSM or secure key storage
        # This is a simplified implementation
        key_file = self.key_store_path / "keys.json"
        if key_file.exists():
            # This would be encrypted in production
            logger.warning("Loading keys from unencrypted storage - use HSM in production")

    def _save_keys(self):
        """Save keys to storage"""
        # In production, keys should be stored in HSM or secure key storage
        logger.warning("Saving keys to unencrypted storage - use HSM in production")


# Convenience functions
def generate_secure_key(algorithm: str = "AES-GCM") -> bytes:
    """Generate a secure encryption key"""
    if algorithm == "AES-GCM":
        return AESGCMCipher.generate_key()
    if algorithm == "ChaCha20-Poly1305":
        return ChaCha20Poly1305.generate_key()
    else:
        raise ValueError("Unsupported algorithm: {algorithm}") from e
def secure_hash(data: bytes, algorithm: str = "SHA256") -> str:
    """Compute secure hash of data"""
    if algorithm == "SHA256":
        return hashlib.sha256(data).hexdigest()
    if algorithm == "SHA3-256":
        return hashlib.sha3_256(data).hexdigest()
    if algorithm == "BLAKE2b":
        return hashlib.blake2b(data).hexdigest()
    else:
        raise ValueError("Unsupported hash algorithm: {algorithm}") from e