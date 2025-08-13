"""
Type aliases for cryptographic operations.

This module provides narrow type aliases for keys, signatures, and other
cryptographic primitives to improve type safety and code clarity.
"""
from typing import TypeAlias, Union

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.rsa import (
    RSAPrivateKey,
    RSAPublicKey,
)
from cryptography.hazmat.primitives.asymmetric.ec import (
    EllipticCurvePrivateKey,
    EllipticCurvePublicKey,
)

# Ed25519 key types (preferred for signatures)
PublicKey: TypeAlias = Ed25519PublicKey
PrivateKey: TypeAlias = Ed25519PrivateKey

# Alternative naming for clarity
Ed25519Private: TypeAlias = Ed25519PrivateKey
Ed25519Public: TypeAlias = Ed25519PublicKey

# RSA key types (for legacy compatibility)
RSAPrivate: TypeAlias = RSAPrivateKey
RSAPublic: TypeAlias = RSAPublicKey

# Elliptic curve key types
ECPrivate: TypeAlias = EllipticCurvePrivateKey
ECPublic: TypeAlias = EllipticCurvePublicKey

# Union types for flexibility
AnyPrivateKey: TypeAlias = Union[Ed25519PrivateKey, RSAPrivateKey, EllipticCurvePrivateKey]
AnyPublicKey: TypeAlias = Union[Ed25519PublicKey, RSAPublicKey, EllipticCurvePublicKey]

# Signature and hash types
Signature: TypeAlias = bytes
HashDigest: TypeAlias = bytes
HashHex: TypeAlias = str

# Key material types
KeyBytes: TypeAlias = bytes
KeyHex: TypeAlias = str
KeyPEM: TypeAlias = bytes

# Encryption types
Ciphertext: TypeAlias = bytes
Plaintext: TypeAlias = bytes
Nonce: TypeAlias = bytes
Salt: TypeAlias = bytes

# ZK proof related types
ProofBytes: TypeAlias = bytes
ProofHex: TypeAlias = str
Commitment: TypeAlias = bytes
Challenge: TypeAlias = bytes
Response: TypeAlias = bytes
