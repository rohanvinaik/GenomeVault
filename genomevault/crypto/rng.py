# genomevault/crypto/rng.py
from __future__ import annotations
import os
from hashlib import shake_256


def secure_bytes(n: int) -> bytes:
    return os.urandom(n)


def xof(label: bytes, seed: bytes, out_len: int) -> bytes:
    """
    SHAKE256 XOF: X = SHAKE256(label || seed).read(out_len)
    """
    sh = shake_256()
    sh.update(len(label).to_bytes(4, "big"))
    sh.update(label)
    sh.update(len(seed).to_bytes(4, "big"))
    sh.update(seed)
    return sh.digest(out_len)


def xof_uint_mod(label: bytes, seed: bytes, modulus: int) -> int:
    # 8 bytes → 64-bit — iterate if you need many samples; caller ensures uniqueness
    r = int.from_bytes(xof(label, seed, 8), "big")
    return r % modulus
