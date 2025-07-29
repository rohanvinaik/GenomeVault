from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from hashlib import blake2b
from typing import Dict, Any


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


@dataclass
class Proof:
    version: str
    algo: str  # e.g., "hash-commit-v1"
    circuit_type: str
    nonce: str  # base64
    commitment: str  # base64
    proof: str  # base64 of hash(commitment || nonce)
    public_inputs: Dict[str, Any]

    def to_base64(self) -> str:
        payload = json.dumps(self.__dict__, separators=(",", ":"), sort_keys=True).encode("utf-8")
        return _b64(payload)

    @staticmethod
    def from_base64(s: str) -> "Proof":
        obj = json.loads(_b64d(s).decode("utf-8"))
        return Proof(**obj)


class ZKProofEngine:
    """Minimal, dependency-light ZK placeholder engine.

    NOTE: This is NOT a cryptographic ZK system. It simulates a proof flow using commitments.
    Replace with a real PLONK/zk-SNARK backend in production.
    """

    VERSION = "0.1.0"
    ALGO = "hash-commit-v1"

    @staticmethod
    def _commit(circuit_type: str, inputs: Dict[str, Any]) -> bytes:
        h = blake2b(digest_size=32)
        h.update(circuit_type.encode("utf-8"))
        h.update(b"::")
        h.update(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        return h.digest()

    def create_proof(self, *, circuit_type: str, inputs: Dict[str, Any]) -> Proof:
        # Derive a commitment over circuit_type and inputs
        commitment = self._commit(circuit_type, inputs)
        # Public inputs expose the commitment only (no raw inputs)
        public_inputs = {"commitment": _b64(commitment)}
        # Derive a per-proof nonce and proof hash
        nonce = blake2b(b"nonce::" + commitment, digest_size=16).digest()
        prf = blake2b(commitment + nonce, digest_size=32).digest()
        return Proof(
            version=self.VERSION,
            algo=self.ALGO,
            circuit_type=circuit_type,
            nonce=_b64(nonce),
            commitment=_b64(commitment),
            proof=_b64(prf),
            public_inputs=public_inputs,
        )

    def verify_proof(self, *, proof_data: str, public_inputs: Dict[str, Any]) -> bool:
        try:
            p = Proof.from_base64(proof_data)
        except Exception:
            return False
        if p.algo != self.ALGO or p.version != self.VERSION:
            return False
        # Ensure provided public_inputs match the embedded ones
        if public_inputs != p.public_inputs:
            return False
        # Recompute proof hash
        commitment = _b64d(p.commitment)
        nonce = _b64d(p.nonce)
        expect = blake2b(commitment + nonce, digest_size=32).digest()
        return _b64(expect) == p.proof
