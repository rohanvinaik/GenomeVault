from __future__ import annotations

"""Post Quantum module."""
from typing import Any, Dict

from genomevault.core.exceptions import GVComputeError


class PQEngine:
    """
    Deterministic post-quantum-like engine (stub):
    Encodes statement+witness, returns a byteproof with domain prefix.
    Good enough for unit tests; replace with real PQ backend later.
    """

    PREFIX = b"PQPROOF:"

    def prove(self, statement: Dict[str, Any], witness: Dict[str, Any]) -> bytes:
        """Prove.

        Args:
            statement: Statement.
            witness: Witness.

        Returns:
            bytes instance.
        """
        s = str(sorted(statement.items())).encode()
        w = str(sorted(witness.items())).encode()
        return self.PREFIX + s + b"|" + w

    def verify(self, statement: Dict[str, Any], proof: bytes) -> bool:
        """Verify.

        Args:
            statement: Statement.
            proof: Zero-knowledge proof.

        Returns:
            Boolean result.
        """
        return isinstance(proof, (bytes, bytearray)) and proof.startswith(self.PREFIX)


def prove(statement: Dict[str, Any], witness: Dict[str, Any]) -> bytes:
    """Prove.

    Args:
        statement: Statement.
        witness: Witness.

    Returns:
        bytes instance.
    """
    return PQEngine().prove(statement, witness)


def verify(statement: Dict[str, Any], proof: bytes) -> bool:
    """Verify.

    Args:
        statement: Statement.
        proof: Zero-knowledge proof.

    Returns:
        Boolean result.

    Raises:
        GVComputeError: When operation fails.
    """
    try:
        return PQEngine().verify(statement, proof)
    except Exception as e:
        raise GVComputeError(f"verify failed: {e!s}")
