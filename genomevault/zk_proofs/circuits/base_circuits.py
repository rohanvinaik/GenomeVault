"""Base Circuits module."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class FieldElement:
    """Represents a field element for zero-knowledge proofs."""

    def __init__(self, value: int, modulus: int = 2**256 - 1):
        """Initialize instance.

        Args:
            value: Value to set.
            modulus: Modulus.
        """
        self.value = value % modulus
        self.modulus = modulus

    def __add__(self, other: "FieldElement") -> "FieldElement":
        return FieldElement((self.value + other.value) % self.modulus, self.modulus)

    def __mul__(self, other: "FieldElement") -> "FieldElement":
        return FieldElement((self.value * other.value) % self.modulus, self.modulus)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FieldElement):
            return False
        return self.value == other.value and self.modulus == other.modulus

    def __repr__(self) -> str:
        """Return detailed representation for debugging."""
        return f"FieldElement({self.value})"


class BaseCircuit(ABC):
    """BaseCircuit implementation."""

    @abstractmethod
    def public_statement(self) -> Dict[str, Any]:
        """Public statement."""

    @abstractmethod
    def witness(self) -> Dict[str, Any]:
        """Witness."""

    def prove(self) -> bytes:
        """Prove.

        Returns:
            bytes instance.
        """
        # Deterministic placeholder proof
        s = str(sorted(self.public_statement().items())).encode()
        w = str(sorted(self.witness().items())).encode()
        return b"CIRCUIT:" + s + b"|" + w

    def verify(self, proof: bytes) -> bool:
        """Verify.

        Args:
            proof: Zero-knowledge proof.

        Returns:
            Boolean result.
        """
        return isinstance(proof, (bytes, bytearray)) and proof.startswith(b"CIRCUIT:")


class MerkleTreeCircuit(BaseCircuit):
    """Circuit for Merkle tree membership proofs."""

    def __init__(self, leaf: Optional[bytes] = None, root: Optional[bytes] = None):
        """Initialize instance.

        Args:
            leaf: Leaf.
            root: Root.
        """
        self.leaf = leaf or b""
        self.root = root or b""
        self.path = []

    def public_statement(self) -> Dict[str, Any]:
        """Public inputs for Merkle proof."""
        return {
            "root": self.root.hex() if self.root else "",
            "leaf_hash": hashlib.sha256(self.leaf).hexdigest() if self.leaf else "",
        }

    def witness(self) -> Dict[str, Any]:
        """Private witness for Merkle proof."""
        return {
            "leaf": self.leaf.hex() if self.leaf else "",
            "path": [p.hex() if isinstance(p, bytes) else str(p) for p in self.path],
        }

    def verify_membership(self, leaf: bytes, path: List[bytes], root: bytes) -> bool:
        """Verify Merkle tree membership."""
        self.leaf = leaf
        self.path = path
        self.root = root

        # Compute root from leaf and path
        current = hashlib.sha256(leaf).digest()
        for sibling in path:
            # Combine with sibling (order matters for real Merkle trees)
            combined = min(current, sibling) + max(current, sibling)
            current = hashlib.sha256(combined).digest()

        return current == root


class RangeProofCircuit(BaseCircuit):
    """Circuit for proving a value is within a range."""

    def __init__(self, bit_width: int = 32):
        """Initialize instance.

        Args:
            bit_width: Bit width.
        """
        self.bit_width = bit_width
        self.value = 0
        self.min_value = 0
        self.max_value = 2**bit_width - 1

    def public_statement(self) -> Dict[str, Any]:
        """Public inputs for range proof."""
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "bit_width": self.bit_width,
        }

    def witness(self) -> Dict[str, Any]:
        """Private witness for range proof."""
        return {
            "value": self.value,
            "binary_representation": bin(self.value)[2:].zfill(self.bit_width),
        }

    def prove_in_range(self, value: int, min_val: int = None, max_val: int = None) -> bool:
        """Prove that value is within the specified range."""
        self.value = value
        if min_val is not None:
            self.min_value = min_val
        if max_val is not None:
            self.max_value = max_val

        return self.min_value <= value <= self.max_value


class ComparisonCircuit(BaseCircuit):
    """Circuit for private comparison operations."""

    def __init__(self, value_a: Optional[int] = None, value_b: Optional[int] = None):
        """Initialize instance.

        Args:
            value_a: Value to set.
            value_b: Value to set.
        """
        self.value_a = value_a or 0
        self.value_b = value_b or 0
        self._result = None

    def public_statement(self) -> Dict[str, Any]:
        """Public inputs for comparison."""
        return {"comparison_type": "less_than", "result": self._result}

    def witness(self) -> Dict[str, Any]:
        """Private witness for comparison."""
        return {"value_a": self.value_a, "value_b": self.value_b}

    def compare(self, value_a: int, value_b: int) -> bool:
        """Perform private comparison."""
        self.value_a = value_a
        self.value_b = value_b
        self._result = value_a < value_b
        return self._result
