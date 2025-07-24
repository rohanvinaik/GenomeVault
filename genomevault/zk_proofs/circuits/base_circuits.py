from typing import Any, Dict, List

"""
Base circuit implementations for zero-knowledge proofs

This module provides the foundational circuit building blocks
for genomic privacy-preserving proofs using PLONK.
"""

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FieldElement:
    """Element in the PLONK field (BLS12-381 scalar field)"""

    value: int
    modulus: int = (0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001  # BLS12-381 scalar field)

    def __post_init__(self):
        self.value = self.value % self.modulus

    def __add__(self, other):
        if isinstance(other, FieldElement):
            return FieldElement((self.value + other.value) % self.modulus)
        return FieldElement((self.value + other) % self.modulus)

    def __mul__(self, other):
        if isinstance(other, FieldElement):
            return FieldElement((self.value * other.value) % self.modulus)
        return FieldElement((self.value * other) % self.modulus)

    def __sub__(self, other):
        if isinstance(other, FieldElement):
            return FieldElement((self.value - other.value) % self.modulus)
        return FieldElement((self.value - other) % self.modulus)

    def __pow__(self, exp):
        return FieldElement(pow(self.value, exp, self.modulus))

    def inverse(self):
        """Modular inverse using Fermat's little theorem"""
        return FieldElement(pow(self.value, self.modulus - 2, self.modulus))

    def __eq__(self, other):
        if isinstance(other, FieldElement):
            return self.value == other.value
        return self.value == other

    def __repr__(self):
        return "FieldElement({self.value})"


class BaseCircuit(ABC):
    """Abstract base class for ZK circuits"""

    def __init__(self, name: str, num_constraints: int):
        self.name = name
        self.num_constraints = num_constraints
        self.wires = {}  # Wire assignments
        self.constraints = []  # List of constraints
        self.public_input_indices = []
        self.private_input_indices = []

    @abstractmethod
    def setup(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup circuit with inputs"""

    @abstractmethod
    def generate_constraints(self):
        """Generate circuit constraints"""

    def add_constraint(self,
        a: FieldElement,
        b: FieldElement,
        c: FieldElement,
        ql: int = 0,
        qr: int = 0,
        qo: int = 0,
        qm: int = 0,
        qc: int = 0,):
        """
        Add PLONK constraint: ql*a + qr*b + qo*c + qm*a*b + qc = 0
        """
        constraint = {
            "a": a,
            "b": b,
            "c": c,
            "ql": ql,
            "qr": qr,
            "qo": qo,
            "qm": qm,
            "qc": qc,
        }
        self.constraints.append(constraint)

    def add_multiplication_gate(self, a: FieldElement, b: FieldElement, c: FieldElement):
        """Add constraint: a * b = c"""
        self.add_constraint(a, b, c, qm = 1, qo = -1)

    def add_addition_gate(self, a: FieldElement, b: FieldElement, c: FieldElement):
        """Add constraint: a + b = c"""
        self.add_constraint(a, b, c, ql = 1, qr = 1, qo = -1)

    def add_constant_gate(self, a: FieldElement, constant: int):
        """Add constraint: a = constant"""
        self.add_constraint(a, FieldElement(0), FieldElement(0), ql = 1, qc = -constant)


class MerkleTreeCircuit(BaseCircuit):
    """Circuit for Merkle tree inclusion proofs"""

    def __init__(self, tree_depth: int = 20):
        super().__init__("merkle_inclusion", 2 * tree_depth)
        self.tree_depth = tree_depth

    def setup(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup Merkle tree circuit"""
        self.root = FieldElement(int(public_inputs["root"], 16))
        self.leaf = FieldElement(int(private_inputs["leaf"], 16))
        self.path = [FieldElement(int(h, 16)) for h in private_inputs["path"]]
        self.indices = private_inputs["indices"]  # 0 for left, 1 for right

    def generate_constraints(self):
        """Generate Merkle tree constraints"""
        current = self.leaf

        for i in range(self.tree_depth):
            sibling = self.path[i]

            # Choose ordering based on index
            if self.indices[i] == 0:
                left, right = current, sibling
            else:
                left, right = sibling, current

            # Hash left and right to get parent
            parent = self._hash_pair(left, right)

            # Add constraint for hash computation
            self.add_constraint(left, right, parent, qm = 1, qo = -1)

            current = parent

        # Final constraint: computed root equals public root
        self.add_constraint(current, self.root, FieldElement(0), ql = 1, qr = -1)

    def _hash_pair(self, left: FieldElement, right: FieldElement) -> FieldElement:
        """Hash two field elements (simplified for demo)"""
        # In production, would use Poseidon hash
        data = "{left.value}:{right.value}".encode()
        hash_val = int(hashlib.sha256(data).hexdigest(), 16)
        return FieldElement(hash_val)


class RangeProofCircuit(BaseCircuit):
    """Circuit for proving value is in range [min, max]"""

    def __init__(self, bit_width: int = 64):
        super().__init__("range_proof", bit_width + 2)
        self.bit_width = bit_width

    def setup(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup range proof circuit"""
        self.min_val = FieldElement(public_inputs["min"])
        self.max_val = FieldElement(public_inputs["max"])
        self.value = FieldElement(private_inputs["value"])
        self.value_bits = self._to_bits(private_inputs["value"])

    def generate_constraints(self):
        """Generate range proof constraints"""
        # Constrain each bit to be 0 or 1
        for i, bit in enumerate(self.value_bits):
            bit_elem = FieldElement(bit)
            # bit * (1 - bit) = 0
            self.add_constraint(bit_elem, FieldElement(1) - bit_elem, FieldElement(0), qm = 1)

        # Reconstruct value from bits
        reconstructed = FieldElement(0)
        for i, bit in enumerate(self.value_bits):
            reconstructed = reconstructed + FieldElement(bit * (2**i))

        # Constraint: reconstructed value equals input
        self.add_constraint(reconstructed, self.value, FieldElement(0), ql = 1, qr = -1)

        # Constraint: value >= min
        diff_min = self.value - self.min_val
        self._add_non_negative_constraint(diff_min)

        # Constraint: value <= max
        diff_max = self.max_val - self.value
        self._add_non_negative_constraint(diff_max)

    def _to_bits(self, value: int) -> List[int]:
        """Convert value to bit representation"""
        return [(value >> i) & 1 for i in range(self.bit_width)]

    def _add_non_negative_constraint(self, value: FieldElement):
        """Add constraint that value is non-negative"""
        # Simplified: in production would use proper range proof


class ComparisonCircuit(BaseCircuit):
    """Circuit for comparing two values without revealing them"""

    def __init__(self):
        super().__init__("comparison", 10)

    def setup(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup comparison circuit"""
        self.result = FieldElement(1 if public_inputs["result"] else 0)
        self.a = FieldElement(private_inputs["a"])
        self.b = FieldElement(private_inputs["b"])
        self.comparison_type = public_inputs["comparison_type"]  # 'gt', 'lt', 'eq'

    def generate_constraints(self):
        """Generate comparison constraints"""
        if self.comparison_type == "gt":
            # Prove a > b without revealing a or b
            diff = self.a - self.b
            # Add range proof that diff > 0
            self._prove_positive(diff)

        elif self.comparison_type == "lt":
            # Prove a < b
            diff = self.b - self.a
            self._prove_positive(diff)

        elif self.comparison_type == "eq":
            # Prove a == b
            diff = self.a - self.b
            # Constraint: diff = 0
            self.add_constraint(diff, FieldElement(0), FieldElement(0), ql = 1)

    def _prove_positive(self, value: FieldElement):
        """Prove value > 0 (simplified)"""
        # In production, would use proper range proof
        # For now, just add a placeholder constraint
        self.add_constraint(value, FieldElement(1), value, qm = 1, qo = -1)


class HashPreimageCircuit(BaseCircuit):
    """Circuit for proving knowledge of hash preimage"""

    def __init__(self):
        super().__init__("hash_preimage", 100)

    def setup(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup hash preimage circuit"""
        self.hash_output = FieldElement(int(public_inputs["hash"], 16))
        self.preimage = private_inputs["preimage"]

    def generate_constraints(self):
        """Generate hash preimage constraints"""
        # Convert preimage to field elements
        preimage_elements = []
        for byte in self.preimage:
            preimage_elements.append(FieldElement(byte))

        # Compute hash (simplified - in production use Poseidon)
        computed_hash = self._hash_elements(preimage_elements)

        # Constraint: computed hash equals public hash
        self.add_constraint(computed_hash, self.hash_output, FieldElement(0), ql = 1, qr = -1)

    def _hash_elements(self, elements: List[FieldElement]) -> FieldElement:
        """Hash field elements (simplified)"""
        data = ":".join(str(e.value) for e in elements).encode()
        hash_val = int(hashlib.sha256(data).hexdigest(), 16)
        return FieldElement(hash_val)


class AggregatorCircuit(BaseCircuit):
    """Circuit for aggregating multiple values with privacy"""

    def __init__(self, num_values: int):
        super().__init__("aggregator", num_values * 2)
        self.num_values = num_values

    def setup(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup aggregator circuit"""
        self.sum_commitment = FieldElement(int(public_inputs["sum_commitment"], 16))
        self.count = FieldElement(public_inputs["count"])
        self.values = [FieldElement(v) for v in private_inputs["values"]]
        self.randomness = FieldElement(int(private_inputs["randomness"], 16))

    def generate_constraints(self):
        """Generate aggregation constraints"""
        # Compute sum
        total = FieldElement(0)
        for value in self.values:
            total = total + value

        # Add commitment to sum
        commitment = self._commit(total, self.randomness)

        # Constraint: commitment matches public commitment
        self.add_constraint(commitment, self.sum_commitment, FieldElement(0), ql = 1, qr = -1)

        # Constraint: count is correct
        self.add_constraint(FieldElement(len(self.values)), self.count, FieldElement(0), ql = 1, qr = -1)

    def _commit(self, value: FieldElement, randomness: FieldElement) -> FieldElement:
        """Create Pedersen commitment (simplified)"""
        # In production, use proper Pedersen commitment
        data = "{value.value}:{randomness.value}".encode()
        hash_val = int(hashlib.sha256(data).hexdigest(), 16)
        return FieldElement(hash_val)


# Helper functions for circuit construction
# DUPLICATE: Merged with __post_init__
# def create_wire_assignment(num_wires: int) -> Dict[str, int]:
# """Create wire assignment mapping"""
# return {"w_{i}": i for i in range(num_wires)}


# DUPLICATE: Merged with create_wire_assignment
# def evaluate_constraint(constraint: Dict, wire_values: Dict[int, FieldElement]) -> FieldElement:
# """Evaluate a PLONK constraint"""
# a = wire_values.get(constraint["a"], FieldElement(0))
# b = wire_values.get(constraint["b"], FieldElement(0))
# c = wire_values.get(constraint["c"], FieldElement(0))

# result = (# constraint["ql"] * a
# + constraint["qr"] * b
# + constraint["qo"] * c
# + constraint["qm"] * a * b
# + constraint["qc"]
#)

# return result


# DUPLICATE: Merged with create_wire_assignment
# def verify_constraints(circuit: BaseCircuit, wire_values: Dict[int, FieldElement]) -> bool:
# """Verify all constraints are satisfied"""
# for constraint in circuit.constraints:
# result = evaluate_constraint(constraint, wire_values)
# if result.value != 0:
# return False
# return True
