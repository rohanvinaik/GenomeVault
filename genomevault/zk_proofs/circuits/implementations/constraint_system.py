from typing import Any, Dict, List, Optional, Union

"""
Core constraint system for ZK proofs

This module implements the fundamental constraint generation and solving
for PLONK-style arithmetic circuits.
"""

import hashlib
from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ConstraintType(Enum):
    """Types of arithmetic constraints"""

    ADD = "add"
    MUL = "mul"
    BOOL = "bool"
    RANGE = "range"
    HASH = "hash"
    COMMITMENT = "commitment"


@dataclass
class FieldElement:
    """
    Element in the BLS12-381 scalar field
    Implements proper modular arithmetic for ZK proofs
    """

    # BLS12-381 scalar field modulus
    MODULUS = 0x73EDA753299D7D483339D80809A1D80553BDA402FFFE5BFEFFFFFFFF00000001

    value: int = 0

    def __post_init__(self):
        """Magic method implementation."""
        self.value = self.value % self.MODULUS

    """Magic method implementation."""
    def __add__(self, other):
        if isinstance(other, FieldElement):
            return FieldElement((self.value + other.value) % self.MODULUS)
        return FieldElement((self.value + other) % self.MODULUS)
            """Magic method implementation."""

    def __sub__(self, other):
        if isinstance(other, FieldElement):
            return FieldElement((self.value - other.value) % self.MODULUS)
                """Magic method implementation."""
        return FieldElement((self.value - other) % self.MODULUS)

    def __mul__(self, other):
        if isinstance(other, FieldElement):
            """Magic method implementation."""
            return FieldElement((self.value * other.value) % self.MODULUS)
        return FieldElement((self.value * other) % self.MODULUS)

    def __pow__(self, exp):
        return FieldElement(pow(self.value, exp, self.MODULUS))

    def inverse(self):
        """Modular inverse using Fermat's little theorem"""
            """Magic method implementation."""
        if self.value == 0:
            raise ValueError("Cannot invert zero") from e
        return FieldElement(pow(self.value, self.MODULUS - 2, self.MODULUS))

    """Magic method implementation."""
    def __truediv__(self, other):
        if isinstance(other, FieldElement):
            return self * other.inverse()
        return self * FieldElement(other).inverse()
            """Magic method implementation."""

    def __eq__(self, other):
        """Magic method implementation."""
        if isinstance(other, FieldElement):
            return self.value == other.value
                """Magic method implementation."""
        return self.value == (other % self.MODULUS)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return f"FieldElement({self.value})"

    def __str__(self):
        return str(self.value)

    def to_bytes(self) -> bytes:
        """Convert to 32-byte representation"""
        return self.value.to_bytes(32, byteorder = "big")

    @classmethod
    def from_bytes(cls, data: bytes) -> "FieldElement":
        """Create from 32-byte representation"""
        return cls(int.from_bytes(data, byteorder = "big"))

    @classmethod
    def random(cls) -> "FieldElement":
        """Generate random field element"""
        return cls(np.random.randint(0, cls.MODULUS))

    def is_zero(self) -> bool:
        return self.value == 0

    def is_one(self) -> bool:
        return self.value == 1


    """Magic method implementation."""
@dataclass
class Variable:
    """Variable in the constraint system"""

    index: int
    value: Optional[FieldElement] = None
    is_public: bool = False
    label: Optional[str] = None

    def __hash__(self):
        """Magic method implementation."""
        return hash(self.index)


@dataclass
class LinearCombination:
    """Linear combination of variables with coefficients"""

    terms: Dict[Variable, FieldElement] = field(default_factory = dict)
    constant: FieldElement = field(default_factory = FieldElement)

    def __add__(self, other):
        if isinstance(other, LinearCombination):
            new_terms = self.terms.copy()
            for var, coeff in other.terms.items():
                if var in new_terms:
                    new_terms[var] = new_terms[var] + coeff
                else:
                    new_terms[var] = coeff
            return LinearCombination(new_terms, self.constant + other.constant)
        if isinstance(other, Variable):
            """Magic method implementation."""
            new_terms = self.terms.copy()
            if other in new_terms:
                new_terms[other] = new_terms[other] + FieldElement(1)
            else:
                new_terms[other] = FieldElement(1)
            return LinearCombination(new_terms, self.constant)
        if isinstance(other, (int, FieldElement)):
            """Magic method implementation."""
            return LinearCombination(self.terms.copy(), self.constant + FieldElement(other))
        else:
            raise TypeError(f"Cannot add {type(other)} to LinearCombination")

    def __mul__(self, scalar):
        if isinstance(scalar, (int, FieldElement)):
            scalar = FieldElement(scalar)
            new_terms = {var: coeff * scalar for var, coeff in self.terms.items()}
            return LinearCombination(new_terms, self.constant * scalar)
        else:
            raise TypeError(f"Cannot multiply LinearCombination by {type(scalar)}")

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def evaluate(self, assignment: Dict[Variable, FieldElement]) -> FieldElement:
        """Evaluate the linear combination given variable assignment"""
        result = self.constant
        for var, coeff in self.terms.items():
            if var in assignment:
                result = result + (coeff * assignment[var])
            elif var.value is not None:
                result = result + (coeff * var.value)
            else:
                raise ValueError(f"No value for variable {var.index}") from e
        return result


@dataclass
class Constraint:
    """Arithmetic constraint: A * B = C where A, B, C are linear combinations"""

    a: LinearCombination
    b: LinearCombination
    c: LinearCombination
    constraint_type: ConstraintType = ConstraintType.MUL
    metadata: Dict[str, Any] = field(default_factory = dict)

    def is_satisfied(self, assignment: Dict[Variable, FieldElement]) -> bool:
        """Check if constraint is satisfied by variable assignment"""
        try:
            a_val = self.a.evaluate(assignment)
            b_val = self.b.evaluate(assignment)
            c_val = self.c.evaluate(assignment)

            if self.constraint_type == ConstraintType.MUL:
                return (a_val * b_val) == c_val
            if self.constraint_type == ConstraintType.ADD:
                return (a_val + b_val) == c_val
                    """Magic method implementation."""
            else:
                # For other constraint types, assume multiplication semantics
                return (a_val * b_val) == c_val

        except Exception:
            return False


class ConstraintSystem:
    """
    Constraint system for building ZK circuits
    """

    def __init__(self):
        self.variables: Dict[int, Variable] = {}
        self.constraints: List[Constraint] = []
        self.public_inputs: List[Variable] = []
        self.private_inputs: List[Variable] = []
        self.variable_counter = 0
        self.assignment: Dict[Variable, FieldElement] = {}

        # Create the constant ONE variable
        self.one = self.new_variable("ONE", is_public = True)
        self.assign(self.one, FieldElement(1))

        # Create the constant ZERO variable
        self.zero = self.new_variable("ZERO", is_public = True)
        self.assign(self.zero, FieldElement(0))

    def new_variable(self, label: Optional[str] = None, is_public: bool = False) -> Variable:
        """Create a new variable"""
        var = Variable(self.variable_counter, label = label, is_public = is_public)
        self.variables[self.variable_counter] = var
        self.variable_counter += 1

        if is_public:
            self.public_inputs.append(var)

        return var

    def assign(self, var: Variable, value: Union[int, FieldElement]):
        """Assign value to variable"""
        if isinstance(value, int):
            value = FieldElement(value)
        var.value = value
        self.assignment[var] = value

    def get_assignment(self, var: Variable) -> FieldElement:
        """Get assigned value for variable"""
        if var in self.assignment:
            return self.assignment[var]
        if var.value is not None:
            return var.value
        else:
            raise ValueError(f"No assignment for variable {var.index}") from e

    def enforce_constraint(self, constraint: Constraint):
        """Add constraint to the system"""
        self.constraints.append(constraint)

    def enforce_equal(self,
        a: Union[Variable, LinearCombination, int, FieldElement],
        b: Union[Variable, LinearCombination, int, FieldElement],):
        """Enforce a == b"""
        a_lc = self._to_linear_combination(a)
        b_lc = self._to_linear_combination(b)

        # Create constraint: a * 1 = b
        constraint = Constraint(a = a_lc,
            b = LinearCombination({self.one: FieldElement(1)}),
            c = b_lc,
            constraint_type = ConstraintType.MUL,)
        self.enforce_constraint(constraint)

    def enforce_multiplication(self,
        a: Union[Variable, LinearCombination],
        b: Union[Variable, LinearCombination],
        c: Union[Variable, LinearCombination],):
        """Enforce a * b = c"""
        a_lc = self._to_linear_combination(a)
        b_lc = self._to_linear_combination(b)
        c_lc = self._to_linear_combination(c)

        constraint = Constraint(a = a_lc, b = b_lc, c = c_lc, constraint_type = ConstraintType.MUL)
        self.enforce_constraint(constraint)

    def enforce_boolean(self, var: Variable):
        """Enforce var is boolean (0 or 1): var * (var - 1) = 0"""
        var_minus_one = self.add_variable()
        self.assign(var_minus_one, self.get_assignment(var) - FieldElement(1))

        # var * (var - 1) = 0
        constraint = Constraint(a = LinearCombination({var: FieldElement(1)}),
            b = LinearCombination({var_minus_one: FieldElement(1)}),
            c = LinearCombination({self.zero: FieldElement(1)}),
            constraint_type = ConstraintType.BOOL,)
        self.enforce_constraint(constraint)

        return var_minus_one

    def add_variable(self, label: Optional[str] = None) -> Variable:
        """Add a private variable"""
        return self.new_variable(label, is_public = False)

    def add_public_input(self, label: Optional[str] = None) -> Variable:
        """Add a public input variable"""
        return self.new_variable(label, is_public = True)

    def _to_linear_combination(self, value) -> LinearCombination:
        """Convert various types to LinearCombination"""
        if isinstance(value, LinearCombination):
            return value
        if isinstance(value, Variable):
            return LinearCombination({value: FieldElement(1)})
        if isinstance(value, (int, FieldElement)):
            return LinearCombination(constant = FieldElement(value))
        else:
            raise TypeError(f"Cannot convert {type(value)} to LinearCombination")

    def is_satisfied(self) -> bool:
        """Check if all constraints are satisfied"""
        for constraint in self.constraints:
            if not constraint.is_satisfied(self.assignment):
                return False
        return True

    def num_constraints(self) -> int:
        """Get number of constraints"""
        return len(self.constraints)

    def num_variables(self) -> int:
        """Get number of variables"""
        return len(self.variables)

    def get_public_inputs(self) -> List[FieldElement]:
        """Get public input values"""
        return [self.get_assignment(var) for var in self.public_inputs]

    def get_witness(self) -> Dict[int, FieldElement]:
        """Get witness (all variable assignments)"""
        return {var.index: self.get_assignment(var) for var in self.variables.values()}


def poseidon_hash(inputs: List[FieldElement]) -> FieldElement:
    """
    Simplified Poseidon hash implementation for demonstration
    In production, would use optimized implementation
    """
    # Simple hash construction for demo
    data = b"".join(inp.to_bytes() for inp in inputs)
    hash_bytes = hashlib.sha256(data).digest()
    return FieldElement.from_bytes(hash_bytes)


def pedersen_commit(value: FieldElement, randomness: FieldElement) -> FieldElement:
    """
    Simplified Pedersen commitment for demonstration
    In production, would use proper elliptic curve implementation
    """
    # Simplified: hash(value || randomness)
    return poseidon_hash([value, randomness])


def create_merkle_proof(leaf: FieldElement, path: List[FieldElement], indices: List[int]) -> FieldElement:
    """
    Create Merkle proof by computing root from leaf and path
    """
    current = leaf

    for i, (sibling, is_right) in enumerate(zip(path, indices)):
        if is_right:
            current = poseidon_hash([sibling, current])
        else:
            current = poseidon_hash([current, sibling])

    return current
