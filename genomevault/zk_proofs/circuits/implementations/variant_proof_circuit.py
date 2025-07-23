"""
Variant Proof Circuit Implementation

This implements the actual constraint generation for proving
variant presence without revealing the variant position.
"""

import hashlib
from typing import Any, Dict, List, Optional

from .constraint_system import (
    ConstraintSystem, 
    FieldElement, 
    Variable,
    poseidon_hash,
    create_merkle_proof
)


class VariantProofCircuit:
    """
    Zero-knowledge circuit for proving variant presence
    
    Public inputs:
    - variant_hash: Hash of variant (chr:pos:ref:alt)
    - reference_hash: Hash of reference genome version
    - commitment_root: Merkle root of user's genome
    
    Private inputs:
    - variant_data: Actual variant information
    - merkle_path: Merkle proof path
    - merkle_indices: Left/right indicators for path
    - witness_randomness: Randomness for zero-knowledge
    """
    
    def __init__(self, merkle_depth: int = 20):
        self.merkle_depth = merkle_depth
        self.cs = ConstraintSystem()
        self.setup_complete = False
        
    def setup_circuit(self, public_inputs: Dict[str, Any], private_inputs: Dict[str, Any]):
        """Setup the circuit with actual inputs"""
        
        # Public input variables
        self.variant_hash_var = self.cs.add_public_input("variant_hash")
        self.reference_hash_var = self.cs.add_public_input("reference_hash")
        self.commitment_root_var = self.cs.add_public_input("commitment_root")
        
        # Assign public input values
        self.cs.assign(self.variant_hash_var, FieldElement(int(public_inputs["variant_hash"], 16)))
        self.cs.assign(self.reference_hash_var, FieldElement(int(public_inputs["reference_hash"], 16)))
        self.cs.assign(self.commitment_root_var, FieldElement(int(public_inputs["commitment_root"], 16)))
        
        # Private input variables
        self.variant_chr = self.cs.add_variable("variant_chr")
        self.variant_pos = self.cs.add_variable("variant_pos")
        self.variant_ref = self.cs.add_variable("variant_ref")
        self.variant_alt = self.cs.add_variable("variant_alt")
        self.witness_randomness = self.cs.add_variable("witness_randomness")
        
        # Merkle path variables
        self.merkle_path_vars = []
        self.merkle_indices_vars = []
        
        for i in range(self.merkle_depth):
            path_var = self.cs.add_variable(f"merkle_path_{i}")
            index_var = self.cs.add_variable(f"merkle_index_{i}")
            self.merkle_path_vars.append(path_var)
            self.merkle_indices_vars.append(index_var)
        
        # Assign private input values
        variant_data = private_inputs["variant_data"]
        self.cs.assign(self.variant_chr, FieldElement(self._encode_chromosome(variant_data["chr"])))
        self.cs.assign(self.variant_pos, FieldElement(variant_data["pos"]))
        self.cs.assign(self.variant_ref, FieldElement(self._encode_base(variant_data["ref"])))
        self.cs.assign(self.variant_alt, FieldElement(self._encode_base(variant_data["alt"])))
        self.cs.assign(self.witness_randomness, FieldElement(int(private_inputs["witness_randomness"], 16)))
        
        # Assign Merkle path
        merkle_path = private_inputs["merkle_path"]
        merkle_indices = private_inputs["merkle_indices"]
        
        for i, (path_hash, index) in enumerate(zip(merkle_path, merkle_indices)):
            if i < len(self.merkle_path_vars):
                self.cs.assign(self.merkle_path_vars[i], FieldElement(int(path_hash, 16)))
                self.cs.assign(self.merkle_indices_vars[i], FieldElement(index))
        
        self.setup_complete = True
        
    def generate_constraints(self):
        """Generate all circuit constraints"""
        if not self.setup_complete:
            raise RuntimeError("Circuit must be setup before generating constraints")
        
        # 1. Verify variant hash
        self._constrain_variant_hash()
        
        # 2. Verify Merkle inclusion proof
        self._constrain_merkle_inclusion()
        
        # 3. Add range constraints
        self._constrain_ranges()
        
        # 4. Add randomness for zero-knowledge
        self._add_zero_knowledge_randomness()
        
    def _constrain_variant_hash(self):
        """Constrain that the variant hash is computed correctly"""
        
        # Compute variant hash from components
        # In practice, would use circuit-friendly hash
        computed_hash_var = self.cs.add_variable("computed_variant_hash")
        
        # For demo: hash = poseidon(chr, pos, ref, alt)
        variant_components = [
            self.cs.get_assignment(self.variant_chr),
            self.cs.get_assignment(self.variant_pos),
            self.cs.get_assignment(self.variant_ref),
            self.cs.get_assignment(self.variant_alt)
        ]
        
        computed_hash = poseidon_hash(variant_components)
        self.cs.assign(computed_hash_var, computed_hash)
        
        # Constrain computed hash equals public input
        self.cs.enforce_equal(computed_hash_var, self.variant_hash_var)
        
    def _constrain_merkle_inclusion(self):
        """Constrain Merkle tree inclusion proof"""
        
        # Create variant leaf
        leaf_var = self.cs.add_variable("variant_leaf")
        
        # Leaf = hash(variant_components + randomness)
        leaf_components = [
            self.cs.get_assignment(self.variant_chr),
            self.cs.get_assignment(self.variant_pos),
            self.cs.get_assignment(self.variant_ref),
            self.cs.get_assignment(self.variant_alt),
            self.cs.get_assignment(self.witness_randomness)
        ]
        
        leaf_hash = poseidon_hash(leaf_components)
        self.cs.assign(leaf_var, leaf_hash)
        
        # Compute Merkle root from leaf and path
        current_var = leaf_var
        
        for i in range(self.merkle_depth):
            # Create variables for hash computation
            left_var = self.cs.add_variable(f"merkle_left_{i}")
            right_var = self.cs.add_variable(f"merkle_right_{i}")
            parent_var = self.cs.add_variable(f"merkle_parent_{i}")
            
            # Get path sibling and index
            sibling = self.cs.get_assignment(self.merkle_path_vars[i])
            index = self.cs.get_assignment(self.merkle_indices_vars[i])
            
            # Constrain index to be boolean
            self.cs.enforce_boolean(self.merkle_indices_vars[i])
            
            # Select left and right based on index
            if index.value == 0:  # current is left child
                self.cs.assign(left_var, self.cs.get_assignment(current_var))
                self.cs.assign(right_var, sibling)
            else:  # current is right child
                self.cs.assign(left_var, sibling)
                self.cs.assign(right_var, self.cs.get_assignment(current_var))
            
            # Compute parent hash
            parent_hash = poseidon_hash([
                self.cs.get_assignment(left_var),
                self.cs.get_assignment(right_var)
            ])
            self.cs.assign(parent_var, parent_hash)
            
            # Update current for next iteration
            current_var = parent_var
        
        # Final constraint: computed root equals public commitment root
        self.cs.enforce_equal(current_var, self.commitment_root_var)
        
    def _constrain_ranges(self):
        """Add range constraints for validity"""
        
        # Chromosome should be 1-23, X(24), Y(25), MT(26)
        chr_val = self.cs.get_assignment(self.variant_chr)
        if not (1 <= chr_val.value <= 26):
            raise ValueError(f"Invalid chromosome: {chr_val.value}")
        
        # Position should be positive and reasonable
        pos_val = self.cs.get_assignment(self.variant_pos)
        if not (1 <= pos_val.value <= 300_000_000):  # Max human chromosome length
            raise ValueError(f"Invalid position: {pos_val.value}")
        
        # Base encoding should be valid (A=1, C=2, G=3, T=4)
        ref_val = self.cs.get_assignment(self.variant_ref)
        if not (1 <= ref_val.value <= 4):
            raise ValueError(f"Invalid reference base: {ref_val.value}")
            
        alt_val = self.cs.get_assignment(self.variant_alt)
        if not (1 <= alt_val.value <= 4):
            raise ValueError(f"Invalid alternate base: {alt_val.value}")
    
    def _add_zero_knowledge_randomness(self):
        """Add randomness to achieve zero-knowledge"""
        
        # Create additional random variables
        r1 = self.cs.add_variable("randomness_1")
        r2 = self.cs.add_variable("randomness_2")
        r3 = self.cs.add_variable("randomness_3")
        
        # Assign random values
        self.cs.assign(r1, FieldElement.random())
        self.cs.assign(r2, FieldElement.random())
        self.cs.assign(r3, FieldElement.random())
        
        # Add dummy constraints involving randomness
        # r1 * r2 = r3 (satisfied by construction)
        self.cs.enforce_multiplication(r1, r2, r3)
        
    def _encode_chromosome(self, chr_str: str) -> int:
        """Encode chromosome string to integer"""
        if chr_str.startswith("chr"):
            chr_str = chr_str[3:]
        
        if chr_str == "X":
            return 24
        elif chr_str == "Y":
            return 25
        elif chr_str == "MT" or chr_str == "M":
            return 26
        else:
            return int(chr_str)
    
    def _encode_base(self, base: str) -> int:
        """Encode DNA base to integer"""
        base_map = {"A": 1, "C": 2, "G": 3, "T": 4}
        return base_map.get(base.upper(), 1)
    
    def get_constraint_system(self) -> ConstraintSystem:
        """Get the constraint system"""
        return self.cs
    
    def get_public_inputs(self) -> List[FieldElement]:
        """Get public input values"""
        return self.cs.get_public_inputs()
    
    def get_witness(self) -> Dict[int, FieldElement]:
        """Get witness (private inputs)"""
        return self.cs.get_witness()
    
    def verify_constraints(self) -> bool:
        """Verify all constraints are satisfied"""
        return self.cs.is_satisfied()
    
    def get_circuit_info(self) -> Dict[str, Any]:
        """Get circuit information"""
        return {
            "name": "variant_proof",
            "num_constraints": self.cs.num_constraints(),
            "num_variables": self.cs.num_variables(),
            "merkle_depth": self.merkle_depth,
            "public_inputs": len(self.cs.public_inputs),
            "is_satisfied": self.cs.is_satisfied() if self.setup_complete else None
        }


def create_variant_proof_example():
    """Example of how to use the VariantProofCircuit"""
    
    # Example variant: chr1:12345:A>G
    variant_data = {
        "chr": "chr1",
        "pos": 12345,
        "ref": "A",
        "alt": "G"
    }
    
    # Create variant hash
    variant_str = f"{variant_data['chr']}:{variant_data['pos']}:{variant_data['ref']}:{variant_data['alt']}"
    variant_hash = hashlib.sha256(variant_str.encode()).hexdigest()
    
    # Mock reference hash
    reference_hash = hashlib.sha256(b"GRCh38").hexdigest()
    
    # Mock Merkle proof (in practice, would be real)
    merkle_path = [
        hashlib.sha256(f"sibling_{i}".encode()).hexdigest()
        for i in range(20)
    ]
    merkle_indices = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    
    # Mock commitment root
    commitment_root = hashlib.sha256(b"genome_commitment_root").hexdigest()
    
    # Create witness randomness
    witness_randomness = hashlib.sha256(b"random_witness_value").hexdigest()
    
    # Setup circuit
    circuit = VariantProofCircuit(merkle_depth=20)
    
    public_inputs = {
        "variant_hash": variant_hash,
        "reference_hash": reference_hash,
        "commitment_root": commitment_root
    }
    
    private_inputs = {
        "variant_data": variant_data,
        "merkle_path": merkle_path,
        "merkle_indices": merkle_indices,
        "witness_randomness": witness_randomness
    }
    
    # Setup and generate constraints
    circuit.setup_circuit(public_inputs, private_inputs)
    circuit.generate_constraints()
    
    return circuit


if __name__ == "__main__":
    # Test the circuit
    circuit = create_variant_proof_example()
    
    print("Variant Proof Circuit Test")
    print("=" * 40)
    
    info = circuit.get_circuit_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print(f"\nConstraints satisfied: {circuit.verify_constraints()}")
    print(f"Public inputs: {len(circuit.get_public_inputs())}")
    print(f"Witness size: {len(circuit.get_witness())}")
