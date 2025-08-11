from __future__ import annotations

"""Recursive Snark module."""
"""
Recursive SNARK implementation for efficient proof composition.
Enables constant verification time regardless of composed proof count.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from genomevault.utils.logging import get_logger
from genomevault.utils.metrics import get_metrics
from genomevault.zk_proofs.backends.gnark_backend import get_backend
from genomevault.zk_proofs.prover import Proof

logger = get_logger(__name__)
metrics = get_metrics()


@dataclass
class RecursiveProof:
    """Recursive proof that aggregates multiple sub-proofs."""

    proof_id: str
    sub_proof_ids: list[str]
    aggregation_proof: bytes
    public_aggregate: dict[str, Any]
    verification_key: str
    metadata: dict[str, Any]

    @property
    def proof_count(self) -> int:
        """Total number of proofs aggregated (including nested)."""
        return self.metadata.get("total_proof_count", len(self.sub_proof_ids))

    @property
    def verification_complexity(self) -> str:
        """Verification complexity class."""
        return "O(1)" if self.metadata.get("uses_accumulator", False) else "O(log n)"


class RecursiveSNARKProver:
    """
    Implements recursive SNARK composition for GenomeVault.
    Achieves O(D log D) circuit overhead for verifying D proofs.
    """

    def __init__(self, max_recursion_depth: int = 10, use_real_backend: bool = True):
        """
        Initialize recursive SNARK prover.

        Args:
            max_recursion_depth: Maximum nesting depth for recursive proofs
            use_real_backend: Whether to use real gnark backend or simulation
        """
        self.max_recursion_depth = max_recursion_depth
        self.accumulator_state = self._initialize_accumulator()
        self.circuit_cache = {}
        self.use_real_backend = use_real_backend

        # Initialize ZK backend
        self.backend = get_backend(use_real=use_real_backend)

        logger.info(f"RecursiveSNARKProver initialized with max depth {max_recursion_depth}")
        logger.info(f"Using {'real gnark' if use_real_backend else 'simulated'} backend")

    def _initialize_accumulator(self) -> dict[str, Any]:
        """Initialize cryptographic accumulator for proof aggregation."""
        return {
            "accumulator_value": np.random.bytes(32),
            "witness_cache": {},
            "proof_count": 0,
            "last_update": time.time(),
        }

    def compose_proofs(
        self, proofs: list[Proof], aggregation_strategy: str = "balanced_tree"
    ) -> RecursiveProof:
        """
        Compose multiple proofs into a single recursive proof.

        Args:
            proofs: List of proofs to compose
            aggregation_strategy: Strategy for proof aggregation
                - "balanced_tree": Binary tree aggregation (O(log n) depth)
                - "accumulator": Accumulator-based (O(1) verification)
                - "sequential": Sequential aggregation (simple but deep)

        Returns:
            Recursive proof aggregating all input proofs
        """
        if not proofs:
            raise ValueError("Cannot compose empty proof list")

        if len(proofs) == 1:
            return self._wrap_single_proof(proofs[0])

        # Choose aggregation strategy
        if aggregation_strategy == "balanced_tree":
            return self._balanced_tree_composition(proofs)
        elif aggregation_strategy == "accumulator":
            return self._accumulator_based_composition(proofs)
        elif aggregation_strategy == "sequential":
            return self._sequential_composition(proofs)
        else:
            raise ValueError(f"Unknown aggregation strategy: {aggregation_strategy}")

    def _balanced_tree_composition(self, proofs: list[Proof]) -> RecursiveProof:
        """
        Compose proofs using balanced binary tree structure.
        Achieves O(log n) verification depth.
        """
        logger.info(f"Composing {len(proofs)} proofs using balanced tree strategy")

        # Build tree bottom-up
        current_level = proofs
        level = 0

        while len(current_level) > 1:
            next_level = []

            # Pair up proofs
            for i in range(0, len(current_level), 2):
                if i + 1 < len(current_level):
                    # Compose pair
                    left, right = current_level[i], current_level[i + 1]
                    composed = self._compose_pair(left, right, level)
                    next_level.append(composed)
                else:
                    # Odd proof, carry forward
                    next_level.append(current_level[i])

            current_level = next_level
            level += 1

            if level > self.max_recursion_depth:
                raise ValueError(f"Exceeded maximum recursion depth: {level}")

        # Convert final proof to RecursiveProof
        final_proof = current_level[0]

        return RecursiveProof(
            proof_id=self._generate_proof_id(proofs),
            sub_proof_ids=[p.proof_id for p in proofs],
            aggregation_proof=(
                final_proof.proof_data
                if isinstance(final_proof, Proof)
                else final_proof.aggregation_proof
            ),
            public_aggregate=self._aggregate_public_inputs(proofs),
            verification_key=self._compute_aggregate_vk(proofs),
            metadata={
                "strategy": "balanced_tree",
                "tree_depth": level,
                "total_proof_count": len(proofs),
                "composition_time": time.time(),
                "uses_accumulator": False,
            },
        )

    def _accumulator_based_composition(self, proofs: list[Proof]) -> RecursiveProof:
        """
        Compose proofs using cryptographic accumulator.
        Achieves O(1) verification time.
        """
        logger.info(f"Composing {len(proofs)} proofs using accumulator strategy")

        # Update accumulator with each proof
        acc_value = self.accumulator_state["accumulator_value"]
        witnesses = []

        for proof in proofs:
            # Add proof to accumulator
            new_acc, witness = self._accumulator_add(acc_value, proof)
            acc_value = new_acc
            witnesses.append(witness)

            # Cache witness for later verification
            self.accumulator_state["witness_cache"][proof.proof_id] = witness

        # Update accumulator state
        self.accumulator_state["accumulator_value"] = acc_value
        self.accumulator_state["proof_count"] += len(proofs)
        self.accumulator_state["last_update"] = time.time()

        # Generate accumulator proof
        acc_proof = self._generate_accumulator_proof(proofs, witnesses, acc_value)

        return RecursiveProof(
            proof_id=self._generate_proof_id(proofs),
            sub_proof_ids=[p.proof_id for p in proofs],
            aggregation_proof=acc_proof,
            public_aggregate=self._aggregate_public_inputs(proofs),
            verification_key=self._compute_aggregate_vk(proofs),
            metadata={
                "strategy": "accumulator",
                "accumulator_value": acc_value.hex(),
                "total_proof_count": len(proofs),
                "composition_time": time.time(),
                "uses_accumulator": True,
                "accumulator_size": len(acc_value),
            },
        )

    def _sequential_composition(self, proofs: list[Proof]) -> RecursiveProof:
        """Simple sequential proof composition (for comparison/testing)."""
        logger.info(f"Composing {len(proofs)} proofs using sequential strategy")

        # Start with first proof
        current = proofs[0]

        # Sequentially compose each subsequent proof
        for i in range(1, len(proofs)):
            current = self._compose_pair(current, proofs[i], i)

        return RecursiveProof(
            proof_id=self._generate_proof_id(proofs),
            sub_proof_ids=[p.proof_id for p in proofs],
            aggregation_proof=(
                current.proof_data if isinstance(current, Proof) else current.aggregation_proof
            ),
            public_aggregate=self._aggregate_public_inputs(proofs),
            verification_key=self._compute_aggregate_vk(proofs),
            metadata={
                "strategy": "sequential",
                "chain_length": len(proofs),
                "total_proof_count": len(proofs),
                "composition_time": time.time(),
                "uses_accumulator": False,
            },
        )

    def _compose_pair(self, left: Proof, right: Proof, level: int) -> Proof:
        """
        Compose two proofs into one using recursive SNARK.

        This implements the formula:
        π_combined = Prove(circuit_verifier, (π_1, π_2), w)
        """
        # Create verifier circuit for the pair
        circuit_key = f"{left.circuit_name}_{right.circuit_name}_{level}"

        if circuit_key not in self.circuit_cache:
            self.circuit_cache[circuit_key] = self._create_verifier_circuit(
                left.circuit_name, right.circuit_name
            )

        verifier_circuit = self.circuit_cache[circuit_key]

        # Public inputs: hashes of sub-proofs
        public_inputs = {
            "left_proof_hash": self._hash_proof(left),
            "right_proof_hash": self._hash_proof(right),
            "aggregated_public": self._merge_public_inputs(left.public_inputs, right.public_inputs),
        }

        # Private inputs: the actual proofs
        private_inputs = {
            "left_proof": left.proof_data,
            "right_proof": right.proof_data,
            "left_vk": left.verification_key,
            "right_vk": right.verification_key,
            "witness_randomness": np.random.bytes(32).hex(),
        }

        # Generate composed proof
        composed_proof_data = self._generate_recursive_proof(
            verifier_circuit, public_inputs, private_inputs
        )

        return Proof(
            proof_id=hashlib.sha256(f"{left.proof_id}_{right.proof_id}".encode()).hexdigest()[:16],
            circuit_name=f"recursive_{level}",
            proof_data=composed_proof_data,
            public_inputs=public_inputs,
            timestamp=time.time(),
            verification_key=self._combine_verification_keys(
                left.verification_key, right.verification_key
            ),
            metadata={
                "type": "recursive",
                "level": level,
                "sub_proofs": [left.proof_id, right.proof_id],
            },
        )

    def _create_verifier_circuit(self, left_circuit: str, right_circuit: str) -> dict[str, Any]:
        """Create SNARK verifier circuit for proof pair."""
        # Circuit that verifies two proofs
        return {
            "name": f"verifier_{left_circuit}_{right_circuit}",
            "constraints": 5000,  # Optimized with custom gates
            "gates": [
                {"type": "poseidon_hash", "inputs": 4},  # Custom Poseidon gate
                {"type": "pairing_check", "curve": "BLS12-381"},
                {"type": "range_check", "bits": 254},
            ],
            "public_input_size": 3,
            "uses_lookups": True,  # Plonky2-style lookups
        }

    def _generate_recursive_proof(
        self,
        circuit: dict[str, Any],
        public_inputs: dict[str, Any],
        private_inputs: dict[str, Any],
    ) -> bytes:
        """Generate proof for recursive verification circuit."""
        if self.use_real_backend:
            # Use real gnark backend for proof generation
            with metrics.time_operation("recursive_proof_generation"):
                proof_data = self.backend.generate_proof(
                    "recursive_verifier", public_inputs, private_inputs
                )
            return proof_data
        else:
            # Fallback to simulation
            proof_components = {
                "pi": np.random.bytes(192).hex(),  # Main proof
                "recursive_commitment": np.random.bytes(32).hex(),
                "accumulator_update": np.random.bytes(32).hex(),
                "circuit_digest": hashlib.sha256(
                    json.dumps(circuit, sort_keys=True).encode()
                ).hexdigest(),
            }

            # FIXED: Never truncate proofs - use proper compression instead
            # Truncating JSON to 384 bytes corrupts the data structure completely.
            # Use deterministic serialization and compression for size optimization.
            
            import zlib
            
            # Serialize deterministically with sorted keys
            proof_json = json.dumps(proof_components, sort_keys=True, separators=(',', ':'))
            proof_data = proof_json.encode('utf-8')
            
            # Compress using zlib for size reduction without data loss
            compressed_proof = zlib.compress(proof_data, level=9)
            
            # Add a header to indicate compressed format
            # Format: b'CPROOF' (6 bytes) + 4-byte original size + compressed data
            original_size = len(proof_data).to_bytes(4, 'big')
            compressed_with_header = b'CPROOF' + original_size + compressed_proof
            
            logger.debug(
                f"Proof compression: {len(proof_data)} bytes → "
                f"{len(compressed_with_header)} bytes "
                f"({100 * len(compressed_with_header) / len(proof_data):.1f}%)"
            )
            
            return compressed_with_header

    def _decompress_proof_data(self, compressed_data: bytes) -> dict[str, Any]:
        """
        Decompress proof data that was compressed in _generate_recursive_proof.
        
        FIXED: Added decompression support for the new compressed proof format.
        Handles both compressed (with CPROOF header) and uncompressed formats.
        """
        import zlib
        
        # Check for compressed format header
        if compressed_data.startswith(b'CPROOF'):
            # Extract original size and compressed data
            original_size = int.from_bytes(compressed_data[6:10], 'big')
            compressed_proof = compressed_data[10:]
            
            # Decompress
            try:
                decompressed = zlib.decompress(compressed_proof)
                
                # Verify size matches
                if len(decompressed) != original_size:
                    logger.warning(
                        f"Decompressed size mismatch: expected {original_size}, "
                        f"got {len(decompressed)}"
                    )
                
                # Parse JSON
                return json.loads(decompressed.decode('utf-8'))
                
            except (zlib.error, json.JSONDecodeError) as e:
                logger.error(f"Failed to decompress/parse proof: {e}")
                # Return empty dict or raise based on requirements
                return {}
        else:
            # Try to parse as uncompressed JSON (backward compatibility)
            try:
                return json.loads(compressed_data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # If it's not JSON, might be raw binary proof
                logger.debug("Proof data is not compressed JSON, treating as raw binary")
                return {"raw_proof": compressed_data.hex()}
    
    def _accumulator_add(self, acc_value: bytes, proof: Proof) -> tuple[bytes, bytes]:
        """Add proof to cryptographic accumulator."""
        # Hash proof into accumulator
        proof_digest = self._hash_proof(proof).encode()

        # Update accumulator value
        new_acc = hashlib.sha256(acc_value + proof_digest).digest()

        # Generate membership witness
        witness = hashlib.sha256(
            proof_digest + acc_value + proof.timestamp.to_bytes(8, "big")
        ).digest()

        return new_acc, witness

    def _generate_accumulator_proof(
        self, proofs: list[Proof], witnesses: list[bytes], final_acc: bytes
    ) -> bytes:
        """Generate proof of correct accumulator construction."""
        # Prove that final_acc correctly accumulates all proofs

        proof_components = {
            "final_accumulator": final_acc.hex(),
            "batch_proof": {
                "commitment": hashlib.sha256(b"".join(w for w in witnesses)).hexdigest(),
                "challenge": np.random.bytes(32).hex(),
                "response": np.random.bytes(128).hex(),
            },
            "proof_count": len(proofs),
            "timestamp": time.time(),
        }

        # FIXED: Never truncate proof data - use compression instead
        import zlib
        proof_json = json.dumps(proof_components, sort_keys=True, separators=(',', ':'))
        proof_data = proof_json.encode('utf-8')
        compressed = zlib.compress(proof_data, level=9)
        return b'CPROOF' + len(proof_data).to_bytes(4, 'big') + compressed

    def _aggregate_public_inputs(self, proofs: list[Proof]) -> dict[str, Any]:
        """Aggregate public inputs from multiple proofs."""
        aggregate = {
            "proof_count": len(proofs),
            "circuit_types": list({p.circuit_name for p in proofs}),
            "timestamp_range": {
                "min": min(p.timestamp for p in proofs),
                "max": max(p.timestamp for p in proofs),
            },
        }

        # Aggregate specific types of public inputs
        if all(p.circuit_name == "polygenic_risk_score" for p in proofs):
            # Aggregate PRS results
            aggregate["prs_summary"] = {
                "count": len(proofs),
                "models": list({p.public_inputs.get("prs_model", "unknown") for p in proofs}),
            }

        return aggregate

    def _merge_public_inputs(
        self, left_public: dict[str, Any], right_public: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge public inputs from two proofs."""
        merged = {"left": left_public, "right": right_public}

        # Special handling for certain input types
        if "proof_count" in left_public and "proof_count" in right_public:
            merged["total_proof_count"] = left_public["proof_count"] + right_public["proof_count"]

        return merged

    def _compute_aggregate_vk(self, proofs: list[Proof]) -> str:
        """Compute aggregate verification key."""
        # In practice, would compute proper aggregate VK
        vk_components = [p.verification_key or p.proof_id for p in proofs]

        aggregate_vk = hashlib.sha256("".join(sorted(vk_components)).encode()).hexdigest()

        return aggregate_vk

    def _combine_verification_keys(self, vk1: str, vk2: str) -> str:
        """Combine two verification keys."""
        return hashlib.sha256(f"{vk1}_{vk2}".encode()).hexdigest()

    def _hash_proof(self, proof: Proof) -> str:
        """Compute hash of proof for aggregation."""
        proof_data = {
            "proof_id": proof.proof_id,
            "circuit_name": proof.circuit_name,
            "public_inputs": proof.public_inputs,
            "proof_data": (
                proof.proof_data.hex()
                if isinstance(proof.proof_data, bytes)
                else str(proof.proof_data)
            ),
        }

        return hashlib.sha256(json.dumps(proof_data, sort_keys=True).encode()).hexdigest()

    def _generate_proof_id(self, proofs: list[Proof]) -> str:
        """Generate ID for recursive proof."""
        content = {
            "sub_proof_ids": sorted([p.proof_id for p in proofs]),
            "timestamp": time.time(),
            "nonce": np.random.bytes(8).hex(),
        }

        return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]

    def _wrap_single_proof(self, proof: Proof) -> RecursiveProof:
        """Wrap single proof as recursive proof."""
        return RecursiveProof(
            proof_id=proof.proof_id,
            sub_proof_ids=[proof.proof_id],
            aggregation_proof=proof.proof_data,
            public_aggregate=proof.public_inputs,
            verification_key=proof.verification_key or self._compute_aggregate_vk([proof]),
            metadata={
                "strategy": "single",
                "total_proof_count": 1,
                "composition_time": time.time(),
                "uses_accumulator": False,
            },
        )

    def verify_recursive_proof(self, recursive_proof: RecursiveProof) -> bool:
        """
        Verify a recursive proof.

        Returns:
            True if proof is valid
        """
        logger.info(
            f"Verifying recursive proof {recursive_proof.proof_id} "
            f"aggregating {recursive_proof.proof_count} proofs"
        )

        if self.use_real_backend:
            # Use real backend for verification
            with metrics.time_operation("recursive_proof_verification"):
                return self.backend.verify_proof(
                    "recursive_verifier",
                    recursive_proof.aggregation_proof,
                    recursive_proof.public_aggregate,
                )
        else:
            # Simulate verification
            # Verification is O(1) for accumulator-based proofs
            if recursive_proof.metadata.get("uses_accumulator", False):
                verification_time = 0.025  # 25ms constant
            else:
                # O(log n) for tree-based
                verification_time = 0.025 + 0.005 * recursive_proof.metadata.get("tree_depth", 1)

            # Simulate verification time
            time.sleep(verification_time)

            return True


# Example usage
if __name__ == "__main__":
    from genomevault.zk_proofs.prover import Prover

    # Initialize provers
    prover = Prover()
    recursive_prover = RecursiveSNARKProver()

    # Generate some base proofs
    proofs = []

    # Generate 10 PRS proofs
    for i in range(10):
        proof = prover.generate_proof(
            circuit_name="polygenic_risk_score",
            public_inputs={
                "prs_model": f"T2D_model_v{i}",
                "score_range": {"min": 0, "max": 1},
                "result_commitment": hashlib.sha256(f"result_{i}".encode()).hexdigest(),
                "genome_commitment": hashlib.sha256(f"genome_{i}".encode()).hexdigest(),
            },
            private_inputs={
                "variants": np.random.randint(0, 2, 1000).tolist(),
                "weights": np.random.rand(1000).tolist(),
                "merkle_proofs": [
                    hashlib.sha256(f"proof_{j}".encode()).hexdigest() for j in range(20)
                ],
                "witness_randomness": np.random.bytes(32).hex(),
            },
        )
        proofs.append(proof)

    logger.info(f"Generated {len(proofs)} base proofs")

    # Test different aggregation strategies
    strategies = ["balanced_tree", "accumulator", "sequential"]

    for strategy in strategies:
        logger.info(f"\nTesting {strategy} aggregation:")

        start_time = time.time()
        recursive_proof = recursive_prover.compose_proofs(proofs, strategy)
        composition_time = time.time() - start_time

        logger.info(f"  Composition time: {composition_time * 1000:.1f}ms")
        logger.info(f"  Proof size: {len(recursive_proof.aggregation_proof)} bytes")
        logger.info(f"  Verification complexity: {recursive_proof.verification_complexity}")

        # Verify the recursive proof
        start_time = time.time()
        valid = recursive_prover.verify_recursive_proof(recursive_proof)
        verification_time = time.time() - start_time

        logger.info(f"  Verification time: {verification_time * 1000:.1f}ms")
        logger.info(f"  Valid: {valid}")
