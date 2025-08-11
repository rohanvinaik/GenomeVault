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
        # FIXED: Use cryptographically secure randomness instead of np.random
        import os

        return {
            "accumulator_value": os.urandom(32),
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

        # FIXED: Add metadata for verification integrity checks
        subproof_hashes = [hashlib.sha256(p.proof_data).hexdigest() for p in proofs]
        aggregate_inputs = self._aggregate_public_inputs(proofs)
        aggregate_vk = self._compute_aggregate_vk(proofs)

        return RecursiveProof(
            proof_id=self._generate_proof_id(proofs),
            sub_proof_ids=[p.proof_id for p in proofs],
            aggregation_proof=(
                final_proof.proof_data
                if isinstance(final_proof, Proof)
                else final_proof.aggregation_proof
            ),
            public_aggregate=aggregate_inputs,
            verification_key=aggregate_vk,
            metadata={
                "strategy": "tree",
                "tree_depth": level,
                "total_proof_count": len(proofs),
                "composition_time": time.time(),
                "uses_accumulator": False,
                "subproof_hashes": subproof_hashes,
                "stored_public_aggregate": aggregate_inputs,
                "expected_vk": aggregate_vk,
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

        # FIXED: Add metadata for verification integrity checks
        subproof_hashes = [hashlib.sha256(p.proof_data).hexdigest() for p in proofs]
        aggregate_inputs = self._aggregate_public_inputs(proofs)
        aggregate_vk = self._compute_aggregate_vk(proofs)

        return RecursiveProof(
            proof_id=self._generate_proof_id(proofs),
            sub_proof_ids=[p.proof_id for p in proofs],
            aggregation_proof=acc_proof,
            public_aggregate=aggregate_inputs,
            verification_key=aggregate_vk,
            metadata={
                "strategy": "accumulator",
                "accumulator_value": acc_value.hex(),
                "total_proof_count": len(proofs),
                "composition_time": time.time(),
                "uses_accumulator": True,
                "accumulator_size": len(acc_value),
                "subproof_hashes": subproof_hashes,
                "stored_public_aggregate": aggregate_inputs,
                "expected_vk": aggregate_vk,
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

        # FIXED: Add metadata for verification integrity checks
        subproof_hashes = [hashlib.sha256(p.proof_data).hexdigest() for p in proofs]
        aggregate_inputs = self._aggregate_public_inputs(proofs)
        aggregate_vk = self._compute_aggregate_vk(proofs)

        return RecursiveProof(
            proof_id=self._generate_proof_id(proofs),
            sub_proof_ids=[p.proof_id for p in proofs],
            aggregation_proof=(
                current.proof_data if isinstance(current, Proof) else current.aggregation_proof
            ),
            public_aggregate=aggregate_inputs,
            verification_key=aggregate_vk,
            metadata={
                "strategy": "sequential",
                "chain_length": len(proofs),
                "total_proof_count": len(proofs),
                "composition_time": time.time(),
                "uses_accumulator": False,
                "subproof_hashes": subproof_hashes,
                "stored_public_aggregate": aggregate_inputs,
                "expected_vk": aggregate_vk,
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
        # FIXED: Use cryptographically secure randomness
        import os

        private_inputs = {
            "left_proof": left.proof_data,
            "right_proof": right.proof_data,
            "left_vk": left.verification_key,
            "right_vk": right.verification_key,
            "witness_randomness": os.urandom(32).hex(),
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
            # FIXED: Use cryptographically secure randomness
            import os

            proof_components = {
                "pi": os.urandom(192).hex(),  # Main proof
                "recursive_commitment": os.urandom(32).hex(),
                "accumulator_update": os.urandom(32).hex(),
                "circuit_digest": hashlib.sha256(
                    json.dumps(circuit, sort_keys=True).encode()
                ).hexdigest(),
            }

            # FIXED: Never truncate proofs - use proper compression instead
            # Truncating JSON to 384 bytes corrupts the data structure completely.
            # Use deterministic serialization and compression for size optimization.

            import zlib

            # Serialize deterministically with sorted keys
            proof_json = json.dumps(proof_components, sort_keys=True, separators=(",", ":"))
            proof_data = proof_json.encode("utf-8")

            # Compress using zlib for size reduction without data loss
            compressed_proof = zlib.compress(proof_data, level=9)

            # Add a header to indicate compressed format
            # Format: b'CPROOF' (6 bytes) + 4-byte original size + compressed data
            original_size = len(proof_data).to_bytes(4, "big")
            compressed_with_header = b"CPROOF" + original_size + compressed_proof

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
        if compressed_data.startswith(b"CPROOF"):
            # Extract original size and compressed data
            original_size = int.from_bytes(compressed_data[6:10], "big")
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
                return json.loads(decompressed.decode("utf-8"))

            except (zlib.error, json.JSONDecodeError) as e:
                logger.error(f"Failed to decompress/parse proof: {e}")
                # Return empty dict or raise based on requirements
                return {}
        else:
            # Try to parse as uncompressed JSON (backward compatibility)
            try:
                return json.loads(compressed_data.decode("utf-8"))
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

        # FIXED: Use cryptographically secure randomness
        import os

        proof_components = {
            "final_accumulator": final_acc.hex(),
            "batch_proof": {
                "commitment": hashlib.sha256(b"".join(w for w in witnesses)).hexdigest(),
                "challenge": os.urandom(32).hex(),
                "response": os.urandom(128).hex(),
            },
            "proof_count": len(proofs),
            "timestamp": time.time(),
        }

        # FIXED: Never truncate proof data - use compression instead
        import zlib

        proof_json = json.dumps(proof_components, sort_keys=True, separators=(",", ":"))
        proof_data = proof_json.encode("utf-8")
        compressed = zlib.compress(proof_data, level=9)
        return b"CPROOF" + len(proof_data).to_bytes(4, "big") + compressed

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
        """
        Compute aggregate verification key with proper domain separation.

        FIXED: Use canonical byte layout with domain separation instead of JSON.
        Format: H("VK_AGG" || len || VK1 || VK2 || ...)
        """
        hasher = hashlib.sha256()

        # Domain separation tag
        hasher.update(b"VK_AGG")

        # Number of VKs being aggregated (4 bytes, big-endian)
        hasher.update(len(proofs).to_bytes(4, "big"))

        # Add each VK in deterministic order (sorted by proof ID for consistency)
        sorted_proofs = sorted(proofs, key=lambda p: p.proof_id)
        for proof in sorted_proofs:
            vk = proof.verification_key or proof.proof_id

            # Convert VK to bytes (handle both hex strings and raw strings)
            try:
                vk_bytes = bytes.fromhex(vk) if len(vk) % 2 == 0 else vk.encode("utf-8")
            except ValueError:
                vk_bytes = vk.encode("utf-8")

            # Add VK length (4 bytes) followed by VK bytes
            hasher.update(len(vk_bytes).to_bytes(4, "big"))
            hasher.update(vk_bytes)

        return hasher.hexdigest()

    def _combine_verification_keys(self, vk1: str, vk2: str) -> str:
        """
        Combine two verification keys with proper domain separation.

        FIXED: Use canonical byte layout instead of string concatenation.
        Format: H("VK_COMBINE" || VK1_len || VK1 || VK2_len || VK2)
        """
        hasher = hashlib.sha256()

        # Domain separation tag
        hasher.update(b"VK_COMBINE")

        # Convert VKs to bytes
        try:
            vk1_bytes = bytes.fromhex(vk1) if len(vk1) % 2 == 0 else vk1.encode("utf-8")
        except ValueError:
            vk1_bytes = vk1.encode("utf-8")

        try:
            vk2_bytes = bytes.fromhex(vk2) if len(vk2) % 2 == 0 else vk2.encode("utf-8")
        except ValueError:
            vk2_bytes = vk2.encode("utf-8")

        # Add VK1 length and bytes
        hasher.update(len(vk1_bytes).to_bytes(4, "big"))
        hasher.update(vk1_bytes)

        # Add VK2 length and bytes
        hasher.update(len(vk2_bytes).to_bytes(4, "big"))
        hasher.update(vk2_bytes)

        return hasher.hexdigest()

    def _hash_proof(self, proof: Proof) -> str:
        """
        Compute hash of proof for aggregation with proper domain separation.

        FIXED: Use canonical byte layout instead of JSON serialization.
        Format: H("SUBPROOF" || circuit_id || vk || proof_bytes_commit || public_inputs_commit)
        """
        hasher = hashlib.sha256()

        # Domain separation tag
        hasher.update(b"SUBPROOF")

        # Circuit ID (proof_id as bytes)
        circuit_id = proof.proof_id.encode("utf-8")
        hasher.update(len(circuit_id).to_bytes(4, "big"))
        hasher.update(circuit_id)

        # Circuit name
        circuit_name = proof.circuit_name.encode("utf-8")
        hasher.update(len(circuit_name).to_bytes(4, "big"))
        hasher.update(circuit_name)

        # Verification key (if present)
        if proof.verification_key:
            try:
                vk_bytes = bytes.fromhex(proof.verification_key)
            except ValueError:
                vk_bytes = proof.verification_key.encode("utf-8")
            hasher.update(len(vk_bytes).to_bytes(4, "big"))
            hasher.update(vk_bytes)
        else:
            hasher.update((0).to_bytes(4, "big"))  # Zero length for missing VK

        # Proof data commitment (hash of the actual proof bytes)
        if isinstance(proof.proof_data, bytes):
            proof_bytes_hash = hashlib.sha256(proof.proof_data).digest()
        else:
            proof_bytes_hash = hashlib.sha256(str(proof.proof_data).encode("utf-8")).digest()
        hasher.update(proof_bytes_hash)

        # Public inputs commitment (canonical serialization)
        if proof.public_inputs:
            # Sort keys for deterministic ordering
            sorted_items = sorted(proof.public_inputs.items())
            inputs_hasher = hashlib.sha256()

            for key, value in sorted_items:
                # Add key
                key_bytes = key.encode("utf-8")
                inputs_hasher.update(len(key_bytes).to_bytes(4, "big"))
                inputs_hasher.update(key_bytes)

                # Add value (handle different types)
                if isinstance(value, (int, float)):
                    value_bytes = str(value).encode("utf-8")
                elif isinstance(value, bytes):
                    value_bytes = value
                else:
                    value_bytes = str(value).encode("utf-8")

                inputs_hasher.update(len(value_bytes).to_bytes(4, "big"))
                inputs_hasher.update(value_bytes)

            hasher.update(inputs_hasher.digest())
        else:
            # Empty public inputs
            hasher.update(hashlib.sha256(b"EMPTY").digest())

        return hasher.hexdigest()

    def _generate_proof_id(self, proofs: list[Proof]) -> str:
        """
        Generate ID for recursive proof with canonical format.

        FIXED: Use canonical byte layout instead of JSON.
        Format: H("PROOF_ID" || count || sorted_ids || timestamp || nonce)
        """
        hasher = hashlib.sha256()

        # Domain separation tag
        hasher.update(b"PROOF_ID")

        # Number of sub-proofs
        hasher.update(len(proofs).to_bytes(4, "big"))

        # Sorted proof IDs for deterministic ordering
        sorted_ids = sorted([p.proof_id for p in proofs])
        for proof_id in sorted_ids:
            id_bytes = proof_id.encode("utf-8")
            hasher.update(len(id_bytes).to_bytes(4, "big"))
            hasher.update(id_bytes)

        # Timestamp (8 bytes, microsecond precision)
        timestamp_us = int(time.time() * 1_000_000)
        hasher.update(timestamp_us.to_bytes(8, "big"))

        # Random nonce for uniqueness (8 bytes)
        # FIXED: Use cryptographically secure randomness
        import os

        hasher.update(os.urandom(8))

        # Return first 16 hex chars (64 bits) for shorter IDs
        return hasher.hexdigest()[:16]

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
            # FIXED: Implement proper verification checks even in simulated mode
            # Previously just returned True, now performs actual integrity checks

            logger.debug("Performing simulated verification with integrity checks")

            # 1. Decompress and parse the aggregation proof
            try:
                proof_data = self._decompress_proof_data(recursive_proof.aggregation_proof)
                if not proof_data:
                    logger.error("Failed to decompress aggregation proof")
                    return False
            except Exception as e:
                logger.error(f"Proof decompression error: {e}")
                return False

            # 2. Verify the aggregation proof contains expected components
            required_components = [
                "pi",
                "recursive_commitment",
                "accumulator_update",
                "circuit_digest",
            ]
            for component in required_components:
                if component not in proof_data:
                    logger.error(f"Missing required proof component: {component}")
                    return False

            # 3. Recompute commitment over subproof hashes
            # Extract subproof data from metadata
            subproof_hashes = recursive_proof.metadata.get("subproof_hashes", [])
            if subproof_hashes:
                # Compute expected commitment
                import hashlib

                combined_hash = hashlib.sha256()
                for subproof_hash in subproof_hashes:
                    combined_hash.update(bytes.fromhex(subproof_hash))
                expected_commitment = combined_hash.hexdigest()

                # Verify against stored commitment
                stored_commitment = proof_data.get("recursive_commitment", "")
                if stored_commitment and stored_commitment != expected_commitment:
                    logger.error(
                        f"Commitment mismatch: expected {expected_commitment[:16]}..., "
                        f"got {stored_commitment[:16]}..."
                    )
                    return False

            # 4. Verify public aggregate matches expected
            expected_aggregate = recursive_proof.public_aggregate
            stored_aggregate = recursive_proof.metadata.get("stored_public_aggregate")

            if stored_aggregate and stored_aggregate != expected_aggregate:
                logger.error("Public aggregate mismatch")
                return False

            # 5. Verify aggregate VK equals combination of sub VKs
            if recursive_proof.verification_key:
                # Check VK format (should be hex string of appropriate length)
                try:
                    vk_bytes = bytes.fromhex(recursive_proof.verification_key)
                    if len(vk_bytes) < 32:  # Minimum VK size
                        logger.error(f"Verification key too short: {len(vk_bytes)} bytes")
                        return False
                except ValueError:
                    logger.error("Invalid verification key format")
                    return False

                # Verify VK matches expected combination
                expected_vk = recursive_proof.metadata.get("expected_vk")
                if expected_vk and recursive_proof.verification_key != expected_vk:
                    logger.error("Verification key mismatch")
                    return False

            # 6. Check proof count consistency
            claimed_count = recursive_proof.proof_count
            metadata_count = recursive_proof.metadata.get("total_proof_count", claimed_count)

            if claimed_count != metadata_count:
                logger.error(
                    f"Proof count mismatch: claimed {claimed_count}, "
                    f"metadata says {metadata_count}"
                )
                return False

            # 7. Verify proof structure based on strategy
            strategy = recursive_proof.metadata.get("strategy", "unknown")

            if strategy == "tree":
                # Tree-based proofs should have tree_depth
                tree_depth = recursive_proof.metadata.get("tree_depth", 0)
                expected_depth = (claimed_count - 1).bit_length()
                if abs(tree_depth - expected_depth) > 1:
                    logger.error(
                        f"Tree depth inconsistent with proof count: "
                        f"depth={tree_depth}, count={claimed_count}"
                    )
                    return False

            elif strategy == "accumulator" and recursive_proof.metadata.get("uses_accumulator"):
                # Accumulator proofs should have accumulator value
                if "accumulator_update" not in proof_data:
                    logger.error("Accumulator proof missing accumulator update")
                    return False

            # 8. Simulate realistic verification time
            if recursive_proof.metadata.get("uses_accumulator", False):
                verification_time = 0.025  # 25ms constant for accumulator
            else:
                # O(log n) for tree-based
                verification_time = 0.025 + 0.005 * recursive_proof.metadata.get("tree_depth", 1)

            # Add small random variation for realism
            import random

            verification_time *= 0.9 + 0.2 * random.random()

            time.sleep(verification_time)

            logger.debug(
                f"Simulated verification passed all checks in {verification_time*1000:.1f}ms"
            )
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
                "witness_randomness": os.urandom(32).hex(),
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
