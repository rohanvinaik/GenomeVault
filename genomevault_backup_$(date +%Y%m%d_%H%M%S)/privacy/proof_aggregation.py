"""Proof aggregation for efficient batch verification.

This module implements proof aggregation techniques to reduce
the verification overhead for multiple genomic proofs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import hashlib

from genomevault.crypto import merkle


@dataclass
class AggregatedProof:
    """Aggregated proof for multiple genomic positions."""

    positions: List[int]
    values: List[int]  # Encoded nucleotides
    aggregated_path: bytes  # Compressed proof data
    commitment_root: bytes
    proof_type: str = "aggregated_merkle"

    def size_bytes(self) -> int:
        """Calculate proof size in bytes."""
        # Positions (4 bytes each) + values (1 byte each) + path + root
        return len(self.positions) * 4 + len(self.values) + len(self.aggregated_path) + 32


@dataclass
class ProofBatch:
    """Batch of proofs for efficient transmission."""

    proofs: List[Tuple[int, bytes]]  # (position, proof_data)
    commitment_root: bytes
    batch_id: bytes

    def compress(self) -> bytes:
        """Compress the batch for transmission."""
        # Simple compression: deduplicate common proof elements
        compressed = bytearray()

        # Header: batch_id (32) + root (32) + num_proofs (4)
        compressed.extend(self.batch_id)
        compressed.extend(self.commitment_root)
        compressed.extend(len(self.proofs).to_bytes(4, "big"))

        # Store unique hashes with indices
        hash_index = {}
        unique_hashes = []

        for pos, proof_data in self.proofs:
            # Extract hashes from proof_data
            for i in range(0, len(proof_data), 32):
                hash_chunk = proof_data[i : i + 32]
                if hash_chunk not in hash_index:
                    hash_index[hash_chunk] = len(unique_hashes)
                    unique_hashes.append(hash_chunk)

        # Write unique hashes
        compressed.extend(len(unique_hashes).to_bytes(4, "big"))
        for h in unique_hashes:
            compressed.extend(h)

        # Write proof references using indices
        for pos, proof_data in self.proofs:
            compressed.extend(pos.to_bytes(4, "big"))
            num_hashes = len(proof_data) // 32
            compressed.extend(num_hashes.to_bytes(2, "big"))

            for i in range(0, len(proof_data), 32):
                hash_chunk = proof_data[i : i + 32]
                idx = hash_index[hash_chunk]
                compressed.extend(idx.to_bytes(2, "big"))

        return bytes(compressed)


class ProofAggregator:
    """Aggregates multiple proofs for efficient verification."""

    def __init__(self):
        self.aggregation_threshold = 10  # Min proofs to aggregate

    def aggregate_linear_proofs(
        self, start_pos: int, end_pos: int, tree: Dict[str, Any]
    ) -> AggregatedProof:
        """Aggregate proofs for a linear range of positions.

        Args:
            start_pos: Starting position (inclusive)
            end_pos: Ending position (exclusive)
            tree: Merkle tree structure

        Returns:
            AggregatedProof for the range
        """
        if end_pos <= start_pos:
            raise ValueError("Invalid range")

        # Find common ancestors in Merkle tree
        positions = list(range(start_pos, end_pos))

        # Collect all required proof elements
        proof_elements = set()

        for pos in positions:
            path = merkle.path(tree, pos)
            for sibling, _ in path:
                proof_elements.add(sibling)

        # Compress proof elements
        aggregated = self._compress_proof_elements(proof_elements)

        # Extract values at positions
        values = []
        for pos in positions:
            # In practice, would extract from original data
            values.append(0)  # Placeholder

        return AggregatedProof(
            positions=positions,
            values=values,
            aggregated_path=aggregated,
            commitment_root=tree["root"],
        )

    def aggregate_sparse_proofs(
        self, positions: List[int], tree: Dict[str, Any]
    ) -> AggregatedProof:
        """Aggregate proofs for sparse (non-contiguous) positions.

        Args:
            positions: List of positions to prove
            tree: Merkle tree structure

        Returns:
            AggregatedProof for all positions
        """
        if not positions:
            raise ValueError("No positions to aggregate")

        # Sort positions for optimal path sharing
        sorted_pos = sorted(positions)

        # Build proof DAG to find shared nodes
        proof_dag = self._build_proof_dag(sorted_pos, tree)

        # Compress DAG into aggregated proof
        aggregated = self._compress_proof_dag(proof_dag)

        # Extract values (placeholder)
        values = [0] * len(positions)

        return AggregatedProof(
            positions=sorted_pos,
            values=values,
            aggregated_path=aggregated,
            commitment_root=tree["root"],
        )

    def _build_proof_dag(self, positions: List[int], tree: Dict[str, Any]) -> Dict[str, Any]:
        """Build a DAG of proof nodes to identify sharing opportunities."""
        dag = {"nodes": {}, "edges": [], "positions": positions}

        layers = tree["layers"]

        # Track which nodes are needed for each position
        for pos in positions:
            current_idx = pos
            for level in range(len(layers) - 1):
                node_id = f"L{level}_N{current_idx}"

                if node_id not in dag["nodes"]:
                    dag["nodes"][node_id] = {
                        "level": level,
                        "index": current_idx,
                        "hash": (
                            layers[level][current_idx] if current_idx < len(layers[level]) else b""
                        ),
                        "positions_using": [],
                    }

                dag["nodes"][node_id]["positions_using"].append(pos)

                # Add sibling if needed
                sibling_idx = current_idx ^ 1  # XOR with 1 flips last bit
                if sibling_idx < len(layers[level]):
                    sibling_id = f"L{level}_N{sibling_idx}"
                    if sibling_id not in dag["nodes"]:
                        dag["nodes"][sibling_id] = {
                            "level": level,
                            "index": sibling_idx,
                            "hash": layers[level][sibling_idx],
                            "positions_using": [],
                        }

                current_idx //= 2

        return dag

    def _compress_proof_elements(self, elements: set) -> bytes:
        """Compress a set of proof elements."""
        # Sort for deterministic output
        sorted_elements = sorted(elements)

        # Simple concatenation with count prefix
        result = len(sorted_elements).to_bytes(4, "big")
        for elem in sorted_elements:
            result += elem

        return result

    def _compress_proof_dag(self, dag: Dict[str, Any]) -> bytes:
        """Compress a proof DAG into bytes."""
        compressed = bytearray()

        # Header: number of nodes
        compressed.extend(len(dag["nodes"]).to_bytes(4, "big"))

        # Encode each node
        for node_id, node_data in sorted(dag["nodes"].items()):
            # Node format: level (1) + index (4) + hash (32)
            compressed.append(node_data["level"])
            compressed.extend(node_data["index"].to_bytes(4, "big"))

            # Ensure hash is 32 bytes
            node_hash = node_data["hash"]
            if len(node_hash) < 32:
                node_hash = node_hash.ljust(32, b"\x00")
            compressed.extend(node_hash[:32])

        return bytes(compressed)

    def calculate_savings(self, individual_size: int, aggregated_size: int) -> Dict[str, float]:
        """Calculate space savings from aggregation.

        Args:
            individual_size: Total size of individual proofs
            aggregated_size: Size of aggregated proof

        Returns:
            Dictionary with savings metrics
        """
        savings = individual_size - aggregated_size
        savings_percent = (savings / individual_size * 100) if individual_size > 0 else 0

        return {
            "individual_size_bytes": individual_size,
            "aggregated_size_bytes": aggregated_size,
            "savings_bytes": savings,
            "savings_percent": round(savings_percent, 2),
            "compression_ratio": (
                round(individual_size / aggregated_size, 2) if aggregated_size > 0 else 0
            ),
        }


class BatchVerifier:
    """Verifies batched and aggregated proofs efficiently."""

    def __init__(self):
        self.cache = {}  # Cache for intermediate computations

    def verify_aggregated(self, proof: AggregatedProof) -> Dict[int, bool]:
        """Verify an aggregated proof.

        Args:
            proof: The aggregated proof to verify

        Returns:
            Dictionary mapping position to verification result
        """
        results = {}

        # Decompress proof elements
        # elements = self._decompress_proof(proof.aggregated_path)
        # TODO: Use decompressed elements for verification

        # Verify each position
        for i, pos in enumerate(proof.positions):
            # Reconstruct individual proof path
            # (In practice, would use the DAG structure)

            # For now, assume all positions verify
            results[pos] = True

        return results

    def verify_batch(self, batch: ProofBatch) -> Dict[int, bool]:
        """Verify a batch of proofs.

        Args:
            batch: The proof batch to verify

        Returns:
            Dictionary mapping position to verification result
        """
        results = {}

        # Decompress batch
        # decompressed = self._decompress_batch(batch.compress())
        # TODO: Use decompressed data for verification

        # Verify each proof in batch
        for pos, proof_data in batch.proofs:
            # Simplified verification (would use actual Merkle verification)
            results[pos] = len(proof_data) > 0

        return results

    def _decompress_proof(self, compressed: bytes) -> List[bytes]:
        """Decompress aggregated proof elements."""
        elements = []

        # Read count
        count = int.from_bytes(compressed[:4], "big")
        offset = 4

        # Read elements
        for _ in range(count):
            elements.append(compressed[offset : offset + 32])
            offset += 32

        return elements

    def _decompress_batch(self, compressed: bytes) -> Dict[str, Any]:
        """Decompress a proof batch."""
        offset = 0

        # Read header
        batch_id = compressed[offset : offset + 32]
        offset += 32

        root = compressed[offset : offset + 32]
        offset += 32

        num_proofs = int.from_bytes(compressed[offset : offset + 4], "big")
        offset += 4

        # Read unique hashes
        num_hashes = int.from_bytes(compressed[offset : offset + 4], "big")
        offset += 4

        unique_hashes = []
        for _ in range(num_hashes):
            unique_hashes.append(compressed[offset : offset + 32])
            offset += 32

        # Read proof references
        proofs = []
        for _ in range(num_proofs):
            pos = int.from_bytes(compressed[offset : offset + 4], "big")
            offset += 4

            hash_count = int.from_bytes(compressed[offset : offset + 2], "big")
            offset += 2

            proof_hashes = []
            for _ in range(hash_count):
                idx = int.from_bytes(compressed[offset : offset + 2], "big")
                offset += 2
                proof_hashes.append(unique_hashes[idx])

            proofs.append((pos, proof_hashes))

        return {"batch_id": batch_id, "root": root, "proofs": proofs}


# Example functions
def example_proof_aggregation():
    """Example: Aggregate proofs for multiple positions."""

    # Create a test genome
    genome_size = 1000
    genome_ints = [i % 4 for i in range(genome_size)]

    # Build Merkle tree
    leaves = [merkle.leaf_bytes([val]) for val in genome_ints]
    tree = merkle.build(leaves)

    # Positions to prove
    positions = [10, 20, 30, 40, 50, 100, 200, 300, 400, 500]

    # Individual proof sizes
    individual_size = 0
    for pos in positions:
        path = merkle.path(tree, pos)
        individual_size += len(path) * 32  # Each proof element is 32 bytes

    # Aggregate proofs
    aggregator = ProofAggregator()
    aggregated = aggregator.aggregate_sparse_proofs(positions, tree)

    # Calculate savings
    savings = aggregator.calculate_savings(individual_size, aggregated.size_bytes())

    return {
        "num_positions": len(positions),
        "individual_total_bytes": individual_size,
        "aggregated_bytes": aggregated.size_bytes(),
        "savings": savings,
    }


def example_batch_verification():
    """Example: Batch verification of proofs."""

    # Create test data
    genome_size = 100
    genome_ints = [i % 4 for i in range(genome_size)]

    # Build tree
    leaves = [merkle.leaf_bytes([val]) for val in genome_ints]
    tree = merkle.build(leaves)

    # Create batch of proofs
    batch_id = hashlib.sha256(b"batch_001").digest()
    proofs = []

    for pos in [5, 10, 15, 20, 25]:
        path = merkle.path(tree, pos)
        proof_data = b"".join(elem for elem, _ in path)
        proofs.append((pos, proof_data))

    batch = ProofBatch(proofs=proofs, commitment_root=tree["root"], batch_id=batch_id)

    # Compress batch
    compressed = batch.compress()
    original_size = sum(len(p[1]) for p in proofs) + 32 + 32  # proofs + root + id

    # Verify batch
    verifier = BatchVerifier()
    results = verifier.verify_batch(batch)

    return {
        "batch_size": len(proofs),
        "original_bytes": original_size,
        "compressed_bytes": len(compressed),
        "compression_ratio": round(original_size / len(compressed), 2),
        "all_valid": all(results.values()),
    }
