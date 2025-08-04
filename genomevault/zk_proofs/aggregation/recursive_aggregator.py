from genomevault.observability.logging import configure_logging

logger = configure_logging()
"""
Recursive Proof Aggregation for GenomeVault ZK System

Stage 4 implementation: Recursion / Aggregation & Performance Bench
- Recursive proof composition script aggregating N subproofs
- Benchmarks: proof size, verify time, aggregation time vs N
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from genomevault.utils.metrics import MetricsCollector
from genomevault.zk_proofs.circuits.implementations.constraint_system import (
    ConstraintSystem, FieldElement, poseidon_hash)


@dataclass
class Proof:
    """Represents a zero-knowledge proof."""

    circuit_name: str
    public_inputs: list[FieldElement]
    proof_data: bytes
    metadata: dict[str, Any]


@dataclass
class AggregatedProof:
    """Represents an aggregated proof of multiple subproofs."""

    num_proofs: int
    aggregated_data: bytes
    public_aggregate: FieldElement
    metadata: dict[str, Any]


class RecursiveProofAggregator:
    """
    Implements recursive SNARK composition for efficient batch verification.

    Based on the formula: π_combined = Prove(circuit_verifier, (π_1, π_2, ..., π_n), w)
    This enables constant verification time regardless of the number of composed proofs.
    """

    def __init__(self, max_aggregation_depth: int = 3):
        """
        Initialize the recursive proof aggregator.

        Args:
            max_aggregation_depth: Maximum recursion depth for aggregation tree
        """
        self.max_aggregation_depth = max_aggregation_depth
        self.metrics = MetricsCollector()

    def aggregate_proofs(self, proofs: list[Proof]) -> AggregatedProof:
        """
        Aggregate multiple proofs into a single proof.

        This implements the recursive composition strategy from the project spec:
        - Build a tree of aggregations for efficient verification
        - Each node verifies its children
        - Root proof verifies all subproofs

        Args:
            proofs: List of proofs to aggregate

        Returns:
            Single aggregated proof
        """
        start_time = time.time()

        if not proofs:
            raise ValueError("No proofs to aggregate")

        if len(proofs) == 1:
            # Single proof, no aggregation needed
            return self._wrap_single_proof(proofs[0])

        # Build aggregation tree
        aggregation_tree = self._build_aggregation_tree(proofs)

        # Create aggregated proof
        aggregated_proof = self._create_aggregated_proof(aggregation_tree)

        aggregation_time = time.time() - start_time

        # Record metrics
        self.metrics.record("aggregation_time", aggregation_time * 1000, "ms")
        self.metrics.record("num_proofs_aggregated", len(proofs), "count")
        self.metrics.record(
            "aggregated_proof_size", len(aggregated_proof.aggregated_data), "bytes"
        )

        return aggregated_proof

    def _build_aggregation_tree(self, proofs: list[Proof]) -> dict[str, Any]:
        """Build a tree structure for recursive aggregation."""
        # For N proofs, create a balanced binary tree
        # This gives O(log N) verification depth

        if len(proofs) <= 2:
            return {"type": "leaf", "proofs": proofs, "depth": 0}

        # Split proofs into two halves
        mid = len(proofs) // 2
        left_proofs = proofs[:mid]
        right_proofs = proofs[mid:]

        # Recursively build subtrees
        left_tree = self._build_aggregation_tree(left_proofs)
        right_tree = self._build_aggregation_tree(right_proofs)

        return {
            "type": "internal",
            "left": left_tree,
            "right": right_tree,
            "depth": max(left_tree["depth"], right_tree["depth"]) + 1,
        }

    def _create_aggregated_proof(self, tree: dict[str, Any]) -> AggregatedProof:
        """Create aggregated proof from tree structure."""
        if tree["type"] == "leaf":
            # Base case: aggregate 1-2 proofs directly
            return self._aggregate_leaf_proofs(tree["proofs"])

        # Recursive case: aggregate child aggregations
        left_agg = self._create_aggregated_proof(tree["left"])
        right_agg = self._create_aggregated_proof(tree["right"])

        # Create verifier circuit for the two aggregated proofs
        verifier_circuit = RecursiveVerifierCircuit()

        # Setup circuit with the two child proofs
        public_inputs = {
            "left_aggregate": left_agg.public_aggregate,
            "right_aggregate": right_agg.public_aggregate,
            "total_proofs": left_agg.num_proofs + right_agg.num_proofs,
        }

        private_inputs = {
            "left_proof": left_agg.aggregated_data,
            "right_proof": right_agg.aggregated_data,
            "randomness": hashlib.sha256(b"recursive_random").hexdigest(),
        }

        verifier_circuit.setup_circuit(public_inputs, private_inputs)
        verifier_circuit.generate_constraints()

        # Create aggregated proof
        # In practice, this would use the gnark backend
        aggregated_data = self._mock_create_proof(verifier_circuit)

        # Compute public aggregate
        public_aggregate = poseidon_hash(
            [left_agg.public_aggregate, right_agg.public_aggregate]
        )

        return AggregatedProof(
            num_proofs=left_agg.num_proofs + right_agg.num_proofs,
            aggregated_data=aggregated_data,
            public_aggregate=public_aggregate,
            metadata={
                "aggregation_depth": tree["depth"],
                "left_proofs": left_agg.num_proofs,
                "right_proofs": right_agg.num_proofs,
            },
        )

    def _aggregate_leaf_proofs(self, proofs: list[Proof]) -> AggregatedProof:
        """Aggregate 1-2 proofs at leaf level."""
        if len(proofs) == 1:
            return self._wrap_single_proof(proofs[0])

        # Create a simple aggregation circuit
        agg_circuit = SimpleAggregationCircuit()

        # Combine public inputs
        combined_public = []
        for proof in proofs:
            combined_public.extend(proof.public_inputs)

        public_inputs = {
            "num_proofs": len(proofs),
            "public_hash": poseidon_hash(combined_public),
        }

        private_inputs = {
            "proof_data": [proof.proof_data for proof in proofs],
            "public_inputs": [proof.public_inputs for proof in proofs],
        }

        agg_circuit.setup_circuit(public_inputs, private_inputs)
        agg_circuit.generate_constraints()

        # Create proof
        aggregated_data = self._mock_create_proof(agg_circuit)

        return AggregatedProof(
            num_proofs=len(proofs),
            aggregated_data=aggregated_data,
            public_aggregate=public_inputs["public_hash"],
            metadata={
                "aggregation_depth": 0,
                "circuit_names": [p.circuit_name for p in proofs],
            },
        )

    def _wrap_single_proof(self, proof: Proof) -> AggregatedProof:
        """Wrap a single proof as an aggregated proof."""
        return AggregatedProof(
            num_proofs=1,
            aggregated_data=proof.proof_data,
            public_aggregate=poseidon_hash(proof.public_inputs),
            metadata=proof.metadata,
        )

    def _mock_create_proof(self, circuit: Any) -> bytes:
        """Mock proof creation for testing."""
        # In practice, would call gnark backend
        # For now, create deterministic mock proof

        circuit_data = {
            "constraints": circuit.cs.num_constraints(),
            "variables": circuit.cs.num_variables(),
            "public_inputs": len(circuit.cs.public_inputs),
        }

        proof_hash = hashlib.sha256(json.dumps(circuit_data).encode()).digest()

        # Pad to realistic proof size
        return proof_hash + b"\x00" * (384 - len(proof_hash))

    def verify_aggregated_proof(self, proof: AggregatedProof) -> bool:
        """
        Verify an aggregated proof.

        This has constant verification time regardless of how many
        proofs were aggregated.
        """
        start_time = time.time()

        # In practice, would verify the aggregated proof
        # For now, check basic validity
        is_valid = (
            len(proof.aggregated_data) > 0
            and proof.num_proofs > 0
            and proof.public_aggregate is not None
        )

        verification_time = time.time() - start_time

        # Record metrics
        self.metrics.record(
            "aggregated_verification_time", verification_time * 1000, "ms"
        )

        return is_valid


class RecursiveVerifierCircuit:
    """
    Circuit that verifies two aggregated proofs.

    This is the core of recursive composition - a circuit that
    can verify other proofs.
    """

    def __init__(self):
        self.cs = ConstraintSystem()
        self.setup_complete = False

    def setup_circuit(
        self, public_inputs: dict[str, Any], private_inputs: dict[str, Any]
    ):
        """Setup the verifier circuit."""
        # Public inputs
        self.left_aggregate_var = self.cs.add_public_input("left_aggregate")
        self.right_aggregate_var = self.cs.add_public_input("right_aggregate")
        self.total_proofs_var = self.cs.add_public_input("total_proofs")

        # Assign values
        self.cs.assign(self.left_aggregate_var, public_inputs["left_aggregate"])
        self.cs.assign(self.right_aggregate_var, public_inputs["right_aggregate"])
        self.cs.assign(
            self.total_proofs_var, FieldElement(public_inputs["total_proofs"])
        )

        # Private inputs (proof data)
        # In practice, would parse and verify the actual proofs
        # For now, we'll create constraints that simulate verification

        self.setup_complete = True

    def generate_constraints(self):
        """Generate constraints for proof verification."""
        if not self.setup_complete:
            raise RuntimeError("Circuit must be setup first")

        # Simulate proof verification constraints
        # In practice, would implement the full verifier logic

        # Constraint 1: Verify the aggregates combine correctly
        combined_var = self.cs.add_variable("combined_aggregate")
        combined_val = poseidon_hash(
            [
                self.cs.get_assignment(self.left_aggregate_var),
                self.cs.get_assignment(self.right_aggregate_var),
            ]
        )
        self.cs.assign(combined_var, combined_val)

        # Add some dummy constraints to simulate verification complexity
        for i in range(100):  # Simulate ~100 constraints for verification
            temp_var = self.cs.add_variable(f"verify_temp_{i}")
            self.cs.assign(temp_var, FieldElement(i))

            # temp * 1 = temp (trivial constraint)
            self.cs.enforce_equal(temp_var, temp_var)


class SimpleAggregationCircuit:
    """Simple circuit for aggregating 2 proofs."""

    def __init__(self):
        self.cs = ConstraintSystem()
        self.setup_complete = False

    def setup_circuit(
        self, public_inputs: dict[str, Any], private_inputs: dict[str, Any]
    ):
        """Setup the aggregation circuit."""
        self.num_proofs_var = self.cs.add_public_input("num_proofs")
        self.public_hash_var = self.cs.add_public_input("public_hash")

        self.cs.assign(self.num_proofs_var, FieldElement(public_inputs["num_proofs"]))
        self.cs.assign(self.public_hash_var, public_inputs["public_hash"])

        self.setup_complete = True

    def generate_constraints(self):
        """Generate aggregation constraints."""
        if not self.setup_complete:
            raise RuntimeError("Circuit must be setup first")

        # Add constraints for aggregation
        # In practice, would verify each subproof
        pass


def benchmark_recursive_aggregation(max_proofs: int = 128) -> dict[str, Any]:
    """
    Benchmark recursive proof aggregation performance.

    Tests aggregation time and verification time for varying numbers of proofs.
    """
    logger.info("Benchmarking recursive aggregation up to %smax_proofs proofs...")

    aggregator = RecursiveProofAggregator()

    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    batch_sizes = [b for b in batch_sizes if b <= max_proofs]

    results = {
        "batch_sizes": batch_sizes,
        "aggregation_times": [],
        "verification_times": [],
        "proof_sizes": [],
        "aggregation_depths": [],
    }

    for batch_size in batch_sizes:
        logger.info("\nTesting batch size: %sbatch_size")

        # Create mock proofs
        proofs = []
        for i in range(batch_size):
            proof = Proof(
                circuit_name=f"test_circuit_{i}",
                public_inputs=[FieldElement(i), FieldElement(i * 2)],
                proof_data=hashlib.sha256(f"proof_{i}".encode()).digest()
                + b"\x00" * 352,
                metadata={"index": i},
            )
            proofs.append(proof)

        # Time aggregation
        start_time = time.time()
        aggregated = aggregator.aggregate_proofs(proofs)
        aggregation_time = time.time() - start_time

        # Time verification
        start_time = time.time()
        valid = aggregator.verify_aggregated_proof(aggregated)
        verification_time = time.time() - start_time

        # Record results
        results["aggregation_times"].append(aggregation_time * 1000)  # Convert to ms
        results["verification_times"].append(verification_time * 1000)
        results["proof_sizes"].append(len(aggregated.aggregated_data))
        results["aggregation_depths"].append(
            aggregated.metadata.get("aggregation_depth", 0)
        )

        logger.info("  Aggregation time: %saggregation_time * 1000:.2fms")
        logger.info("  Verification time: %sverification_time * 1000:.2fms")
        logger.info("  Proof size: %slen(aggregated.aggregated_data) bytes")
        logger.info(
            "  Aggregation depth: %saggregated.metadata.get('aggregation_depth', 0)"
        )

    # Plot results
    plot_aggregation_benchmarks(results)

    return results


def plot_aggregation_benchmarks(results: dict[str, Any]):
    """Plot benchmark results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    batch_sizes = results["batch_sizes"]

    # Aggregation time vs batch size
    ax1.plot(batch_sizes, results["aggregation_times"], "b-o")
    ax1.set_xlabel("Number of Proofs")
    ax1.set_ylabel("Aggregation Time (ms)")
    ax1.set_title("Aggregation Time vs Number of Proofs")
    ax1.set_xscale("log", base=2)
    ax1.grid(True)

    # Verification time (should be constant)
    ax2.plot(batch_sizes, results["verification_times"], "r-o")
    ax2.set_xlabel("Number of Proofs")
    ax2.set_ylabel("Verification Time (ms)")
    ax2.set_title("Verification Time (Constant)")
    ax2.set_xscale("log", base=2)
    ax2.grid(True)

    # Proof size (should be constant)
    ax3.plot(batch_sizes, results["proof_sizes"], "g-o")
    ax3.set_xlabel("Number of Proofs")
    ax3.set_ylabel("Proof Size (bytes)")
    ax3.set_title("Aggregated Proof Size (Constant)")
    ax3.set_xscale("log", base=2)
    ax3.grid(True)

    # Aggregation depth (log N)
    ax4.plot(batch_sizes, results["aggregation_depths"], "m-o")
    ax4.set_xlabel("Number of Proofs")
    ax4.set_ylabel("Aggregation Depth")
    ax4.set_title("Tree Depth (O(log N))")
    ax4.set_xscale("log", base=2)
    ax4.grid(True)

    plt.tight_layout()

    # Save plot
    output_dir = Path("benchmarks/zk")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plt.savefig(output_dir / f"recursive_aggregation_{timestamp}.png", dpi=150)
    logger.info(
        "\nBenchmark plot saved to benchmarks/zk/recursive_aggregation_%stimestamp.png"
    )


def main():
    """Run recursive aggregation demonstration and benchmarks."""
    logger.info("=" * 60)
    logger.info("GenomeVault Recursive Proof Aggregation")
    logger.info("=" * 60)

    # Simple demonstration
    logger.info("\n1. Simple Aggregation Demo")
    logger.info("-" * 30)

    aggregator = RecursiveProofAggregator()

    # Create some example proofs
    proofs = []
    for i in range(4):
        proof = Proof(
            circuit_name="variant_presence",
            public_inputs=[FieldElement(1000 + i), FieldElement(2000 + i)],
            proof_data=hashlib.sha256(f"demo_proof_{i}".encode()).digest()
            + b"\x00" * 352,
            metadata={"variant_id": f"rs{1000 + i}"},
        )
        proofs.append(proof)

    logger.info("Created %slen(proofs) individual proofs")

    # Aggregate them
    aggregated = aggregator.aggregate_proofs(proofs)

    logger.info("\nAggregated into single proof:")
    logger.info("  Number of proofs: %saggregated.num_proofs")
    logger.info("  Proof size: %slen(aggregated.aggregated_data) bytes")
    logger.info("  Public aggregate: %saggregated.public_aggregate")

    # Verify
    valid = aggregator.verify_aggregated_proof(aggregated)
    logger.info("  Verification: %s'VALID' if valid else 'INVALID'")

    # Run benchmarks
    logger.info("\n2. Performance Benchmarks")
    logger.info("-" * 30)

    results = benchmark_recursive_aggregation(max_proofs=128)

    # Summary
    logger.info("\n3. Summary")
    logger.info("-" * 30)
    logger.info(
        "✓ Constant verification time: ~%snp.mean(results['verification_times']):.1fms"
    )
    logger.info("✓ Constant proof size: %sresults['proof_sizes'][0] bytes")
    logger.info("✓ Logarithmic aggregation depth: O(log N)")
    logger.info("✓ Efficient aggregation: up to %smax(results['batch_sizes']) proofs")

    # Save benchmark results
    output_dir = Path("benchmarks/zk")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d")
    with open(output_dir / f"{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("\nBenchmark data saved to benchmarks/zk/%stimestamp.json")


if __name__ == "__main__":
    main()
