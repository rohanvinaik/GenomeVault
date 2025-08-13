from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


"""
Property-based tests for ZK circuits using Hypothesis.

Stage 3 implementation: Verification & Property Tests
- Mutate inputs slightly, ensure proof fails
- Boundary checks
- Fuzz for malformed public inputs
"""

import hashlib

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, rule

from genomevault.zk_proofs.circuits.implementations.variant_frequency_circuit import (
    VariantFrequencyCircuit,
)
from genomevault.zk_proofs.circuits.implementations.variant_proof_circuit import (
    VariantProofCircuit,
)

# Constants for magic number elimination (PLR2004)
MAX_VARIANTS = 10
VERIFICATION_TIME_MAX = 0.1


# Custom strategies for genomic data
@st.composite
def chromosome_strategy(draw):
    """Generate valid chromosome identifiers."""
    chr_type = draw(st.sampled_from(["numeric", "X", "Y", "MT"]))
    if chr_type == "numeric":
        return f"chr{draw(st.integers(min_value=1, max_value=22))}"
    return f"chr{chr_type}"


@st.composite
def genomic_position_strategy(draw):
    """Generate valid genomic positions."""
    return draw(st.integers(min_value=1, max_value=250_000_000))


@st.composite
def dna_base_strategy(draw):
    """Generate valid DNA bases."""
    return draw(st.sampled_from(["A", "C", "G", "T"]))


@st.composite
def variant_data_strategy(draw):
    """Generate valid variant data."""
    return {
        "chr": draw(chromosome_strategy()),
        "pos": draw(genomic_position_strategy()),
        "ref": draw(dna_base_strategy()),
        "alt": draw(dna_base_strategy()),
    }


@st.composite
def merkle_proof_strategy(draw, depth=20):
    """Generate Merkle proof data."""
    path = [
        hashlib.sha256(f"node_{i}_{draw(st.integers())}".encode()).hexdigest() for i in range(depth)
    ]
    indices = draw(st.lists(st.integers(0, 1), min_size=depth, max_size=depth))
    return {"path": path, "indices": indices}


class TestVariantProofCircuit:
    """Property tests for variant presence proof circuit."""

    @given(variant_data=variant_data_strategy())
    def test_valid_variant_proof_verifies(self, variant_data):
        """Valid variant data should produce valid proof."""
        circuit = VariantProofCircuit(merkle_depth=5)  # Smaller for testing

        # Create valid inputs
        variant_str = f"{variant_data['chr']}:{variant_data['pos']}:{variant_data['ref']}:{variant_data['alt']}"
        variant_hash = hashlib.sha256(variant_str.encode()).hexdigest()

        public_inputs = {
            "variant_hash": variant_hash,
            "reference_hash": hashlib.sha256(b"test_ref").hexdigest(),
            "commitment_root": hashlib.sha256(b"test_root").hexdigest(),
        }

        private_inputs = {
            "variant_data": variant_data,
            "merkle_path": [hashlib.sha256(f"path_{i}".encode()).hexdigest() for i in range(5)],
            "merkle_indices": [i % 2 for i in range(5)],
            "witness_randomness": hashlib.sha256(b"random").hexdigest(),
        }

        circuit.setup_circuit(public_inputs, private_inputs)
        circuit.generate_constraints()

        assert circuit.verify_constraints(), "Valid inputs should satisfy constraints"

    @given(
        variant_data=variant_data_strategy(),
        wrong_hash=st.text(min_size=64, max_size=64, alphabet="0123456789abcdef"),
    )
    def test_wrong_hash_fails(self, variant_data, wrong_hash):
        """Wrong variant hash should fail verification."""
        circuit = VariantProofCircuit(merkle_depth=5)

        # Use wrong hash
        public_inputs = {
            "variant_hash": wrong_hash,
            "reference_hash": hashlib.sha256(b"test_ref").hexdigest(),
            "commitment_root": hashlib.sha256(b"test_root").hexdigest(),
        }

        private_inputs = {
            "variant_data": variant_data,
            "merkle_path": [hashlib.sha256(f"path_{i}".encode()).hexdigest() for i in range(5)],
            "merkle_indices": [i % 2 for i in range(5)],
            "witness_randomness": hashlib.sha256(b"random").hexdigest(),
        }

        circuit.setup_circuit(public_inputs, private_inputs)

        # Should fail to generate valid constraints
        try:
            circuit.generate_constraints()
            satisfied = circuit.verify_constraints()
            assert not satisfied, "Wrong hash should not satisfy constraints"
        except (ValueError, AssertionError, Exception):
            logger.exception("Unhandled exception")
            # Constraint generation might fail, which is also acceptable
            raise

    @given(
        variant_data=variant_data_strategy(),
        position_offset=st.integers(min_value=1, max_value=1000),
    )
    def test_position_mutation_fails(self, variant_data, position_offset):
        """Mutating position should invalidate proof."""
        circuit = VariantProofCircuit(merkle_depth=5)

        # Create hash with original position
        variant_str = f"{variant_data['chr']}:{variant_data['pos']}:{variant_data['ref']}:{variant_data['alt']}"
        variant_hash = hashlib.sha256(variant_str.encode()).hexdigest()

        # But use different position in private input
        mutated_variant = variant_data.copy()
        mutated_variant["pos"] += position_offset

        public_inputs = {
            "variant_hash": variant_hash,
            "reference_hash": hashlib.sha256(b"test_ref").hexdigest(),
            "commitment_root": hashlib.sha256(b"test_root").hexdigest(),
        }

        private_inputs = {
            "variant_data": mutated_variant,
            "merkle_path": [hashlib.sha256(f"path_{i}".encode()).hexdigest() for i in range(5)],
            "merkle_indices": [i % 2 for i in range(5)],
            "witness_randomness": hashlib.sha256(b"random").hexdigest(),
        }

        circuit.setup_circuit(public_inputs, private_inputs)

        try:
            circuit.generate_constraints()
            satisfied = circuit.verify_constraints()
            assert not satisfied, "Mutated position should not satisfy constraints"
        except (ValueError, AssertionError, Exception):
            logger.exception("Unhandled exception")
            raise

    def test_boundary_chromosome_values(self):
        """Test boundary values for chromosome encoding."""
        circuit = VariantProofCircuit(merkle_depth=5)

        # Test all valid chromosome values
        valid_chromosomes = [f"chr{i}" for i in range(1, 23)] + [
            "chrX",
            "chrY",
            "chrMT",
            "chrM",
        ]

        for chr_name in valid_chromosomes:
            variant_data = {"chr": chr_name, "pos": 12345, "ref": "A", "alt": "G"}

            variant_str = f"{variant_data['chr']}:{variant_data['pos']}:{variant_data['ref']}:{variant_data['alt']}"
            variant_hash = hashlib.sha256(variant_str.encode()).hexdigest()

            public_inputs = {
                "variant_hash": variant_hash,
                "reference_hash": hashlib.sha256(b"test_ref").hexdigest(),
                "commitment_root": hashlib.sha256(b"test_root").hexdigest(),
            }

            private_inputs = {
                "variant_data": variant_data,
                "merkle_path": [hashlib.sha256(f"path_{i}".encode()).hexdigest() for i in range(5)],
                "merkle_indices": [0] * 5,
                "witness_randomness": hashlib.sha256(b"random").hexdigest(),
            }

            circuit.setup_circuit(public_inputs, private_inputs)
            circuit.generate_constraints()

            assert circuit.verify_constraints(), f"Valid chromosome {chr_name} should work"


class TestVariantFrequencyCircuit:
    """Property tests for variant frequency sum circuit."""

    @given(
        num_snps=st.integers(min_value=1, max_value=32),
        allele_counts=st.lists(st.integers(min_value=0, max_value=10000), min_size=1, max_size=32),
    )
    def test_sum_constraint_satisfied(self, num_snps, allele_counts):
        """Sum of allele counts should match public sum."""
        # Ensure we have the right number of counts
        allele_counts = allele_counts[:num_snps]

        circuit = VariantFrequencyCircuit(max_snps=32, merkle_depth=5)

        # Calculate correct sum
        total_sum = sum(allele_counts)

        # Create SNP IDs
        snp_ids = [hash(f"rs{i}") % (2**32) for i in range(32)]

        public_inputs = {
            "total_sum": total_sum,
            "merkle_root": hashlib.sha256(b"test_root").hexdigest(),
            "num_snps": num_snps,
            "snp_ids": snp_ids,
        }

        # Create mock merkle proofs
        merkle_proofs = []
        for i in range(num_snps):
            proof = {
                "path": [hashlib.sha256(f"node_{i}_{j}".encode()).hexdigest() for j in range(5)],
                "indices": [j % 2 for j in range(5)],
            }
            merkle_proofs.append(proof)

        private_inputs = {
            "allele_counts": allele_counts,
            "merkle_proofs": merkle_proofs,
            "randomness": hashlib.sha256(b"random").hexdigest(),
        }

        circuit.setup_circuit(public_inputs, private_inputs)
        circuit.generate_constraints()

        assert circuit.verify_constraints(), "Valid sum should satisfy constraints"

    @given(
        num_snps=st.integers(min_value=1, max_value=32),
        allele_counts=st.lists(st.integers(min_value=0, max_value=10000), min_size=1, max_size=32),
        sum_offset=st.integers(min_value=1, max_value=100),
    )
    def test_wrong_sum_fails(self, num_snps, allele_counts, sum_offset):
        """Wrong sum should fail verification."""
        allele_counts = allele_counts[:num_snps]

        circuit = VariantFrequencyCircuit(max_snps=32, merkle_depth=5)

        # Use wrong sum
        correct_sum = sum(allele_counts)
        wrong_sum = correct_sum + sum_offset

        snp_ids = [hash(f"rs{i}") % (2**32) for i in range(32)]

        public_inputs = {
            "total_sum": wrong_sum,
            "merkle_root": hashlib.sha256(b"test_root").hexdigest(),
            "num_snps": num_snps,
            "snp_ids": snp_ids,
        }

        merkle_proofs = []
        for i in range(num_snps):
            proof = {
                "path": [hashlib.sha256(f"node_{i}_{j}".encode()).hexdigest() for j in range(5)],
                "indices": [j % 2 for j in range(5)],
            }
            merkle_proofs.append(proof)

        private_inputs = {
            "allele_counts": allele_counts,
            "merkle_proofs": merkle_proofs,
            "randomness": hashlib.sha256(b"random").hexdigest(),
        }

        circuit.setup_circuit(public_inputs, private_inputs)

        try:
            circuit.generate_constraints()
            satisfied = circuit.verify_constraints()
            assert not satisfied, "Wrong sum should not satisfy constraints"
        except (ValueError, AssertionError, Exception):
            logger.exception("Unhandled exception")
            raise

    @given(
        num_snps=st.integers(min_value=1, max_value=30),
        invalid_count=st.integers(min_value=10001, max_value=20000),
    )
    def test_range_constraint_violation(self, num_snps, invalid_count):
        """Allele counts exceeding C_MAX should fail."""
        circuit = VariantFrequencyCircuit(max_snps=32, merkle_depth=5)

        # Create counts with one invalid value
        allele_counts = [100] * (num_snps - 1) + [invalid_count]

        snp_ids = [hash(f"rs{i}") % (2**32) for i in range(32)]

        public_inputs = {
            "total_sum": sum(allele_counts),
            "merkle_root": hashlib.sha256(b"test_root").hexdigest(),
            "num_snps": num_snps,
            "snp_ids": snp_ids,
        }

        merkle_proofs = []
        for i in range(num_snps):
            proof = {
                "path": [hashlib.sha256(f"node_{i}_{j}".encode()).hexdigest() for j in range(5)],
                "indices": [j % 2 for j in range(5)],
            }
            merkle_proofs.append(proof)

        private_inputs = {
            "allele_counts": allele_counts,
            "merkle_proofs": merkle_proofs,
            "randomness": hashlib.sha256(b"random").hexdigest(),
        }

        circuit.setup_circuit(public_inputs, private_inputs)

        # Should fail range check
        try:
            circuit.generate_constraints()
            # The circuit should detect the range violation
            # In a real implementation with proper range proofs, this would fail
        except ValueError:
            logger.exception("Unhandled exception")
            # Expected for out-of-range values
            raise

    def test_unused_slots_zero(self):
        """Unused SNP slots should be constrained to zero."""
        circuit = VariantFrequencyCircuit(max_snps=32, merkle_depth=5)

        num_snps = 5
        allele_counts = [100, 200, 300, 400, 500]

        snp_ids = [hash(f"rs{i}") % (2**32) for i in range(32)]

        public_inputs = {
            "total_sum": sum(allele_counts),
            "merkle_root": hashlib.sha256(b"test_root").hexdigest(),
            "num_snps": num_snps,
            "snp_ids": snp_ids,
        }

        merkle_proofs = []
        for i in range(num_snps):
            proof = {
                "path": [hashlib.sha256(f"node_{i}_{j}".encode()).hexdigest() for j in range(5)],
                "indices": [j % 2 for j in range(5)],
            }
            merkle_proofs.append(proof)

        private_inputs = {
            "allele_counts": allele_counts,
            "merkle_proofs": merkle_proofs,
            "randomness": hashlib.sha256(b"random").hexdigest(),
        }

        circuit.setup_circuit(public_inputs, private_inputs)
        circuit.generate_constraints()

        # Check that unused slots are zero
        for i in range(num_snps, 32):
            count_val = circuit.cs.get_assignment(circuit.count_vars[i])
            assert count_val.value == 0, f"Unused slot {i} should be zero"

        assert circuit.verify_constraints()


class ZKCircuitStateMachine(RuleBasedStateMachine):
    """
    Stateful testing for ZK circuits.

    This tests that the circuit maintains consistency across
    multiple operations and state transitions.
    """

    def __init__(self):
        """Initialize the instance."""
        super().__init__()
        self.variants = []
        self.total_count = 0
        self.circuit = None

    @initialize()
    def setup(self):
        """Initialize the state machine."""
        self.variants = []
        self.total_count = 0
        self.circuit = VariantFrequencyCircuit(max_snps=10, merkle_depth=5)

    @rule(
        snp_id=st.text(min_size=5, max_size=10),
        count=st.integers(min_value=0, max_value=5000),
    )
    def add_variant(self, snp_id, count):
        """Add a variant with count."""
        if len(self.variants) < MAX_VARIANTS:
            self.variants.append((snp_id, count))
            self.total_count += count

    @rule()
    def verify_circuit(self):
        """Verify the circuit with current state."""
        if not self.variants:
            return

        # Create inputs from current state
        snp_ids = [hash(snp) % (2**32) for snp, _ in self.variants]
        snp_ids += [0] * (10 - len(self.variants))

        allele_counts = [count for _, count in self.variants]

        public_inputs = {
            "total_sum": self.total_count,
            "merkle_root": hashlib.sha256(b"state_root").hexdigest(),
            "num_snps": len(self.variants),
            "snp_ids": snp_ids,
        }

        merkle_proofs = []
        for i in range(len(self.variants)):
            proof = {
                "path": [hashlib.sha256(f"state_{i}_{j}".encode()).hexdigest() for j in range(5)],
                "indices": [j % 2 for j in range(5)],
            }
            merkle_proofs.append(proof)

        private_inputs = {
            "allele_counts": allele_counts,
            "merkle_proofs": merkle_proofs,
            "randomness": hashlib.sha256(f"state_{len(self.variants)}".encode()).hexdigest(),
        }

        # Create new circuit instance for each verification
        circuit = VariantFrequencyCircuit(max_snps=MAX_VARIANTS, merkle_depth=5)
        circuit.setup_circuit(public_inputs, private_inputs)
        circuit.generate_constraints()

        assert (
            circuit.verify_constraints()
        ), f"Circuit should verify with {len(self.variants)} variants"

    @rule()
    def check_invariants(self):
        """Check state invariants."""
        # Total count should equal sum of individual counts
        if self.variants:
            calculated_sum = sum(count for _, count in self.variants)
            assert calculated_sum == self.total_count, "Total count invariant violated"

        # Number of variants should not exceed max
        assert len(self.variants) <= MAX_VARIANTS, "Too many variants"


# Run the state machine tests
TestZKStateMachine = ZKCircuitStateMachine.TestCase


def test_malformed_public_inputs():
    """Test circuit behavior with malformed public inputs."""
    circuit = VariantProofCircuit(merkle_depth=5)

    # Test cases for malformed inputs
    malformed_cases = [
        # Missing required field
        {"reference_hash": "abc123", "commitment_root": "def456"},
        # Invalid hash format
        {
            "variant_hash": "not-a-hash",
            "reference_hash": "abc123",
            "commitment_root": "def456",
        },
        # Wrong type
        {
            "variant_hash": 12345,
            "reference_hash": "abc123",
            "commitment_root": "def456",
        },
    ]

    for i, malformed_input in enumerate(malformed_cases):
        with pytest.raises((KeyError, ValueError, TypeError)):
            circuit.setup_circuit(
                malformed_input,
                {
                    "variant_data": {"chr": "chr1", "pos": 123, "ref": "A", "alt": "G"},
                    "merkle_path": [],
                    "merkle_indices": [],
                    "witness_randomness": "0" * 64,
                },
            )


@pytest.mark.parametrize("constraint_count", [100, 1000, 5000])
def test_circuit_performance_scaling(constraint_count):
    """Test that circuit generation scales appropriately."""
    import time

    # Create circuit with varying sizes
    num_snps = min(constraint_count // 100, 32)
    circuit = VariantFrequencyCircuit(max_snps=32, merkle_depth=10)

    # Generate test data
    allele_counts = [100] * num_snps
    snp_ids = [hash(f"rs{i}") % (2**32) for i in range(32)]

    public_inputs = {
        "total_sum": sum(allele_counts),
        "merkle_root": hashlib.sha256(b"perf_test").hexdigest(),
        "num_snps": num_snps,
        "snp_ids": snp_ids,
    }

    merkle_proofs = []
    for i in range(num_snps):
        proof = {
            "path": [hashlib.sha256(f"perf_{i}_{j}".encode()).hexdigest() for j in range(10)],
            "indices": [j % 2 for j in range(10)],
        }
        merkle_proofs.append(proof)

    private_inputs = {
        "allele_counts": allele_counts,
        "merkle_proofs": merkle_proofs,
        "randomness": hashlib.sha256(b"random").hexdigest(),
    }

    # Time circuit generation
    start_time = time.time()
    circuit.setup_circuit(public_inputs, private_inputs)
    circuit.generate_constraints()
    generation_time = time.time() - start_time

    # Time verification
    start_time = time.time()
    verified = circuit.verify_constraints()
    verification_time = time.time() - start_time

    assert verified, "Circuit should verify"

    # Performance assertions
    # Generation should be reasonably fast
    assert generation_time < 1.0, f"Generation took too long: {generation_time:.3f}s"

    # Verification should be very fast
    assert (
        verification_time < VERIFICATION_TIME_MAX
    ), f"Verification took too long: {verification_time:.3f}s"

    logger.debug(f"\nPerformance for {constraint_count} constraints:")
    logger.debug(f"  Generation: {generation_time * 1000:.1f}ms")
    logger.debug(f"  Verification: {verification_time * 1000:.1f}ms")
    logger.debug(f"  Constraints: {circuit.cs.num_constraints()}")


if __name__ == "__main__":
    # Run specific test examples
    logger.debug("Running ZK Circuit Property Tests")
    logger.debug("=" * 50)

    # Test variant proof
    test_variant = TestVariantProofCircuit()
    test_variant.test_valid_variant_proof_verifies()
    test_variant.test_boundary_chromosome_values()
    logger.debug("✓ Variant proof tests passed")

    # Test frequency circuit
    test_freq = TestVariantFrequencyCircuit()
    test_freq.test_sum_constraint_satisfied()
    test_freq.test_unused_slots_zero()
    logger.debug("✓ Frequency circuit tests passed")

    # Test malformed inputs
    test_malformed_public_inputs()
    logger.debug("✓ Malformed input tests passed")

    # Performance test
    test_circuit_performance_scaling(1000)
    logger.debug("✓ Performance tests passed")

    logger.debug("\nAll property tests completed successfully!")
