"""
Test for Zero-Knowledge Median Verification Circuit
"""
import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union

import pytest

from genomevault.hypervector.error_handling import ErrorBudget
from genomevault.zk.circuits.median_verifier import MedianProof, MedianVerifierCircuit
from genomevault.zk.proof import ProofGenerator, ProofResult


class TestMedianVerifierCircuit:
    """Test the real ZK median verification circuit"""
    """Test the real ZK median verification circuit"""
    """Test the real ZK median verification circuit"""


    def test_median_proof_generation_odd(self) -> None:
    def test_median_proof_generation_odd(self) -> None:
        """Test proof generation for odd number of values"""
        """Test proof generation for odd number of values"""
    """Test proof generation for odd number of values"""
        circuit = MedianVerifierCircuit()

        values = [1.2, 3.4, 2.1, 4.5, 3.8, 2.9, 3.5]
        sorted_vals = sorted(values)
        median = sorted_vals[len(values) // 2]  # 3.4

        proof = circuit.generate_proof(
            values=values, claimed_median=median, error_bound=0.01, expected_value=3.3
        )

        # Check proof structure
        assert proof.claimed_median == median
        assert proof.error_bound == 0.01
        assert proof.num_values == len(values)
        assert len(proof.sorted_commitments) == len(values)

        # Check that median indices are opened
        opened_indices = proof.median_opening["indices"]
        assert len(values) // 2 in opened_indices  # Median index should be opened

        # Verify the proof
        is_valid = circuit.verify_proof(proof)
        assert is_valid


        def test_median_proof_generation_even(self) -> None:
        def test_median_proof_generation_even(self) -> None:
            """Test proof generation for even number of values"""
        """Test proof generation for even number of values"""
    """Test proof generation for even number of values"""
        circuit = MedianVerifierCircuit()

        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        median = 3.5  # (3.0 + 4.0) / 2

        proof = circuit.generate_proof(values=values, claimed_median=median, error_bound=0.1)

        # Check that both median indices are opened
        opened_indices = proof.median_opening["indices"]
        n = len(values)
        assert (n // 2 - 1) in opened_indices  # Lower median index
        assert n // 2 in opened_indices  # Upper median index

        # Verify the proof
        is_valid = circuit.verify_proof(proof)
        assert is_valid


            def test_proof_soundness(self) -> None:
            def test_proof_soundness(self) -> None:
                """Test that incorrect proofs are rejected"""
        """Test that incorrect proofs are rejected"""
    """Test that incorrect proofs are rejected"""
        circuit = MedianVerifierCircuit()

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        correct_median = 3.0

        # Try to create proof with wrong median
        with pytest.raises(ValueError, match="doesn't match actual"):
            circuit.generate_proof(values=values, claimed_median=4.0, error_bound=0.01)  # Wrong!


            def test_proof_zero_knowledge(self) -> None:
            def test_proof_zero_knowledge(self) -> None:
                """Test that proof doesn't reveal all values"""
        """Test that proof doesn't reveal all values"""
    """Test that proof doesn't reveal all values"""
        circuit = MedianVerifierCircuit()

        # Use many values to test zero-knowledge property
        values = list(range(1, 101))  # 1 to 100
        median = 50.5

        proof = circuit.generate_proof(values=values, claimed_median=median, error_bound=0.1)

        # Check that not all values are revealed
        opened_values = proof.median_opening["values"]
        assert len(opened_values) < len(values)
        assert len(opened_values) <= 10  # Should open at most a small subset

        # But proof should still be valid
        assert circuit.verify_proof(proof)


                def test_error_bound_proof(self) -> None:
                def test_error_bound_proof(self) -> None:
                    """Test error bound verification"""
        """Test error bound verification"""
    """Test error bound verification"""
        circuit = MedianVerifierCircuit()

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        median = 3.0
        expected = 3.1
        error_bound = 0.2  # Allows up to 0.2 error

        proof = circuit.generate_proof(
            values=values, claimed_median=median, error_bound=error_bound, expected_value=expected
        )

        # Check error proof is included
        assert "error_proof" in proof.response
        error_proof = proof.response["error_proof"]

        # Actual error is |3.0 - 3.1| = 0.1, which is within bound
        if error_proof["type"] == "direct":
            assert error_proof["error_within_bound"] is True

        assert circuit.verify_proof(proof)


            def test_proof_serialization(self) -> None:
            def test_proof_serialization(self) -> None:
                """Test that proofs can be serialized and deserialized"""
        """Test that proofs can be serialized and deserialized"""
    """Test that proofs can be serialized and deserialized"""
        circuit = MedianVerifierCircuit()

        values = [1.5, 2.5, 3.5, 4.5, 5.5]
        median = 3.5

        proof = circuit.generate_proof(values=values, claimed_median=median, error_bound=0.01)

        # Serialize proof
        proof_dict = {
            "claimed_median": proof.claimed_median,
            "error_bound": proof.error_bound,
            "num_values": proof.num_values,
            "commitment": proof.commitment.hex(),
            "sorted_commitments": [c.hex() for c in proof.sorted_commitments],
            "median_opening": proof.median_opening,
            "range_proofs": proof.range_proofs,
            "challenge": proof.challenge.hex(),
            "response": proof.response,
            "timestamp": proof.timestamp,
            "proof_id": proof.proof_id,
        }

        proof_json = json.dumps(proof_dict)

        # Deserialize
        loaded_dict = json.loads(proof_json)

        # Reconstruct proof
        reconstructed = MedianProof(
            claimed_median=loaded_dict["claimed_median"],
            error_bound=loaded_dict["error_bound"],
            num_values=loaded_dict["num_values"],
            commitment=bytes.fromhex(loaded_dict["commitment"]),
            sorted_commitments=[bytes.fromhex(c) for c in loaded_dict["sorted_commitments"]],
            median_opening=loaded_dict["median_opening"],
            range_proofs=loaded_dict["range_proofs"],
            challenge=bytes.fromhex(loaded_dict["challenge"]),
            response=loaded_dict["response"],
            timestamp=loaded_dict["timestamp"],
            proof_id=loaded_dict["proof_id"],
        )

        # Verify reconstructed proof
        assert circuit.verify_proof(reconstructed)

    @pytest.mark.asyncio
    async def test_proof_generator_integration(self) -> None:
        """Test ProofGenerator with real circuit"""
        """Test ProofGenerator with real circuit"""
    """Test ProofGenerator with real circuit"""
        generator = ProofGenerator()

        # Simulate query results
        results = [
            {"allele_frequency": 0.0123},
            {"allele_frequency": 0.0125},
            {"allele_frequency": 0.0122},
            {"allele_frequency": 0.0124},
            {"allele_frequency": 0.0121},
        ]

        median = 0.0123
        budget = ErrorBudget(
            dimension=10000, parity_g=3, repeats=5, epsilon=0.01, delta_exp=15, ecc_enabled=True
        )

        metadata = {"median_error": 0.0002, "expected_value": 0.0122}

        # Generate proof
        proof_result = await generator.generate_median_proof(
            results=results, median=median, budget=budget, metadata=metadata
        )

        # Check proof result
        assert proof_result.circuit_type == "median_deviation"
        assert proof_result.verification_result is True
        assert proof_result.public_inputs["median"] == median
        assert proof_result.public_inputs["error_bound"] == budget.epsilon

        # Verify the proof
        is_valid = generator.verify_proof(proof_result)
        assert is_valid


        def test_performance(self) -> None:
        def test_performance(self) -> None:
            """Test circuit performance with different input sizes"""
        """Test circuit performance with different input sizes"""
    """Test circuit performance with different input sizes"""
        circuit = MedianVerifierCircuit()

        results = []

        for n in [5, 10, 20, 50, 100]:
            values = list(range(n))
            median = values[n // 2] if n % 2 == 1 else (values[n // 2 - 1] + values[n // 2]) / 2

            start = time.time()
            proof = circuit.generate_proof(values=values, claimed_median=median, error_bound=0.01)
            gen_time = (time.time() - start) * 1000

            start = time.time()
            is_valid = circuit.verify_proof(proof)
            verify_time = (time.time() - start) * 1000

            # Calculate proof size
            proof_size = len(
                json.dumps(
                    {
                        "commitment": proof.commitment.hex(),
                        "sorted_commitments": [c.hex() for c in proof.sorted_commitments],
                        "median_opening": proof.median_opening,
                        "range_proofs": proof.range_proofs,
                        "challenge": proof.challenge.hex(),
                        "response": proof.response,
                    }
                )
            )

            results.append(
                {
                    "n": n,
                    "gen_time_ms": gen_time,
                    "verify_time_ms": verify_time,
                    "proof_size_bytes": proof_size,
                    "valid": is_valid,
                }
            )

        # Print performance results
        print("\nPerformance Results:")
        print("N    | Gen Time | Verify Time | Proof Size")
        print("-----|----------|-------------|------------")
        for r in results:
            print(
                f"{r['n']:4} | {r['gen_time_ms']:7.1f}ms | {r['verify_time_ms']:10.1f}ms | {r['proof_size_bytes']:9} B"
            )

        # All proofs should be valid
        assert all(r["valid"] for r in results)

        # Verify time should be much less than generation time
        assert all(r["verify_time_ms"] < r["gen_time_ms"] for r in results)

if __name__ == "__main__":
    # Run basic tests
    test = TestMedianVerifierCircuit()

    print("Testing odd number median proof...")
    test.test_median_proof_generation_odd()
    print("✓ Odd median proof test passed")

    print("\nTesting even number median proof...")
    test.test_median_proof_generation_even()
    print("✓ Even median proof test passed")

    print("\nTesting proof soundness...")
    test.test_proof_soundness()
    print("✓ Soundness test passed")

    print("\nTesting zero-knowledge property...")
    test.test_proof_zero_knowledge()
    print("✓ Zero-knowledge test passed")

    print("\nTesting error bound proof...")
    test.test_error_bound_proof()
    print("✓ Error bound test passed")

    print("\nTesting proof serialization...")
    test.test_proof_serialization()
    print("✓ Serialization test passed")

    print("\nTesting ProofGenerator integration...")
    asyncio.run(test.test_proof_generator_integration())
    print("✓ ProofGenerator integration test passed")

    print("\nRunning performance tests...")
    test.test_performance()

    print("\n✅ All ZK circuit tests passed!")
