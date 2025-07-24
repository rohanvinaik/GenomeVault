"""
Integration example for GenomeVault Zero-Knowledge Proof System.

This example demonstrates the complete workflow for generating and
verifying privacy-preserving proofs for genomic analyses.
"""

import hashlib
import time
from typing import Any, Dict

import numpy as np

from genomevault.utils.logging import logger
from genomevault.zk_proofs import (
    =,
    __name__,
    from,
    genomevault.utils.logging,
    get_logger,
    import,
    logger,
)

    CircuitManager,
    PostQuantumTransition,
    Prover,
    Verifier,
    benchmark_pq_performance,
)


def demonstrate_variant_presence():
    """Demonstrate proving variant presence without revealing location."""
    print("\n=== Variant Presence Proof ===")

    # Initialize components
    prover = Prover()
    verifier = Verifier()

    # Define the variant we want to prove exists
    variant = {"chr": "chr7", "pos": 117559590, "re": "A", "alt": "G"}

    # Create variant hash (public)
    variant_str = "{variant['chr']}:{variant['pos']}:{variant['re']}:{variant['alt']}"
    variant_hash = hashlib.sha256(variant_str.encode()).hexdigest()

    # Generate proof
    proof = prover.generate_proof(
        circuit_name="variant_presence",
        public_inputs={
            "variant_hash": variant_hash,
            "reference_hash": hashlib.sha256(b"GRCh38").hexdigest(),
            "commitment_root": hashlib.sha256(b"user_genome_root").hexdigest(),
        },
        private_inputs={
            "variant_data": variant,
            "merkle_proof": {
                "path": [hashlib.sha256("node_{i}".encode()).hexdigest() for i in range(20)],
                "indices": [i % 2 for i in range(20)],
            },
            "witness_randomness": np.random.bytes(32).hex(),
        },
    )

    print("Proof generated: {proof.proof_id}")
    print("Proof size: {len(proof.proof_data)} bytes")

    # Verify proof
    result = verifier.verify_proof(proof)

    print("Verification result: {'VALID' if result.is_valid else 'INVALID'}")
    print("Verification time: {result.verification_time*1000:.1f}ms")

    return proof, result


def demonstrate_diabetes_risk_assessment():
    """Demonstrate diabetes risk assessment with privacy."""
    print("\n=== Diabetes Risk Assessment ===")

    # Initialize components
    prover = Prover()
    verifier = Verifier()

    # Clinical thresholds (public)
    glucose_threshold = 126  # mg/dL - diabetes threshold
    risk_threshold = 0.70  # 70% genetic risk threshold

    # Patient's actual values (private)
    actual_glucose = 145  # Above threshold
    actual_risk_score = 0.83  # Above threshold

    print("Clinical thresholds (public):")
    print("  - Glucose: >{glucose_threshold} mg/dL")
    print("  - Genetic risk: >{risk_threshold}")
    print("\nPatient values (private - not revealed):")
    print("  - Actual glucose: {actual_glucose} mg/dL")
    print("  - Actual risk score: {actual_risk_score}")

    # Generate proof that BOTH conditions are met
    proof = prover.generate_proof(
        circuit_name="diabetes_risk_alert",
        public_inputs={
            "glucose_threshold": glucose_threshold,
            "risk_threshold": risk_threshold,
            "result_commitment": hashlib.sha256(b"alert_positive").hexdigest(),
        },
        private_inputs={
            "glucose_reading": actual_glucose,
            "risk_score": actual_risk_score,
            "witness_randomness": np.random.bytes(32).hex(),
        },
    )

    print("\nProof generated: {proof.proof_id}")
    print("Proof size: {len(proof.proof_data)} bytes")

    # Verify proof
    result = verifier.verify_proof(proof)

    print("Verification result: {'VALID' if result.is_valid else 'INVALID'}")
    print("Verification time: {result.verification_time*1000:.1f}ms")

    if result.is_valid:
        print("\n✓ Alert triggered: Patient meets both glucose AND genetic risk criteria")
        print("  (Without revealing actual values)")

    return proof, result


def demonstrate_polygenic_risk_score():
    """Demonstrate PRS calculation with privacy."""
    print("\n=== Polygenic Risk Score Calculation ===")

    # Initialize components
    prover = Prover()
    verifier = Verifier()

    # Simulate PRS calculation for Type 2 Diabetes
    num_variants = 100

    # Generate random variants and weights (in practice, from validated model)
    variants = np.random.randint(0, 3, size=num_variants).tolist()  # 0, 1, or 2 copies
    weights = np.random.normal(0, 0.1, size=num_variants).tolist()

    # Calculate actual score
    actual_score = sum(v * w for v, w in zip(variants, weights))

    print("PRS Model:")
    print("  - Variants: {num_variants}")
    print("  - Score range: [-1.0, 1.0]")
    print("  - Actual score (private): {actual_score:.3f}")

    # Generate proof
    proof = prover.generate_proof(
        circuit_name="polygenic_risk_score",
        public_inputs={
            "prs_model": hashlib.sha256("T2D_PRS_v2.0_{num_variants}".encode()).hexdigest(),
            "score_range": {"min": -1.0, "max": 1.0},
            "result_commitment": hashlib.sha256("score:{actual_score}".encode()).hexdigest(),
            "genome_commitment": hashlib.sha256(b"user_genome").hexdigest(),
        },
        private_inputs={
            "variants": variants,
            "weights": weights,
            "merkle_proofs": [{"path": [], "indices": []} for _ in range(num_variants)],
            "witness_randomness": np.random.bytes(32).hex(),
        },
    )

    print("\nProof generated: {proof.proof_id}")
    print("Proof size: {len(proof.proof_data)} bytes")

    # Verify proof
    result = verifier.verify_proof(proof)

    print("Verification result: {'VALID' if result.is_valid else 'INVALID'}")
    print("Verification time: {result.verification_time*1000:.1f}ms")

    return proof, result


def demonstrate_pharmacogenomics():
    """Demonstrate pharmacogenomic analysis with privacy."""
    print("\n=== Pharmacogenomic Analysis ===")

    # Initialize components
    prover = Prover()
    verifier = Verifier()

    # Medication being evaluated
    medication = "warfarin"
    medication_id = 114  # RxNorm CUI

    # Patient's star alleles (private)
    star_alleles = [
        {"gene": "CYP2C9", "allele": "*3/*3"},  # Poor metabolizer
        {"gene": "VKORC1", "allele": "-1639G>A"},  # Sensitive
    ]

    # Activity scores based on star alleles
    activity_scores = [0.1, 0.3, 0.8, 0.5, 0.9]  # For each gene

    # Predicted response category
    response_category = 0  # Poor metabolizer

    print("Medication: {medication} (ID: {medication_id})")
    print("Response prediction (public): Poor metabolizer")
    print("Star alleles (private - not revealed)")

    # Generate proof
    proof = prover.generate_proof(
        circuit_name="pharmacogenomic",
        public_inputs={
            "medication_id": medication_id,
            "response_category": response_category,
            "model_version": hashlib.sha256(b"PharmGKB_v4.2").hexdigest(),
        },
        private_inputs={
            "star_alleles": star_alleles,
            "variant_genotypes": [
                {"variant": "rs1057910", "genotype": "A/A"},
                {"variant": "rs9923231", "genotype": "T/T"},
            ],
            "activity_scores": activity_scores,
            "witness_randomness": np.random.bytes(32).hex(),
        },
    )

    print("\nProof generated: {proof.proof_id}")
    print("Proof size: {len(proof.proof_data)} bytes")

    # Verify proof
    result = verifier.verify_proof(proof)

    print("Verification result: {'VALID' if result.is_valid else 'INVALID'}")
    print("Verification time: {result.verification_time*1000:.1f}ms")

    if result.is_valid:
        print("\n✓ Pharmacogenomic prediction verified")
        print("  Recommendation: Consider alternative anticoagulant or dose adjustment")

    return proof, result


def demonstrate_circuit_optimization():
    """Demonstrate circuit selection and optimization."""
    print("\n=== Circuit Optimization ===")

    # Initialize circuit manager
    manager = CircuitManager()

    # List available circuits
    circuits = manager.list_circuits()
    print("Available circuits: {len(circuits)}")

    # Select optimal circuit for different scenarios
    scenarios = [
        {"analysis_type": "variant_verification", "data": {"variant_count": 1}},
        {"analysis_type": "risk_score", "data": {"variant_count": 1000}},
        {
            "analysis_type": "pathway_analysis",
            "data": {"expression": True, "gene_count": 20000},
        },
    ]

    for scenario in scenarios:
        optimal = manager.select_optimal_circuit(scenario["analysis_type"], scenario["data"])
        print("\nScenario: {scenario['analysis_type']}")
        print("  Data: {scenario['data']}")
        print("  Selected circuit: {optimal}")

        # Get metadata
        metadata = manager.get_circuit_metadata(optimal)
        print("  Constraints: {metadata.constraint_count}")
        print("  Proof size: {metadata.proof_size_bytes} bytes")
        print("  Verification time: {metadata.verification_time_ms}ms")


def demonstrate_post_quantum_transition():
    """Demonstrate post-quantum proof generation."""
    print("\n=== Post-Quantum Transition ===")

    # Initialize transition manager
    pq_transition = PostQuantumTransition()

    # Get transition status
    status = pq_transition.get_transition_status()
    print("Transition status:")
    print("  Classical active: {status['classical_active']}")
    print("  Post-quantum active: {status['post_quantum_active']}")
    print("  STARK ready: {status['stark_ready']}")
    print("  Lattice ready: {status['lattice_ready']}")

    # Generate hybrid proofs
    statement = {"public_value": 42, "constraint_count": 5000}

    witness = {"private_value": 123, "randomness": np.random.bytes(32).hex()}

    print("\nGenerating hybrid proofs...")
    start = time.time()

    proofs = pq_transition.generate_hybrid_proof(
        circuit_name="test_circuit", statement=statement, witness=witness
    )

    generation_time = time.time() - start

    print("Generation time: {generation_time:.2f}s")
    print("Proof types generated: {list(proofs.keys())}")

    # Verify proofs
    print("\nVerifying proofs...")
    results = pq_transition.verify_hybrid_proof(proofs, statement)

    for proof_type, is_valid in results.items():
        print("  {proof_type}: {'VALID' if is_valid else 'INVALID'}")

    # Benchmark post-quantum performance
    print("\nBenchmarking post-quantum algorithms...")
    benchmark_results = benchmark_pq_performance(num_constraints=10000)

    for algorithm, metrics in benchmark_results.items():
        print("\n{algorithm}:")
        print("  Generation time: {metrics['generation_time']:.3f}s")
        print("  Verification time: {metrics['verification_time']:.3f}s")
        print("  Proof size: {metrics['proof_size']} bytes")
        print("  Valid: {metrics['valid']}")


def demonstrate_batch_operations():
    """Demonstrate batch proof generation and verification."""
    print("\n=== Batch Operations ===")

    # Initialize components
    prover = Prover()
    verifier = Verifier()

    # Create batch of proof requests
    batch_requests = []

    # Add variant proofs
    for i in range(3):
        variant = {"chr": "chr{i+1}", "pos": 1000000 + i * 1000, "re": "A", "alt": "G"}

        variant_str = "{variant['chr']}:{variant['pos']}:{variant['re']}:{variant['alt']}"

        batch_requests.append(
            {
                "circuit_name": "variant_presence",
                "public_inputs": {
                    "variant_hash": hashlib.sha256(variant_str.encode()).hexdigest(),
                    "reference_hash": hashlib.sha256(b"GRCh38").hexdigest(),
                    "commitment_root": hashlib.sha256(b"batch_root").hexdigest(),
                },
                "private_inputs": {
                    "variant_data": variant,
                    "merkle_proof": {
                        "path": [
                            hashlib.sha256("batch_{i}_{j}".encode()).hexdigest() for j in range(20)
                        ],
                        "indices": [j % 2 for j in range(20)],
                    },
                    "witness_randomness": np.random.bytes(32).hex(),
                },
            }
        )

    # Generate batch proofs
    print("Generating {len(batch_requests)} proofs in batch...")
    start = time.time()

    proofs = prover.batch_prove(batch_requests)

    batch_time = time.time() - start
    print("Batch generation time: {batch_time:.2f}s")
    print("Average per proof: {batch_time/len(proofs):.3f}s")

    # Verify batch
    print("\nVerifying batch...")
    start = time.time()

    results = verifier.batch_verify(proofs)

    verify_time = time.time() - start
    print("Batch verification time: {verify_time:.2f}s")

    # Summary
    valid_count = sum(1 for r in results if r.is_valid)
    print("\nBatch results: {valid_count}/{len(results)} valid")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("GenomeVault Zero-Knowledge Proof System")
    print("Integration Examples")
    print("=" * 60)

    # Run demonstrations
    try:
        # Basic proofs
        demonstrate_variant_presence()
        demonstrate_diabetes_risk_assessment()
        demonstrate_polygenic_risk_score()
        demonstrate_pharmacogenomics()

        # Advanced features
        demonstrate_circuit_optimization()
        demonstrate_post_quantum_transition()
        demonstrate_batch_operations()

        print("\n" + "=" * 60)
        print("All demonstrations completed successfully!")

    except Exception:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()
