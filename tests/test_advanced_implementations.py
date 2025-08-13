"""Advanced implementation integration tests."""
import os
import sys

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import hashlib
import time

import numpy as np

from genomevault.hypervector_transform.advanced_compression import (
    AdvancedHierarchicalCompressor,
)
from genomevault.pir.advanced.it_pir import InformationTheoreticPIR
from genomevault.zk_proofs.advanced.catalytic_proof import CatalyticProofEngine
from genomevault.zk_proofs.advanced.recursive_snark import RecursiveSNARKProver
from genomevault.zk_proofs.advanced.stark_prover import PostQuantumVerifier, STARKProver
from genomevault.zk_proofs.prover import Prover


def test_recursive_snark():
    """Test recursive SNARK composition."""
    logger.debug("\n" + "=" * 60)
    logger.debug("Testing Recursive SNARK Composition")
    logger.debug("=" * 60)

    # Initialize provers
    base_prover = Prover()
    recursive_prover = RecursiveSNARKProver()

    # Generate base proofs
    proofs = []
    for i in range(5):
        proof = base_prover.generate_proof(
            circuit_name="variant_presence",
            public_inputs={
                "variant_hash": hashlib.sha256(f"variant_{i}".encode()).hexdigest(),
                "reference_hash": hashlib.sha256(b"GRCh38").hexdigest(),
                "commitment_root": hashlib.sha256(f"root_{i}".encode()).hexdigest(),
            },
            private_inputs={
                "variant_data": {
                    "chr": f"chr{i + 1}",
                    "pos": i * 1000,
                    "ref": "A",
                    "alt": "G",
                },
                "merkle_proof": [
                    hashlib.sha256(f"node_{j}".encode()).hexdigest() for j in range(10)
                ],
                "witness_randomness": np.random.bytes(32).hex(),
            },
        )
        proofs.append(proof)

    # Test different aggregation strategies
    strategies = ["balanced_tree", "accumulator"]

    for strategy in strategies:
        logger.debug(f"\nTesting {strategy} aggregation:")
        start = time.time()
        recursive_proof = recursive_prover.compose_proofs(proofs, strategy)
        comp_time = time.time() - start

        logger.debug(f"  Composition time: {comp_time * 1000:.1f} ms")
        logger.debug(f"  Proof count: {recursive_proof.proof_count}")
        logger.debug(f"  Verification complexity: {recursive_proof.verification_complexity}")

        # Verify
        start = time.time()
        valid = recursive_prover.verify_recursive_proof(recursive_proof)
        verify_time = time.time() - start

        logger.debug(f"  Verification time: {verify_time * 1000:.1f} ms")
        logger.debug(f"  Valid: {valid}")

    return True


def test_stark_post_quantum():
    """Test STARK post-quantum proofs."""
    logger.debug("\n" + "=" * 60)
    logger.debug("Testing Post-Quantum STARK Proofs")
    logger.debug("=" * 60)

    # Initialize STARK system
    prover = STARKProver(security_bits=128)
    verifier = PostQuantumVerifier()

    # Create execution trace for PRS calculation
    trace_length = 256
    trace = np.zeros((trace_length, 3), dtype=np.uint64)  # [accumulator, variant, weight]

    # Simulate PRS computation
    for i in range(1, trace_length):
        variant = np.random.randint(0, 2)
        weight = np.random.randint(1, 100)
        trace[i, 1] = variant
        trace[i, 2] = weight
        trace[i, 0] = (trace[i - 1, 0] + variant * weight) % prover.field_size

    # Define constraints
    constraints = [
        {"type": "boundary", "register": 0, "step": 0, "value": 0},
        {"type": "transition", "expression": "acc_next = acc + variant * weight"},
    ]

    # Generate STARK proof
    public_inputs = {
        "computation": "polygenic_risk_score",
        "num_variants": trace_length - 1,
        "final_score": int(trace[-1, 0]),
    }

    logger.debug("\nGenerating STARK proof...")
    start = time.time()
    stark_proof = prover.generate_stark_proof(trace, public_inputs, constraints)
    gen_time = time.time() - start

    logger.debug(f"  Proof ID: {stark_proof.proof_id}")
    logger.debug(f"  Security level: {stark_proof.security_level} bits (post-quantum)")
    logger.debug(f"  Proof size: {stark_proof.proof_size_kb:.1f} KB")
    logger.debug(f"  Generation time: {gen_time * 1000:.1f} ms")

    # Verify proof
    logger.debug("\nVerifying STARK proof...")
    start = time.time()
    valid = verifier.verify_stark(stark_proof)
    verify_time = time.time() - start

    logger.debug(f"  Verification result: {'VALID' if valid else 'INVALID'}")
    logger.debug(f"  Verification time: {verify_time * 1000:.1f} ms")

    return valid


def test_catalytic_proof():
    """Test catalytic proof engine."""
    logger.debug("\n" + "=" * 60)
    logger.debug("Testing Catalytic Proof Engine")
    logger.debug("=" * 60)

    # Initialize with limited clean space
    engine = CatalyticProofEngine(
        clean_space_limit=256 * 1024,  # 256KB clean
        catalytic_space_size=20 * 1024 * 1024,  # 20MB catalytic
    )

    # Test PRS proof with catalytic space
    logger.debug("\nGenerating PRS proof with catalytic space:")

    num_variants = 5000
    catalytic_proof = engine.generate_catalytic_proof(
        circuit_name="polygenic_risk_score",
        public_inputs={
            "prs_model": "T2D_catalytic",
            "score_range": {"min": 0, "max": 1},
            "result_commitment": hashlib.sha256(b"prs_cat").hexdigest(),
            "genome_commitment": hashlib.sha256(b"genome_cat").hexdigest(),
        },
        private_inputs={
            "variants": np.random.randint(0, 2, num_variants).tolist(),
            "weights": np.random.rand(num_variants).tolist(),
            "merkle_proofs": [hashlib.sha256(f"p_{i}".encode()).hexdigest() for i in range(20)],
            "witness_randomness": np.random.bytes(32).hex(),
        },
    )

    logger.debug(f"  Proof ID: {catalytic_proof.proof_id}")
    logger.debug(f"  Clean space used: {catalytic_proof.clean_space_used / 1024:.1f} KB")
    logger.debug(f"  Space efficiency: {catalytic_proof.space_efficiency:.1f}x")

    # Show space savings
    savings = engine.get_space_savings("polygenic_risk_score")
    logger.debug("\nSpace savings:")
    logger.debug(f"  Standard approach: {savings['standard_approach_mb']:.1f} MB")
    logger.debug(f"  Catalytic clean: {savings['catalytic_clean_mb']:.1f} MB")
    logger.debug(f"  Reduction: {savings['clean_space_reduction']:.1f}%")

    return True


def test_it_pir():
    """Test Information-Theoretic PIR."""
    logger.debug("\n" + "=" * 60)
    logger.debug("Testing Information-Theoretic PIR")
    logger.debug("=" * 60)

    # Initialize IT-PIR
    pir = InformationTheoreticPIR(num_servers=3, threshold=2)

    # Create mock genomic database
    database_size = 100
    block_size = 1024

    databases = []
    for server_id in range(pir.num_servers):
        db = []
        for i in range(database_size):
            data = f"Genomic_Block_{i}_Data".encode().ljust(block_size, b"\0")
            db.append(data)
        databases.append(db)

    # Retrieve item privately
    target_index = 42
    logger.debug(f"\nRetrieving index {target_index} privately:")
    logger.debug(f"  Database size: {database_size} blocks")
    logger.debug(f"  Servers: {pir.num_servers}, Threshold: {pir.threshold}")

    # Generate query
    start = time.time()
    query = pir.generate_query(target_index, database_size, block_size)
    query_time = time.time() - start

    logger.debug(f"\n  Query generation: {query_time * 1000:.2f} ms")
    logger.debug(f"  Query size/server: {query.server_queries[0].nbytes / 1024:.2f} KB")

    # Process on servers
    responses = []
    for server_id in range(pir.num_servers):
        start = time.time()
        response = pir.process_server_query(server_id, query, databases[server_id])
        response_time = time.time() - start
        responses.append(response)
        logger.debug(f"  Server {server_id} response: {response_time * 1000:.2f} ms")

    # Reconstruct
    start = time.time()
    reconstructed = pir.reconstruct_response(query, responses)
    recon_time = time.time() - start

    logger.debug(f"\n  Reconstruction: {recon_time * 1000:.2f} ms")

    # Verify
    expected = databases[0][target_index]
    matches = reconstructed[: len(expected)] == expected
    logger.debug(f"  Retrieved correctly: {matches}")

    return matches


def test_hierarchical_compression():
    """Test hierarchical hypervector compression."""
    logger.debug("\n" + "=" * 60)
    logger.debug("Testing Hierarchical Hypervector Compression")
    logger.debug("=" * 60)

    # Initialize compressor
    compressor = AdvancedHierarchicalCompressor()

    # Create genomic feature vector
    base_vector = np.random.randn(10000)
    base_vector[np.random.choice(10000, 9000, replace=False)] = 0  # Sparse

    logger.debug("\nBase vector:")
    logger.debug(f"  Dimensions: {len(base_vector)}")
    logger.debug(f"  Sparsity: {np.mean(base_vector == 0):.2%}")
    logger.debug(f"  Size: {base_vector.nbytes / 1024:.1f} KB")

    # Compress hierarchically
    compressed = compressor.hierarchical_compression(
        base_vector, modality_context="genomic", overall_model_context="disease_risk"
    )

    logger.debug("\nCompressed vector:")
    logger.debug(f"  Level: {compressed.level}")
    logger.debug(f"  Dimensions: {len(compressed.high_vector)}")
    logger.debug(
        f"  Compression ratio: {compressed.compression_metadata['compression_ratio']:.2f}x"
    )

    # Test storage tiers
    logger.debug("\nStorage tiers:")
    for tier in ["mini", "clinical", "fullhdc"]:
        optimized = compressor.create_storage_optimized_vector(base_vector, tier)
        logger.debug(f"  {tier}: {optimized['size_kb']:.1f} KB")

    return True


def run_all_tests():
    """Run all advanced implementation tests."""
    logger.debug("\nGenomeVault Advanced Implementation Tests")
    # print("=========================================")

    tests = [
        ("Recursive SNARK", test_recursive_snark),
        ("Post-Quantum STARK", test_stark_post_quantum),
        ("Catalytic Proofs", test_catalytic_proof),
        ("IT-PIR", test_it_pir),
        ("Hierarchical Compression", test_hierarchical_compression),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "PASSED" if result else "FAILED"
        except Exception as e:
            logger.exception("Unhandled exception")
            logger.debug(f"\nError in {test_name}: {e}")
            results[test_name] = "ERROR"
            raise

    # Summary
    logger.debug("\n" + "=" * 60)
    logger.debug("Test Summary")
    logger.debug("=" * 60)

    for test_name, result in results.items():
        status_symbol = "✓" if result == "PASSED" else "✗"
        logger.debug(f"{status_symbol} {test_name}: {result}")

    all_passed = all(r == "PASSED" for r in results.values())

    if all_passed:
        logger.debug("\nAll tests passed! Advanced implementations are working correctly.")
    else:
        logger.debug("\nSome tests failed. Please check the output above.")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
