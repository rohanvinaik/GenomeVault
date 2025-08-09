#!/usr/bin/env python3

"""Test the ZK proof system end-to-end."""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_zk_system():
    """Test ZK proof generation and verification."""
    import numpy as np

    from genomevault.zk_proofs.prover import Prover
    from genomevault.zk_proofs.verifier import Verifier

    print("Initializing ZK proof system...")
    prover = Prover()
    verifier = Verifier()

    print("\nGenerating variant presence proof...")
    variant_proof = prover.generate_proof(
        circuit_name="variant_presence",
        public_inputs={
            "variant_hash": hashlib.sha256(b"chr1:12345:A:G").hexdigest(),
            "reference_hash": hashlib.sha256(b"GRCh38").hexdigest(),
            "commitment_root": hashlib.sha256(b"genome_root").hexdigest(),
        },
        private_inputs={
            "variant_data": {"chr": "chr1", "pos": 12345, "ref": "A", "alt": "G"},
            "merkle_proof": ["hash1", "hash2", "hash3"],
            "witness_randomness": np.random.bytes(32).hex(),
        },
    )

    print(f"  Proof ID: {variant_proof.proof_id}")
    print(f"  Proof size: {len(variant_proof.proof_data)} bytes")

    print("\nVerifying proof...")
    result = verifier.verify_proof(variant_proof)
    print(f"  Verification result: {result.is_valid}")
    print(f"  Verification time: {result.verification_time*1000:.1f}ms")

    print("\nGenerating diabetes risk proof...")
    diabetes_proof = prover.generate_proof(
        circuit_name="diabetes_risk_alert",
        public_inputs={
            "glucose_threshold": 126,
            "risk_threshold": 0.75,
            "result_commitment": hashlib.sha256(b"alert_status").hexdigest(),
        },
        private_inputs={
            "glucose_reading": 140,
            "risk_score": 0.82,
            "witness_randomness": np.random.bytes(32).hex(),
        },
    )

    print(f"  Proof ID: {diabetes_proof.proof_id}")
    print(f"  Proof size: {len(diabetes_proof.proof_data)} bytes")

    print("\nVerifying diabetes proof...")
    result = verifier.verify_proof(diabetes_proof)
    print(f"  Verification result: {result.is_valid}")
    print(f"  Verification time: {result.verification_time*1000:.1f}ms")

    print("\n✓ ZK proof system working correctly!")
    return True


if __name__ == "__main__":
    try:
        test_zk_system()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
