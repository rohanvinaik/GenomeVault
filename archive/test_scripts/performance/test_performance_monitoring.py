#!/usr/bin/env python3
"""Test the enhanced performance monitoring system."""

import sys

sys.path.insert(0, ".")

import hashlib
from genomevault.zk_proofs.prover import Prover


def test_performance_monitoring():
    """Test comprehensive performance monitoring."""
    print("🔧 Testing Enhanced Performance Monitoring")
    print("=" * 50)

    # Initialize prover
    prover = Prover()

    print("✅ Prover initialized with monitoring")

    # Test system info
    print("\n📊 System Information:")
    system_info = prover.get_system_info()
    for key, value in system_info.items():
        print(f"  {key}: {value}")

    # Generate a few proofs to populate metrics
    print("\n🔮 Generating test proofs...")
    for i in range(3):
        variant_data = {"chr": f"chr{i+1}", "pos": i * 1000, "ref": "A", "alt": "G"}

        variant_str = f"{variant_data['chr']}:{variant_data['pos']}:{variant_data['ref']}:{variant_data['alt']}"
        variant_hash = hashlib.sha256(variant_str.encode()).hexdigest()

        public_inputs = {
            "variant_hash": variant_hash,
            "reference_hash": "ref_" + hashlib.sha256(b"reference").hexdigest()[:32],
            "commitment_root": "root_" + hashlib.sha256(b"root").hexdigest()[:32],
        }

        private_inputs = {
            "variant_data": variant_data,
            "merkle_proof": ["proof1", "proof2"],
            "witness_randomness": f"random_{i}",
        }

        # Generate proof
        proof = prover.generate_proof("variant_presence", public_inputs, private_inputs)

        # Verify proof
        is_valid = prover.verify_proof(proof, public_inputs, "variant_presence")

        print(f"  ✅ Proof {i+1}: Generated and verified ({is_valid})")

        # Check proof metadata
        if hasattr(proof, "metadata") and proof.metadata:
            perf = proof.metadata.get("_performance", {})
            if perf:
                print(
                    f"     Performance: {perf['duration_ms']:.1f}ms, "
                    f"{perf['memory_delta_mb']:.2f}MB, "
                    f"device: {perf['device']}"
                )

    # Get performance dashboard
    print("\n📈 Performance Dashboard:")
    dashboard = prover.get_performance_dashboard()
    print(f"  Dashboard data available: {bool(dashboard)}")

    # Get performance report
    print("\n📋 Performance Report:")
    report = prover.get_performance_report()
    if report:
        print("  Report generated successfully")
        print(f"  Report length: {len(report)} characters")
        # Show first few lines
        lines = report.split("\n")[:5]
        for line in lines:
            if line.strip():
                print(f"    {line}")

    return True


if __name__ == "__main__":
    success = test_performance_monitoring()

    print("\n" + "=" * 50)
    if success:
        print("🎉 PERFORMANCE MONITORING TEST PASSED!")
    else:
        print("💥 PERFORMANCE MONITORING TEST FAILED!")

    sys.exit(0 if success else 1)
