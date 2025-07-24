#!/usr/bin/env python3
"""
Run clinical validation with REAL GenomeVault components
"""
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from clinical_validation import ClinicalValidator, NHANESDataSource, PimaDataSource


def main():
    """Run full clinical validation"""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("🧬 GenomeVault Clinical Validation")
    print("Using REAL Zero-Knowledge Proofs and Privacy-Preserving Components")
    print("=" * 70)

    # Create validator with actual GenomeVault components
    validator = ClinicalValidator()

    # Add data sources
    print("\n📊 Loading clinical data sources...")
    validator.add_data_source(NHANESDataSource())
    validator.add_data_source(PimaDataSource())

    # Run validation with REAL components
    print("\n🔐 Running validation with actual ZK proofs...")
    results = validator.run_full_clinical_validation()

    # Summary
    print("\n" + "=" * 70)
    print("✅ VALIDATION COMPLETE")
    print(f"Components tested: {', '.join(results['components_tested'])}")
    print(f"Data sources processed: {len(results['data_sources'])}")

    # Show ZK proof performance
    if results["zk_proof_metrics"]:
        print("\n🔐 Zero-Knowledge Proof Performance:")
        for source, metrics in results["zk_proof_metrics"].items():
            print(f"  {source}:")
            print(f"    - Generation time: {metrics['avg_generation_time_ms']:.1f} ms")
            print(f"    - Verification time: {metrics['avg_verification_time_ms']:.1f} ms")
            print(f"    - Proof size: {metrics['avg_proof_size_bytes']:.0f} bytes")

    print("\n📄 Full report saved to: genomevault_clinical_validation_report.md")

    return results


if __name__ == "__main__":
    main()
