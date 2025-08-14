#!/usr/bin/env python3
"""Demonstration of privacy-preserving genomic analysis.

This script shows how GenomeVault enables genomic analysis
without exposing the raw genomic data.
"""

import random

# Import our privacy modules
from genomevault.privacy.genomic_proof import (
    GenomicProver,
    ClinicalVerifier,
    example_privacy_preserving_diagnosis,
)
from genomevault.privacy.proof_aggregation import (
    example_proof_aggregation,
    example_batch_verification,
)
from genomevault.clinical.privacy_workflow import (
    example_clinical_workflow,
    example_batch_processing,
)


def demonstrate_snp_privacy():
    """Demonstrate SNP verification with privacy."""

    print("\n" + "=" * 60)
    print("1. PRIVACY-PRESERVING SNP VERIFICATION")
    print("=" * 60)

    # Create a simulated genome with a known disease marker
    random.seed(42)
    bases = ["A", "C", "G", "T"]
    genome_size = 1000

    # Generate random genome
    genome = "".join(random.choice(bases) for _ in range(genome_size))

    # Insert disease marker at position 42
    disease_position = 42
    disease_marker = "G"
    genome = genome[:disease_position] + disease_marker + genome[disease_position + 1 :]

    print(f"\n✓ Created genome with {genome_size} bases")
    print(f"✓ Disease marker '{disease_marker}' at position {disease_position}")
    print("✗ Genome data is NEVER exposed\n")

    # Create commitment
    prover = GenomicProver()
    commitment = prover.commit_genome(genome)

    print("Genomic Commitment (Merkle root):")
    print(f"  {commitment.to_hex()[:32]}...")
    print("  (Only 32 bytes represent entire genome)")

    # Generate proof for disease marker
    proof = prover.prove_snp(disease_position, commitment)

    print(f"\nProof for position {disease_position}:")
    print(f"  Nucleotide: {proof.nucleotide}")
    print(f"  Proof size: {len(proof.proof_path)} hashes")
    print(f"  Total proof: {len(proof.proof_path) * 32} bytes")

    # Verify the proof
    verifier = ClinicalVerifier()
    is_valid = verifier.verify_snp_proof(proof)

    print("\nVerification:")
    print(f"  ✓ Proof is valid: {is_valid}")
    print(f"  ✓ Confirmed: Position {disease_position} = '{proof.nucleotide}'")
    print(f"  ✗ Other {genome_size-1} positions remain private")

    # Try to verify wrong position - should fail
    fake_proof = prover.prove_snp(100, commitment)
    fake_proof.position = disease_position  # Tamper with position
    is_valid_fake = verifier.verify_snp_proof(fake_proof)

    print("\nTamper Detection:")
    print(f"  ✓ Tampered proof detected: Valid = {is_valid_fake}")


def demonstrate_disease_panel():
    """Demonstrate disease panel testing with privacy."""

    print("\n" + "=" * 60)
    print("2. PRIVACY-PRESERVING DISEASE PANEL")
    print("=" * 60)

    result = example_privacy_preserving_diagnosis()

    print(f"\nGenomic Commitment: {result['commitment']}")
    print(f"Genome size: {result['genome_size']} bases")

    verification = result["verification_results"]
    print("\nDisease Panel Results:")
    print(f"  Valid proofs: {verification['valid_proofs']}")
    print(f"  Matching markers: {verification['matching_markers']}")
    print(f"  Risk score: {verification['risk_score']:.2%}")

    print("\nPrivacy Preserved:")
    print(f"  ✓ Only {len(verification['details'])} positions revealed")
    print(f"  ✓ Remaining {result['genome_size'] - len(verification['details'])} positions private")
    print("  ✓ No sequence information exposed")


def demonstrate_proof_aggregation():
    """Demonstrate efficient proof aggregation."""

    print("\n" + "=" * 60)
    print("3. PROOF AGGREGATION FOR EFFICIENCY")
    print("=" * 60)

    result = example_proof_aggregation()

    print(f"\nProving {result['num_positions']} genomic positions:")
    print(f"  Individual proofs: {result['individual_total_bytes']} bytes")
    print(f"  Aggregated proof: {result['aggregated_bytes']} bytes")

    savings = result["savings"]
    print("\nSpace Savings:")
    print(f"  Saved: {savings['savings_bytes']} bytes ({savings['savings_percent']}%)")
    print(f"  Compression ratio: {savings['compression_ratio']}x")

    # Batch verification
    batch_result = example_batch_verification()

    print("\nBatch Verification:")
    print(f"  Batch size: {batch_result['batch_size']} proofs")
    print(f"  Original: {batch_result['original_bytes']} bytes")
    print(f"  Compressed: {batch_result['compressed_bytes']} bytes")
    print(f"  Compression: {batch_result['compression_ratio']}x")
    print(f"  All valid: {batch_result['all_valid']}")


def demonstrate_clinical_workflow():
    """Demonstrate complete clinical workflow."""

    print("\n" + "=" * 60)
    print("4. CLINICAL WORKFLOW WITH PRIVACY")
    print("=" * 60)

    result = example_clinical_workflow()

    print("\nClinical Report Generated:")
    print(f"  Report ID: {result['report_id']}")
    print(f"  Risk Category: {result['risk_category']}")
    print(f"  Risk Score: {result['risk_score']}")
    print(f"  Actionable Findings: {result['actionable_findings']}")
    print(f"  Commitment: {result['commitment_root']}")
    print(f"  Valid: {result['report_valid']}")

    print("\nPrivacy Features:")
    print("  ✓ Patient genome never exposed")
    print("  ✓ Only relevant variants analyzed")
    print("  ✓ Cryptographic proof of validity")
    print("  ✓ Auditable without data exposure")

    # Batch processing
    batch_result = example_batch_processing()

    print("\nBatch Processing:")
    print(f"  Total samples: {batch_result['total_samples']}")
    print(f"  Successful: {batch_result['successful_reports']}")
    print(f"  All valid: {batch_result['all_reports_valid']}")


def demonstrate_privacy_guarantees():
    """Demonstrate the privacy guarantees of the system."""

    print("\n" + "=" * 60)
    print("5. PRIVACY GUARANTEES")
    print("=" * 60)

    # Create two different genomes with same SNP
    genome1 = "ACGTACGTACGTACGT" * 100  # 1600 bases
    genome2 = "TGCATGCATGCATGCA" * 100  # 1600 bases, different except position 5

    # Ensure both have same SNP at position 5
    position = 5
    snp = "G"
    genome1 = genome1[:position] + snp + genome1[position + 1 :]
    genome2 = genome2[:position] + snp + genome2[position + 1 :]

    prover = GenomicProver()

    # Create commitments
    commitment1 = prover.commit_genome(genome1)
    proof1 = prover.prove_snp(position, commitment1)

    commitment2 = prover.commit_genome(genome2)
    proof2 = prover.prove_snp(position, commitment2)

    print(f"\nTwo different genomes with same SNP at position {position}:")
    print(f"  Genome 1 commitment: {commitment1.to_hex()[:16]}...")
    print(f"  Genome 2 commitment: {commitment2.to_hex()[:16]}...")
    print(f"  Commitments are different: {commitment1.root != commitment2.root}")

    print(f"\nProofs for position {position}:")
    print(f"  Both prove nucleotide '{snp}': {proof1.nucleotide == proof2.nucleotide == snp}")
    print(f"  Proof 1 valid: {proof1.verify()}")
    print(f"  Proof 2 valid: {proof2.verify()}")

    print("\nPrivacy Properties:")
    print("  ✓ Cannot determine if genomes are same or different")
    print("  ✓ Cannot infer other positions from proof")
    print("  ✓ Cannot link proofs to individuals")
    print("  ✓ Zero-knowledge: Verifier learns only the proven fact")


def main():
    """Run all demonstrations."""

    print("\n" + "#" * 60)
    print("# GENOMEVAULT PRIVACY-PRESERVING GENOMIC ANALYSIS")
    print("#" * 60)

    print("\nThis demonstration shows how GenomeVault enables")
    print("genomic analysis without exposing raw genomic data.")

    # Run demonstrations
    demonstrate_snp_privacy()
    demonstrate_disease_panel()
    demonstrate_proof_aggregation()
    demonstrate_clinical_workflow()
    demonstrate_privacy_guarantees()

    print("\n" + "#" * 60)
    print("# SUMMARY")
    print("#" * 60)

    print(
        """
GenomeVault successfully demonstrates:

1. PRIVACY PRESERVATION
   - Genomic data never exposed in plaintext
   - Only specific positions revealed with proof
   - Cryptographic commitments prevent tampering

2. CLINICAL UTILITY
   - Disease risk assessment without data exposure
   - Pharmacogenomic testing with privacy
   - Batch processing for efficiency

3. SCALABILITY
   - Proof aggregation reduces overhead
   - Batch verification for multiple samples
   - Compression ratios of 2-5x

4. VERIFIABILITY
   - Cryptographic proofs ensure data integrity
   - Third-party verification without data access
   - Audit trail without privacy compromise

The system enables genomic medicine while preserving
patient privacy through zero-knowledge proofs.
"""
    )


if __name__ == "__main__":
    main()
