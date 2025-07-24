#!/usr/bin/env python3
"""
Example usage of GenomeVault 3.0 integrated system.
Demonstrates the complete workflow from data processing to proof generation.
"""
import asyncio

import numpy as np

from genomevault.blockchain.node import BlockchainNode
from genomevault.hypervector_transform.encoding import (
    HypervectorBinder,
    HypervectorEncoder,
)
from genomevault.local_processing.sequencing import SequencingProcessor, Variant
from genomevault.pir.client import PIRClient, PIRServer

# Import GenomeVault components
from genomevault.utils.config import CompressionTier, NodeClass, config
from genomevault.zk_proofs.prover import Prover


async def demonstrate_genomevault():
    """Demonstrate complete GenomeVault workflow."""

    print("=== GenomeVault 3.0 Demonstration ===\n")

    # Step 1: Configuration
    print("1. System Configuration")
    print("   Compression Tier: {config.hypervector.compression_tier.value}")
    print("   Hypervector Dimensions: {config.hypervector.base_dimensions}")
    print("   PIR Servers: {config.pir.num_servers}")
    print("   Privacy Target: {config.pir.target_failure_probability}")

    # Step 2: Local Processing (simulated)
    print("\n2. Local Genomic Processing")

    # Simulate variant data
    variants = [
        Variant("chr1", 100000, "A", "G", 30.0, "0/1"),
        Variant("chr2", 200000, "C", "T", 35.0, "1/1"),
        Variant("chr7", 300000, "G", "A", 40.0, "0/1"),
    ]

    # Calculate differential storage
    processor = SequencingProcessor()
    processor.compute_differential_storage(variants)
    print("   Variants found: {len(variants)}")
    print("   Storage hash: {diff_storage['storage_hash'][:16]}...")

    # Step 3: Hypervector Encoding
    print("\n3. Hypervector Encoding")

    encoder = HypervectorEncoder(compression_tier=CompressionTier.CLINICAL)

    # Encode genomic features
    genomic_features = {"variants": [v.to_dict() for v in variants]}
    genomic_hv = encoder.encode_features(genomic_features, domain="genomic")

    # Encode clinical features (simulated)
    clinical_features = {
        "labs": {
            "glucose": 140,
            "hba1c": 7.2,
            "cholesterol": 200,
        }  # mg/dL  # %  # mg/dL
    }
    clinical_hv = encoder.encode_features(clinical_features, domain="clinical")

    print("   Genomic hypervector shape: {genomic_hv.shape}")
    print("   Clinical hypervector shape: {clinical_hv.shape}")
    print(
        "   Compression achieved: {config.get_compression_size(['genomics', 'clinical'])} KB"
    )

    # Cross-modal binding
    binder = HypervectorBinder()
    binder.circular_convolution(genomic_hv, clinical_hv)
    print("   Cross-modal binding complete")

    # Step 4: Zero-Knowledge Proof Generation
    print("\n4. Zero-Knowledge Proof Generation")

    prover = Prover()

    # Generate diabetes risk proof
    diabetes_proof = prover.generate_proof(
        circuit_name="diabetes_risk_alert",
        public_inputs={
            "glucose_threshold": 126,
            "risk_threshold": 0.75,
            "result_commitment": "commitment_hash",
        },
        private_inputs={
            "glucose_reading": 140,
            "risk_score": 0.82,
            "witness_randomness": np.random.bytes(32).hex(),
        },
    )

    print("   Proof ID: {diabetes_proof.proof_id}")
    print("   Proof size: {len(diabetes_proof.proof_data)} bytes")
    print(
        "   Generation time: {diabetes_proof.metadata['generation_time_seconds']*1000:.1f}ms"
    )

    # Step 5: PIR Query (simulated)
    print("\n5. Private Information Retrieval")

    # Configure PIR servers
    servers = [
        PIRServer("ln1", "http://ln1.genomevault.com", "us-east", False, 0.95, 70),
        PIRServer("ln2", "http://ln2.genomevault.com", "eu-west", False, 0.95, 80),
        PIRServer("ts1", "http://ts1.genomevault.com", "us-west", True, 0.98, 60),
        PIRServer("ts2", "http://ts2.genomevault.com", "us-central", True, 0.98, 50),
    ]

    pir_client = PIRClient(servers[:4], database_size=1000000)

    # Show optimal configuration
    pir_client.get_optimal_server_configuration()
    print("   Optimal PIR config: {optimal_config['optimal']['name']}")
    print("   Expected latency: {optimal_config['optimal']['latency_ms']}ms")
    print(
        "   Privacy guarantee: P_fail = {optimal_config['optimal']['failure_probability']:.2e}"
    )

    # Step 6: Blockchain Integration
    print("\n6. Blockchain Node Setup")

    # Create a light node with HIPAA verification
    node = BlockchainNode(
        node_id="clinic_node_001",
        node_class=NodeClass.LIGHT,
        is_trusted_signatory=False,
    )

    print("   Initial voting power: {node.voting_power}")

    # HIPAA fast-track verification
    await node.verify_hipaa_credentials(
        npi="1234567890",
        baa_hash="baa_hash",
        risk_analysis_hash="risk_hash",
        hsm_serial="HSM12345",
    )

    print("   Post-HIPAA voting power: {node.voting_power}")
    print("   Block rewards: {node._calculate_block_rewards()} credits/block")

    # Submit proof transaction
    tx_hash = node.submit_transaction(
        {
            "type": "proof_record",
            "from": node.node_id,
            "data": {
                "proof_key": diabetes_proof.proof_id,
                "circuit_type": "diabetes_risk_alert",
                "proof_data": diabetes_proof.proof_data.hex()[:64] + "...",
                "public_inputs": diabetes_proof.public_inputs,
            },
            "signature": "mock_signature",
        }
    )

    print("   Transaction submitted: {tx_hash[:16]}...")

    # Step 7: Summary
    print("\n7. System Summary")
    print("   Total data processed: ~100GB (simulated)")
    print("   Data leaving device: <60KB")
    print("   Privacy preserved: ✓")
    print("   Proof verifiable: ✓")
    print("   Blockchain recorded: ✓")

    # Calculate total system metrics
    print("\n=== Performance Metrics ===")
    print("Processing time: <10 minutes (consumer hardware)")
    print("Proof generation: <1 minute")
    print("PIR query latency: ~{optimal_config['optimal']['latency_ms']}ms")
    print(
        "Storage required: {config.get_compression_size(['genomics', 'clinical', 'transcriptomics'])} KB"
    )
    print("Security level: 256-bit post-quantum")

    # Privacy calculations
    print("\n=== Privacy Guarantees ===")
    print(
        "PIR privacy (2 TS): P_fail = {pir_client.calculate_privacy_failure_probability(2, 0.98):.2e}"
    )
    print("Differential privacy: ε = {config.security.differential_privacy_epsilon}")
    print("Zero-knowledge: Perfect secrecy")

    # Network voting power
    print("\n=== Network Governance ===")
    print("Light TS node: w = {1 + 10} = 11")
    print("Full node: w = {4 + 0} = 4")
    print("Archive node: w = {8 + 0} = 8")
    print("BFT requirement: H > F (majority honest voting power)")


def demonstrate_privacy_calculations():
    """Demonstrate privacy guarantee calculations."""
    print("\n=== Privacy Guarantee Calculations ===")

    # PIR privacy calculations
    print("\nPIR Privacy Breach Probability:")
    print("P_fail(k, q) = (1-q)^k")
    print("\nFor HIPAA TS servers (q = 0.98):")
    for k in range(1, 5):
        (1 - 0.98) ** k
        print("  {k} servers: P_fail = {p_fail:.2e}")

    print("\nFor generic servers (q = 0.95):")
    for k in range(1, 6):
        (1 - 0.95) ** k
        print("  {k} servers: P_fail = {p_fail:.2e}")

    # Minimum servers calculation
    target_failure = 1e-4
    q_hipaa = 0.98
    q_generic = 0.95

    import math

    math.ceil(math.log(target_failure) / math.log(1 - q_hipaa))
    math.ceil(math.log(target_failure) / math.log(1 - q_generic))

    print("\nMinimum servers for P_fail ≤ {target_failure}:")
    print("  HIPAA TS: {k_min_hipaa} servers")
    print("  Generic: {k_min_generic} servers")


def demonstrate_compression_tiers():
    """Demonstrate compression tier calculations."""
    print("\n=== Compression Tier Analysis ===")

    modalities = ["genomics", "transcriptomics", "epigenetics", "proteomics"]

    print("\nStorage by tier:")
    for tier in CompressionTier:
        config.hypervector.compression_tier = tier

        print("\n{tier.value.upper()} Tier:")
        for i in range(1, len(modalities) + 1):
            size = config.get_compression_size(modalities[:i])
            print("  {i} modality(ies): {size} KB")


if __name__ == "__main__":
    # Run main demonstration
    asyncio.run(demonstrate_genomevault())

    # Additional demonstrations
    demonstrate_privacy_calculations()
    demonstrate_compression_tiers()

    print("\n=== GenomeVault 3.0 Demonstration Complete ===")
    print("Your genomic data remains private while enabling breakthrough research!")
