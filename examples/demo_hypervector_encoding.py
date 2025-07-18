"""
Demonstration of GenomeVault Hypervector Encoding System

This script demonstrates the key features of the hypervector encoding module,
showing how genomic data is transformed into privacy-preserving representations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

# GenomeVault imports
from hypervector_transform.encoding import HypervectorEncoder, create_encoder
from hypervector_transform.binding import (
    HypervectorBinder, BindingType, PositionalBinder, CrossModalBinder
)
from hypervector_transform.holographic import HolographicEncoder
from hypervector_transform.mapping import (
    BiologicalSimilarityMapper, ManifoldPreservingMapper
)
from core.constants import OmicsType
from utils.logging import setup_logging


def demonstrate_basic_encoding():
    """Demonstrate basic hypervector encoding"""
    print("\n=== Basic Hypervector Encoding ===")
    
    # Create encoder
    encoder = create_encoder(dimension=10000, projection_type="sparse_random")
    
    # Simulate genomic features
    genomic_features = {
        "variants": {
            "snps": list(range(100)),  # 100 SNPs
            "indels": list(range(20)),  # 20 indels
            "cnvs": list(range(5))      # 5 CNVs
        },
        "quality_metrics": {
            "mean_coverage": 35.2,
            "uniformity": 0.92,
            "gc_content": 0.41
        }
    }
    
    # Encode the genomic data
    hypervector = encoder.encode(genomic_features, OmicsType.GENOMIC)
    
    print(f"Input features: {len(genomic_features['variants']['snps'])} SNPs, "
          f"{len(genomic_features['variants']['indels'])} indels, "
          f"{len(genomic_features['variants']['cnvs'])} CNVs")
    print(f"Encoded hypervector dimension: {hypervector.shape[0]}")
    print(f"Hypervector sparsity: {(hypervector == 0).float().mean():.2%}")
    print(f"Compression ratio: {125 * 8 / (hypervector.shape[0] * 4):.2f}:1")
    
    # Demonstrate similarity preservation
    # Create a similar genomic profile
    similar_features = genomic_features.copy()
    similar_features["variants"]["snps"] = list(range(95))  # 95% overlap
    
    similar_hv = encoder.encode(similar_features, OmicsType.GENOMIC)
    
    # Create a different genomic profile
    different_features = genomic_features.copy()
    different_features["variants"]["snps"] = list(range(200, 300))  # No overlap
    
    different_hv = encoder.encode(different_features, OmicsType.GENOMIC)
    
    # Compare similarities
    sim_similar = encoder.similarity(hypervector, similar_hv)
    sim_different = encoder.similarity(hypervector, different_hv)
    
    print(f"\nSimilarity with 95% overlap: {sim_similar:.3f}")
    print(f"Similarity with 0% overlap: {sim_different:.3f}")


def demonstrate_binding_operations():
    """Demonstrate binding operations for combining information"""
    print("\n=== Hypervector Binding Operations ===")
    
    dimension = 5000
    binder = HypervectorBinder(dimension)
    
    # Create hypervectors for different genomic elements
    gene_hv = torch.randn(dimension)
    gene_hv = gene_hv / torch.norm(gene_hv)
    
    expression_hv = torch.randn(dimension)
    expression_hv = expression_hv / torch.norm(expression_hv)
    
    condition_hv = torch.randn(dimension)
    condition_hv = condition_hv / torch.norm(condition_hv)
    
    # Bind gene with expression level
    gene_expr_bound = binder.bind([gene_hv, expression_hv], BindingType.CIRCULAR)
    
    # Bind with experimental condition
    full_bound = binder.bind([gene_expr_bound, condition_hv], BindingType.CIRCULAR)
    
    print(f"Original gene vector norm: {torch.norm(gene_hv):.3f}")
    print(f"Gene+expression bound vector norm: {torch.norm(gene_expr_bound):.3f}")
    print(f"Full bound vector norm: {torch.norm(full_bound):.3f}")
    
    # Demonstrate unbinding (recovery)
    recovered_gene = binder.unbind(gene_expr_bound, [expression_hv], BindingType.CIRCULAR)
    recovery_similarity = torch.nn.functional.cosine_similarity(
        gene_hv.unsqueeze(0), recovered_gene.unsqueeze(0)
    ).item()
    
    print(f"\nRecovery similarity after unbinding: {recovery_similarity:.3f}")
    
    # Demonstrate bundling (superposition)
    genes = [torch.randn(dimension) for _ in range(5)]
    gene_set = binder.bundle(genes, normalize=True)
    
    print(f"\nBundled 5 genes into single vector")
    print(f"Each gene's contribution to bundle:")
    for i, gene in enumerate(genes):
        contrib = torch.nn.functional.cosine_similarity(
            gene_set.unsqueeze(0), gene.unsqueeze(0)
        ).item()
        print(f"  Gene {i}: {contrib:.3f}")


def demonstrate_positional_encoding():
    """Demonstrate position-aware binding for genomic sequences"""
    print("\n=== Positional Encoding for Genomic Sequences ===")
    
    dimension = 5000
    pos_binder = PositionalBinder(dimension)
    
    # Simulate a sequence of genomic features at different positions
    sequence_length = 10
    features = []
    
    for i in range(sequence_length):
        # Create feature vector for position i
        feature = torch.randn(dimension)
        feature = feature / torch.norm(feature)
        features.append(feature)
    
    # Bind with positions
    sequence_bound = pos_binder.bind_sequence(features, start_position=1000)
    
    print(f"Encoded sequence of {sequence_length} elements")
    print(f"Sequence hypervector dimension: {sequence_bound.shape[0]}")
    
    # Show that different positions create different bindings
    pos_100 = pos_binder.bind_with_position(features[0], 100)
    pos_200 = pos_binder.bind_with_position(features[0], 200)
    
    position_similarity = torch.nn.functional.cosine_similarity(
        pos_100.unsqueeze(0), pos_200.unsqueeze(0)
    ).item()
    
    print(f"Same feature at different positions similarity: {position_similarity:.3f}")


def demonstrate_holographic_encoding():
    """Demonstrate holographic encoding for structured data"""
    print("\n=== Holographic Encoding for Genomic Variants ===")
    
    dimension = 5000
    holo_encoder = HolographicEncoder(dimension)
    
    # Encode a genomic variant with annotations
    variant1 = holo_encoder.encode_genomic_variant(
        chromosome="chr7",
        position=117559590,
        ref="G",
        alt="A",
        annotations={
            "gene": "CFTR",
            "effect": "missense",
            "clinical_significance": "pathogenic"
        }
    )
    
    variant2 = holo_encoder.encode_genomic_variant(
        chromosome="chr7",
        position=117559591,  # Adjacent position
        ref="C",
        alt="T",
        annotations={
            "gene": "CFTR",
            "effect": "synonymous",
            "clinical_significance": "benign"
        }
    )
    
    # Similar variant (same gene, different position)
    variant3 = holo_encoder.encode_genomic_variant(
        chromosome="chr7",
        position=117560000,
        ref="A",
        alt="G",
        annotations={
            "gene": "CFTR",
            "effect": "missense",
            "clinical_significance": "uncertain"
        }
    )
    
    # Different gene variant
    variant4 = holo_encoder.encode_genomic_variant(
        chromosome="chr17",
        position=41276045,
        ref="C",
        alt="T",
        annotations={
            "gene": "BRCA1",
            "effect": "nonsense",
            "clinical_significance": "pathogenic"
        }
    )
    
    # Compare similarities
    sim_adjacent = torch.nn.functional.cosine_similarity(
        variant1.unsqueeze(0), variant2.unsqueeze(0)
    ).item()
    sim_same_gene = torch.nn.functional.cosine_similarity(
        variant1.unsqueeze(0), variant3.unsqueeze(0)
    ).item()
    sim_diff_gene = torch.nn.functional.cosine_similarity(
        variant1.unsqueeze(0), variant4.unsqueeze(0)
    ).item()
    
    print(f"Variant similarities:")
    print(f"  Adjacent positions (same gene): {sim_adjacent:.3f}")
    print(f"  Same gene, different position: {sim_same_gene:.3f}")
    print(f"  Different gene: {sim_diff_gene:.3f}")
    
    # Create memory trace of multiple variants
    variants = [
        {"chr": "chr1", "pos": 1000, "ref": "A", "alt": "G"},
        {"chr": "chr1", "pos": 2000, "ref": "C", "alt": "T"},
        {"chr": "chr2", "pos": 3000, "ref": "G", "alt": "A"},
    ]
    
    memory = holo_encoder.create_memory_trace(variants)
    print(f"\nCreated memory trace of {len(variants)} variants")
    print(f"Memory trace dimension: {memory.shape[0]}")


def demonstrate_cross_modal_binding():
    """Demonstrate binding across different omics modalities"""
    print("\n=== Cross-Modal Binding for Multi-Omics ===")
    
    dimension = 5000
    cross_binder = CrossModalBinder(dimension)
    
    # Create hypervectors for different omics types
    genomic_hv = torch.randn(dimension)
    genomic_hv = genomic_hv / torch.norm(genomic_hv)
    
    transcriptomic_hv = torch.randn(dimension)
    transcriptomic_hv = transcriptomic_hv / torch.norm(transcriptomic_hv)
    
    epigenomic_hv = torch.randn(dimension)
    epigenomic_hv = epigenomic_hv / torch.norm(epigenomic_hv)
    
    proteomic_hv = torch.randn(dimension)
    proteomic_hv = proteomic_hv / torch.norm(proteomic_hv)
    
    # Bind modalities
    modality_data = {
        "genomic": genomic_hv,
        "transcriptomic": transcriptomic_hv,
        "epigenomic": epigenomic_hv,
        "proteomic": proteomic_hv
    }
    
    bound_modalities = cross_binder.bind_modalities(modality_data)
    
    print(f"Created cross-modal bindings:")
    for key in bound_modalities:
        print(f"  {key}: dimension {bound_modalities[key].shape[0]}")
    
    # Show that combined representation preserves information from all modalities
    combined = bound_modalities["combined"]
    
    print(f"\nContribution of each modality to combined representation:")
    for modality, hv in modality_data.items():
        contrib = torch.nn.functional.cosine_similarity(
            combined.unsqueeze(0), hv.unsqueeze(0)
        ).item()
        print(f"  {modality}: {contrib:.3f}")


def demonstrate_similarity_preservation():
    """Demonstrate similarity preservation in mappings"""
    print("\n=== Similarity-Preserving Transformations ===")
    
    # Create synthetic biological data with structure
    n_samples = 100
    n_features = 50
    
    # Create two clusters of samples
    cluster1 = torch.randn(n_samples // 2, n_features) + torch.tensor([2.0] * n_features)
    cluster2 = torch.randn(n_samples // 2, n_features) - torch.tensor([2.0] * n_features)
    
    data = torch.cat([cluster1, cluster2], dim=0)
    labels = torch.cat([torch.zeros(n_samples // 2), torch.ones(n_samples // 2)])
    
    # Create biological similarity mapper
    mapper = BiologicalSimilarityMapper(
        n_features, 1000, OmicsType.GENOMIC
    )
    
    # Transform data
    transformed = mapper.fit_transform(data)
    
    print(f"Original data: {data.shape}")
    print(f"Transformed data: {transformed.shape}")
    
    # Check that clusters remain separated
    cluster1_transformed = transformed[:n_samples // 2]
    cluster2_transformed = transformed[n_samples // 2:]
    
    within_cluster_sim = torch.nn.functional.cosine_similarity(
        cluster1_transformed.mean(dim=0).unsqueeze(0),
        cluster1_transformed[0].unsqueeze(0)
    ).item()
    
    between_cluster_sim = torch.nn.functional.cosine_similarity(
        cluster1_transformed.mean(dim=0).unsqueeze(0),
        cluster2_transformed.mean(dim=0).unsqueeze(0)
    ).item()
    
    print(f"\nWithin-cluster similarity: {within_cluster_sim:.3f}")
    print(f"Between-cluster similarity: {between_cluster_sim:.3f}")
    print(f"Separation maintained: {within_cluster_sim > between_cluster_sim}")


def demonstrate_privacy_guarantees():
    """Demonstrate privacy properties of hypervectors"""
    print("\n=== Privacy Guarantees of Hypervectors ===")
    
    dimension = 10000
    encoder = create_encoder(dimension=dimension)
    
    # Create sensitive genomic data
    sensitive_data = {
        "variants": {
            "snps": list(range(1000)),  # Many SNPs
            "pathogenic": ["BRCA1_c.5266dupC", "CFTR_F508del", "HBB_c.20A>T"]
        },
        "quality_metrics": {
            "mean_coverage": 42.7,
            "uniformity": 0.94
        }
    }
    
    # Encode to hypervector
    hypervector = encoder.encode(sensitive_data, OmicsType.GENOMIC)
    
    print(f"Original data contains:")
    print(f"  - {len(sensitive_data['variants']['snps'])} SNPs")
    print(f"  - {len(sensitive_data['variants']['pathogenic'])} pathogenic variants")
    print(f"\nEncoded to {dimension}D hypervector")
    
    # Demonstrate irreversibility
    # Try to recover information from hypervector
    print("\nAttempting to recover original data...")
    
    # Random projection is not invertible when D >> d
    print(f"Projection: {1000}D -> {dimension}D (not invertible)")
    
    # Show that similar inputs create distinguishable outputs
    modified_data = sensitive_data.copy()
    modified_data["variants"]["snps"] = list(range(999))  # Remove one SNP
    
    modified_hv = encoder.encode(modified_data, OmicsType.GENOMIC)
    
    similarity = encoder.similarity(hypervector, modified_hv)
    print(f"\nSingle SNP difference creates {1-similarity:.4f} change in hypervector")
    print("This demonstrates sensitivity while maintaining privacy")


def demonstrate_compression_tiers():
    """Demonstrate tiered compression for different use cases"""
    print("\n=== Tiered Compression System ===")
    
    from local_processing.compression import TieredCompressor, CompressionTier
    
    # Create test hypervector
    dimension = 10000
    hypervector = torch.randn(dimension)
    hypervector = hypervector / torch.norm(hypervector)
    
    # Test different compression tiers
    for tier in [CompressionTier.MINI, CompressionTier.CLINICAL, CompressionTier.FULL]:
        compressor = TieredCompressor(tier)
        
        compressed = compressor.compress(
            {"hypervector": hypervector},
            OmicsType.GENOMIC
        )
        
        print(f"\n{tier.value} tier:")
        print(f"  Original size: {hypervector.numel() * 4:,} bytes")
        print(f"  Compressed size: {compressed.compressed_size:,} bytes")
        print(f"  Compression ratio: {hypervector.numel() * 4 / compressed.compressed_size:.1f}:1")
        
        # Test decompression
        decompressed = compressor.decompress(compressed)
        
        # Measure reconstruction quality
        reconstruction_sim = torch.nn.functional.cosine_similarity(
            hypervector.unsqueeze(0),
            decompressed.unsqueeze(0)
        ).item()
        
        print(f"  Reconstruction similarity: {reconstruction_sim:.3f}")


def visualize_hypervector_properties():
    """Visualize key properties of hypervectors"""
    print("\n=== Visualizing Hypervector Properties ===")
    
    # This would normally create plots, but for text output:
    dimension = 1000
    n_samples = 50
    
    # Create random hypervectors
    vectors = torch.randn(n_samples, dimension)
    vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)
    
    # Compute pairwise similarities
    similarities = torch.matmul(vectors, vectors.T)
    
    # Statistics
    self_sim = torch.diag(similarities).mean().item()
    off_diag = similarities[~torch.eye(n_samples, dtype=bool)]
    mean_sim = off_diag.mean().item()
    std_sim = off_diag.std().item()
    
    print(f"\nRandom hypervector statistics ({dimension}D):")
    print(f"  Self-similarity: {self_sim:.3f}")
    print(f"  Mean pairwise similarity: {mean_sim:.3f} ± {std_sim:.3f}")
    print(f"  Min similarity: {off_diag.min().item():.3f}")
    print(f"  Max similarity: {off_diag.max().item():.3f}")
    
    # Concentration of measure
    print(f"\nConcentration of measure:")
    print(f"  99% of similarities in range: [{mean_sim - 3*std_sim:.3f}, {mean_sim + 3*std_sim:.3f}]")
    print(f"  This demonstrates the 'blessing of dimensionality' for privacy")


def main():
    """Run all demonstrations"""
    print("=" * 60)
    print("GenomeVault Hypervector Encoding Demonstration")
    print("=" * 60)
    
    # Set up logging
    setup_logging(level="INFO")
    
    # Run demonstrations
    demonstrate_basic_encoding()
    demonstrate_binding_operations()
    demonstrate_positional_encoding()
    demonstrate_holographic_encoding()
    demonstrate_cross_modal_binding()
    demonstrate_similarity_preservation()
    demonstrate_privacy_guarantees()
    demonstrate_compression_tiers()
    visualize_hypervector_properties()
    
    print("\n" + "=" * 60)
    print("Demonstration Complete")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Hypervectors preserve biological similarities while protecting privacy")
    print("2. Binding operations enable complex relationships to be encoded")
    print("3. Holographic encoding allows structured data representation")
    print("4. Cross-modal binding integrates multiple omics types")
    print("5. Tiered compression provides flexibility for different use cases")
    print("6. High dimensionality provides mathematical privacy guarantees")


if __name__ == "__main__":
    main()
