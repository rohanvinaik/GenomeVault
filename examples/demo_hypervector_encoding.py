"""
Demonstration of GenomeVault Hypervector Encoding System

This script demonstrates the key features of the hypervector encoding module,
showing how genomic data is transformed into privacy-preserving representations.
"""

import torch

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

from genomevault.core.constants import OmicsType
from genomevault.hypervector_transform.binding import (
    BindingType,
    CrossModalBinder,
    HypervectorBinder,
    PositionalBinder,
)

# GenomeVault imports
from genomevault.hypervector_transform.encoding import create_encoder
from genomevault.hypervector_transform.holographic import HolographicEncoder
from genomevault.hypervector_transform.mapping import BiologicalSimilarityMapper
from genomevault.utils.logging import setup_logging


def demonstrate_basic_encoding():
    """Demonstrate basic hypervector encoding"""
    logger.info("\n=== Basic Hypervector Encoding ===")

    # Create encoder
    encoder = create_encoder(dimension=10000, projection_type="sparse_random")

    # Simulate genomic features
    genomic_features = {
        "variants": {
            "snps": list(range(100)),  # 100 SNPs
            "indels": list(range(20)),  # 20 indels
            "cnvs": list(range(5)),  # 5 CNVs
        },
        "quality_metrics": {
            "mean_coverage": 35.2,
            "uniformity": 0.92,
            "gc_content": 0.41,
        },
    }

    # Encode the genomic data
    hypervector = encoder.encode(genomic_features, OmicsType.GENOMIC)

    logger.info(
        "Input features: {len(genomic_features['variants']['snps'])} SNPs, "
        "{len(genomic_features['variants']['indels'])} indels, "
        "{len(genomic_features['variants']['cnvs'])} CNVs"
    )
    logger.info("Encoded hypervector dimension: {hypervector.shape[0]}")
    logger.info("Hypervector sparsity: {(hypervector == 0).float().mean():.2%}")
    logger.info("Compression ratio: {125 * 8 / (hypervector.shape[0] * 4):.2f}:1")

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
    encoder.similarity(hypervector, similar_hv)
    encoder.similarity(hypervector, different_hv)

    logger.info("\nSimilarity with 95% overlap: {sim_similar:.3f}")
    logger.info("Similarity with 0% overlap: {sim_different:.3f}")


def demonstrate_binding_operations():
    """Demonstrate binding operations for combining information"""
    logger.info("\n=== Hypervector Binding Operations ===")

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
    binder.bind([gene_expr_bound, condition_hv], BindingType.CIRCULAR)

    logger.info("Original gene vector norm: {torch.norm(gene_hv):.3f}")
    logger.info("Gene+expression bound vector norm: {torch.norm(gene_expr_bound):.3f}")
    logger.info("Full bound vector norm: {torch.norm(full_bound):.3f}")

    # Demonstrate unbinding (recovery)
    recovered_gene = binder.unbind(gene_expr_bound, [expression_hv], BindingType.CIRCULAR)
    torch.nn.functional.cosine_similarity(gene_hv.unsqueeze(0), recovered_gene.unsqueeze(0)).item()

    logger.info("\nRecovery similarity after unbinding: {recovery_similarity:.3f}")

    # Demonstrate bundling (superposition)
    genes = [torch.randn(dimension) for _ in range(5)]
    gene_set = binder.bundle(genes, normalize=True)

    logger.info("\nBundled 5 genes into single vector")
    logger.info("Each gene's contribution to bundle:")
    for i, gene in enumerate(genes):
        torch.nn.functional.cosine_similarity(gene_set.unsqueeze(0), gene.unsqueeze(0)).item()
        logger.info("  Gene {i}: {contrib:.3f}")


def demonstrate_positional_encoding():
    """Demonstrate position-aware binding for genomic sequences"""
    logger.info("\n=== Positional Encoding for Genomic Sequences ===")

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
    pos_binder.bind_sequence(features, start_position=1000)

    logger.info("Encoded sequence of {sequence_length} elements")
    logger.info("Sequence hypervector dimension: {sequence_bound.shape[0]}")

    # Show that different positions create different bindings
    pos_100 = pos_binder.bind_with_position(features[0], 100)
    pos_200 = pos_binder.bind_with_position(features[0], 200)

    torch.nn.functional.cosine_similarity(pos_100.unsqueeze(0), pos_200.unsqueeze(0)).item()

    logger.info("Same feature at different positions similarity: {position_similarity:.3f}")


def demonstrate_holographic_encoding():
    """Demonstrate holographic encoding for structured data"""
    logger.info("\n=== Holographic Encoding for Genomic Variants ===")

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
            "clinical_significance": "pathogenic",
        },
    )

    variant2 = holo_encoder.encode_genomic_variant(
        chromosome="chr7",
        position=117559591,  # Adjacent position
        ref="C",
        alt="T",
        annotations={
            "gene": "CFTR",
            "effect": "synonymous",
            "clinical_significance": "benign",
        },
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
            "clinical_significance": "uncertain",
        },
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
            "clinical_significance": "pathogenic",
        },
    )

    # Compare similarities
    torch.nn.functional.cosine_similarity(variant1.unsqueeze(0), variant2.unsqueeze(0)).item()
    torch.nn.functional.cosine_similarity(variant1.unsqueeze(0), variant3.unsqueeze(0)).item()
    torch.nn.functional.cosine_similarity(variant1.unsqueeze(0), variant4.unsqueeze(0)).item()

    logger.info("Variant similarities:")
    logger.info("  Adjacent positions (same gene): {sim_adjacent:.3f}")
    logger.info("  Same gene, different position: {sim_same_gene:.3f}")
    logger.info("  Different gene: {sim_diff_gene:.3f}")

    # Create memory trace of multiple variants
    variants = [
        {"chr": "chr1", "pos": 1000, "re": "A", "alt": "G"},
        {"chr": "chr1", "pos": 2000, "re": "C", "alt": "T"},
        {"chr": "chr2", "pos": 3000, "re": "G", "alt": "A"},
    ]

    holo_encoder.create_memory_trace(variants)
    logger.info("\nCreated memory trace of {len(variants)} variants")
    logger.info("Memory trace dimension: {memory.shape[0]}")


def demonstrate_cross_modal_binding():
    """Demonstrate binding across different omics modalities"""
    logger.info("\n=== Cross-Modal Binding for Multi-Omics ===")

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
        "proteomic": proteomic_hv,
    }

    bound_modalities = cross_binder.bind_modalities(modality_data)

    logger.info("Created cross-modal bindings:")
    for key in bound_modalities:
        logger.info("  {key}: dimension {bound_modalities[key].shape[0]}")

    # Show that combined representation preserves information from all modalities
    combined = bound_modalities["combined"]

    logger.info("\nContribution of each modality to combined representation:")
    for modality, hv in modality_data.items():
        torch.nn.functional.cosine_similarity(combined.unsqueeze(0), hv.unsqueeze(0)).item()
        logger.info("  {modality}: {contrib:.3f}")


def demonstrate_similarity_preservation():
    """Demonstrate similarity preservation in mappings"""
    logger.info("\n=== Similarity-Preserving Transformations ===")

    # Create synthetic biological data with structure
    n_samples = 100
    n_features = 50

    # Create two clusters of samples
    cluster1 = torch.randn(n_samples // 2, n_features) + torch.tensor([2.0] * n_features)
    cluster2 = torch.randn(n_samples // 2, n_features) - torch.tensor([2.0] * n_features)

    data = torch.cat([cluster1, cluster2], dim=0)
    torch.cat([torch.zeros(n_samples // 2), torch.ones(n_samples // 2)])

    # Create biological similarity mapper
    mapper = BiologicalSimilarityMapper(n_features, 1000, OmicsType.GENOMIC)

    # Transform data
    transformed = mapper.fit_transform(data)

    logger.info("Original data: {data.shape}")
    logger.info("Transformed data: {transformed.shape}")

    # Check that clusters remain separated
    cluster1_transformed = transformed[: n_samples // 2]
    cluster2_transformed = transformed[n_samples // 2 :]

    torch.nn.functional.cosine_similarity(
        cluster1_transformed.mean(dim=0).unsqueeze(0),
        cluster1_transformed[0].unsqueeze(0),
    ).item()

    torch.nn.functional.cosine_similarity(
        cluster1_transformed.mean(dim=0).unsqueeze(0),
        cluster2_transformed.mean(dim=0).unsqueeze(0),
    ).item()

    logger.info("\nWithin-cluster similarity: {within_cluster_sim:.3f}")
    logger.info("Between-cluster similarity: {between_cluster_sim:.3f}")
    logger.info("Separation maintained: {within_cluster_sim > between_cluster_sim}")


def demonstrate_privacy_guarantees():
    """Demonstrate privacy properties of hypervectors"""
    logger.info("\n=== Privacy Guarantees of Hypervectors ===")

    dimension = 10000
    encoder = create_encoder(dimension=dimension)

    # Create sensitive genomic data
    sensitive_data = {
        "variants": {
            "snps": list(range(1000)),  # Many SNPs
            "pathogenic": ["BRCA1_c.5266dupC", "CFTR_F508del", "HBB_c.20A>T"],
        },
        "quality_metrics": {"mean_coverage": 42.7, "uniformity": 0.94},
    }

    # Encode to hypervector
    hypervector = encoder.encode(sensitive_data, OmicsType.GENOMIC)

    logger.info("Original data contains:")
    logger.info("  - {len(sensitive_data['variants']['snps'])} SNPs")
    logger.info("  - {len(sensitive_data['variants']['pathogenic'])} pathogenic variants")
    logger.info("\nEncoded to {dimension}D hypervector")

    # Demonstrate irreversibility
    # Try to recover information from hypervector
    logger.info("\nAttempting to recover original data...")

    # Random projection is not invertible when D >> d
    logger.info("Projection: {1000}D -> {dimension}D (not invertible)")

    # Show that similar inputs create distinguishable outputs
    modified_data = sensitive_data.copy()
    modified_data["variants"]["snps"] = list(range(999))  # Remove one SNP

    modified_hv = encoder.encode(modified_data, OmicsType.GENOMIC)

    encoder.similarity(hypervector, modified_hv)
    logger.info("\nSingle SNP difference creates {1-similarity:.4f} change in hypervector")
    logger.info("This demonstrates sensitivity while maintaining privacy")


def demonstrate_compression_tiers():
    """Demonstrate tiered compression for different use cases"""
    logger.info("\n=== Tiered Compression System ===")

    from local_processing.compression import CompressionTier, TieredCompressor

    # Create test hypervector
    dimension = 10000
    hypervector = torch.randn(dimension)
    hypervector = hypervector / torch.norm(hypervector)

    # Test different compression tiers
    for tier in [CompressionTier.MINI, CompressionTier.CLINICAL, CompressionTier.FULL]:
        compressor = TieredCompressor(tier)

        compressed = compressor.compress({"hypervector": hypervector}, OmicsType.GENOMIC)

        logger.info("\n{tier.value} tier:")
        logger.info("  Original size: {hypervector.numel() * 4:,} bytes")
        logger.info("  Compressed size: {compressed.compressed_size:,} bytes")
        logger.info(
            "  Compression ratio: {hypervector.numel() * 4 / compressed.compressed_size:.1f}:1"
        )

        # Test decompression
        decompressed = compressor.decompress(compressed)

        # Measure reconstruction quality
        torch.nn.functional.cosine_similarity(
            hypervector.unsqueeze(0), decompressed.unsqueeze(0)
        ).item()

        logger.info("  Reconstruction similarity: {reconstruction_sim:.3f}")


def visualize_hypervector_properties():
    """Visualize key properties of hypervectors"""
    logger.info("\n=== Visualizing Hypervector Properties ===")

    # This would normally create plots, but for text output:
    dimension = 1000
    n_samples = 50

    # Create random hypervectors
    vectors = torch.randn(n_samples, dimension)
    vectors = vectors / torch.norm(vectors, dim=1, keepdim=True)

    # Compute pairwise similarities
    similarities = torch.matmul(vectors, vectors.T)

    # Statistics
    torch.diag(similarities).mean().item()
    off_diag = similarities[~torch.eye(n_samples, dtype=bool)]
    off_diag.mean().item()
    off_diag.std().item()

    logger.info("\nRandom hypervector statistics ({dimension}D):")
    logger.info("  Self-similarity: {self_sim:.3f}")
    logger.info("  Mean pairwise similarity: {mean_sim:.3f} ± {std_sim:.3f}")
    logger.info("  Min similarity: {off_diag.min().item():.3f}")
    logger.info("  Max similarity: {off_diag.max().item():.3f}")

    # Concentration of measure
    logger.info("\nConcentration of measure:")
    logger.info(
        "  99% of similarities in range: [{mean_sim - 3*std_sim:.3f}, {mean_sim + 3*std_sim:.3f}]"
    )
    logger.info("  This demonstrates the 'blessing of dimensionality' for privacy")


def main():
    """Run all demonstrations"""
    logger.info("=" * 60)
    logger.info("GenomeVault Hypervector Encoding Demonstration")
    logger.info("=" * 60)

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

    logger.info("\n" + "=" * 60)
    logger.info("Demonstration Complete")
    logger.info("=" * 60)
    logger.info("\nKey Takeaways:")
    logger.info("1. Hypervectors preserve biological similarities while protecting privacy")
    logger.info("2. Binding operations enable complex relationships to be encoded")
    logger.info("3. Holographic encoding allows structured data representation")
    logger.info("4. Cross-modal binding integrates multiple omics types")
    logger.info("5. Tiered compression provides flexibility for different use cases")
    logger.info("6. High dimensionality provides mathematical privacy guarantees")


if __name__ == "__main__":
    main()
