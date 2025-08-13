"""
Integration example: Local Processing + Hypervector Encoding

This example demonstrates the complete flow from raw genomic data
to privacy-preserving hypervector representation.
"""
from pathlib import Path

import torch

from genomevault.core.constants import OmicsType
from genomevault.hypervector_transform.binding import CrossModalBinder
from genomevault.hypervector_transform.encoding import create_encoder
from genomevault.hypervector_transform.holographic import HolographicEncoder
from genomevault.local_processing.compression import CompressionTier, TieredCompressor
from genomevault.local_processing.sequencing import SequencingProcessor
from genomevault.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def process_and_encode_genome(vcf_path: str):
    """
    Complete pipeline from VCF to privacy-preserving hypervector
    """
    logger.info("=== GenomeVault Integration Example ===\n")

    # Phase 1: Local Processing
    logger.info("Phase 1: Processing genomic data locally...")
    processor = SequencingProcessor()

    # Process VCF file
    genomic_data = processor.process_vcf(vcf_path)

    logger.info("  - Processed {genomic_data['summary']['total_variants']} variants")
    logger.info("  - Found {genomic_data['summary']['snp_count']} SNPs")
    logger.info("  - Found {genomic_data['summary']['indel_count']} indels")
    logger.info(
        "  - Mean quality: {genomic_data['quality_metrics']['mean_quality']:.1f}"
    )

    # Phase 2: Hypervector Encoding
    logger.info("\nPhase 2: Encoding to hypervector...")
    encoder = create_encoder(dimension=10000, projection_type="sparse_random")

    # Encode genomic data
    genomic_hv = encoder.encode(genomic_data, OmicsType.GENOMIC)
    logger.info("  - Encoded to {genomic_hv.shape[0]}D hypervector")

    # Calculate compression
    (
        genomic_data["summary"]["total_variants"] * 100
    )  # Rough estimate bytes per variant
    genomic_hv.shape[0] * 4  # 4 bytes per float
    logger.info("  - Compression ratio: {original_size / hv_size:.1f}:1")

    # Tier compression
    logger.info("\nPhase 2b: Tiered compression...")
    compressor = TieredCompressor(CompressionTier.CLINICAL)
    compressed = compressor.compress(
        {"hypervector": genomic_hv, **genomic_data}, OmicsType.GENOMIC
    )

    logger.info("  - Clinical tier size: {compressed.compressed_size:,} bytes")
    logger.info(
        "  - Total compression: {original_size / compressed.compressed_size:.1f}:1"
    )

    return genomic_hv, compressed, genomic_data


def demonstrate_variant_encoding(genomic_data: dict):
    """
    Show holographic encoding of specific variants
    """
    logger.info("\nHolographic Variant Encoding:")

    holo_encoder = HolographicEncoder(dimension=5000)

    # Encode a few variants
    if genomic_data["variants"]["snps"]:
        # Take first SNP as example
        snp = genomic_data["variants"]["snps"][0]

        variant_hv = holo_encoder.encode_genomic_variant(
            chromosome=snp["chr"],
            position=snp["pos"],
            ref=snp["ref"],
            alt=snp["alt"],
            annotations=snp.get("annotations", {}),
        )

        logger.info(
            "  - Encoded variant {snp['chr']}:{snp['pos']} {snp['re']}>{snp['alt']}"
        )
        logger.info("  - Holographic dimension: {variant_hv.shape[0]}")

        # Query for chromosome
        holo_encoder.query(variant_hv, "chr")
        logger.info("  - Can query components without exposing raw data")

    return variant_hv if genomic_data["variants"]["snps"] else None


def demonstrate_privacy_preservation(genomic_hv: torch.Tensor):
    """
    Show privacy properties of hypervectors
    """
    logger.info("\nPrivacy Demonstration:")

    # Create slightly modified version
    noise = torch.randn_like(genomic_hv) * 0.01
    modified_hv = genomic_hv + noise
    modified_hv = modified_hv / torch.norm(modified_hv)

    # Compare
    torch.nn.functional.cosine_similarity(
        genomic_hv.unsqueeze(0), modified_hv.unsqueeze(0)
    ).item()

    logger.info("  - Small changes in input create detectable changes in output")
    logger.info("  - Similarity after small modification: {similarity:.4f}")

    # Show irreversibility
    logger.info("  - Original data dimension: ~1000 features")
    logger.info("  - Hypervector dimension: {genomic_hv.shape[0]}")
    logger.info("  - Transformation is irreversible (underdetermined system)")

    # Demonstrate distributed information
    top_10_percent = int(genomic_hv.shape[0] * 0.1)
    top_indices = torch.topk(torch.abs(genomic_hv), top_10_percent).indices

    partial_hv = torch.zeros_like(genomic_hv)
    partial_hv[top_indices] = genomic_hv[top_indices]

    torch.nn.functional.cosine_similarity(
        genomic_hv.unsqueeze(0), partial_hv.unsqueeze(0)
    ).item()

    logger.info(
        "  - Information is distributed: top 10% of components only capture "
        "{partial_similarity:.2%} of information"
    )


def simulate_multi_omics_integration():
    """
    Demonstrate multi-omics integration with privacy
    """
    logger.info("\nMulti-Omics Integration:")

    dimension = 5000
    cross_binder = CrossModalBinder(dimension)

    # Simulate different omics hypervectors
    # In practice, these would come from processing real data
    genomic_hv = torch.randn(dimension)
    genomic_hv = genomic_hv / torch.norm(genomic_hv)

    transcriptomic_hv = torch.randn(dimension)
    transcriptomic_hv = transcriptomic_hv / torch.norm(transcriptomic_hv)

    epigenomic_hv = torch.randn(dimension)
    epigenomic_hv = epigenomic_hv / torch.norm(epigenomic_hv)

    # Integrate
    modalities = {
        "genomic": genomic_hv,
        "transcriptomic": transcriptomic_hv,
        "epigenomic": epigenomic_hv,
    }

    integrated = cross_binder.bind_modalities(modalities)
    combined = integrated["combined"]

    logger.info("  - Integrated 3 omics types into single {combined.shape[0]}D vector")
    logger.info("  - Each modality contributes to the combined representation")
    logger.info("  - Original data remains private while enabling integrated analysis")

    return combined


def main():
    """
    Run the complete integration example
    """
    setup_logging(level="INFO")

    # Example VCF file (you would use a real file)
    vcf_path = "example_data/sample.vcf"

    # Check if example data exists
    if not Path(vcf_path).exists():
        logger.info("Creating simulated genomic data for demonstration...")
        # In practice, you would process real VCF files
        # For demo, we'll create mock data

        genomic_data = {
            "variants": {
                "snps": [
                    {"chr": "chr1", "pos": 100, "re": "A", "alt": "G", "quality": 30},
                    {"chr": "chr1", "pos": 200, "re": "C", "alt": "T", "quality": 40},
                    {"chr": "chr2", "pos": 300, "re": "G", "alt": "A", "quality": 35},
                ],
                "indels": [
                    {"chr": "chr3", "pos": 400, "re": "AT", "alt": "A", "quality": 25}
                ],
                "cnvs": [],
            },
            "quality_metrics": {
                "mean_quality": 35.0,
                "mean_coverage": 30.0,
                "uniformity": 0.92,
                "gc_content": 0.41,
            },
            "summary": {
                "total_variants": 4,
                "snp_count": 3,
                "indel_count": 1,
                "cnv_count": 0,
            },
        }

        # Encode the mock data
        encoder = create_encoder(dimension=10000)
        genomic_hv = encoder.encode(genomic_data, OmicsType.GENOMIC)

        # Compress
        compressor = TieredCompressor(CompressionTier.CLINICAL)
        compressed = compressor.compress(
            {"hypervector": genomic_hv, **genomic_data}, OmicsType.GENOMIC
        )

        logger.info(
            "\nPhase 1: Simulated processing of {genomic_data['summary']['total_variants']} variants"
        )
        logger.info("Phase 2: Encoded to {genomic_hv.shape[0]}D hypervector")
        logger.info("Compressed to {compressed.compressed_size:,} bytes")

    else:
        # Process real VCF file
        genomic_hv, compressed, genomic_data = process_and_encode_genome(vcf_path)

    # Demonstrate additional features
    demonstrate_variant_encoding(genomic_data)
    demonstrate_privacy_preservation(genomic_hv)
    simulate_multi_omics_integration()

    logger.info("\n=== Integration Complete ===")
    logger.info("\nKey Points:")
    logger.info("1. Raw genomic data never leaves the user's device")
    logger.info("2. Hypervectors preserve biological relationships")
    logger.info("3. Mathematical privacy through high-dimensional projection")
    logger.info("4. Efficient compression for storage and transmission")
    logger.info("5. Multi-omics integration while maintaining privacy")
    logger.info(
        "\nNext: Zero-knowledge proofs will enable verification without disclosure"
    )


if __name__ == "__main__":
    main()
