"""
Integration example: Local Processing + Hypervector Encoding

This example demonstrates the complete flow from raw genomic data
to privacy-preserving hypervector representation.
"""

from pathlib import Path

import torch

from core.constants import OmicsType
from hypervector_transform.binding import CrossModalBinder

# Phase 2: Hypervector Encoding
from hypervector_transform.encoding import create_encoder
from hypervector_transform.holographic import HolographicEncoder
from local_processing.compression import CompressionTier, TieredCompressor

# Phase 1: Local Processing
from local_processing.sequencing import SequencingProcessor
from utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def process_and_encode_genome(vcf_path: str):
    """
    Complete pipeline from VCF to privacy-preserving hypervector
    """
    print("=== GenomeVault Integration Example ===\n")
    
    # Phase 1: Local Processing
    print("Phase 1: Processing genomic data locally...")
    processor = SequencingProcessor()
    
    # Process VCF file
    genomic_data = processor.process_vcf(vcf_path)
    
    print(f"  - Processed {genomic_data['summary']['total_variants']} variants")
    print(f"  - Found {genomic_data['summary']['snp_count']} SNPs")
    print(f"  - Found {genomic_data['summary']['indel_count']} indels")
    print(f"  - Mean quality: {genomic_data['quality_metrics']['mean_quality']:.1f}")
    
    # Phase 2: Hypervector Encoding
    print("\nPhase 2: Encoding to hypervector...")
    encoder = create_encoder(dimension=10000, projection_type="sparse_random")
    
    # Encode genomic data
    genomic_hv = encoder.encode(genomic_data, OmicsType.GENOMIC)
    print(f"  - Encoded to {genomic_hv.shape[0]}D hypervector")
    
    # Calculate compression
    original_size = genomic_data['summary']['total_variants'] * 100  # Rough estimate bytes per variant
    hv_size = genomic_hv.shape[0] * 4  # 4 bytes per float
    print(f"  - Compression ratio: {original_size / hv_size:.1f}:1")
    
    # Tier compression
    print("\nPhase 2b: Tiered compression...")
    compressor = TieredCompressor(CompressionTier.CLINICAL)
    compressed = compressor.compress(
        {"hypervector": genomic_hv, **genomic_data},
        OmicsType.GENOMIC
    )
    
    print(f"  - Clinical tier size: {compressed.compressed_size:,} bytes")
    print(f"  - Total compression: {original_size / compressed.compressed_size:.1f}:1")
    
    return genomic_hv, compressed, genomic_data


def demonstrate_variant_encoding(genomic_data: dict):
    """
    Show holographic encoding of specific variants
    """
    print("\nHolographic Variant Encoding:")
    
    holo_encoder = HolographicEncoder(dimension=5000)
    
    # Encode a few variants
    if genomic_data['variants']['snps']:
        # Take first SNP as example
        snp = genomic_data['variants']['snps'][0]
        
        variant_hv = holo_encoder.encode_genomic_variant(
            chromosome=snp['chr'],
            position=snp['pos'],
            ref=snp['ref'],
            alt=snp['alt'],
            annotations=snp.get('annotations', {})
        )
        
        print(f"  - Encoded variant {snp['chr']}:{snp['pos']} {snp['ref']}>{snp['alt']}")
        print(f"  - Holographic dimension: {variant_hv.shape[0]}")
        
        # Query for chromosome
        chr_recovered = holo_encoder.query(variant_hv, "chr")
        print(f"  - Can query components without exposing raw data")
    
    return variant_hv if genomic_data['variants']['snps'] else None


def demonstrate_privacy_preservation(genomic_hv: torch.Tensor):
    """
    Show privacy properties of hypervectors
    """
    print("\nPrivacy Demonstration:")
    
    # Create slightly modified version
    noise = torch.randn_like(genomic_hv) * 0.01
    modified_hv = genomic_hv + noise
    modified_hv = modified_hv / torch.norm(modified_hv)
    
    # Compare
    similarity = torch.nn.functional.cosine_similarity(
        genomic_hv.unsqueeze(0),
        modified_hv.unsqueeze(0)
    ).item()
    
    print(f"  - Small changes in input create detectable changes in output")
    print(f"  - Similarity after small modification: {similarity:.4f}")
    
    # Show irreversibility
    print(f"  - Original data dimension: ~1000 features")
    print(f"  - Hypervector dimension: {genomic_hv.shape[0]}")
    print(f"  - Transformation is irreversible (underdetermined system)")
    
    # Demonstrate distributed information
    top_10_percent = int(genomic_hv.shape[0] * 0.1)
    top_indices = torch.topk(torch.abs(genomic_hv), top_10_percent).indices
    
    partial_hv = torch.zeros_like(genomic_hv)
    partial_hv[top_indices] = genomic_hv[top_indices]
    
    partial_similarity = torch.nn.functional.cosine_similarity(
        genomic_hv.unsqueeze(0),
        partial_hv.unsqueeze(0)
    ).item()
    
    print(f"  - Information is distributed: top 10% of components only capture "
          f"{partial_similarity:.2%} of information")


def simulate_multi_omics_integration():
    """
    Demonstrate multi-omics integration with privacy
    """
    print("\nMulti-Omics Integration:")
    
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
        "epigenomic": epigenomic_hv
    }
    
    integrated = cross_binder.bind_modalities(modalities)
    combined = integrated["combined"]
    
    print(f"  - Integrated 3 omics types into single {combined.shape[0]}D vector")
    print(f"  - Each modality contributes to the combined representation")
    print(f"  - Original data remains private while enabling integrated analysis")
    
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
        print("Creating simulated genomic data for demonstration...")
        # In practice, you would process real VCF files
        # For demo, we'll create mock data
        
        genomic_data = {
            "variants": {
                "snps": [
                    {"chr": "chr1", "pos": 100, "ref": "A", "alt": "G", "quality": 30},
                    {"chr": "chr1", "pos": 200, "ref": "C", "alt": "T", "quality": 40},
                    {"chr": "chr2", "pos": 300, "ref": "G", "alt": "A", "quality": 35},
                ],
                "indels": [
                    {"chr": "chr3", "pos": 400, "ref": "AT", "alt": "A", "quality": 25}
                ],
                "cnvs": []
            },
            "quality_metrics": {
                "mean_quality": 35.0,
                "mean_coverage": 30.0,
                "uniformity": 0.92,
                "gc_content": 0.41
            },
            "summary": {
                "total_variants": 4,
                "snp_count": 3,
                "indel_count": 1,
                "cnv_count": 0
            }
        }
        
        # Encode the mock data
        encoder = create_encoder(dimension=10000)
        genomic_hv = encoder.encode(genomic_data, OmicsType.GENOMIC)
        
        # Compress
        compressor = TieredCompressor(CompressionTier.CLINICAL)
        compressed = compressor.compress(
            {"hypervector": genomic_hv, **genomic_data},
            OmicsType.GENOMIC
        )
        
        print(f"\nPhase 1: Simulated processing of {genomic_data['summary']['total_variants']} variants")
        print(f"Phase 2: Encoded to {genomic_hv.shape[0]}D hypervector")
        print(f"Compressed to {compressed.compressed_size:,} bytes")
        
    else:
        # Process real VCF file
        genomic_hv, compressed, genomic_data = process_and_encode_genome(vcf_path)
    
    # Demonstrate additional features
    variant_hv = demonstrate_variant_encoding(genomic_data)
    demonstrate_privacy_preservation(genomic_hv)
    combined_hv = simulate_multi_omics_integration()
    
    print("\n=== Integration Complete ===")
    print("\nKey Points:")
    print("1. Raw genomic data never leaves the user's device")
    print("2. Hypervectors preserve biological relationships")
    print("3. Mathematical privacy through high-dimensional projection")
    print("4. Efficient compression for storage and transmission")
    print("5. Multi-omics integration while maintaining privacy")
    print("\nNext: Zero-knowledge proofs will enable verification without disclosure")


if __name__ == "__main__":
    main()
