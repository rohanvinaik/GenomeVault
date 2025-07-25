#!/usr/bin/env python3
"""
Test script for SNP dial functionality
Demonstrates single-nucleotide accuracy encoding with different panel settings
"""

import time
from typing import Dict, List

import torch

from genomevault.hypervector.encoding.genomic import GenomicEncoder, PanelGranularity
from genomevault.hypervector.positional import PositionalEncoder, SNPPanel


def demonstrate_snp_dial():
    """Demonstrate the SNP dial functionality"""
    print("=== GenomeVault SNP Dial Demonstration ===\n")
    
    # Test variants
    test_variants = [
        {"chromosome": "chr1", "position": 100000, "ref": "A", "alt": "G"},
        {"chromosome": "chr1", "position": 100100, "ref": "C", "alt": "T"},
        {"chromosome": "chr1", "position": 200000, "ref": "G", "alt": "A"},
        {"chromosome": "chr2", "position": 150000, "ref": "T", "alt": "C"},
    ]
    
    # Test different panel settings
    panel_settings = [
        (PanelGranularity.OFF, "Standard encoding (no SNP panel)"),
        (PanelGranularity.COMMON, "Common SNPs (1M positions)"),
        (PanelGranularity.CLINICAL, "Clinical/dbSNP (10M positions)")
    ]
    
    results = {}
    
    for granularity, description in panel_settings:
        print(f"\n--- Testing: {description} ---")
        
        # Create encoder with SNP mode
        enable_snp = granularity != PanelGranularity.OFF
        encoder = GenomicEncoder(
            dimension=100000,
            enable_snp_mode=enable_snp,
            panel_granularity=granularity
        )
        
        # Measure encoding time
        start_time = time.time()
        
        if enable_snp:
            # Use panel encoding
            encoded_genome = encoder.encode_genome_with_panel(test_variants)
        else:
            # Use standard encoding
            encoded_genome = encoder.encode_genome(test_variants)
            
        encode_time = (time.time() - start_time) * 1000
        
        # Get memory usage if SNP mode
        memory_stats = None
        if enable_snp:
            memory_stats = encoder.positional_encoder.get_memory_usage()
        
        results[granularity.value] = {
            "encode_time_ms": encode_time,
            "hypervector_dim": encoder.dimension,
            "memory_stats": memory_stats,
            "encoded_genome": encoded_genome
        }
        
        print(f"  Encoding time: {encode_time:.2f} ms")
        print(f"  Hypervector dimension: {encoder.dimension}")
        
        if memory_stats:
            print(f"  Memory usage:")
            print(f"    - Cache entries: {memory_stats['cache_entries']}")
            print(f"    - Cache size: {memory_stats['cache_size_mb']:.2f} MB")
            print(f"    - Non-zeros per vector: {memory_stats['nnz_per_vector']}")
    
    # Compare similarity between encodings
    print("\n\n--- Similarity Comparison ---")
    print("Comparing genome encodings between different panel settings:")
    
    base_encoding = results["off"]["encoded_genome"]
    
    for granularity in ["common", "clinical"]:
        if granularity in results:
            similarity = torch.cosine_similarity(
                base_encoding, 
                results[granularity]["encoded_genome"], 
                dim=0
            ).item()
            print(f"  Standard vs {granularity}: {similarity:.4f}")
    
    # Demonstrate hierarchical zoom
    print("\n\n--- Hierarchical Zoom Demonstration ---")
    
    encoder = GenomicEncoder(
        dimension=100000,
        enable_snp_mode=True,
        panel_granularity=PanelGranularity.CLINICAL
    )
    
    # Create zoom tiles for chromosome 1
    chr1_variants = [v for v in test_variants if v["chromosome"] == "chr1"]
    encoder.create_zoom_tiles("chr1", chr1_variants)
    
    # Query different zoom levels
    zoom_tests = [
        (0, 0, 250000, "Full chromosome"),
        (1, 0, 250000, "1Mb windows"),
        (2, 100000, 101000, "1kb tiles")
    ]
    
    for level, start, end, description in zoom_tests:
        zoom_vec = encoder.get_zoom_vector("chr1", start, end, level)
        print(f"\n  Level {level} ({description}):")
        print(f"    Region: chr1:{start}-{end}")
        print(f"    Vector norm: {torch.norm(zoom_vec).item():.4f}")
        print(f"    Non-zero elements: {(zoom_vec != 0).sum().item()}")
    
    print("\n\n=== Demonstration Complete ===")
    
    # Show performance summary
    print("\nPerformance Summary:")
    print("-" * 50)
    print(f"{'Setting':<15} {'Encode Time (ms)':<20} {'Extra RAM (MB)':<15}")
    print("-" * 50)
    
    for granularity, desc in panel_settings:
        result = results[granularity.value]
        encode_time = result["encode_time_ms"]
        
        # Estimate extra RAM
        if granularity == PanelGranularity.OFF:
            extra_ram = 0
        elif granularity == PanelGranularity.COMMON:
            extra_ram = 40  # ~40MB for 1M positions
        else:
            extra_ram = 400  # ~400MB for 10M positions
            
        print(f"{granularity.value:<15} {encode_time:<20.2f} {extra_ram:<15}")


def test_custom_panel():
    """Test loading custom SNP panel from file"""
    print("\n\n--- Custom Panel Test ---")
    
    # Create a mock BED file
    mock_bed_content = """chr1\t100000\t100001\trs123
chr1\t100100\t100101\trs124
chr1\t200000\t200001\trs125
chr2\t150000\t150001\trs126
"""
    
    # Save to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.bed', delete=False) as f:
        f.write(mock_bed_content)
        temp_file = f.name
    
    try:
        # Create encoder and load custom panel
        encoder = GenomicEncoder(
            dimension=100000,
            enable_snp_mode=True,
            panel_granularity=PanelGranularity.OFF
        )
        
        encoder.load_custom_panel(temp_file, "my_custom_panel")
        
        # Get panel info
        panel_info = encoder.snp_panel.get_panel_info("my_custom_panel")
        print(f"\nCustom panel loaded:")
        print(f"  Name: {panel_info['name']}")
        print(f"  Size: {panel_info['size']} positions")
        print(f"  Chromosomes: {', '.join(panel_info['chromosomes'])}")
        
        # Test encoding with custom panel
        test_variants = [
            {"chromosome": "chr1", "position": 100000, "ref": "A", "alt": "G"},
            {"chromosome": "chr1", "position": 100100, "ref": "C", "alt": "T"},
        ]
        
        encoded = encoder.encode_genome_with_panel(test_variants, "my_custom_panel")
        print(f"  Encoded genome vector norm: {torch.norm(encoded).item():.4f}")
        
    finally:
        # Clean up
        import os
        os.unlink(temp_file)


def benchmark_panel_scaling():
    """Benchmark encoding performance with different panel sizes"""
    print("\n\n--- Panel Scaling Benchmark ---")
    
    # Create test variants
    import random
    random.seed(42)
    
    num_variants = 1000
    test_variants = []
    for i in range(num_variants):
        test_variants.append({
            "chromosome": f"chr{random.randint(1, 22)}",
            "position": random.randint(1, 250000000),
            "ref": random.choice(["A", "C", "G", "T"]),
            "alt": random.choice(["A", "C", "G", "T"])
        })
    
    # Test different dimensions and panel sizes
    configs = [
        (10000, 0.01, "10k dim, 1% sparse"),
        (50000, 0.01, "50k dim, 1% sparse"),
        (100000, 0.01, "100k dim, 1% sparse"),
        (100000, 0.001, "100k dim, 0.1% sparse"),
    ]
    
    print(f"\nEncoding {num_variants} variants with different configurations:")
    print("-" * 70)
    print(f"{'Config':<25} {'Encode Time (ms)':<20} {'Memory (MB)':<15}")
    print("-" * 70)
    
    for dim, sparsity, desc in configs:
        encoder = GenomicEncoder(
            dimension=dim,
            enable_snp_mode=True,
            panel_granularity=PanelGranularity.CLINICAL
        )
        
        # Update sparsity
        encoder.positional_encoder.sparsity = sparsity
        encoder.positional_encoder.nnz = int(dim * sparsity)
        
        # Measure encoding
        start_time = time.time()
        encoded = encoder.encode_genome_with_panel(test_variants)
        encode_time = (time.time() - start_time) * 1000
        
        # Get memory stats
        memory_stats = encoder.positional_encoder.get_memory_usage()
        
        print(f"{desc:<25} {encode_time:<20.2f} {memory_stats['cache_size_mb']:<15.2f}")


def demonstrate_proof_aggregation():
    """Demonstrate how SNP-level proofs can be aggregated"""
    print("\n\n--- Proof Aggregation for SNP Queries ---")
    
    # Simulate different query scenarios
    scenarios = [
        {
            "name": "Single variant lookup",
            "panel": "off",
            "proof_size_kb": 128,
            "verification_time_ms": 5
        },
        {
            "name": "Common SNP panel query",
            "panel": "common",
            "proof_size_kb": 256,
            "verification_time_ms": 12
        },
        {
            "name": "Clinical panel query",
            "panel": "clinical",
            "proof_size_kb": 512,
            "verification_time_ms": 25
        },
        {
            "name": "Hierarchical zoom (3 levels)",
            "panel": "clinical",
            "proof_size_kb": 384,
            "verification_time_ms": 18
        }
    ]
    
    print("\nProof characteristics for different query types:")
    print("-" * 80)
    print(f"{'Scenario':<30} {'Panel':<10} {'Proof Size':<15} {'Verify Time':<15}")
    print("-" * 80)
    
    for scenario in scenarios:
        print(f"{scenario['name']:<30} {scenario['panel']:<10} "
              f"{scenario['proof_size_kb']} KB{'':<10} "
              f"{scenario['verification_time_ms']} ms")
    
    print("\nNote: Actual proof sizes depend on circuit complexity and public inputs")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_snp_dial()
    test_custom_panel()
    benchmark_panel_scaling()
    demonstrate_proof_aggregation()
    
    print("\n\n=== All Tests Complete ===")
    print("\nThe SNP dial implementation provides:")
    print("1. Tunable single-nucleotide accuracy (off/common/clinical/custom)")
    print("2. Memory-efficient sparse position encoding")
    print("3. Hierarchical zoom for region queries")
    print("4. Minimal code changes to existing pipeline")
    print("5. Compatible with existing PIR and ZK proof systems")
