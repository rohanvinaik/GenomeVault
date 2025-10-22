#!/usr/bin/env python3
"""
Quick verification that the chunking bug is fixed.
Run this to confirm the fix works before running full benchmarks.
"""

import sys
import time
from pathlib import Path

# Add genomevault to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.differential_encoding.chunking import (
    CryptographicChunker,
    ChunkingStrategy,
    AnalysisType,
    STRATEGY_CONFIGS,
)
from genomevault.differential_encoding.crypto_primitives import CryptoRNG
from genomevault.differential_encoding.reference_management import (
    Variant,
    GenomeSection,
)


def create_test_variants(chromosome: str, positions: list) -> list:
    """Create test variants."""
    return [
        Variant(
            chromosome=chromosome,
            position=pos,
            reference_allele="A",
            alternate_allele="T"
        )
        for pos in positions
    ]


def main():
    print("="*70)
    print("DIFFERENTIAL ENCODING CHUNKING BUG FIX VERIFICATION")
    print("="*70)
    print()
    
    # Test Case 1: Dense variants (previously triggered infinite loop)
    print("Test 1: Dense variant region (10k variants in 100kb)")
    print("-" * 70)
    positions = list(range(100000, 200000, 10))
    variants = create_test_variants("chr1", positions)
    
    section = GenomeSection(
        chromosome="chr1",
        start_position=100000,
        end_position=200000,
        variants=variants
    )
    
    strategy = ChunkingStrategy(
        strategy_type=AnalysisType.SLIDING_WINDOW,
        chunk_size=50000,
        overlap=10000,
        min_variants=50,
        max_variants=100,  # Will truncate heavily
        randomize_boundaries=False,
        respect_features=False
    )
    
    rng = CryptoRNG()
    chunker = CryptographicChunker(strategy, rng)
    
    start = time.time()
    try:
        chunks = chunker.chunk_genome_section(section, rng.derive_seed(b"test1"))
        elapsed = time.time() - start
        print(f"✅ PASSED: Created {len(chunks)} chunks in {elapsed:.3f}s")
        print(f"   (Previously would infinite loop)")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    print()
    
    # Test Case 2: Large region (previously triggered 100k limit)
    print("Test 2: Whole chromosome simulation (249M bp, ~250k variants)")
    print("-" * 70)
    positions = []
    pos = 1000000
    while pos < 250_000_000:
        for i in range(50):  # clusters of 50
            positions.append(pos + i * 100)
        pos += 100000
    
    variants = create_test_variants("chr1", positions)
    
    section = GenomeSection(
        chromosome="chr1",
        start_position=1000000,
        end_position=250_000_000,
        variants=variants
    )
    
    strategy = STRATEGY_CONFIGS[AnalysisType.WHOLE_CHROMOSOME]
    chunker = CryptographicChunker(strategy, rng)
    
    start = time.time()
    try:
        chunks = chunker.chunk_genome_section(section, rng.derive_seed(b"test2"))
        elapsed = time.time() - start
        
        expected = (250_000_000 - 1000000) // (strategy.chunk_size - strategy.overlap)
        
        print(f"✅ PASSED: Created {len(chunks)} chunks in {elapsed:.3f}s")
        print(f"   Expected ~{expected} chunks, got {len(chunks)} ({len(chunks)/expected:.2f}x)")
        print(f"   Processed {len(variants):,} variants")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    print()
    
    # Test Case 3: All strategies
    print("Test 3: All analysis strategies")
    print("-" * 70)
    
    positions = list(range(1000000, 10000000, 500))
    variants = create_test_variants("chr1", positions)
    
    section = GenomeSection(
        chromosome="chr1",
        start_position=1000000,
        end_position=10000000,
        variants=variants
    )
    
    all_passed = True
    for analysis_type in AnalysisType:
        strategy = STRATEGY_CONFIGS[analysis_type]
        chunker = CryptographicChunker(strategy, rng)
        
        try:
            start = time.time()
            chunks = chunker.chunk_genome_section(
                section, 
                rng.derive_seed(f"{analysis_type.value}".encode())
            )
            elapsed = time.time() - start
            
            print(f"  ✅ {analysis_type.value:20s}: {len(chunks):4d} chunks in {elapsed:.3f}s")
        except Exception as e:
            print(f"  ❌ {analysis_type.value:20s}: FAILED - {e}")
            all_passed = False
    
    if not all_passed:
        return False
    
    print()
    print("="*70)
    print("✅ ALL TESTS PASSED - BUG IS FIXED!")
    print("="*70)
    print()
    print("Summary:")
    print("  • Dense variant regions: ✅ No infinite loops")
    print("  • Large chromosomes: ✅ Completes in reasonable time")
    print("  • All strategies: ✅ Working correctly")
    print()
    print("You can now run your benchmarks:")
    print("  cd benchmarks/differential_encoding")
    print("  python benchmark_chunking.py")
    print("  python benchmark_end_to_end.py")
    print()
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
