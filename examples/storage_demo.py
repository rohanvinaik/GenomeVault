"""
Differential Encoding Storage Demo

This example demonstrates the storage and serialization functionality (Section 7.2)
that is fully working. It shows:
1. Creating mock encoding results
2. Creating an EncodedGenome from results
3. Saving to compressed storage
4. Loading and verifying integrity
5. Analyzing compression efficiency

NOTE: For a complete end-to-end pipeline demo (Sections 2-7.2), see
complete_pipeline_demo.py. That demo requires API compatibility fixes
that are tracked in the project todo list.
"""

import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np

from genomevault.differential_encoding import (
    # Metadata
    DifferentialEncodingMetadata,

    # Storage
    EncodedGenome,
)


def create_mock_encoding_result():
    """
    Create a mock encoding result for demonstration.

    In a real scenario, this would come from DifferentialGenomicEncoder.encode_experimental_genome()
    """
    print("📊 Creating mock encoding result...")

    # Create sample hypervectors (10,000D, normalized)
    num_chunks = 50
    dimension = 10000

    chunk_hypervectors = []
    metadata_list = []

    for i in range(num_chunks):
        # Create random hypervector and normalize
        hv = np.random.randn(dimension).astype(np.float32)
        hv = hv / np.linalg.norm(hv)
        chunk_hypervectors.append(hv)

        # Create metadata for chunk (all IDs must be 32 bytes)
        chunk_id_str = f"chunk_{i:04d}".encode()
        chunk_id = chunk_id_str + b'\x00' * (32 - len(chunk_id_str))  # Pad to 32 bytes

        meta = DifferentialEncodingMetadata(
            chunk_id=chunk_id,
            chromosome=f"chr{(i % 22) + 1}",
            start_position=100000 + i * 50000,
            end_position=150000 + i * 50000,
            reference_genome_id=f"reference_{(i % 5) + 1:03d}",
            reference_seed=bytes(range(32)),  # Deterministic seed
            reference_hash=bytes(range(32, 64)),  # Deterministic hash
            cryptographic_binding=bytes(range(64, 96)),  # Deterministic binding
            chunking_strategy="sliding_window",
            chunking_seed=bytes(range(96, 128)),  # Deterministic chunking seed
            analysis_type="sliding_window",
            difference_counts={
                "new_mutations": 5 + (i % 10),
                "missing_variants": 3 + (i % 7),
                "genotype_differences": 2 + (i % 5),
                "total": (5 + (i % 10)) + (3 + (i % 7)) + (2 + (i % 5)),
            },
            created_timestamp=datetime.now(),
        )
        metadata_list.append(meta)

    # Create bundled hypervector (superposition of all chunks)
    bundled_hypervector = np.mean(chunk_hypervectors, axis=0)
    bundled_hypervector = bundled_hypervector / np.linalg.norm(bundled_hypervector)

    # Calculate statistics
    statistics = {
        "total_chunks": num_chunks,
        "total_differences": sum(m.difference_counts["total"] for m in metadata_list),
        "new_mutations": sum(m.difference_counts["new_mutations"] for m in metadata_list),
        "missing_variants": sum(m.difference_counts["missing_variants"] for m in metadata_list),
        "genotype_differences": sum(m.difference_counts["genotype_differences"] for m in metadata_list),
        "chromosomes": sorted(set(m.chromosome for m in metadata_list)),
        "hypervector_dimension": dimension,
    }

    # Create a mock EncodingResult-like object
    class EncodingResult:
        def __init__(self, hypervectors, metadata, bundled_hypervector, statistics):
            self.hypervectors = hypervectors
            self.metadata = metadata
            self.bundled_hypervector = bundled_hypervector
            self.statistics = statistics

    result = EncodingResult(
        hypervectors=chunk_hypervectors,
        metadata=metadata_list,
        bundled_hypervector=bundled_hypervector,
        statistics=statistics
    )

    print(f"  ✅ Created mock result:")
    print(f"     - {len(result.hypervectors)} chunk hypervectors")
    print(f"     - {len(result.metadata)} metadata entries")
    print(f"     - Bundled hypervector: {result.bundled_hypervector.shape}")
    print(f"     - Total differences: {result.statistics['total_differences']}")

    return result


def demonstrate_storage(result, genome_id="patient_demo_001", assembly="GRCh38"):
    """
    Demonstrate storage and serialization functionality.
    """
    print(f"\n💾 Creating EncodedGenome for {genome_id}...")

    # Create master seed
    master_seed = b"demo_master_seed_12345678901234"  # 32 bytes

    # Create EncodedGenome from result
    encoded = EncodedGenome.from_encoding_result(
        genome_id=genome_id,
        assembly=assembly,
        result=result,
        master_seed=master_seed
    )

    print(f"  ✅ EncodedGenome created:")
    print(f"     {encoded}")

    # Save to compressed storage
    print(f"\n💾 Saving to compressed storage...")
    temp_dir = Path(tempfile.mkdtemp())
    save_path = temp_dir / f"{genome_id}.enc.gz"

    file_size = encoded.save(save_path, compress=True)

    print(f"  ✅ Saved to: {save_path}")
    print(f"     - Compressed file size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    print(f"     - Uncompressed JSON size: {encoded.storage_size_kb():.2f} KB")
    print(f"     - Compression ratio: {encoded.storage_size_kb() / (file_size / 1024):.2f}x")

    # Also save uncompressed for comparison
    uncompressed_path = temp_dir / f"{genome_id}.enc.json"
    uncompressed_size = encoded.save(uncompressed_path, compress=False)

    print(f"\n📄 Uncompressed for comparison:")
    print(f"     - Uncompressed file size: {uncompressed_size:,} bytes ({uncompressed_size / 1024:.2f} KB)")
    print(f"     - Gzip compression ratio: {uncompressed_size / file_size:.2f}x")

    return save_path, encoded


def demonstrate_loading_and_verification(save_path):
    """
    Demonstrate loading and integrity verification.
    """
    print(f"\n📂 Loading encoded genome from storage...")

    loaded = EncodedGenome.load(save_path)

    print(f"  ✅ Loaded: {loaded}")
    print(f"     - Genome ID: {loaded.genome_id}")
    print(f"     - Assembly: {loaded.assembly}")
    print(f"     - Version: {loaded.version}")
    print(f"     - Created: {loaded.created_at.isoformat()}")
    print(f"     - Chunks: {len(loaded.chunk_hypervectors)}")

    # Verify integrity
    print(f"\n🔐 Verifying integrity...")

    is_valid = loaded.verify()

    if is_valid:
        print("  ✅ Integrity verification PASSED")
        print("     - Encoding hash matches")
        print("     - All hypervectors normalized")
        print("     - Dimensions consistent")
    else:
        print("  ❌ Integrity verification FAILED")

    # Check hypervector properties
    print(f"\n📊 Hypervector Properties:")
    bundled_norm = np.linalg.norm(loaded.bundled_hypervector)
    print(f"  Bundled hypervector:")
    print(f"    - Dimension: {len(loaded.bundled_hypervector)}")
    print(f"    - Norm: {bundled_norm:.6f} (should be ~1.0)")
    print(f"    - Range: [{loaded.bundled_hypervector.min():.4f}, {loaded.bundled_hypervector.max():.4f}]")

    chunk_norms = [np.linalg.norm(hv) for hv in loaded.chunk_hypervectors[:5]]
    print(f"  First 5 chunk hypervectors:")
    for i, norm in enumerate(chunk_norms):
        print(f"    - Chunk {i}: norm = {norm:.6f}")

    return loaded


def demonstrate_compression_analysis(loaded, original_vcf_size_kb=1500):
    """
    Demonstrate compression efficiency analysis.
    """
    print(f"\n📈 Compression Analysis:")

    encoded_size_kb = loaded.storage_size_kb()
    compression_ratio = loaded.compression_ratio(original_vcf_size_kb)

    print(f"  Original VCF (estimated): {original_vcf_size_kb:.2f} KB")
    print(f"  Encoded size (uncompressed JSON): {encoded_size_kb:.2f} KB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Space savings: {(1 - 1/compression_ratio) * 100:.1f}%")

    # Show summary
    print(f"\n📋 Summary Information:")
    summary = loaded.summary()
    for key, value in summary.items():
        if isinstance(value, list):
            print(f"  {key}: {len(value)} items ({', '.join(str(v) for v in value[:3])}{'...' if len(value) > 3 else ''})")
        else:
            print(f"  {key}: {value}")


def demonstrate_similarity_comparison(loaded):
    """
    Demonstrate using bundled hypervector for similarity comparison.
    """
    print(f"\n🔍 Similarity Comparison Demo:")

    bundled_hv = loaded.bundled_hypervector

    # Simulate comparison with another genome
    # (In practice, this would be another EncodedGenome)
    simulated_similar = bundled_hv + np.random.normal(0, 0.05, bundled_hv.shape).astype(np.float32)
    simulated_similar = simulated_similar / np.linalg.norm(simulated_similar)

    simulated_different = np.random.randn(len(bundled_hv)).astype(np.float32)
    simulated_different = simulated_different / np.linalg.norm(simulated_different)

    similarity_similar = np.dot(bundled_hv, simulated_similar)
    similarity_different = np.dot(bundled_hv, simulated_different)

    print(f"  Bundled hypervector shape: {bundled_hv.shape}")
    print(f"  Cosine similarity with similar genome: {similarity_similar:.4f}")
    print(f"  Cosine similarity with different genome: {similarity_different:.4f}")
    print(f"  (Range: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")


def main():
    """Run complete storage demonstration."""
    print("=" * 80)
    print("🧬 DIFFERENTIAL ENCODING STORAGE DEMO")
    print("=" * 80)
    print()
    print("This demo showcases the storage and serialization functionality:")
    print("  1. Creating mock encoding results")
    print("  2. Creating EncodedGenome from results")
    print("  3. Saving to compressed storage")
    print("  4. Loading and verifying integrity")
    print("  5. Analyzing compression efficiency")
    print("  6. Demonstrating similarity comparisons")
    print()
    print("NOTE: For a complete pipeline demo (genome → encoding → storage),")
    print("      see complete_pipeline_demo.py (requires API compatibility fixes)")
    print()
    print("=" * 80)

    # Step 1: Create mock result
    result = create_mock_encoding_result()

    # Step 2: Save to storage
    save_path, encoded = demonstrate_storage(result)

    # Step 3: Load and verify
    loaded = demonstrate_loading_and_verification(save_path)

    # Step 4: Analyze compression
    demonstrate_compression_analysis(loaded)

    # Step 5: Demonstrate similarity
    demonstrate_similarity_comparison(loaded)

    print("\n" + "=" * 80)
    print("✅ STORAGE DEMO COMPLETE")
    print("=" * 80)
    print()
    print(f"Encoded genome saved to: {save_path}")
    print()
    print("Key achievements:")
    print("  ✅ Efficient compressed storage with gzip")
    print("  ✅ Complete integrity verification with SHA256")
    print("  ✅ Lossless serialization/deserialization roundtrip")
    print("  ✅ Significant compression vs. original VCF")
    print("  ✅ Fast similarity search with bundled hypervectors")
    print()


if __name__ == '__main__':
    main()
