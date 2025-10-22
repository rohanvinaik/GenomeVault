"""
Complete Differential Encoding Pipeline Demo

This example demonstrates the complete end-to-end workflow:
1. Creating experimental and reference genomes with variants
2. Setting up the differential encoding pipeline
3. Encoding an experimental genome with cryptographic verification
4. Bundling chunk hypervectors into a genome-level representation
5. Saving the encoded genome to compressed storage
6. Loading and verifying the encoded genome
7. Analyzing compression ratios and storage efficiency

This showcases Sections 2-7.2 of the differential encoding specification.
"""

import tempfile
from pathlib import Path
import numpy as np

from genomevault.differential_encoding import (
    # Section 2: Cryptographic Primitives
    CryptoRNG,
    compute_reference_hash,

    # Section 3: Reference Management
    Variant,
    ReferenceGenome,
    SecureReferenceGenomeManager,

    # Section 4: Chunking
    AnalysisType,
    Genome,

    # Section 6.2: Hypervector Encoder
    DifferentialHypervectorEncoder,

    # Section 7.1: Pipeline
    DifferentialGenomicEncoder,

    # Section 7.2: Storage
    EncodedGenome,
)


def create_sample_genomes():
    """
    Create sample experimental and reference genomes for demonstration.

    Returns:
        tuple: (experimental_genome, reference_genomes)
    """
    print("📊 Creating sample genomic data...")

    # Create experimental genome with variants across multiple chromosomes
    experimental_variants = {
        'chr1': [
            Variant(chromosome='chr1', position=100000 + i*10000,
                   ref='A', alt='G', genotype='0/1', quality=99.0,
                   info={'IMPACT': 'HIGH' if i % 3 == 0 else 'MODERATE'})
            for i in range(20)
        ],
        'chr2': [
            Variant(chromosome='chr2', position=200000 + i*10000,
                   ref='C', alt='T', genotype='1/1', quality=98.0,
                   info={'Consequence': 'missense_variant'})
            for i in range(15)
        ],
    }

    experimental_genome = Genome(
        genome_id='patient_001',
        assembly='GRCh38',
        chromosomes=experimental_variants
    )

    # Create reference genomes for the pool
    reference_genomes = []

    for ref_id in range(1, 6):  # 5 reference genomes
        ref_variants = {
            'chr1': [
                Variant(chromosome='chr1', position=100000 + i*10000 + (ref_id * 100),
                       ref='A', alt='T', genotype='0/1', quality=95.0)
                for i in range(18)
            ],
            'chr2': [
                Variant(chromosome='chr2', position=200000 + i*10000 + (ref_id * 100),
                       ref='G', alt='A', genotype='0/1', quality=94.0)
                for i in range(12)
            ],
        }

        # Create reference with proper hash
        temp_ref = ReferenceGenome(
            genome_id=f'reference_{ref_id:03d}',
            assembly='GRCh38',
            variants=ref_variants,
            cryptographic_hash='temp'
        )

        actual_hash = compute_reference_hash(temp_ref)

        ref_genome = ReferenceGenome(
            genome_id=f'reference_{ref_id:03d}',
            assembly='GRCh38',
            variants=ref_variants,
            cryptographic_hash=actual_hash
        )

        reference_genomes.append(ref_genome)

    print(f"  ✅ Created experimental genome: {experimental_genome.genome_id}")
    print(f"     - {sum(len(v) for v in experimental_genome.chromosomes.values())} variants across {len(experimental_genome.chromosomes)} chromosomes")
    print(f"  ✅ Created {len(reference_genomes)} reference genomes")

    return experimental_genome, reference_genomes


def setup_pipeline(reference_genomes):
    """
    Set up the differential encoding pipeline components.

    Args:
        reference_genomes: List of ReferenceGenome objects

    Returns:
        DifferentialGenomicEncoder: Configured pipeline encoder
    """
    print("\n🔧 Setting up differential encoding pipeline...")

    # 1. Create reference manager with reference pool
    temp_dir = Path(tempfile.mkdtemp())
    reference_manager = SecureReferenceGenomeManager(
        reference_dir=temp_dir
    )

    # Add reference genomes to the pool
    for ref_genome in reference_genomes:
        reference_manager.pool.add_reference(ref_genome)

    # 2. Create hypervector encoder (10,000D)
    hypervector_encoder = DifferentialHypervectorEncoder(
        dimension=10000,
        seed=42
    )

    # 3. Create cryptographic RNG for reproducibility
    crypto_rng = CryptoRNG()

    # 4. Create pipeline encoder
    pipeline = DifferentialGenomicEncoder(
        reference_manager=reference_manager,
        hypervector_encoder=hypervector_encoder,
        crypto_rng=crypto_rng
    )

    print(f"  ✅ Reference manager: {reference_manager.reference_count} genomes")
    print(f"  ✅ Hypervector encoder: {hypervector_encoder.dimension}D")
    print(f"  ✅ Pipeline ready")

    return pipeline


def encode_genome(pipeline, experimental_genome, analysis_type):
    """
    Encode an experimental genome using the pipeline.

    Args:
        pipeline: DifferentialGenomicEncoder instance
        experimental_genome: Genome to encode
        analysis_type: Type of analysis (e.g., AnalysisType.SLIDING_WINDOW)

    Returns:
        EncodingResult: Encoding result with hypervectors and metadata
    """
    print(f"\n🧬 Encoding genome: {experimental_genome.genome_id}")
    print(f"   Analysis type: {analysis_type.value}")

    # Create deterministic master seed for reproducibility
    master_seed = b"demo_master_seed" + b"_" * 16  # 32 bytes

    # Progress callback for monitoring
    def progress_callback(current, total, chunk):
        if current % 10 == 0 or current == total - 1:
            print(f"  📦 Processing chunk {current + 1}/{total}: {chunk.chromosome}:{chunk.start_position}-{chunk.end_position}")

    # Encode the genome
    result = pipeline.encode_experimental_genome(
        experimental_genome=experimental_genome,
        analysis_type=analysis_type,
        master_seed=master_seed,
        bundle_chunks=True,
        progress_callback=progress_callback
    )

    print(f"\n  ✅ Encoding complete:")
    print(f"     - {len(result.hypervectors)} chunk hypervectors")
    print(f"     - {len(result.metadata)} metadata entries")
    print(f"     - Bundled hypervector: {result.bundled_hypervector.shape if result.bundled_hypervector is not None else 'None'}")
    print(f"     - Total differences: {result.statistics.get('total_differences', 0)}")
    print(f"       • New mutations: {result.statistics.get('new_mutations', 0)}")
    print(f"       • Missing variants: {result.statistics.get('missing_variants', 0)}")
    print(f"       • Genotype differences: {result.statistics.get('genotype_differences', 0)}")

    return result, master_seed


def save_and_load(experimental_genome, result, master_seed):
    """
    Save encoded genome to storage and load it back.

    Args:
        experimental_genome: Original genome
        result: EncodingResult from pipeline
        master_seed: Master seed used for encoding

    Returns:
        tuple: (saved_path, loaded_genome, file_size_bytes)
    """
    print("\n💾 Saving encoded genome to storage...")

    # Create EncodedGenome from result
    encoded = EncodedGenome.from_encoding_result(
        genome_id=experimental_genome.genome_id,
        assembly=experimental_genome.assembly,
        result=result,
        master_seed=master_seed
    )

    print(f"  📊 EncodedGenome created: {encoded}")

    # Save to temporary file with compression
    temp_dir = Path(tempfile.mkdtemp())
    save_path = temp_dir / f"{experimental_genome.genome_id}.enc.gz"

    file_size = encoded.save(save_path, compress=True)

    print(f"  ✅ Saved to: {save_path}")
    print(f"     - File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    print(f"     - Uncompressed JSON size: {encoded.storage_size_kb():.2f} KB")

    # Load it back
    print("\n📂 Loading encoded genome from storage...")
    loaded = EncodedGenome.load(save_path)

    print(f"  ✅ Loaded: {loaded}")
    print(f"     - Genome ID: {loaded.genome_id}")
    print(f"     - Assembly: {loaded.assembly}")
    print(f"     - Chunks: {len(loaded.chunk_hypervectors)}")
    print(f"     - Created: {loaded.created_at.isoformat()}")

    return save_path, loaded, file_size


def verify_and_analyze(loaded, original_vcf_size_kb=1500):
    """
    Verify integrity and analyze compression efficiency.

    Args:
        loaded: Loaded EncodedGenome
        original_vcf_size_kb: Original VCF file size in KB (for comparison)
    """
    print("\n🔐 Verifying encoding integrity...")

    is_valid = loaded.verify()

    if is_valid:
        print("  ✅ Integrity verification PASSED")
        print("     - Encoding hash matches")
        print("     - All hypervectors normalized")
        print("     - Metadata validated")
    else:
        print("  ❌ Integrity verification FAILED")
        return

    # Analyze compression
    print(f"\n📈 Compression Analysis:")

    encoded_size_kb = loaded.storage_size_kb()
    compression_ratio = loaded.compression_ratio(original_vcf_size_kb)

    print(f"  Original VCF (estimated): {original_vcf_size_kb:.2f} KB")
    print(f"  Encoded size (uncompressed): {encoded_size_kb:.2f} KB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")
    print(f"  Space savings: {(1 - 1/compression_ratio) * 100:.1f}%")

    # Summary information
    print(f"\n📋 Summary:")
    summary = loaded.summary()
    for key, value in summary.items():
        if key == 'encoding_hash':
            print(f"  {key}: {value}")
        elif isinstance(value, list):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")


def demonstrate_similarity_search(pipeline, loaded):
    """
    Demonstrate similarity search using bundled hypervector.

    Args:
        pipeline: DifferentialGenomicEncoder instance
        loaded: Loaded EncodedGenome
    """
    print("\n🔍 Demonstrating similarity search...")

    # The bundled hypervector can be used for similarity search
    bundled_hv = loaded.bundled_hypervector

    print(f"  Bundled hypervector shape: {bundled_hv.shape}")
    print(f"  Bundled hypervector norm: {np.linalg.norm(bundled_hv):.6f}")

    # Simulate similarity comparison with another genome
    # (In practice, this would be another encoded genome)
    simulated_other = bundled_hv + np.random.normal(0, 0.1, bundled_hv.shape)
    simulated_other = simulated_other / np.linalg.norm(simulated_other)

    similarity = np.dot(bundled_hv, simulated_other)

    print(f"\n  Cosine similarity with simulated genome: {similarity:.4f}")
    print(f"  (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")

    # Show chunk-level similarities
    print(f"\n  📊 Chunk-level similarity distribution:")
    chunk_similarities = []
    for i, chunk_hv in enumerate(loaded.chunk_hypervectors[:5]):  # First 5 chunks
        sim_other = chunk_hv + np.random.normal(0, 0.1, chunk_hv.shape)
        sim_other = sim_other / np.linalg.norm(sim_other)
        sim = np.dot(chunk_hv, sim_other)
        chunk_similarities.append(sim)
        print(f"     Chunk {i+1}: {sim:.4f}")


def main():
    """Run complete pipeline demonstration."""
    print("=" * 80)
    print("🧬 COMPLETE DIFFERENTIAL ENCODING PIPELINE DEMO")
    print("=" * 80)
    print()
    print("This demo showcases the complete workflow:")
    print("  1. Creating sample genomic data")
    print("  2. Setting up the encoding pipeline")
    print("  3. Encoding an experimental genome")
    print("  4. Bundling chunk hypervectors")
    print("  5. Saving to compressed storage")
    print("  6. Loading and verifying integrity")
    print("  7. Analyzing compression efficiency")
    print("  8. Demonstrating similarity search")
    print()
    print("=" * 80)

    # Step 1: Create sample data
    experimental_genome, reference_genomes = create_sample_genomes()

    # Step 2: Setup pipeline
    pipeline = setup_pipeline(reference_genomes)

    # Step 3: Encode genome
    result, master_seed = encode_genome(
        pipeline,
        experimental_genome,
        AnalysisType.SLIDING_WINDOW
    )

    # Step 4: Save and load
    save_path, loaded, file_size = save_and_load(
        experimental_genome,
        result,
        master_seed
    )

    # Step 5: Verify and analyze
    verify_and_analyze(loaded, original_vcf_size_kb=1500)

    # Step 6: Demonstrate similarity search
    demonstrate_similarity_search(pipeline, loaded)

    print("\n" + "=" * 80)
    print("✅ DEMO COMPLETE")
    print("=" * 80)
    print()
    print(f"Encoded genome saved to: {save_path}")
    print(f"File size: {file_size:,} bytes")
    print()
    print("Key achievements:")
    print("  ✅ Cryptographically secure differential encoding")
    print("  ✅ Hyperdimensional vector representation (10,000D)")
    print("  ✅ Efficient compressed storage with integrity verification")
    print("  ✅ Privacy-preserving similarity search capability")
    print("  ✅ Complete audit trail with metadata")
    print()


if __name__ == '__main__':
    main()
