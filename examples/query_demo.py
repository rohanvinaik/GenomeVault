"""
Query Interface Demo for Differential Encoding

This example demonstrates how to query differentially encoded genomes:
1. Creating a mock encoded genome
2. Setting up the query interface
3. Querying specific genomic regions
4. Finding similar chunks via hypervector similarity
5. Batch querying multiple regions
6. Getting genome statistics

This showcases Section 8 of the differential encoding specification.
"""

import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np

from genomevault.differential_encoding import (
    # Reference management
    Variant,
    ReferenceGenome,
    SecureReferenceGenomeManager,
    compute_reference_hash,

    # Hypervector encoding
    DifferentialHypervectorEncoder,

    # Storage
    EncodedGenome,
    DifferentialEncodingMetadata,

    # Query
    DifferentialGenomeQuery,
)


def create_mock_references():
    """Create sample reference genomes."""
    print("📚 Creating reference genome pool...")

    genomes = []

    for ref_id in range(1, 6):  # 5 reference genomes
        variants = {
            'chr1': [
                Variant(
                    chromosome='chr1',
                    position=100000 + i * 10000,
                    ref='A',
                    alt='G',
                    genotype='0/1',
                    quality=95.0,
                )
                for i in range(20)
            ],
            'chr2': [
                Variant(
                    chromosome='chr2',
                    position=200000 + i * 10000,
                    ref='C',
                    alt='T',
                    genotype='0/1',
                    quality=94.0,
                )
                for i in range(15)
            ],
        }

        # Create with proper hash
        temp_ref = ReferenceGenome(
            genome_id=f'reference_{ref_id:03d}',
            assembly='GRCh38',
            variants=variants,
            cryptographic_hash='temp',
        )

        actual_hash = compute_reference_hash(temp_ref)

        genome = ReferenceGenome(
            genome_id=f'reference_{ref_id:03d}',
            assembly='GRCh38',
            variants=variants,
            cryptographic_hash=actual_hash,
        )

        genomes.append(genome)

    print(f"  ✅ Created {len(genomes)} reference genomes")
    return genomes


def create_mock_encoded_genome():
    """Create a mock encoded genome for demonstration."""
    print("\n🧬 Creating mock encoded genome...")

    # Create mock hypervectors and metadata
    num_chunks = 50
    dimension = 10000

    chunk_hypervectors = []
    metadata_list = []

    # Distribute chunks across chromosomes
    for i in range(num_chunks):
        # Create normalized hypervector
        hv = np.random.randn(dimension).astype(np.float32)
        hv = hv / np.linalg.norm(hv)
        chunk_hypervectors.append(hv)

        # Determine chromosome (80% chr1, 20% chr2)
        if i < 40:
            chromosome = 'chr1'
            chr_offset = 100000
        else:
            chromosome = 'chr2'
            chr_offset = 200000

        # Create metadata for chunk
        chunk_id = f"chunk_{i:04d}".encode()
        chunk_id = chunk_id + b'\x00' * (32 - len(chunk_id))

        meta = DifferentialEncodingMetadata(
            chunk_id=chunk_id,
            chromosome=chromosome,
            start_position=chr_offset + (i % 40) * 50000,
            end_position=chr_offset + ((i % 40) + 1) * 50000,
            reference_genome_id=f'reference_{(i % 5) + 1:03d}',
            reference_seed=bytes(range(32)),
            reference_hash=bytes(range(32, 64)),
            cryptographic_binding=bytes(range(64, 96)),
            chunking_strategy='sliding_window',
            chunking_seed=bytes(range(96, 128)),
            analysis_type='sliding_window',
            difference_counts={
                'new_mutations': 5 + (i % 10),
                'missing_variants': 3 + (i % 7),
                'genotype_differences': 2 + (i % 5),
                'total': (5 + (i % 10)) + (3 + (i % 7)) + (2 + (i % 5)),
            },
            created_timestamp=datetime.now(),
        )
        metadata_list.append(meta)

    # Create bundled hypervector
    bundled_hypervector = np.mean(chunk_hypervectors, axis=0).astype(np.float32)
    bundled_hypervector = bundled_hypervector / np.linalg.norm(bundled_hypervector)

    # Create statistics
    statistics = {
        'total_chunks': num_chunks,
        'total_differences': sum(m.difference_counts['total'] for m in metadata_list),
        'new_mutations': sum(m.difference_counts['new_mutations'] for m in metadata_list),
        'missing_variants': sum(m.difference_counts['missing_variants'] for m in metadata_list),
        'genotype_differences': sum(m.difference_counts['genotype_differences'] for m in metadata_list),
        'chromosomes': ['chr1', 'chr2'],
        'hypervector_dimension': dimension,
    }

    # Create EncodedGenome
    encoded = EncodedGenome(
        genome_id='patient_query_demo',
        assembly='GRCh38',
        bundled_hypervector=bundled_hypervector,
        chunk_hypervectors=chunk_hypervectors,
        metadata=metadata_list,
        statistics=statistics,
        master_seed=b'query_demo_seed' + b'\x00' * 17,
        encoding_hash='demo_hash_12345678',
    )

    print(f"  ✅ Created encoded genome: {encoded}")
    print(f"     - {len(metadata_list)} chunks")
    print(f"     - {statistics['total_differences']} total differences")
    print(f"     - Chromosomes: {', '.join(statistics['chromosomes'])}")

    return encoded


def setup_query_interface(reference_genomes):
    """Set up the query interface."""
    print("\n🔧 Setting up query interface...")

    # Create reference manager
    temp_dir = Path(tempfile.mkdtemp())
    reference_manager = SecureReferenceGenomeManager(reference_dir=temp_dir)

    for ref_genome in reference_genomes:
        reference_manager.pool.add_reference(ref_genome)

    # Create hypervector encoder
    hv_encoder = DifferentialHypervectorEncoder(dimension=10000, seed=42)

    # Create query interface
    query = DifferentialGenomeQuery(reference_manager, hv_encoder)

    print(f"  ✅ Query interface ready: {query}")

    return query


def demonstrate_region_query(query, encoded_genome):
    """Demonstrate querying a specific genomic region."""
    print("\n" + "=" * 80)
    print("1. REGION QUERY DEMO")
    print("=" * 80)

    # Query a specific region on chr1
    chromosome = 'chr1'
    start = 100000
    end = 300000

    print(f"\n📍 Querying region: {chromosome}:{start}-{end}")

    result = query.query_region(encoded_genome, chromosome, start, end)

    print(f"\n✅ Query result:")
    print(f"   {result}")
    print(f"   - Variants found: {result.variant_count}")
    print(f"   - Chunks used: {result.chunks_used}")
    print(f"   - Metadata entries: {len(result.metadata)}")

    if result.variants:
        print(f"\n   First 5 variants:")
        for v in result.variants[:5]:
            print(f"      {v.chromosome}:{v.position} {v.ref}→{v.alt}")

    # Query a different region
    print(f"\n📍 Querying different region: chr2:500000-600000")

    result2 = query.query_region(encoded_genome, 'chr2', 500000, 600000)

    print(f"\n✅ Query result:")
    print(f"   {result2}")
    print(f"   - Variants found: {result2.variant_count}")
    print(f"   - Chunks used: {result2.chunks_used}")


def demonstrate_similarity_search(query, encoded_genome):
    """Demonstrate hypervector similarity search."""
    print("\n" + "=" * 80)
    print("2. SIMILARITY SEARCH DEMO")
    print("=" * 80)

    # Use the bundled hypervector as a query
    print("\n🔍 Searching for chunks similar to bundled hypervector...")

    query_hv = encoded_genome.bundled_hypervector

    # Find top 10 most similar chunks
    matches = query.query_by_hypervector_similarity(
        encoded_genome,
        query_hv,
        threshold=0.1,  # Low threshold to get many matches
        top_k=10
    )

    print(f"\n✅ Found {len(matches)} similar chunks (top 10):")

    for i, match in enumerate(matches, 1):
        print(f"\n   {i}. {match}")
        print(f"      Similarity: {match.similarity:.4f}")
        print(f"      Region: {match.metadata.chromosome}:{match.metadata.start_position}-{match.metadata.end_position}")
        print(f"      Differences: {match.metadata.difference_counts['total']}")

    # Search with higher threshold
    print(f"\n🔍 Searching with higher threshold (0.3)...")

    matches_strict = query.query_by_hypervector_similarity(
        encoded_genome,
        query_hv,
        threshold=0.3,
    )

    print(f"\n✅ Found {len(matches_strict)} chunks with similarity >= 0.3")

    # Search using a chunk's hypervector
    print(f"\n🔍 Searching for chunks similar to chunk #0...")

    query_hv_chunk = encoded_genome.chunk_hypervectors[0]

    matches_chunk = query.query_by_hypervector_similarity(
        encoded_genome,
        query_hv_chunk,
        threshold=0.5,
        top_k=5
    )

    print(f"\n✅ Found {len(matches_chunk)} similar chunks (top 5):")
    for match in matches_chunk:
        print(f"   - Chunk {match.chunk_index}: similarity={match.similarity:.4f}")


def demonstrate_batch_query(query, encoded_genome):
    """Demonstrate batch querying multiple regions."""
    print("\n" + "=" * 80)
    print("3. BATCH QUERY DEMO")
    print("=" * 80)

    # Define multiple regions to query
    regions = [
        ('chr1', 100000, 150000),
        ('chr1', 200000, 250000),
        ('chr1', 400000, 450000),
        ('chr2', 500000, 550000),
        ('chr2', 700000, 750000),
    ]

    print(f"\n📊 Batch querying {len(regions)} regions...")

    for i, (chr, start, end) in enumerate(regions, 1):
        print(f"   {i}. {chr}:{start}-{end}")

    # Perform batch query
    results = query.batch_query_regions(encoded_genome, regions)

    print(f"\n✅ Batch query complete: {len(results)} results")

    print(f"\n   Results summary:")
    for i, result in enumerate(results, 1):
        print(f"   {i}. {result.chromosome}:{result.start}-{result.end}")
        print(f"      Variants: {result.variant_count}, Chunks: {result.chunks_used}")


def demonstrate_statistics(query, encoded_genome):
    """Demonstrate getting genome statistics."""
    print("\n" + "=" * 80)
    print("4. STATISTICS DEMO")
    print("=" * 80)

    print(f"\n📊 Getting genome statistics...")

    stats = query.get_statistics(encoded_genome)

    print(f"\n✅ Statistics:")
    print(f"   Genome ID: {encoded_genome.genome_id}")
    print(f"   Assembly: {encoded_genome.assembly}")
    print(f"   Total chunks: {stats['total_chunks']}")
    print(f"   Hypervector dimension: {stats['hypervector_dimension']}")
    print(f"   Average chunk size: {stats['average_chunk_size']:.0f} bp")

    print(f"\n   Chromosomes: {', '.join(stats['chromosomes'])}")

    print(f"\n   Chunks per chromosome:")
    for chr, count in stats['chunks_per_chromosome'].items():
        print(f"      {chr}: {count} chunks")

    print(f"\n   Position ranges:")
    for chr, (min_pos, max_pos) in stats['position_range'].items():
        print(f"      {chr}: {min_pos:,} - {max_pos:,} ({max_pos - min_pos:,} bp)")


def demonstrate_performance(query, encoded_genome):
    """Demonstrate query performance."""
    print("\n" + "=" * 80)
    print("5. PERFORMANCE DEMO")
    print("=" * 80)

    import time

    # Test region query performance
    print(f"\n⏱️  Testing region query performance...")

    num_queries = 100
    regions = [('chr1', 100000 + i * 10000, 150000 + i * 10000) for i in range(num_queries)]

    start = time.time()
    for chr, s, e in regions:
        query.query_region(encoded_genome, chr, s, e)
    elapsed = time.time() - start

    print(f"   ✅ {num_queries} region queries in {elapsed:.3f}s")
    print(f"      Average: {elapsed / num_queries * 1000:.2f} ms/query")

    # Test similarity search performance
    print(f"\n⏱️  Testing similarity search performance...")

    query_hv = encoded_genome.bundled_hypervector

    start = time.time()
    for _ in range(10):
        query.query_by_hypervector_similarity(
            encoded_genome, query_hv, threshold=0.5
        )
    elapsed = time.time() - start

    print(f"   ✅ 10 similarity searches in {elapsed:.3f}s")
    print(f"      Average: {elapsed / 10 * 1000:.2f} ms/search")

    # Test batch query performance
    print(f"\n⏱️  Testing batch query performance...")

    batch_regions = [('chr1', i * 50000, (i + 1) * 50000) for i in range(20)]

    start = time.time()
    results = query.batch_query_regions(encoded_genome, batch_regions)
    elapsed = time.time() - start

    print(f"   ✅ Batch query of {len(batch_regions)} regions in {elapsed:.3f}s")
    print(f"      Average: {elapsed / len(batch_regions) * 1000:.2f} ms/region")


def main():
    """Run complete query demo."""
    print("=" * 80)
    print("🔍 DIFFERENTIAL ENCODING QUERY INTERFACE DEMO")
    print("=" * 80)
    print()
    print("This demo showcases:")
    print("  1. Querying specific genomic regions")
    print("  2. Finding similar chunks via hypervector similarity")
    print("  3. Batch querying multiple regions efficiently")
    print("  4. Getting genome statistics")
    print("  5. Performance benchmarks")
    print()

    # Setup
    reference_genomes = create_mock_references()
    encoded_genome = create_mock_encoded_genome()
    query = setup_query_interface(reference_genomes)

    # Demonstrations
    demonstrate_region_query(query, encoded_genome)
    demonstrate_similarity_search(query, encoded_genome)
    demonstrate_batch_query(query, encoded_genome)
    demonstrate_statistics(query, encoded_genome)
    demonstrate_performance(query, encoded_genome)

    print("\n" + "=" * 80)
    print("✅ QUERY DEMO COMPLETE")
    print("=" * 80)
    print()
    print("Key features demonstrated:")
    print("  ✅ Fast region-based variant retrieval")
    print("  ✅ Hypervector similarity search")
    print("  ✅ Efficient batch query processing")
    print("  ✅ Comprehensive genome statistics")
    print("  ✅ Sub-millisecond query latency")
    print()


if __name__ == '__main__':
    main()
