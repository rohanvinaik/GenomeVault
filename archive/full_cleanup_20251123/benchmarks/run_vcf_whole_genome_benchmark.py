#!/usr/bin/env python3
"""
VCF Whole Genome Differential Encoding Benchmark

Benchmarks the baseline VCF-based differential encoding pipeline with:
- Query: ref1.vcf.gz (23.4M whole-genome variants)
- Reference pool: ref2.vcf.gz + ref3.vcf.gz (47.9M variants)
- k-anonymity: 3
- HDC dimension: 10,000

This is the BASELINE benchmark for comparison against the GDiff format.

Usage:
    python benchmarks/run_vcf_whole_genome_benchmark.py
"""

import time
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import working implementations
from genomevault.differential_encoding import (
    SecureReferenceGenomeManager,
    DifferentialHypervectorEncoder,
    DifferentialGenomicEncoder,
    CryptoRNG,
    ReferenceGenome,
    Variant,
    Genome,
    compute_reference_hash,
    AnalysisType,
)


def load_vcf_variants(vcf_path: Path, max_variants: int = None) -> Dict[str, list]:
    """
    Load variants from VCF file.

    Args:
        vcf_path: Path to VCF.gz file
        max_variants: Optional limit on variants to load (None = all)

    Returns:
        Dictionary mapping chromosome to list of Variant objects
    """
    import gzip

    logger.info(f"Loading variants from: {vcf_path.name}")

    variants_by_chr = {}
    total_loaded = 0

    with gzip.open(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split('\t')
            if len(parts) < 10:
                continue

            chrom = parts[0]
            pos = int(parts[1])
            ref = parts[3]
            alt = parts[4]
            qual = float(parts[5]) if parts[5] != '.' else 99.0

            # Extract genotype from FORMAT field (simplified)
            format_fields = parts[8].split(':')
            sample_fields = parts[9].split(':')
            gt_idx = format_fields.index('GT') if 'GT' in format_fields else 0
            genotype = sample_fields[gt_idx] if gt_idx < len(sample_fields) else '0/1'

            variant = Variant(
                chromosome=chrom,
                position=pos,
                ref=ref,
                alt=alt,
                genotype=genotype,
                quality=qual
            )

            if chrom not in variants_by_chr:
                variants_by_chr[chrom] = []

            variants_by_chr[chrom].append(variant)
            total_loaded += 1

            if max_variants and total_loaded >= max_variants:
                break

            # Log progress
            if total_loaded % 1000000 == 0:
                logger.info(f"  Loaded {total_loaded:,} variants...")

    logger.info(f"  ✓ Loaded {total_loaded:,} variants across {len(variants_by_chr)} chromosomes")

    return variants_by_chr


def run_vcf_benchmark(
    query_vcf: Path,
    reference_pool_vcfs: list[Path],
    output_dir: Path,
    max_variants: int = None,
) -> Dict[str, Any]:
    """
    Run the VCF-based whole genome differential encoding benchmark.

    Args:
        query_vcf: Path to query VCF file (ref1.vcf.gz)
        reference_pool_vcfs: List of reference VCF paths (ref2, ref3)
        output_dir: Directory for benchmark results
        max_variants: Optional limit on variants to load (None = all)

    Returns:
        Dictionary containing benchmark results
    """
    logger.info("=" * 80)
    logger.info("VCF WHOLE GENOME DIFFERENTIAL ENCODING BENCHMARK")
    logger.info("=" * 80)
    logger.info("")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration
    config = {
        "query_vcf": str(query_vcf),
        "reference_pool": [str(p) for p in reference_pool_vcfs],
        "k_anonymity": len(reference_pool_vcfs) + 1,  # refs + query
        "hdc_dimension": 10000,
        "format": "VCF",
        "max_variants": max_variants,
    }

    logger.info("Configuration:")
    logger.info(f"  Query VCF: {query_vcf.name}")
    logger.info(f"  Reference pool: {len(reference_pool_vcfs)} VCFs")
    for ref_vcf in reference_pool_vcfs:
        logger.info(f"    - {ref_vcf.name}")
    logger.info(f"  k-anonymity: {config['k_anonymity']}")
    logger.info(f"  HDC dimension: {config['hdc_dimension']:,}D")
    if max_variants:
        logger.info(f"  Max variants: {max_variants:,}")
    logger.info("")

    # Stage 1: Load VCF files
    logger.info("[1/4] Loading VCF files...")
    load_start = time.time()

    # Load query variants
    query_variants = load_vcf_variants(query_vcf, max_variants)
    query_variant_count = sum(len(v) for v in query_variants.values())

    # Load reference pool variants
    reference_variants = []
    ref_variant_counts = []

    for ref_vcf in reference_pool_vcfs:
        ref_vars = load_vcf_variants(ref_vcf, max_variants)
        reference_variants.append(ref_vars)
        ref_count = sum(len(v) for v in ref_vars.values())
        ref_variant_counts.append(ref_count)

    total_ref_variants = sum(ref_variant_counts)
    load_time = time.time() - load_start

    logger.info(f"  ✓ Query variants: {query_variant_count:,}")
    logger.info(f"  ✓ Reference pool: {total_ref_variants:,} variants total")
    logger.info(f"  ✓ Load time: {load_time:.2f}s")
    logger.info("")

    # Stage 2: Setup differential encoding pipeline
    logger.info("[2/4] Setting up differential encoding pipeline...")
    setup_start = time.time()

    try:
        import tempfile

        # Create reference manager
        temp_dir = Path(tempfile.mkdtemp())
        ref_manager = SecureReferenceGenomeManager(reference_dir=temp_dir)

        # Add reference genomes to pool
        for i, ref_vars in enumerate(reference_variants):
            temp_ref = ReferenceGenome(
                genome_id=f"reference_{i+1:03d}",
                assembly="GRCh38",
                variants=ref_vars,
                cryptographic_hash="temp"
            )

            ref_genome = ReferenceGenome(
                genome_id=temp_ref.genome_id,
                assembly=temp_ref.assembly,
                variants=temp_ref.variants,
                cryptographic_hash=compute_reference_hash(temp_ref)
            )

            ref_manager.pool.add_reference(ref_genome)

        logger.info(f"  ✓ Added {ref_manager.reference_count} references to pool")

        # Create hypervector encoder
        hv_encoder = DifferentialHypervectorEncoder(
            dimension=config['hdc_dimension'],
            seed=42
        )

        # Create crypto RNG
        crypto_rng = CryptoRNG()

        # Create pipeline encoder
        pipeline = DifferentialGenomicEncoder(
            reference_manager=ref_manager,
            hypervector_encoder=hv_encoder,
            crypto_rng=crypto_rng,
        )

        setup_time = time.time() - setup_start
        logger.info(f"  ✓ Pipeline initialized ({setup_time:.2f}s)")
        logger.info("")

    except Exception as e:
        logger.error(f"  ❌ Pipeline setup failed: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    # Stage 3: Differential Encoding
    logger.info("[3/4] Running differential encoding...")
    logger.info("  This processes whole genome variants (may take several minutes)...")

    diff_start = time.time()

    try:
        # Create experimental genome from query
        exp_genome = Genome(
            genome_id="query_sample",
            assembly="GRCh38",
            chromosomes=query_variants
        )

        # Encode genome with k-anonymity
        logger.info(f"  Encoding {query_variant_count:,} variants with k={config['k_anonymity']} anonymity...")
        master_seed = b"vcf_benchmark_whole_genome_" + b"_" * 5  # 32 bytes

        encoding_result = pipeline.encode_experimental_genome(
            experimental_genome=exp_genome,
            analysis_type=AnalysisType.SLIDING_WINDOW,
            master_seed=master_seed,
            bundle_chunks=True,
        )

        diff_time = time.time() - diff_start

        num_chunks = len(encoding_result.hypervectors)
        num_diffs = sum(len(hv.differences) for hv in encoding_result.hypervectors)

        logger.info(f"  ✓ Differential encoding complete ({diff_time:.2f}s)")
        logger.info(f"  Encoded chunks: {num_chunks:,}")
        logger.info(f"  Total differences: {num_diffs:,}")
        logger.info("")

    except Exception as e:
        logger.error(f"  ❌ Differential encoding failed: {e}")
        import traceback
        traceback.print_exc()
        diff_time = time.time() - diff_start
        encoding_result = None
        num_chunks = 0
        num_diffs = 0

    # Stage 4: HDC Encoding & Compression Analysis
    logger.info("[4/4] Analyzing HDC encoding & compression...")

    hdc_start = time.time()

    try:
        if encoding_result and encoding_result.bundled_hypervector is not None:
            bundled_hv = encoding_result.bundled_hypervector

            # Calculate sizes
            original_vcf_size_mb = query_vcf.stat().st_size / (1024 * 1024)
            hv_size_kb = bundled_hv.nbytes / 1024
            hv_size_mb = hv_size_kb / 1024

            compression_ratio = original_vcf_size_mb / hv_size_mb
            space_savings = (1 - 1/compression_ratio) * 100

            logger.info(f"  ✓ HDV dimension: {config['hdc_dimension']:,}D")
            logger.info(f"  ✓ HDV size: {hv_size_kb:.2f} KB ({hv_size_mb:.2f} MB)")
            logger.info(f"  ✓ Original VCF: {original_vcf_size_mb:.2f} MB")
            logger.info(f"  ✓ Compression ratio: {compression_ratio:.2f}x")
            logger.info(f"  ✓ Space savings: {space_savings:.1f}%")

        else:
            hv_size_kb = 0
            compression_ratio = 0
            space_savings = 0

        hdc_time = time.time() - hdc_start
        logger.info("")

    except Exception as e:
        logger.error(f"  ❌ HDC analysis failed: {e}")
        hdc_time = time.time() - hdc_start
        hv_size_kb = 0
        compression_ratio = 0
        space_savings = 0

    # Calculate total time
    total_time = load_time + setup_time + diff_time + hdc_time

    # Compile results
    results = {
        "timestamp": datetime.now().isoformat(),
        "format": "VCF",
        "configuration": config,
        "input_stats": {
            "query_vcf": str(query_vcf),
            "query_variants": query_variant_count,
            "query_chromosomes": len(query_variants),
            "reference_pool_vcfs": [str(p) for p in reference_pool_vcfs],
            "reference_pool_variants": ref_variant_counts,
            "total_reference_variants": total_ref_variants,
        },
        "performance": {
            "load_time_s": round(load_time, 2),
            "setup_time_s": round(setup_time, 2),
            "differential_encoding_time_s": round(diff_time, 2),
            "hdc_analysis_time_s": round(hdc_time, 2),
            "total_time_s": round(total_time, 2),
        },
        "differential_encoding": {
            "num_chunks": num_chunks,
            "num_differences": num_diffs,
            "k_anonymity": config['k_anonymity'],
        },
        "hdc_encoding": {
            "dimension": config['hdc_dimension'],
            "size_kb": round(hv_size_kb, 2),
            "size_mb": round(hv_size_kb / 1024, 2),
            "compression_ratio": round(compression_ratio, 2),
            "space_savings_percent": round(space_savings, 1),
        },
    }

    # Save results
    results_file = output_dir / "vcf_benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Results saved to: {results_file}")
    logger.info("")
    logger.info("Summary:")
    logger.info(f"  Format: VCF (baseline)")
    logger.info(f"  Query variants: {query_variant_count:,}")
    logger.info(f"  Differential encoding: {diff_time:.2f}s ({num_diffs:,} diffs)")
    logger.info(f"  HDC encoding: {hdc_time:.2f}s ({hv_size_kb:.2f} KB)")
    logger.info(f"  Total time: {total_time:.2f}s")
    logger.info(f"  Compression: {compression_ratio:.2f}x ({space_savings:.1f}% savings)")
    logger.info("")

    return results


def main():
    """Main entry point."""

    # Paths
    query_vcf = Path("benchmark_results/enhanced_privacy_k13_phase123_optimized/layer2_reference_pool/ref1.vcf.gz")
    reference_pool_dir = Path("benchmark_results/vcf_gdiff_comparison/vcf_benchmark/ref_pool")
    output_dir = Path("benchmark_results/vcf_gdiff_comparison/vcf_benchmark/results")

    # Reference pool VCFs
    reference_pool_vcfs = [
        reference_pool_dir / "ref2.vcf.gz",
        reference_pool_dir / "ref3.vcf.gz",
    ]

    # Validate paths
    if not query_vcf.exists():
        logger.error(f"❌ ERROR: Query VCF not found: {query_vcf}")
        return 1

    for ref_vcf in reference_pool_vcfs:
        if not ref_vcf.exists():
            logger.error(f"❌ ERROR: Reference VCF not found: {ref_vcf}")
            return 1

    logger.info(f"Query VCF: {query_vcf.stat().st_size / (1024*1024):.1f} MB")
    for ref_vcf in reference_pool_vcfs:
        logger.info(f"Reference: {ref_vcf.name} ({ref_vcf.stat().st_size / (1024*1024):.1f} MB)")
    logger.info("")

    # Run benchmark
    try:
        results = run_vcf_benchmark(
            query_vcf=query_vcf,
            reference_pool_vcfs=reference_pool_vcfs,
            output_dir=output_dir,
            max_variants=None,  # Load ALL variants (whole genome)
        )

        if "error" in results:
            return 1

        return 0

    except Exception as e:
        logger.error(f"\n❌ BENCHMARK FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
