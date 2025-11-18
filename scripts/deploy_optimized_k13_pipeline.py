#!/usr/bin/env python3
"""
Deploy optimized k=13 enhanced privacy pipeline.

Uses the existing consensus and applies all Phase 1 optimizations:
- Sambamba parallel sorting (2-3× faster)
- Parallel BCFtools variant calling (1.5-2× faster)
- Optimized minimap2 parameters (2.3× faster)
- Minimap2 index caching (save 30-60 sec per reference)
- Metal GPU HDC encoding (43× faster)

Expected: ~18 min per reference vs 7.5 hours baseline (25× speedup)
Total: ~3.6 hours for 12 references vs 90 hours baseline
"""

import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_enhanced_privacy_pipeline_optimized import OptimizedEnhancedPrivacyPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Configuration
    output_dir = "benchmark_results/enhanced_privacy_k13_optimized_phase1"
    num_references = 12
    threads = 10  # M1 Max has 10 cores

    print("=" * 80)
    print("Deploying Optimized k=13 Enhanced Privacy Pipeline")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Number of references: {num_references}")
    print(f"Threads: {threads}")
    print()
    print("Phase 1 Optimizations Enabled:")
    print("  ✅ Sambamba parallel sorting (10 threads, 4GB RAM)")
    print("  ✅ Parallel BCFtools (5 threads)")
    print("  ✅ Minimap2 optimizations + index caching")
    print("  ✅ Metal GPU acceleration for HDC")
    print()
    print("Expected Performance:")
    print("  Baseline: ~7.5 hours per reference (90 hours total)")
    print("  Optimized: ~18 minutes per reference (3.6 hours total)")
    print("  Speedup: ~25× faster")
    print("=" * 80)
    print()

    # Create pipeline instance with all optimizations enabled
    pipeline = OptimizedEnhancedPrivacyPipeline(
        output_dir=output_dir,
        num_references=num_references,
        num_threads=threads,
        use_metal_gpu=True,
        enable_minimap2_optimizations=True,
        enable_sambamba=True,
        enable_parallel_bcftools=True
    )

    # Use existing consensus from baseline run
    existing_consensus = Path("benchmark_results/enhanced_privacy_k13_20251025_183857/layer1_consensus/consensus.fa")

    if not existing_consensus.exists():
        print(f"ERROR: Consensus not found at {existing_consensus}")
        print("Please ensure the baseline consensus exists before running optimized pipeline.")
        return 1

    print(f"✅ Using existing consensus: {existing_consensus}")
    print()

    # Copy consensus to output directory
    import shutil
    output_consensus = pipeline.layer1_dir / "consensus.fa"
    if not output_consensus.exists():
        print(f"Copying consensus to output directory...")
        shutil.copy2(existing_consensus, output_consensus)
        # Also copy index if it exists
        if existing_consensus.with_suffix('.fa.fai').exists():
            shutil.copy2(existing_consensus.with_suffix('.fa.fai'),
                        output_consensus.with_suffix('.fa.fai'))
        print(f"✅ Consensus copied")
        print()

    # Define reference pool FASTQ samples (k=12)
    # Using diverse samples from downloaded data
    fastq_samples = [
        ("ref1", "data/downloaded/fastq/ERR3239276_1.fastq.gz",
                 "data/downloaded/fastq/ERR3239276_2.fastq.gz"),  # European
        ("ref2", "data/downloaded/fastq/ERR3239454_1.fastq.gz",
                 "data/downloaded/fastq/ERR3239454_2.fastq.gz"),  # European
        ("ref3", "data/downloaded/fastq/ERR3239475_1.fastq.gz",
                 "data/downloaded/fastq/ERR3239475_2.fastq.gz"),  # European
        ("ref4", "data/downloaded/fastq/european/ERR3239548/ERR3239548_1.fastq.gz",
                 "data/downloaded/fastq/european/ERR3239548/ERR3239548_2.fastq.gz"),
        ("ref5", "data/downloaded/fastq/european/ERR3239590/ERR3239590_1.fastq.gz",
                 "data/downloaded/fastq/european/ERR3239590/ERR3239590_2.fastq.gz"),
        ("ref6", "data/downloaded/fastq/european/ERR3239920/ERR3239920_1.fastq.gz",
                 "data/downloaded/fastq/european/ERR3239920/ERR3239920_2.fastq.gz"),
        ("ref7", "data/downloaded/fastq/east_asian/ERR3239578/ERR3239578_1.fastq.gz",
                 "data/downloaded/fastq/east_asian/ERR3239578/ERR3239578_2.fastq.gz"),
        ("ref8", "data/downloaded/fastq/east_asian/ERR3239612/ERR3239612_1.fastq.gz",
                 "data/downloaded/fastq/east_asian/ERR3239612/ERR3239612_2.fastq.gz"),
        ("ref9", "data/downloaded/fastq/african/european/ERR3239756/ERR3239756_1.fastq.gz",
                 "data/downloaded/fastq/african/european/ERR3239756/ERR3239756_2.fastq.gz"),
        ("ref10", "data/downloaded/fastq/african/european/ERR3239778/ERR3239778_1.fastq.gz",
                  "data/downloaded/fastq/african/european/ERR3239778/ERR3239778_2.fastq.gz"),
        ("ref11", "data/downloaded/fastq/south_asian/european/ERR3239912/ERR3239912_1.fastq.gz",
                  "data/downloaded/fastq/south_asian/european/ERR3239912/ERR3239912_2.fastq.gz"),
        ("ref12", "data/downloaded/fastq/south_asian/european/ERR3239934/ERR3239934_1.fastq.gz",
                  "data/downloaded/fastq/south_asian/european/ERR3239934/ERR3239934_2.fastq.gz"),
    ]

    # Query sample
    query_r1 = Path("data/downloaded/fastq/ERR3239334_1.fastq.gz")
    query_r2 = Path("data/downloaded/fastq/ERR3239334_2.fastq.gz")

    # Verify all files exist
    print("Verifying input files...")
    missing = []
    for name, r1, r2 in fastq_samples:
        if not Path(r1).exists():
            missing.append(f"{name} R1: {r1}")
        if not Path(r2).exists():
            missing.append(f"{name} R2: {r2}")

    if not query_r1.exists():
        missing.append(f"Query R1: {query_r1}")
    if not query_r2.exists():
        missing.append(f"Query R2: {query_r2}")

    if missing:
        print("ERROR: Missing input files:")
        for f in missing:
            print(f"  ❌ {f}")
        return 1

    print(f"✅ All {len(fastq_samples)} reference samples verified")
    print(f"✅ Query sample verified")
    print()

    # Run Layer 2: Reference Pool (this is where the time is spent)
    print("=" * 80)
    print("Starting optimized reference pool processing...")
    print("=" * 80)
    print()

    layer2_metrics = pipeline.run_layer2_reference_pool(
        consensus_ref=output_consensus,
        fastq_samples=fastq_samples
    )

    # Run Layer 3: Query Alignment
    layer3_metrics = pipeline.run_layer3_query_alignment(
        consensus_ref=output_consensus,
        query_r1=query_r1,
        query_r2=query_r2
    )

    # Run Layer 4: GenomeVault Core
    reference_vcfs = [
        pipeline.layer2_dir / f"{name}.vcf.gz"
        for name, _, _ in fastq_samples
    ]
    query_vcf = Path(layer3_metrics["query_vcf"])

    layer4_metrics = pipeline.run_layer4_genomevault_core(
        query_vcf=query_vcf,
        reference_vcfs=reference_vcfs
    )

    # Print final summary
    total_time = (layer2_metrics["total_time_sec"] +
                  layer3_metrics["total_time_sec"] +
                  layer4_metrics.get("differential_encoding_time_sec", 0))

    print()
    print("=" * 80)
    print("OPTIMIZED PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Layer 2 (Reference Pool): {layer2_metrics['total_time_sec']/3600:.2f} hours")
    print(f"  Average per reference: {layer2_metrics['avg_time_per_reference_sec']/60:.1f} minutes")
    print(f"Layer 3 (Query): {layer3_metrics['total_time_sec']/60:.1f} minutes")
    print(f"Layer 4 (GenomeVault): {layer4_metrics.get('differential_encoding_time_sec', 0):.1f} seconds")
    print()
    print(f"Baseline comparison:")
    print(f"  Baseline estimate: 90 hours (7.5 hours × 12 refs)")
    print(f"  Optimized actual: {total_time/3600:.2f} hours")
    print(f"  Speedup: {90/(total_time/3600):.1f}×")
    print()
    print(f"Results saved to: {output_dir}/")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
