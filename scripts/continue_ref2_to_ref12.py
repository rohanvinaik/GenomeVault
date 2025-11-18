#!/usr/bin/env python3
"""
Continue k=13 pipeline from ref2 to ref12 (ref1 already complete).

Uses same Phase 1-3 optimizations as initial deployment:
- Sambamba parallel sorting (10 threads, 8GB RAM)
- Parallel BCFtools (5 threads)
- Minimap2 optimizations + index caching
- Metal GPU HDC acceleration
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
    # Same configuration as original deployment
    output_dir = "benchmark_results/enhanced_privacy_k13_phase123_optimized"
    num_references = 12
    threads = 10
    sambamba_memory = "8G"
    vcf_workers = 5

    print("=" * 80)
    print("🔄 CONTINUING k=13 PIPELINE: ref2-ref12")
    print("=" * 80)
    print(f"✅ ref1 already complete (7.5 hours, 613 MB VCF)")
    print(f"⏳ Processing ref2-ref12 (11 references remaining)")
    print()
    print("Using same Phase 1-3 optimizations:")
    print("  ✅ Sambamba parallel sorting (10 threads, 8GB RAM)")
    print("  ✅ Parallel BCFtools (5 threads)")
    print("  ✅ Minimap2 optimizations + index caching")
    print("  ✅ Metal GPU HDC acceleration")
    print()
    print("Estimated time: ~7.5 hours per reference = 82.5 hours total")
    print("=" * 80)
    print()

    # Create pipeline instance
    pipeline = OptimizedEnhancedPrivacyPipeline(
        output_dir=output_dir,
        num_references=num_references,
        num_threads=threads,
        sambamba_memory=sambamba_memory,
        use_metal_gpu=True,
        enable_minimap2_optimizations=True,
        enable_sambamba=True,
        enable_parallel_bcftools=True,
        enable_chromosome_parallel_sort=False,  # Disabled for whole-genome
        enable_parallel_vcf_parsing=False,
        vcf_parse_workers=vcf_workers
    )

    # Use existing consensus
    existing_consensus = Path("benchmark_results/enhanced_privacy_k13_20251025_183857/layer1_consensus/consensus.fa")

    if not existing_consensus.exists():
        print(f"ERROR: Consensus not found at {existing_consensus}")
        return 1

    # Copy consensus to output directory if not already there
    import shutil
    output_consensus = pipeline.layer1_dir / "consensus.fa"
    if not output_consensus.exists():
        print(f"Copying consensus to output directory...")
        shutil.copy2(existing_consensus, output_consensus)
        if existing_consensus.with_suffix('.fa.fai').exists():
            shutil.copy2(existing_consensus.with_suffix('.fa.fai'),
                        output_consensus.with_suffix('.fa.fai'))

    # Define ONLY ref2-ref12 (skip ref1 which is already complete)
    fastq_samples = [
        ("ref2", "data/downloaded/fastq/ERR3239454_1.fastq.gz",
                 "data/downloaded/fastq/ERR3239454_2.fastq.gz"),
        ("ref3", "data/downloaded/fastq/ERR3239475_1.fastq.gz",
                 "data/downloaded/fastq/ERR3239475_2.fastq.gz"),
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

    # Verify all FASTQ files exist before starting
    print("Verifying FASTQ files...")
    missing = []
    for sample_name, r1, r2 in fastq_samples:
        if not Path(r1).exists():
            missing.append(f"{sample_name}: {r1}")
        if not Path(r2).exists():
            missing.append(f"{sample_name}: {r2}")

    if missing:
        print("❌ ERROR: Missing FASTQ files:")
        for m in missing:
            print(f"  - {m}")
        return 1

    print(f"✅ All {len(fastq_samples)} reference pairs found")
    print()

    # Check which references are already complete
    print("Checking for already-completed references...")
    completed = []
    remaining = []

    for sample_name, r1, r2 in fastq_samples:
        vcf_file = pipeline.layer2_dir / f"{sample_name}.vcf.gz"
        vcf_index = pipeline.layer2_dir / f"{sample_name}.vcf.gz.csi"

        if vcf_file.exists() and vcf_index.exists():
            completed.append(sample_name)
            print(f"  ✅ {sample_name} already complete")
        else:
            remaining.append((sample_name, r1, r2))

    if completed:
        print(f"\n✅ {len(completed)} references already complete: {', '.join(completed)}")

    if not remaining:
        print("\n✅ All references already complete!")
        return 0

    print(f"\n⏳ Processing {len(remaining)} remaining references: {', '.join([s[0] for s in remaining])}")
    print()

    # Run Layer 2 for remaining references only
    print("Starting Layer 2 processing...")
    layer2_results = pipeline.run_layer2_reference_pool(
        consensus_ref=output_consensus,
        fastq_samples=remaining
    )

    # Print summary
    print("\n" + "=" * 80)
    print("✅ CONTINUATION COMPLETE!")
    print("=" * 80)
    print(f"Successfully processed: {len(remaining)} references")
    print(f"Total time: {layer2_results.get('total_time_sec', 0) / 3600:.1f} hours")
    print()
    print("Next steps:")
    print("1. Verify all 12 references complete: ls -lh benchmark_results/enhanced_privacy_k13_phase123_optimized/layer2_reference_pool/")
    print("2. Generate benchmark report: python scripts/generate_final_benchmark_report.py")
    print("3. Continue to Layer 3 (query processing)")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
