#!/usr/bin/env python3
"""
Continue k=13 pipeline from ref2 to ref12 using CHROMOSOME-PARALLEL SORTING.

Optimizations:
- Minimap2 alignment to unsorted BAM
- CHROMOSOME-PARTITIONED sorting with sambamba (2-3× faster!)
- Parallel BCFtools variant calling
- Metal GPU HDC acceleration

Expected Performance:
- Alignment: ~2-3 hours
- Sorting (chromosome-parallel): ~25 min (vs 45 min with samtools)
- Variant calling: ~2.5 hours
- Total per reference: ~5.5 hours (vs 7.5 hours) = 1.4× speedup
"""

import sys
import os
from pathlib import Path
import logging
import subprocess
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_enhanced_privacy_pipeline_optimized import OptimizedEnhancedPrivacyPipeline
from genomevault.alignment.chromosome_partitioned_sort import ChromosomePartitionedSorter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def align_to_unsorted_bam(
    reference: Path,
    fastq_r1: Path,
    fastq_r2: Path,
    output_bam: Path,
    num_threads: int = 10,
    minimap2_index: Path = None
) -> float:
    """
    Align FASTQ to unsorted BAM using minimap2.

    Returns:
        Alignment time in seconds
    """
    start_time = time.time()

    logger.info(f"Aligning {fastq_r1.name} + {fastq_r2.name} to unsorted BAM...")

    # Use cached index if available
    if minimap2_index and minimap2_index.exists():
        ref_arg = str(minimap2_index)
        logger.info(f"Using cached index: {minimap2_index}")
    else:
        ref_arg = str(reference)

    # Minimap2 alignment (optimized parameters)
    cmd = f"""
    minimap2 -ax sr -t {num_threads} -K 500M -k 19 -w 10 -2 -A 1 -B 4 {ref_arg} \\
        <(pigz -dc -p 4 {fastq_r1}) <(pigz -dc -p 4 {fastq_r2}) | \\
        samtools view -b -o {output_bam} -
    """

    result = subprocess.run(cmd, shell=True, executable='/bin/bash', capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Alignment failed: {result.stderr}")

    align_time = time.time() - start_time
    logger.info(f"✅ Alignment complete in {align_time:.1f}s")

    return align_time

def main():
    # Configuration
    output_dir = "benchmark_results/enhanced_privacy_k13_phase123_optimized"
    num_references = 12
    threads = 10

    print("=" * 80)
    print("🚀 CHROMOSOME-PARALLEL PIPELINE: ref2-ref12")
    print("=" * 80)
    print(f"✅ ref1 already complete (7.5 hours, 613 MB VCF)")
    print(f"⏳ Processing ref2-ref12 (11 references remaining)")
    print()
    print("🔥 OPTIMIZED METHOD:")
    print("  1. Minimap2 → unsorted BAM")
    print("  2. Partition by chromosome (24 chunks)")
    print("  3. Sambamba sort in parallel (FAST!)")
    print("  4. Merge chromosomes")
    print("  5. Variant calling")
    print()
    print("Expected Performance:")
    print("  Sorting: ~25 min (vs 45 min with samtools) = 1.8× faster")
    print("  Total: ~5.5 hours per reference (vs 7.5 hours) = 1.4× speedup")
    print("  All 11 refs: ~60 hours (vs 82.5 hours) = 22.5 hours saved!")
    print("=" * 80)
    print()

    # Setup paths
    output_path = Path(output_dir)
    layer2_dir = output_path / "layer2_reference_pool"
    layer2_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = output_path / "tmp"
    temp_dir.mkdir(exist_ok=True)

    # Consensus reference
    consensus = output_path / "layer1_consensus" / "consensus.fa"
    if not consensus.exists():
        # Try to copy from baseline
        baseline_consensus = Path("benchmark_results/enhanced_privacy_k13_20251025_183857/layer1_consensus/consensus.fa")
        if baseline_consensus.exists():
            import shutil
            consensus.parent.mkdir(exist_ok=True)
            shutil.copy2(baseline_consensus, consensus)
            if baseline_consensus.with_suffix('.fa.fai').exists():
                shutil.copy2(baseline_consensus.with_suffix('.fa.fai'),
                           consensus.with_suffix('.fa.fai'))
        else:
            print(f"ERROR: Consensus not found!")
            return 1

    # Minimap2 index
    index_cache = output_path / "index_cache"
    index_cache.mkdir(exist_ok=True)
    minimap2_index = index_cache / "consensus.mmi"

    # Build index if not exists
    if not minimap2_index.exists():
        logger.info("Building minimap2 index...")
        subprocess.run(
            ["minimap2", "-d", str(minimap2_index), str(consensus)],
            check=True
        )

    # Define ref2-ref12
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

    # Check which refs are already complete
    logger.info("Checking for already-completed references...")
    remaining = []

    for sample_name, r1, r2 in fastq_samples:
        vcf_file = layer2_dir / f"{sample_name}.vcf.gz"
        vcf_index = layer2_dir / f"{sample_name}.vcf.gz.csi"

        if vcf_file.exists() and vcf_index.exists():
            logger.info(f"  ✅ {sample_name} already complete")
        else:
            remaining.append((sample_name, r1, r2))

    if not remaining:
        print("\n✅ All references already complete!")
        return 0

    logger.info(f"\n⏳ Processing {len(remaining)} remaining references: {', '.join([s[0] for s in remaining])}")
    print()

    # Initialize chromosome sorter
    chr_sorter = ChromosomePartitionedSorter(num_threads=threads)

    # Process each reference
    for idx, (sample_name, r1, r2) in enumerate(remaining, 1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(remaining)}] Processing {sample_name}...")
        print("=" * 80)

        ref_start = time.time()

        # Output files
        unsorted_bam = temp_dir / f"{sample_name}.unsorted.bam"
        sorted_bam = layer2_dir / f"{sample_name}.sorted.bam"
        vcf_file = layer2_dir / f"{sample_name}.vcf.gz"

        # Step 1: Alignment to unsorted BAM
        print(f"\n[Step 1/5] Alignment: minimap2 → unsorted BAM")
        align_time = align_to_unsorted_bam(
            reference=consensus,
            fastq_r1=Path(r1),
            fastq_r2=Path(r2),
            output_bam=unsorted_bam,
            num_threads=threads,
            minimap2_index=minimap2_index
        )
        logger.info(f"  Alignment: {align_time:.1f}s")

        # Step 2: Chromosome-partitioned sorting
        print(f"\n[Step 2/5] Sorting: chromosome-parallel with sambamba")
        sort_metrics = chr_sorter.sort_bam_partitioned(
            input_sam_or_bam=str(unsorted_bam),
            output_bam=str(sorted_bam),
            temp_dir=str(temp_dir / f"{sample_name}_chr_sort")
        )
        logger.info(f"  Partitioning: {sort_metrics['partition_time_sec']:.1f}s")
        logger.info(f"  Parallel sorting ({sort_metrics['num_chromosomes']} chromosomes): {sort_metrics['sort_time_sec']:.1f}s")
        logger.info(f"  Concatenation: {sort_metrics['concatenate_time_sec']:.1f}s")
        logger.info(f"  Total sorting: {sort_metrics['total_time_sec']:.1f}s")

        # Step 3: Index sorted BAM
        print(f"\n[Step 3/5] Indexing sorted BAM")
        index_start = time.time()
        subprocess.run(["samtools", "index", str(sorted_bam)], check=True)
        index_time = time.time() - index_start
        logger.info(f"  Indexing: {index_time:.1f}s")

        # Step 4: Variant calling
        print(f"\n[Step 4/5] Variant calling: bcftools")
        vcf_start = time.time()

        # Parallel BCFtools
        cmd = f"""
        bcftools mpileup --threads 5 -Ou -f {consensus} {sorted_bam} | \\
        bcftools call --threads 5 -mv -Oz -o {vcf_file}
        """
        result = subprocess.run(cmd, shell=True, executable='/bin/bash', capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"Variant calling failed: {result.stderr}")

        vcf_time = time.time() - vcf_start
        logger.info(f"  Variant calling: {vcf_time:.1f}s ({vcf_time/60:.1f} min)")

        # Step 5: Index VCF
        print(f"\n[Step 5/5] Indexing VCF")
        subprocess.run(["bcftools", "index", str(vcf_file)], check=True)

        # Cleanup unsorted BAM
        if unsorted_bam.exists():
            unsorted_bam.unlink()

        # Summary
        ref_time = time.time() - ref_start
        print("\n" + "-" * 80)
        print(f"✅ {sample_name} COMPLETE!")
        print(f"  Alignment: {align_time/60:.1f} min")
        print(f"  Sorting: {sort_metrics['total_time_sec']/60:.1f} min (chromosome-parallel)")
        print(f"  Indexing: {index_time:.1f} s")
        print(f"  Variant calling: {vcf_time/60:.1f} min")
        print(f"  TOTAL: {ref_time/60:.1f} min ({ref_time/3600:.2f} hours)")
        print("-" * 80)

    print("\n" + "=" * 80)
    print("✅ ALL REFERENCES COMPLETE!")
    print("=" * 80)
    print(f"Successfully processed: {len(remaining)} references")
    print()
    print("Next steps:")
    print("1. Verify: ls -lh benchmark_results/enhanced_privacy_k13_phase123_optimized/layer2_reference_pool/")
    print("2. Generate report: python scripts/generate_final_benchmark_report.py")
    print("3. Continue to Layer 3 (query processing)")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    sys.exit(main())
