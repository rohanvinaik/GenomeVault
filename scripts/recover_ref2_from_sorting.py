#!/usr/bin/env python3
"""
Recovery script: Continue ref2 from sorting stage (alignment already complete).

Picks up from existing 27 GB unsorted BAM and continues with:
1. Chromosome-partitioned sorting (FIXED - uses streaming)
2. Variant calling
3. Then continues with ref3-ref12
"""

import sys
import os
import subprocess
import time
from pathlib import Path
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.alignment.chromosome_partitioned_sort import ChromosomePartitionedSorter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("🔧 RECOVERY: ref2 from sorting stage")
    print("=" * 80)
    print("✅ Alignment already complete (7.2 hours)")
    print("📂 Using existing: ref2.unsorted.bam (27 GB)")
    print()
    print("Continuing with:")
    print("  1. Chromosome-partitioned sorting (FIXED streaming method)")
    print("  2. Variant calling")
    print("  3. Then ref3-ref12")
    print("=" * 80)
    print()

    # Paths
    output_dir = Path("benchmark_results/enhanced_privacy_k13_phase123_optimized")
    layer2_dir = output_dir / "layer2_reference_pool"
    temp_dir = output_dir / "tmp"

    consensus = output_dir / "layer1_consensus" / "consensus.fa"
    unsorted_bam = temp_dir / "ref2.unsorted.bam"
    sorted_bam = layer2_dir / "ref2.sorted.bam"
    vcf_file = layer2_dir / "ref2.vcf.gz"

    # Verify files exist
    if not unsorted_bam.exists():
        print(f"❌ ERROR: Unsorted BAM not found: {unsorted_bam}")
        return 1

    if not consensus.exists():
        print(f"❌ ERROR: Consensus not found: {consensus}")
        return 1

    print(f"✅ Unsorted BAM found: {unsorted_bam.stat().st_size / 1e9:.1f} GB")
    print()

    # Initialize chromosome sorter
    chr_sorter = ChromosomePartitionedSorter(num_threads=10)

    # Step 1: Chromosome-partitioned sorting
    print("=" * 80)
    print("[Step 2/5] Sorting: chromosome-parallel with sambamba (STREAMING)")
    print("=" * 80)
    print()

    sort_start = time.time()
    sort_metrics = chr_sorter.sort_bam_partitioned(
        input_sam_or_bam=str(unsorted_bam),
        output_bam=str(sorted_bam),
        temp_dir=str(temp_dir / "ref2_chr_sort")
    )
    sort_time = time.time() - sort_start

    logger.info(f"  Partitioning: {sort_metrics['partition_time_sec']:.1f}s ({sort_metrics['partition_time_sec']/60:.1f} min)")
    logger.info(f"  Parallel sorting ({sort_metrics['num_chromosomes']} chromosomes): {sort_metrics['sort_time_sec']:.1f}s ({sort_metrics['sort_time_sec']/60:.1f} min)")
    logger.info(f"  Concatenation: {sort_metrics['concatenate_time_sec']:.1f}s")
    logger.info(f"  Total sorting: {sort_time:.1f}s ({sort_time/60:.1f} min)")
    print()

    # Step 2: Index sorted BAM
    print("=" * 80)
    print("[Step 3/5] Indexing sorted BAM")
    print("=" * 80)
    print()

    index_start = time.time()
    subprocess.run(["samtools", "index", str(sorted_bam)], check=True)
    index_time = time.time() - index_start
    logger.info(f"  Indexing: {index_time:.1f}s")
    print()

    # Step 3: Variant calling
    print("=" * 80)
    print("[Step 4/5] Variant calling: bcftools")
    print("=" * 80)
    print()

    vcf_start = time.time()
    cmd = f"""
    bcftools mpileup --threads 5 -Ou -f {consensus} {sorted_bam} | \\
    bcftools call --threads 5 -mv -Oz -o {vcf_file}
    """
    result = subprocess.run(cmd, shell=True, executable='/bin/bash', capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Variant calling failed: {result.stderr}")

    vcf_time = time.time() - vcf_start
    logger.info(f"  Variant calling: {vcf_time:.1f}s ({vcf_time/60:.1f} min)")
    print()

    # Step 4: Index VCF
    print("=" * 80)
    print("[Step 5/5] Indexing VCF")
    print("=" * 80)
    print()

    subprocess.run(["bcftools", "index", str(vcf_file)], check=True)
    print()

    # Cleanup unsorted BAM
    if unsorted_bam.exists():
        logger.info(f"Cleaning up unsorted BAM: {unsorted_bam}")
        unsorted_bam.unlink()

    # Summary
    total_time = sort_time + index_time + vcf_time
    print("\n" + "=" * 80)
    print("✅ ref2 COMPLETE!")
    print("=" * 80)
    print(f"  Sorting: {sort_time/60:.1f} min (chromosome-parallel)")
    print(f"  Indexing: {index_time:.1f} s")
    print(f"  Variant calling: {vcf_time/60:.1f} min")
    print(f"  TOTAL (from sorting): {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print("=" * 80)
    print()

    print("Next: Continue with ref3-ref12 using the same pipeline")
    print("  python3 scripts/continue_ref2_to_ref12_chromosome_parallel.py")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
