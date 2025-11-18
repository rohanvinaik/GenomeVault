#!/usr/bin/env python3
"""
Recovery script: Continue ref2 from already-partitioned chromosome BAMs.

Partitioning is COMPLETE (726M reads, 1.5 hours). Now sort the 24 chromosome
BAMs in parallel using samtools (sambamba is broken).
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ref2_recovery_sorting.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("🔧 RECOVERY: ref2 from partitioned chromosome BAMs")
    print("=" * 80)
    print("✅ Alignment complete (7.2 hours)")
    print("✅ Partitioning complete (726M reads, 1.5 hours)")
    print("📂 Using 24 chromosome BAM files")
    print()
    print("Continuing with:")
    print("  1. Parallel chromosome sorting (samtools, 10 cores)")
    print("  2. Concatenate sorted chromosomes")
    print("  3. Variant calling")
    print("=" * 80)
    print()

    # Paths
    output_dir = Path("benchmark_results/enhanced_privacy_k13_phase123_optimized")
    layer2_dir = output_dir / "layer2_reference_pool"
    temp_dir = output_dir / "tmp"
    chr_sort_dir = temp_dir / "ref2_chr_sort"

    consensus = output_dir / "layer1_consensus" / "consensus.fa"
    sorted_bam = layer2_dir / "ref2.sorted.bam"
    vcf_file = layer2_dir / "ref2.vcf.gz"

    # Verify consensus exists
    if not consensus.exists():
        print(f"❌ ERROR: Consensus not found: {consensus}")
        return 1

    # Verify chromosome partition directory exists
    if not chr_sort_dir.exists():
        print(f"❌ ERROR: Chromosome partition directory not found: {chr_sort_dir}")
        return 1

    # Count chromosome BAM files
    chr_bams_unsorted = list(chr_sort_dir.glob("*.unsorted.bam"))
    logger.info(f"Found {len(chr_bams_unsorted)} unsorted chromosome BAMs")

    if len(chr_bams_unsorted) == 0:
        print(f"❌ ERROR: No chromosome BAM files found in {chr_sort_dir}")
        return 1

    # Sort by chromosome name to maintain order
    chr_bams_unsorted.sort(key=lambda x: x.stem.split('.')[0])

    # Prepare list of (chr_name, unsorted_bam) tuples
    chr_bams = []
    for bam_file in chr_bams_unsorted:
        chr_name = bam_file.stem.replace('.unsorted', '')
        chr_bams.append((chr_name, str(bam_file)))

    logger.info(f"Chromosomes: {', '.join([c[0] for c in chr_bams[:5]])}...")

    # Initialize chromosome sorter
    chr_sorter = ChromosomePartitionedSorter(num_threads=10)

    # Step 1: Sort chromosomes in parallel (skipping partitioning!)
    print("=" * 80)
    print("[Step 1/4] Parallel sorting (10 chromosomes at a time, samtools)")
    print("=" * 80)
    print()

    sort_start = time.time()

    # Use internal method directly to skip partitioning
    sorted_bams = chr_sorter._parallel_sort_chromosomes(chr_bams, str(chr_sort_dir))
    sort_time = time.time() - sort_start

    logger.info(f"  Parallel sorting ({len(sorted_bams)} chromosomes): {sort_time:.1f}s ({sort_time/60:.1f} min)")
    print()

    # Step 2: Concatenate sorted chromosomes
    print("=" * 80)
    print("[Step 2/4] Concatenating sorted chromosomes")
    print("=" * 80)
    print()

    concat_start = time.time()
    chr_sorter._concatenate_sorted_bams(sorted_bams, str(sorted_bam))
    concat_time = time.time() - concat_start

    logger.info(f"  Concatenation: {concat_time:.1f}s")
    print()

    # Step 3: Variant calling
    print("=" * 80)
    print("[Step 3/4] Variant calling: bcftools")
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
    print("[Step 4/4] Indexing VCF")
    print("=" * 80)
    print()

    subprocess.run(["bcftools", "index", str(vcf_file)], check=True)
    print()

    # Cleanup chromosome BAM files
    logger.info("Cleaning up chromosome BAM files...")
    for bam_file in chr_sort_dir.glob("*.bam"):
        bam_file.unlink()
    logger.info(f"Removed {len(list(chr_sort_dir.glob('*.bam')))} files")

    # Summary
    total_time = sort_time + concat_time + vcf_time
    print("\n" + "=" * 80)
    print("✅ ref2 COMPLETE!")
    print("=" * 80)
    print(f"  Parallel sorting: {sort_time/60:.1f} min ({len(sorted_bams)} chromosomes)")
    print(f"  Concatenation: {concat_time:.1f} s")
    print(f"  Variant calling: {vcf_time/60:.1f} min")
    print(f"  TOTAL (from sorting): {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"  GRAND TOTAL (alignment + partitioning + sorting + VCF): ~9.5 hours")
    print("=" * 80)
    print()

    print("Next: Continue with ref3-ref12")
    print("=" * 80)
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
