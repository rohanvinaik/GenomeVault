#!/usr/bin/env python3
"""
k=11 Parallel Per-Guide Alignment Pipeline (Memory-Safe)

Strategy:
- Align experimental FASTQ to EACH guide FASTA separately (11 jobs)
- Use streaming SAM->BAM conversion (no large temp files)
- Run in parallel batches to control memory
- Select best alignment per read for GDiff encoding

Memory safety:
- Each alignment: ~8 GB RAM (minimap2) + ~2 GB (sambamba) = ~10 GB
- Max 5 parallel jobs = ~50 GB peak RAM (safe for 64 GB system)
- Streaming conversion: no 300+ GB SAM files
- Direct minimap2 -> sambamba pipe
"""

import sys
import time
import logging
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

# Add genomevault to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def align_to_single_guide(
    guide_fasta: Path,
    guide_num: int,
    experimental_r1: Path,
    experimental_r2: Path,
    output_bam: Path,
    threads: int = 8
) -> Tuple[int, float, bool]:
    """
    Align experimental FASTQ to a single guide FASTA.
    Uses streaming: minimap2 -> samtools -> sambamba (no huge SAM file).

    Returns: (guide_num, elapsed_time, success)
    """
    start = time.time()
    output_bam.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Guide {guide_num}] Starting alignment to {guide_fasta.name}")
    logger.info(f"[Guide {guide_num}]   Output: {output_bam}")

    try:
        # Streaming pipeline: minimap2 | samtools view -b | sambamba sort
        # This avoids writing huge SAM files to disk

        cmd = f"""
        minimap2 -ax sr -t {threads} \
            -k 19 -w 10 -A 2 -B 4 -O 6 -E 1 \
            {guide_fasta} \
            {experimental_r1} {experimental_r2} | \
        samtools view -@ {threads//2} -b -h | \
        sambamba sort -t {threads//2} -m 4G \
            -o {output_bam} /dev/stdin
        """

        logger.info(f"[Guide {guide_num}] Streaming: minimap2 | samtools | sambamba")

        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )

        elapsed = time.time() - start
        bam_size = output_bam.stat().st_size / (1024**3)  # GB

        logger.info(f"[Guide {guide_num}] ✓ Complete in {elapsed/60:.1f} min")
        logger.info(f"[Guide {guide_num}]   BAM size: {bam_size:.2f} GB")

        return (guide_num, elapsed, True)

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        logger.error(f"[Guide {guide_num}] ✗ Failed after {elapsed/60:.1f} min")
        logger.error(f"[Guide {guide_num}]   Error: {e.stderr[:500]}")
        return (guide_num, elapsed, False)


def main():
    logger.info("="*80)
    logger.info("k=11 Parallel Per-Guide Alignment (Memory-Safe)")
    logger.info("="*80)

    # Paths
    guide_dir = Path("/Volumes/1TBStorage/guide_strands")
    experimental_dir = Path("data/experimental_strands/ERR3239334")
    alignment_dir = experimental_dir / "per_guide_alignments"

    guide_fastas = [guide_dir / f"ref{i}.fa.gz" for i in range(1, 12)]
    experimental_r1 = Path("data/downloaded/fastq/ERR3239334_1.fastq.gz")
    experimental_r2 = Path("data/downloaded/fastq/ERR3239334_2.fastq.gz")

    # Verify inputs
    logger.info("\nVerifying inputs...")
    for i, fasta in enumerate(guide_fastas, 1):
        if not fasta.exists():
            logger.error(f"Guide FASTA not found: {fasta}")
            return 1
        logger.info(f"  ✓ ref{i}: {fasta.name}")

    if not experimental_r1.exists() or not experimental_r2.exists():
        logger.error("Experimental FASTQ files not found")
        return 1
    logger.info(f"  ✓ Experimental: {experimental_r1.name}, {experimental_r2.name}")

    # Memory safety check
    logger.info("\nMemory safety configuration:")
    logger.info("  Max parallel jobs: 5 (to stay under 64 GB RAM)")
    logger.info("  Per-job RAM: ~10 GB (minimap2 8GB + sambamba 2GB)")
    logger.info("  Peak RAM: ~50 GB (safe)")
    logger.info("  Strategy: Streaming conversion (NO huge SAM files)")

    # Run alignments in parallel batches
    logger.info("\n" + "="*80)
    logger.info("STAGE 1: Parallel Per-Guide Alignments")
    logger.info("="*80)

    total_start = time.time()
    results = []

    max_workers = 5  # Conservative for memory safety

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for i, guide_fasta in enumerate(guide_fastas, 1):
            output_bam = alignment_dir / f"ref{i}.sorted.bam"

            # Skip if already exists
            if output_bam.exists():
                logger.info(f"[Guide {i}] Already exists, skipping: {output_bam}")
                continue

            future = executor.submit(
                align_to_single_guide,
                guide_fasta=guide_fasta,
                guide_num=i,
                experimental_r1=experimental_r1,
                experimental_r2=experimental_r2,
                output_bam=output_bam,
                threads=8
            )
            futures.append(future)

        # Collect results as they complete
        for future in as_completed(futures):
            guide_num, elapsed, success = future.result()
            results.append((guide_num, elapsed, success))

            completed = len(results)
            total = len(futures)
            logger.info(f"\n{'='*80}")
            logger.info(f"Progress: {completed}/{total} guides completed")
            logger.info(f"{'='*80}\n")

    total_elapsed = time.time() - total_start

    # Summary
    logger.info("\n" + "="*80)
    logger.info("ALIGNMENT SUMMARY")
    logger.info("="*80)

    successful = [r for r in results if r[2]]
    failed = [r for r in results if not r[2]]

    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logger.info(f"Successful: {len(successful)}/{len(results)}")

    if failed:
        logger.error(f"Failed guides: {[r[0] for r in failed]}")
        return 1

    logger.info("\n✓ All alignments complete!")
    logger.info(f"✓ Output directory: {alignment_dir}")
    logger.info("\nNext step: Run GDiff encoder with per-guide BAMs")

    return 0


if __name__ == "__main__":
    sys.exit(main())
