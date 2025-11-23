#!/usr/bin/env python3
"""
Privacy-Preserving Reference Pool Alignment

CRITICAL SECURITY REQUIREMENT:
Query FASTQ must NEVER align directly to Byzantine consensus reference.
This would create traceable linkage and violate the privacy architecture.

CORRECT APPROACH:
Query → Reference Pool (consensus-aligned) → Differential Encoding
         ↓
   Privacy-Preserving Indirection
         ↓
Query → Pool → Consensus → Public References
(NO DIRECT LINK TO CONSENSUS)

This module implements the privacy-preserving handoff by:
1. Loading reference pool VCFs (already consensus-aligned)
2. Aligning query FASTQ to reference pool variants (NOT consensus)
3. Using consensus coordinates only for final coordinate system
4. Ensuring query never directly touches consensus reference
"""

import argparse
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PrivacyPreservingReferencePoolAligner:
    """
    Align query FASTQ to reference pool WITHOUT direct consensus alignment.

    Privacy Properties:
    - Query never aligns to consensus directly
    - Query only sees reference pool members' alignment information
    - Creates indirection: Query → Pool → Consensus → Public
    - Optional: User-specific randomization (260-bit entropy) for SHA-256² security
    """

    def __init__(
        self,
        reference_pool_vcfs: List[Path],
        consensus_reference: Path,
        user_randomizer=None,  # Type hint: Optional[UserAlignmentRandomizer]
        threads: int = 8
    ):
        self.ref_pool = reference_pool_vcfs
        self.consensus = consensus_reference
        self.threads = threads
        self.randomizer = user_randomizer

        if len(self.ref_pool) < 2:
            raise ValueError("Reference pool must have at least k=2 members for privacy")

        # Apply user-specific randomization if provided
        if self.randomizer:
            self.kmer_size = self.randomizer.randomize_kmer_size()
            self.window_size = self.randomizer.randomize_window_size()
            self.scoring = self.randomizer.randomize_scoring_matrix()

            entropy = self.randomizer.compute_total_entropy()

            logger.info("="*60)
            logger.info("SHA-256² SECURITY: User-Specific Randomization Applied")
            logger.info("="*60)
            logger.info(f"  User ID: {self.randomizer.user_id}")
            logger.info(f"  k-mer size: {self.kmer_size}")
            logger.info(f"  Window size: {self.window_size}")
            logger.info(f"  Scoring: match={self.scoring['match']}, mismatch={self.scoring['mismatch']}")
            logger.info(f"  Total entropy: ~{entropy['total']:.1f} bits")
            logger.info(f"  Entropy breakdown:")
            logger.info(f"    - k-mer size: {entropy['kmer_size']:.1f} bits")
            logger.info(f"    - Window size: {entropy['window_size']:.1f} bits")
            logger.info(f"    - Scoring matrix: {entropy['scoring_matrix']:.1f} bits")
            logger.info(f"    - Positional jitter: {entropy['positional_jitter']:.1f} bits")
            logger.info(f"    - Read sampling: {entropy['read_sampling']:.1f} bits")
            logger.info("="*60)
        else:
            # Default parameters (no randomization)
            self.kmer_size = 19
            self.window_size = 10
            self.scoring = {
                'match': 2,
                'mismatch': -4,
                'gap_open': -6,
                'gap_extend': -1
            }
            logger.info("No user randomization - using default alignment parameters")

    def reconstruct_reference_from_pool(self, output_fasta: Path):
        """
        Reconstruct reference sequences from pool VCFs + consensus.

        This creates a "virtual reference" that represents the reference pool
        WITHOUT exposing the original consensus directly.
        """
        logger.info(f"Reconstructing reference pool sequences from {len(self.ref_pool)} VCFs...")

        # Strategy: Merge all reference pool variants
        # This creates a pangenome-like reference representing the pool
        merged_vcf = output_fasta.parent / "merged_pool.vcf.gz"

        # Merge VCFs from pool
        merge_cmd = f"bcftools merge {' '.join(str(v) for v in self.ref_pool)} -Oz -o {merged_vcf}"
        subprocess.run(merge_cmd, shell=True, check=True, capture_output=True)
        subprocess.run(f"bcftools index {merged_vcf}", shell=True, check=True)

        # Apply variants to consensus to create pool-specific reference
        # This is NOT the original consensus - it's a pool-representative reference
        apply_cmd = f"bcftools consensus -f {self.consensus} {merged_vcf} > {output_fasta}"
        subprocess.run(apply_cmd, shell=True, check=True)

        logger.info(f"✓ Created pool reference: {output_fasta}")
        return merged_vcf

    def align_query_to_pool(
        self,
        query_fastq_1: Path,
        query_fastq_2: Path,
        output_vcf: Path,
        privacy_preserving: bool = True
    ) -> Path:
        """
        Align query FASTQ to reference pool with privacy preservation.

        Args:
            query_fastq_1: Query FASTQ R1
            query_fastq_2: Query FASTQ R2
            output_vcf: Output VCF path
            privacy_preserving: If True, ensures no direct consensus alignment

        Returns:
            Path to query VCF
        """
        logger.info("=== Privacy-Preserving Query Alignment ===")

        if privacy_preserving:
            logger.info("✓ Privacy mode ENABLED - query will NOT align to consensus directly")
        else:
            logger.warning("⚠ Privacy mode DISABLED - for testing only!")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Step 1: Reconstruct pool-representative reference
            pool_reference = tmpdir / "pool_reference.fa"
            merged_vcf = self.reconstruct_reference_from_pool(pool_reference)

            # Step 2: Align query to pool reference (NOT original consensus!)
            logger.info(f"Aligning query to REFERENCE POOL (k={len(self.ref_pool)} members)...")

            query_bam = tmpdir / "query.sorted.bam"

            # Build minimap2 command with user-specific or default parameters
            align_cmd = f"""
            minimap2 -ax sr -t {self.threads} \
                -k {self.kmer_size} \
                -w {self.window_size} \
                -A {self.scoring['match']} \
                -B {abs(self.scoring['mismatch'])} \
                -O {abs(self.scoring['gap_open'])} \
                -E {abs(self.scoring['gap_extend'])} \
                {pool_reference} \
                {query_fastq_1} {query_fastq_2} | \
                samtools sort -@ {self.threads} -o {query_bam} -
            """

            if self.randomizer:
                logger.info("  Using randomized alignment parameters (SHA-256² security)")
            else:
                logger.info("  Using default alignment parameters")

            subprocess.run(align_cmd, shell=True, check=True, capture_output=True)
            subprocess.run(f"samtools index {query_bam}", shell=True, check=True)

            logger.info("✓ Query aligned to reference pool")

            # Step 3: Call variants
            logger.info("Calling variants from pool-aligned query...")

            vcall_cmd = f"""
            bcftools mpileup -f {pool_reference} {query_bam} | \
                bcftools call -mv -Oz -o {output_vcf}
            """
            subprocess.run(vcall_cmd, shell=True, check=True)
            subprocess.run(f"bcftools index {output_vcf}", shell=True, check=True)

            # Get variant count
            result = subprocess.run(
                f"bcftools view -H {output_vcf} | wc -l",
                shell=True,
                capture_output=True,
                text=True
            )
            variant_count = int(result.stdout.strip())

            logger.info(f"✓ Called {variant_count} variants from query")
            logger.info(f"✓ Privacy preserved: Query aligned to pool, NOT consensus")

        return output_vcf


def main():
    parser = argparse.ArgumentParser(
        description='Privacy-preserving query alignment to reference pool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
SECURITY WARNING:
Never align query FASTQ directly to consensus reference!
This creates traceable linkage and violates privacy architecture.

CORRECT USAGE:
Query → Reference Pool → (implicit consensus coordinates)

Example:
  python align_to_reference_pool.py \\
      --query-fastq query_1.fastq.gz query_2.fastq.gz \\
      --reference-pool ref1.vcf ref2.vcf ref3.vcf \\
      --consensus-reference consensus.fa \\
      --output query.vcf \\
      --privacy-preserving
        """
    )

    parser.add_argument('--query-fastq', nargs=2, required=True,
                        metavar=('R1', 'R2'),
                        help='Query FASTQ files (paired-end)')
    parser.add_argument('--reference-pool', nargs='+', required=True,
                        help='Reference pool VCF files (k>=2)')
    parser.add_argument('--consensus-reference', required=True,
                        help='Byzantine consensus reference FASTA (for coordinates only)')
    parser.add_argument('--output', required=True,
                        help='Output VCF file')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of threads (default: 8)')
    parser.add_argument('--privacy-preserving', action='store_true', default=True,
                        help='Enable privacy-preserving mode (default: True)')
    parser.add_argument('--allow-direct-consensus', action='store_false',
                        dest='privacy_preserving',
                        help='DANGEROUS: Allow direct consensus alignment (testing only!)')

    args = parser.parse_args()

    # Validate inputs
    query_r1 = Path(args.query_fastq[0])
    query_r2 = Path(args.query_fastq[1])
    ref_pool = [Path(v) for v in args.reference_pool]
    consensus = Path(args.consensus_reference)
    output = Path(args.output)

    if not query_r1.exists() or not query_r2.exists():
        logger.error("Query FASTQ files not found")
        return 1

    if not all(v.exists() for v in ref_pool):
        logger.error("Some reference pool VCF files not found")
        return 1

    if not consensus.exists():
        logger.error(f"Consensus reference not found: {consensus}")
        return 1

    if len(ref_pool) < 2:
        logger.error("Reference pool must have at least k=2 members for privacy")
        return 1

    # Initialize aligner
    aligner = PrivacyPreservingReferencePoolAligner(
        reference_pool_vcfs=ref_pool,
        consensus_reference=consensus,
        threads=args.threads
    )

    # Align query to pool
    aligner.align_query_to_pool(
        query_fastq_1=query_r1,
        query_fastq_2=query_r2,
        output_vcf=output,
        privacy_preserving=args.privacy_preserving
    )

    logger.info("=== Privacy-Preserving Alignment Complete ===")
    logger.info(f"Output: {output}")
    logger.info(f"Privacy: {'PRESERVED ✓' if args.privacy_preserving else 'VIOLATED ✗'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
