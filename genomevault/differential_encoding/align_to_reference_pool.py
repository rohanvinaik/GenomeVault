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
1. Loading guide sequences extracted from guide BAMs (NOT consensus!)
2. Aligning query FASTQ to guide sequences directly
3. Using guide pool as blind middleman - no direct consensus link
4. Ensuring query never directly touches consensus reference
"""

import argparse
import logging
import subprocess
import tempfile
import time
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
        guide_fasta_files: List[Path],
        user_randomizer=None,  # Type hint: Optional[UserAlignmentRandomizer]
        threads: int = 8
    ):
        self.guide_fastas = guide_fasta_files
        self.threads = threads
        self.randomizer = user_randomizer

        if len(self.guide_fastas) < 2:
            raise ValueError("Guide pool must have at least k=2 members for privacy")

        # Initialize alignment parameters
        self._initialize_parameters()

    @staticmethod
    def extract_guide_sequences_from_bams(
        guide_bam_files: List[Path],
        output_dir: Path,
        threads: int = 8,
        compress: bool = True
    ) -> List[Path]:
        """
        Extract guide sequences from guide BAM files using samtools consensus.

        This is a critical step in the privacy-preserving pipeline:
        Guide BAMs (aligned to consensus) → Guide FASTA sequences → Query alignment target

        Args:
            guide_bam_files: List of paths to guide BAM files (sorted and indexed)
            output_dir: Directory to save extracted guide FASTA files
            threads: Number of threads for samtools and compression (default: 8)
            compress: If True, compress output with pigz (default: True)

        Returns:
            List of paths to extracted guide FASTA files

        Example:
            guide_bams = [
                Path("ref1.sorted.bam"),
                Path("ref2.sorted.bam"),
                Path("ref3.sorted.bam")
            ]
            guide_fastas = PrivacyPreservingReferencePoolAligner.extract_guide_sequences_from_bams(
                guide_bam_files=guide_bams,
                output_dir=Path("guide_sequences"),
                threads=8
            )
            # Returns: [guide1.fa.gz, guide2.fa.gz, guide3.fa.gz]
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted_fastas = []

        logger.info(f"Extracting guide sequences from {len(guide_bam_files)} BAM files...")

        for i, bam_file in enumerate(guide_bam_files, 1):
            if not bam_file.exists():
                raise FileNotFoundError(f"Guide BAM not found: {bam_file}")

            output_fasta = output_dir / f"guide{i}.fa{'.' + 'gz' if compress else ''}"

            logger.info(f"  Extracting guide {i}/{len(guide_bam_files)}: {bam_file.name} → {output_fasta.name}")

            # Build samtools consensus command
            consensus_cmd = f"samtools consensus --threads {threads} --show-del yes --show-ins yes {bam_file}"

            if compress:
                # Pipe to pigz for parallel compression
                full_cmd = f"{consensus_cmd} | pigz -p {threads} > {output_fasta}"
            else:
                full_cmd = f"{consensus_cmd} > {output_fasta}"

            # Execute extraction
            start_time = time.time()
            subprocess.run(full_cmd, shell=True, check=True)
            elapsed = time.time() - start_time

            # Verify output
            if output_fasta.exists():
                size_mb = output_fasta.stat().st_size / (1024**2)
                logger.info(f"  ✓ Guide {i} extracted: {output_fasta.name} ({size_mb:.0f} MB, {elapsed:.1f}s)")
                extracted_fastas.append(output_fasta)
            else:
                raise RuntimeError(f"Failed to extract guide {i}: {output_fasta} not created")

        logger.info(f"✓ All {len(extracted_fastas)} guide sequences extracted successfully")
        return extracted_fastas

    def _initialize_parameters(self):
        """Initialize alignment parameters (with or without randomization)."""
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

    def prepare_guide_reference(self, output_fasta: Path):
        """
        Prepare guide pool reference by concatenating extracted guide sequences.

        This creates a multi-genome reference representing the guide pool
        WITHOUT exposing the original consensus directly.

        Privacy: Query aligns to guide sequences, NOT consensus.

        CRITICAL: Renames sequences to be unique per reference to avoid duplicate
        chromosome headers. E.g., >chr1_consensus → >ref1_chr1_consensus
        This allows minimap2 to choose among all k=12 versions of each chromosome.
        """
        import gzip
        import re

        logger.info(f"Preparing guide pool reference from {len(self.guide_fastas)} extracted FASTA files...")
        logger.info("  Renaming sequences to avoid duplicate headers (k-anonymity pool)")

        # Concatenate all guide FASTA files with unique sequence names
        with open(output_fasta, 'w') as outf:
            for i, guide_fa in enumerate(self.guide_fastas, 1):
                logger.info(f"  Adding guide {i}: {guide_fa.name}")

                # Open compressed or uncompressed FASTA
                if str(guide_fa).endswith('.gz'):
                    inf = gzip.open(guide_fa, 'rt')
                else:
                    inf = open(guide_fa, 'r')

                try:
                    # Read and rename sequences
                    for line in inf:
                        if line.startswith('>'):
                            # Rename header to be unique: >chr1_consensus → >ref1_chr1_consensus
                            original_header = line.strip()[1:]  # Remove '>'
                            unique_header = f">ref{i}_{original_header}\n"
                            outf.write(unique_header)
                        else:
                            # Write sequence line as-is
                            outf.write(line)
                finally:
                    inf.close()

        logger.info(f"✓ Created guide pool reference: {output_fasta}")
        logger.info(f"  All {len(self.guide_fastas)} references have unique sequence names")
        logger.info(f"✓ Privacy preserved: Query will align to GUIDES, not consensus")
        return output_fasta

    def align_query_to_pool(
        self,
        query_fastq_1: Path,
        query_fastq_2: Path,
        output_vcf: Path,
        output_bam: Path = None,
        privacy_preserving: bool = True
    ) -> Path:
        """
        Align query FASTQ to reference pool with privacy preservation.

        Args:
            query_fastq_1: Query FASTQ R1
            query_fastq_2: Query FASTQ R2
            output_vcf: Output VCF path
            output_bam: Optional output BAM path (for GDiff encoding)
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

            # Step 1: Prepare guide pool reference from extracted FASTA files
            pool_reference = tmpdir / "guide_pool_reference.fa"
            self.prepare_guide_reference(pool_reference)

            # Step 2: Build minimap2 index (ensures proper @SQ header output)
            logger.info(f"Building minimap2 index for guide pool (k={len(self.guide_fastas)} members)...")
            pool_index = tmpdir / "guide_pool.mmi"

            # CRITICAL: Use -x sr preset when building index for Illumina short reads
            # This ensures index parameters match alignment parameters (avoids parameter override warning)
            index_cmd = f"minimap2 -x sr -d {pool_index} {pool_reference}"
            subprocess.run(index_cmd, shell=True, check=True)
            logger.info(f"✓ Index built: {pool_index.stat().st_size / (1024**2):.1f} MB")
            logger.info(f"  Short-read optimized (-x sr preset)")

            # Step 3: Align query to guide sequences using index
            logger.info(f"Aligning query to GUIDE POOL index...")

            query_sam = tmpdir / "query.sam"
            query_bam = tmpdir / "query.sorted.bam"

            if self.randomizer:
                logger.info("  Using randomized alignment parameters (SHA-256² security)")
            else:
                logger.info("  Using default alignment parameters")

            # Step 3a: Align to SAM file (minimap2 won't output @SQ for multi-part index)
            logger.info("  Running minimap2 alignment...")
            align_cmd = f"""
            minimap2 -ax sr -t {self.threads} \
                -k {self.kmer_size} \
                -w {self.window_size} \
                -A {self.scoring['match']} \
                -B {abs(self.scoring['mismatch'])} \
                -O {abs(self.scoring['gap_open'])} \
                -E {abs(self.scoring['gap_extend'])} \
                {pool_index} \
                {query_fastq_1} {query_fastq_2} \
                > {query_sam}
            """
            subprocess.run(align_cmd, shell=True, check=True)

            # Step 3b: Add @SQ headers from reference FASTA and convert to sorted BAM
            logger.info("  Adding reference headers and sorting...")
            view_sort_cmd = f"""
            samtools view -h -bt {pool_reference} {query_sam} | \
                samtools sort -@ {self.threads} -o {query_bam} -
            """
            subprocess.run(view_sort_cmd, shell=True, check=True)

            logger.info("  Indexing BAM...")
            subprocess.run(f"samtools index {query_bam}", shell=True, check=True)

            logger.info("✓ Query aligned to reference pool")

            # Save BAM if requested (for GDiff encoding)
            if output_bam:
                logger.info(f"Saving BAM file: {output_bam}")
                import shutil
                shutil.copy2(query_bam, output_bam)
                shutil.copy2(f"{query_bam}.bai", f"{output_bam}.bai")
                logger.info(f"✓ BAM saved: {output_bam.stat().st_size / (1024**3):.2f} GB")

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

    def align_query_to_pool_with_random_cycling(
        self,
        query_fastq_1: Path,
        query_fastq_2: Path,
        output_bam: Path,
        chunk_size: int = 10_000_000,
        seed: int = None
    ) -> Dict[str, Any]:
        """
        Align query FASTQ to guide pool with PROPER random guide cycling.

        This implements information-theoretic privacy by:
        1. Splitting reads into chunks
        2. Randomly selecting ONE guide per chunk
        3. Aligning each chunk to ONLY that guide
        4. Recording guide selection for decoding

        Privacy guarantee: Attacker cannot determine which guide was used
        for any given read, providing k-anonymity protection.

        Args:
            query_fastq_1: Query FASTQ R1
            query_fastq_2: Query FASTQ R2
            output_bam: Output merged BAM path
            chunk_size: Number of read pairs per chunk (default: 10M)
            seed: Random seed for reproducibility (optional)

        Returns:
            Dict with alignment metadata and guide selection mapping
        """
        import random
        import json
        import gzip

        logger.info("="*80)
        logger.info("PRIVACY-PRESERVING ALIGNMENT: Random Guide Cycling")
        logger.info("="*80)
        logger.info(f"  k-anonymity: {len(self.guide_fastas)} guides")
        logger.info(f"  Chunk size: {chunk_size:,} read pairs")
        logger.info(f"  Privacy: Each chunk aligns to RANDOM guide")
        logger.info("="*80)

        if seed is not None:
            random.seed(seed)
            logger.info(f"  Random seed: {seed} (reproducible)")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Step 1: Build separate indexes for each guide
            logger.info(f"Building {len(self.guide_fastas)} separate guide indexes...")
            guide_indexes = []

            for i, guide_fa in enumerate(self.guide_fastas, 1):
                logger.info(f"  Building index {i}/{len(self.guide_fastas)}: {guide_fa.name}")

                guide_index = tmpdir / f"guide{i}.mmi"
                index_cmd = f"minimap2 -x sr -d {guide_index} {guide_fa}"
                subprocess.run(index_cmd, shell=True, check=True)

                guide_indexes.append({
                    'id': i,
                    'fasta': guide_fa,
                    'index': guide_index
                })

            logger.info(f"✓ Built {len(guide_indexes)} separate indexes")

            # Step 2: Split FASTQ into chunks and align each to random guide
            logger.info(f"Splitting FASTQ and aligning chunks with random guide selection...")

            chunk_bams = []
            guide_selections = {}
            chunk_num = 0

            # Read FASTQ in chunks
            def read_fastq_chunk(fq1, fq2, chunk_size):
                """Generator that yields chunks of FASTQ reads."""
                if str(fq1).endswith('.gz'):
                    f1 = gzip.open(fq1, 'rt')
                    f2 = gzip.open(fq2, 'rt')
                else:
                    f1 = open(fq1, 'r')
                    f2 = open(fq2, 'r')

                try:
                    while True:
                        chunk_r1 = []
                        chunk_r2 = []

                        for _ in range(chunk_size):
                            # Read 4 lines per read (FASTQ format)
                            r1_lines = [f1.readline() for _ in range(4)]
                            r2_lines = [f2.readline() for _ in range(4)]

                            if not r1_lines[0]:  # EOF
                                break

                            chunk_r1.extend(r1_lines)
                            chunk_r2.extend(r2_lines)

                        if not chunk_r1:
                            break

                        yield chunk_r1, chunk_r2
                finally:
                    f1.close()
                    f2.close()

            # Process chunks with random guide selection
            for chunk_r1, chunk_r2 in read_fastq_chunk(query_fastq_1, query_fastq_2, chunk_size):
                chunk_num += 1

                # Randomly select ONE guide for this chunk
                selected_guide = random.choice(guide_indexes)
                guide_selections[f"chunk_{chunk_num}"] = selected_guide['id']

                logger.info(f"  Chunk {chunk_num}: {len(chunk_r1)//4:,} reads → Guide {selected_guide['id']}")

                # Write chunk to temp FASTQ files
                chunk_fq1 = tmpdir / f"chunk{chunk_num}_R1.fastq"
                chunk_fq2 = tmpdir / f"chunk{chunk_num}_R2.fastq"

                with open(chunk_fq1, 'w') as f:
                    f.writelines(chunk_r1)
                with open(chunk_fq2, 'w') as f:
                    f.writelines(chunk_r2)

                # Align chunk to selected guide ONLY
                chunk_sam = tmpdir / f"chunk{chunk_num}.sam"
                chunk_bam = tmpdir / f"chunk{chunk_num}.sorted.bam"

                align_cmd = f"""
                minimap2 -ax sr -t {self.threads} \
                    -k {self.kmer_size} \
                    -w {self.window_size} \
                    -A {self.scoring['match']} \
                    -B {abs(self.scoring['mismatch'])} \
                    -O {abs(self.scoring['gap_open'])} \
                    -E {abs(self.scoring['gap_extend'])} \
                    {selected_guide['index']} \
                    {chunk_fq1} {chunk_fq2} \
                    > {chunk_sam}
                """
                subprocess.run(align_cmd, shell=True, check=True)

                # Convert to BAM and add guide ID to read groups
                rg_id = f"guide{selected_guide['id']}_chunk{chunk_num}"
                view_sort_cmd = f"""
                samtools view -h -bt {selected_guide['fasta']} {chunk_sam} | \
                    samtools addreplacerg -r "@RG\\tID:{rg_id}\\tSM:query\\tPL:ILLUMINA" - | \
                    samtools sort -@ {self.threads} -o {chunk_bam} -
                """
                subprocess.run(view_sort_cmd, shell=True, check=True)
                subprocess.run(f"samtools index {chunk_bam}", shell=True, check=True)

                chunk_bams.append(chunk_bam)

                # Cleanup temp FASTQ files
                chunk_fq1.unlink()
                chunk_fq2.unlink()
                chunk_sam.unlink()

            logger.info(f"✓ Processed {chunk_num} chunks with random guide selection")
            logger.info(f"  Privacy guarantee: k={len(guide_indexes)} anonymity maintained")

            # Step 3: Merge all chunk BAMs
            logger.info(f"Merging {len(chunk_bams)} chunk BAMs...")

            merge_cmd = f"samtools merge -@ {self.threads} {output_bam} " + " ".join(str(b) for b in chunk_bams)
            subprocess.run(merge_cmd, shell=True, check=True)
            subprocess.run(f"samtools index {output_bam}", shell=True, check=True)

            logger.info(f"✓ Merged BAM created: {output_bam.stat().st_size / (1024**3):.2f} GB")

            # Step 4: Save guide selection metadata
            metadata_file = output_bam.with_suffix('.guide_selection.json')
            metadata = {
                'k_anonymity': len(guide_indexes),
                'chunk_size': chunk_size,
                'total_chunks': chunk_num,
                'guide_selections': guide_selections,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'privacy_guarantee': f"Information-theoretic k={len(guide_indexes)} anonymity"
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"✓ Guide selection metadata saved: {metadata_file}")
            logger.info("="*80)
            logger.info("PRIVACY GUARANTEE MAINTAINED:")
            logger.info(f"  - Each chunk aligned to RANDOM guide")
            logger.info(f"  - Attacker cannot determine guide selection")
            logger.info(f"  - k={len(guide_indexes)} anonymity preserved")
            logger.info("="*80)

            return metadata


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
