"""
FASTQ Processing and Region Identification for Differential Encoding

This module handles FASTQ input and identifies genomic regions for differential encoding.
Bridges the gap between raw sequencing data and the differential encoding pipeline.

Key Features:
- FASTQ alignment to reference genome
- Automatic region detection and extraction
- Multi-reference region extraction for k-anonymity
- Seamless integration with existing pipeline
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import subprocess
import tempfile

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GenomicRegion:
    """
    Identified genomic region from FASTQ alignment.

    Attributes:
        chromosome: Chromosome identifier (e.g., 'chr22', '22', 'X')
        start: Start position (0-based)
        end: End position (exclusive)
        coverage: Average coverage depth in this region
        confidence: Confidence score (0.0-1.0) based on alignment quality
        variant_count: Number of variants detected in region
    """
    chromosome: str
    start: int
    end: int
    coverage: float
    confidence: float
    variant_count: int = 0

    def __repr__(self) -> str:
        return (
            f"GenomicRegion({self.chromosome}:{self.start}-{self.end}, "
            f"coverage={self.coverage:.1f}×, confidence={self.confidence:.2f})"
        )

    @property
    def length(self) -> int:
        """Region length in base pairs."""
        return self.end - self.start


@dataclass
class AlignmentResult:
    """
    Result of FASTQ alignment and region detection.

    Attributes:
        regions: List of identified genomic regions
        alignment_file: Path to alignment file (BAM/SAM)
        vcf_file: Optional path to called variants
        stats: Alignment statistics
    """
    regions: List[GenomicRegion]
    alignment_file: Path
    vcf_file: Optional[Path] = None
    stats: Dict[str, Any] = None

    def get_primary_region(self) -> Optional[GenomicRegion]:
        """Get the region with highest coverage."""
        if not self.regions:
            return None
        return max(self.regions, key=lambda r: r.coverage)


class FASTQProcessor:
    """
    Processes FASTQ files to identify genomic regions for differential encoding.

    Workflow:
    1. Align FASTQ reads to reference genome (Minimap2)
    2. Identify covered regions
    3. Call variants (optional)
    4. Extract identified regions from reference pool
    """

    def __init__(
        self,
        reference_genome: Path,
        aligner: str = "minimap2",
        min_coverage: float = 5.0,
        min_confidence: float = 0.7,
        threads: int = 4,
    ):
        """
        Initialize FASTQ processor.

        Args:
            reference_genome: Path to reference genome FASTA
            aligner: Alignment tool to use ('minimap2' or 'bwa')
            min_coverage: Minimum coverage to consider a region
            min_confidence: Minimum confidence score for region detection
            threads: Number of threads for alignment
        """
        self.reference_genome = Path(reference_genome)
        self.aligner = aligner
        self.min_coverage = min_coverage
        self.min_confidence = min_confidence
        self.threads = threads

        # Verify reference exists
        if not self.reference_genome.exists():
            raise FileNotFoundError(f"Reference genome not found: {reference_genome}")

        # Check aligner availability
        self._check_aligner()

        logger.info(
            f"Initialized FASTQProcessor with {aligner}, "
            f"min_coverage={min_coverage}×, threads={threads}"
        )

    def _check_aligner(self):
        """Verify aligner is installed and accessible."""
        try:
            result = subprocess.run(
                [self.aligner, "--version"],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode != 0:
                raise RuntimeError(f"{self.aligner} not found or not executable")
            logger.debug(f"Found {self.aligner}: {result.stdout.strip()}")
        except FileNotFoundError:
            raise RuntimeError(
                f"{self.aligner} not found. Install with: "
                f"{'conda install -c bioconda minimap2' if self.aligner == 'minimap2' else 'conda install -c bioconda bwa'}"
            )

    def process_fastq(
        self,
        fastq_r1: Path,
        fastq_r2: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> AlignmentResult:
        """
        Process FASTQ file(s) and identify genomic regions.

        Args:
            fastq_r1: Path to R1 FASTQ file (or single-end)
            fastq_r2: Optional path to R2 FASTQ file (paired-end)
            output_dir: Optional output directory (uses temp if not specified)

        Returns:
            AlignmentResult with identified regions and alignment files
        """
        logger.info(f"Processing FASTQ: {fastq_r1.name}")

        # Create output directory
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="genomevault_align_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Align reads
        logger.info("Step 1: Aligning reads to reference genome...")
        alignment_file = self._align_reads(fastq_r1, fastq_r2, output_dir)

        # Step 2: Identify covered regions
        logger.info("Step 2: Identifying covered genomic regions...")
        regions = self._identify_regions(alignment_file)

        # Step 3: Call variants (optional, for better differential encoding)
        logger.info("Step 3: Calling variants...")
        vcf_file = self._call_variants(alignment_file, output_dir)

        # Step 4: Compute stats
        stats = self._compute_stats(alignment_file, regions)

        result = AlignmentResult(
            regions=regions,
            alignment_file=alignment_file,
            vcf_file=vcf_file,
            stats=stats,
        )

        logger.info(f"Identified {len(regions)} genomic region(s)")
        if regions:
            primary = result.get_primary_region()
            logger.info(f"Primary region: {primary}")

        return result

    def _align_reads(
        self,
        fastq_r1: Path,
        fastq_r2: Optional[Path],
        output_dir: Path,
    ) -> Path:
        """
        Align reads using minimap2 or BWA.

        Returns:
            Path to sorted BAM file
        """
        output_bam = output_dir / "aligned.sorted.bam"

        if self.aligner == "minimap2":
            # Minimap2 alignment
            cmd = [
                "minimap2",
                "-ax", "sr",  # Short read mode
                "-t", str(self.threads),
                str(self.reference_genome),
                str(fastq_r1),
            ]
            if fastq_r2:
                cmd.append(str(fastq_r2))

            # Pipe to samtools for sorting
            logger.debug(f"Running: {' '.join(cmd)} | samtools sort")

            with open(output_bam, 'wb') as out_f:
                # Run minimap2
                p1 = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

                # Pipe to samtools sort
                p2 = subprocess.Popen(
                    ["samtools", "sort", "-@", str(self.threads), "-o", str(output_bam)],
                    stdin=p1.stdout,
                    stderr=subprocess.PIPE,
                )

                p1.stdout.close()
                p2.communicate()

                if p2.returncode != 0:
                    raise RuntimeError("Alignment failed")

        elif self.aligner == "bwa":
            # BWA alignment (similar structure)
            raise NotImplementedError("BWA support coming soon")

        # Index BAM file
        subprocess.run(
            ["samtools", "index", str(output_bam)],
            check=True,
            capture_output=True,
        )

        logger.info(f"Alignment complete: {output_bam}")
        return output_bam

    def _identify_regions(self, bam_file: Path) -> List[GenomicRegion]:
        """
        Identify covered genomic regions from alignment.

        Uses samtools depth to find regions with sufficient coverage.
        """
        regions = []

        # Get coverage depth
        logger.debug("Computing coverage depth...")
        result = subprocess.run(
            ["samtools", "depth", "-a", str(bam_file)],
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse coverage and identify regions
        current_region = None
        coverage_sum = 0
        position_count = 0

        for line in result.stdout.split('\n'):
            if not line.strip():
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                continue

            chrom, pos, depth = parts[0], int(parts[1]), int(parts[2])

            if depth >= self.min_coverage:
                if current_region is None:
                    # Start new region
                    current_region = {
                        'chromosome': chrom,
                        'start': pos,
                        'end': pos,
                    }
                    coverage_sum = depth
                    position_count = 1
                elif current_region['chromosome'] == chrom and pos <= current_region['end'] + 1000:
                    # Extend current region (allow 1kb gaps)
                    current_region['end'] = pos
                    coverage_sum += depth
                    position_count += 1
                else:
                    # Save current region and start new
                    if position_count > 0:
                        avg_coverage = coverage_sum / position_count
                        confidence = min(1.0, avg_coverage / 30.0)  # 30× = 100% confidence

                        if confidence >= self.min_confidence:
                            regions.append(GenomicRegion(
                                chromosome=current_region['chromosome'],
                                start=current_region['start'],
                                end=current_region['end'],
                                coverage=avg_coverage,
                                confidence=confidence,
                            ))

                    # Start new region
                    current_region = {
                        'chromosome': chrom,
                        'start': pos,
                        'end': pos,
                    }
                    coverage_sum = depth
                    position_count = 1
            else:
                # Below threshold, end current region
                if current_region is not None and position_count > 0:
                    avg_coverage = coverage_sum / position_count
                    confidence = min(1.0, avg_coverage / 30.0)

                    if confidence >= self.min_confidence:
                        regions.append(GenomicRegion(
                            chromosome=current_region['chromosome'],
                            start=current_region['start'],
                            end=current_region['end'],
                            coverage=avg_coverage,
                            confidence=confidence,
                        ))

                    current_region = None
                    coverage_sum = 0
                    position_count = 0

        # Save final region
        if current_region is not None and position_count > 0:
            avg_coverage = coverage_sum / position_count
            confidence = min(1.0, avg_coverage / 30.0)

            if confidence >= self.min_confidence:
                regions.append(GenomicRegion(
                    chromosome=current_region['chromosome'],
                    start=current_region['start'],
                    end=current_region['end'],
                    coverage=avg_coverage,
                    confidence=confidence,
                ))

        logger.debug(f"Identified {len(regions)} regions meeting criteria")
        return regions

    def _call_variants(self, bam_file: Path, output_dir: Path) -> Optional[Path]:
        """
        Call variants using bcftools.

        Returns:
            Path to VCF file, or None if variant calling fails
        """
        vcf_file = output_dir / "variants.vcf.gz"

        try:
            # Call variants with bcftools
            logger.debug("Calling variants with bcftools...")

            # bcftools mpileup | bcftools call
            p1 = subprocess.Popen(
                [
                    "bcftools", "mpileup",
                    "-f", str(self.reference_genome),
                    str(bam_file),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            with open(vcf_file, 'wb') as out_f:
                p2 = subprocess.Popen(
                    [
                        "bcftools", "call",
                        "-mv", "-Oz",
                        "-o", str(vcf_file),
                    ],
                    stdin=p1.stdout,
                    stderr=subprocess.PIPE,
                )

                p1.stdout.close()
                p2.communicate()

            # Index VCF
            subprocess.run(
                ["bcftools", "index", str(vcf_file)],
                check=True,
                capture_output=True,
            )

            logger.info(f"Variants called: {vcf_file}")
            return vcf_file

        except Exception as e:
            logger.warning(f"Variant calling failed: {e}")
            return None

    def _compute_stats(
        self,
        bam_file: Path,
        regions: List[GenomicRegion],
    ) -> Dict[str, Any]:
        """Compute alignment statistics."""
        try:
            result = subprocess.run(
                ["samtools", "flagstat", str(bam_file)],
                capture_output=True,
                text=True,
                check=True,
            )

            stats = {
                'flagstat': result.stdout,
                'num_regions': len(regions),
                'total_region_length': sum(r.length for r in regions),
                'avg_coverage': np.mean([r.coverage for r in regions]) if regions else 0.0,
            }

            return stats

        except Exception as e:
            logger.warning(f"Failed to compute stats: {e}")
            return {}


def create_default_processor(reference_genome: Path) -> FASTQProcessor:
    """
    Create FASTQ processor with sensible defaults.

    Args:
        reference_genome: Path to reference genome FASTA

    Returns:
        Configured FASTQProcessor
    """
    return FASTQProcessor(
        reference_genome=reference_genome,
        aligner="minimap2",
        min_coverage=5.0,  # 5× minimum coverage
        min_confidence=0.7,  # 70% confidence threshold
        threads=4,
    )
