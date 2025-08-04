# Standard library imports
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
import pysam
from Bio import SeqIO

from genomevault.crypto.hashing import secure_hash
from genomevault.utils.config import get_config
# Local imports
from genomevault.utils.logging import get_logger

"""
GenomeVault Sequencing Data Processing

Handles genomic sequencing data processing including alignment, variant calling,
and reference-based differential storage.
"""


# sequencing.py

logger = get_logger(__name__)
config = get_config()


@dataclass
class QualityMetrics:
    """Quality metrics for sequencing data"""

    total_reads: int = 0
    total_bases: int = 0
    q30_bases: int = 0
    q20_bases: int = 0
    gc_content: float = 0.0
    mean_quality: float = 0.0
    mean_read_length: float = 0.0
    duplicate_rate: float = 0.0
    adapter_contamination: float = 0.0
    coverage_mean: float = 0.0
    coverage_std: float = 0.0
    coverage_uniformity: float = 0.0


@dataclass
class Variant:
    """Genomic variant representation"""

    chromosome: str
    position: int
    reference: str
    alternate: str
    quality: float
    genotype: str  # "0/0", "0/1", "1/1"
    depth: int
    allele_frequency: float = 0.0
    annotations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "chr": self.chromosome,
            "pos": self.position,
            "ref": self.reference,
            "alt": self.alternate,
            "qual": self.quality,
            "gt": self.genotype,
            "dp": self.depth,
            "af": self.allele_frequency,
            "ann": self.annotations,
        }

    def get_id(self) -> str:
        """Get unique variant identifier"""
        return "{self.chromosome}:{self.position}:{self.reference}>{self.alternate}"


@dataclass
class GenomicProfile:
    """Complete genomic profile with variants and metadata"""

    sample_id: str
    reference_genome: str
    variants: list[Variant]
    quality_metrics: QualityMetrics
    processing_metadata: dict[str, Any]
    checksum: str = ""

    def __post_init__(self):
        """Calculate checksum after initialization"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate SHA256 checksum of profile"""
        data = {
            "sample_id": self.sample_id,
            "reference": self.reference_genome,
            "variant_count": len(self.variants),
            "variant_hashes": [
                secure_hash(v.get_id().encode()) for v in self.variants[:100]
            ],
        }
        return secure_hash(json.dumps(data, sort_keys=True).encode())


class SequencingProcessor:
    """Main processor for sequencing data"""

    SUPPORTED_FORMATS = {".fastq", ".fq", ".fastq.gz", ".fq.gz", ".bam", ".cram"}
    DEFAULT_REFERENCE = "GRCh38"

    def __init__(
        self,
        reference_path: Path | None = None,
        temp_dir: Path | None = None,
        max_threads: int | None = None,
    ):
        """
        Initialize sequencing processor

        Args:
            reference_path: Path to reference genome
            temp_dir: Temporary directory for processing
            max_threads: Maximum threads to use
        """
        self.reference_path = reference_path or self._get_default_reference()
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "genomevault"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.max_threads = max_threads or config.processing.max_cores
        self.quality_threshold = config.processing.min_quality_score
        self.coverage_threshold = config.processing.min_coverage

        # Validate tools are available
        self._validate_tools()

    def _get_default_reference(self) -> Path:
        """Get default reference genome path"""
        ref_dir = config.storage.data_dir / "references"
        ref_dir.mkdir(parents=True, exist_ok=True)
        return ref_dir / "{self.DEFAULT_REFERENCE}.fa"

    def _validate_tools(self):
        """Validate required tools are installed"""
        required_tools = {
            "bwa": "BWA aligner",
            "samtools": "SAMtools",
            "bcftools": "BCFtools",
            "gatk": "GATK toolkit",
        }

        missing_tools = []
        for tool, name in required_tools.items():
            try:
                subprocess.run([tool, "--version"], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                from genomevault.observability.logging import configure_logging

                logger = configure_logging()
                logger.exception("Unhandled exception")
                missing_tools.append(name)
                raise RuntimeError("Unspecified error")

        if missing_tools:
            logger.warning("Missing tools: %s', '.join(missing_tools)")
            logger.warning("Some processing features may be unavailable")

    @_log_operation("process_sequencing_data")
    def process(self, input_path: Path, sample_id: str) -> GenomicProfile:
        """
        Process sequencing data to generate genomic profile

        Args:
            input_path: Path to input sequencing data
            sample_id: Sample identifier

        Returns:
            GenomicProfile with variants and quality metrics
        """
        logger.info("Processing sequencing data for sample %ssample_id")

        # Validate input
        if not input_path.exists():
            raise FileNotFoundError("Input file not found: {input_path}")

        if input_path.suffix not in self.SUPPORTED_FORMATS:
            raise ValueError("Unsupported file format: {input_path.suffix}")

        # Create working directory
        work_dir = self.temp_dir / "seq_{sample_id}_{os.getpid()}"
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Step 1: Quality control
            logger.info("Running quality control...")
            qc_metrics = self._run_quality_control(input_path, work_dir)

            # Step 2: Alignment (if needed)
            if input_path.suffix in {".fastq", ".fq", ".fastq.gz", ".fq.gz"}:
                logger.info("Aligning reads to reference...")
                aligned_bam = self._align_reads(input_path, work_dir)
            else:
                aligned_bam = input_path

            # Step 3: Mark duplicates
            logger.info("Marking duplicates...")
            dedup_bam = self._mark_duplicates(aligned_bam, work_dir)

            # Step 4: Calculate coverage
            logger.info("Calculating coverage...")
            coverage_metrics = self._calculate_coverage(dedup_bam)
            qc_metrics.coverage_mean = coverage_metrics["mean"]
            qc_metrics.coverage_std = coverage_metrics["std"]
            qc_metrics.coverage_uniformity = coverage_metrics["uniformity"]

            # Step 5: Variant calling
            logger.info("Calling variants...")
            variants = self._call_variants(dedup_bam, work_dir)

            # Step 6: Variant annotation
            logger.info("Annotating variants...")
            annotated_variants = self._annotate_variants(variants)

            # Create genomic profile
            profile = GenomicProfile(
                sample_id=sample_id,
                reference_genome=self.DEFAULT_REFERENCE,
                variants=annotated_variants,
                quality_metrics=qc_metrics,
                processing_metadata={
                    "processor_version": "1.0.0",
                    "reference_path": str(self.reference_path),
                    "quality_threshold": self.quality_threshold,
                    "coverage_threshold": self.coverage_threshold,
                },
            )

            logger.info("Processing complete. Found %slen(profile.variants) variants")
            return profile

        finally:
            # Cleanup
            if work_dir.exists():
                import shutil

                shutil.rmtree(work_dir)

    def _run_quality_control(self, input_path: Path, work_dir: Path) -> QualityMetrics:
        """Run quality control on sequencing data"""
        metrics = QualityMetrics()

        if input_path.suffix in {".fastq", ".fq", ".fastq.gz", ".fq.gz"}:
            # Process FASTQ files
            open_func = _gzip.open if input_path.suffix.endswith(".gz") else open

            with open_func(input_path, "rt") as f:
                for i, record in enumerate(SeqIO.parse(f, "fastq")):
                    metrics.total_reads += 1
                    metrics.total_bases += len(record.seq)

                    # Quality metrics
                    qualities = record.letter_annotations["phred_quality"]
                    metrics.mean_quality += np.mean(qualities)
                    metrics.q30_bases += sum(1 for q in qualities if q >= 30)
                    metrics.q20_bases += sum(1 for q in qualities if q >= 20)

                    # GC content
                    gc_count = record.seq.count("G") + record.seq.count("C")
                    metrics.gc_content += gc_count / len(record.seq)

                    # Sample first 10000 reads for efficiency
                    if i >= 10000:
                        break

            # Average metrics
            if metrics.total_reads > 0:
                metrics.mean_quality /= metrics.total_reads
                metrics.gc_content /= metrics.total_reads
                metrics.mean_read_length = metrics.total_bases / metrics.total_reads

        else:
            # Process BAM/CRAM files
            with pysam.AlignmentFile(str(input_path), "rb") as bam:
                for i, read in enumerate(bam):
                    metrics.total_reads += 1
                    metrics.total_bases += read.query_length or 0

                    if read.query_qualities:
                        qualities = read.query_qualities
                        metrics.mean_quality += np.mean(qualities)
                        metrics.q30_bases += sum(1 for q in qualities if q >= 30)
                        metrics.q20_bases += sum(1 for q in qualities if q >= 20)

                    if i >= 10000:
                        break

            if metrics.total_reads > 0:
                metrics.mean_quality /= metrics.total_reads
                metrics.mean_read_length = metrics.total_bases / metrics.total_reads

        return metrics

    def _align_reads(self, fastq_path: Path, work_dir: Path) -> Path:
        """Align reads to reference genome using BWA"""
        output_bam = work_dir / "aligned.bam"

        # Check if reference is indexed
        if not (self.reference_path.with_suffix(".fa.bwt")).exists():
            logger.info("Indexing reference genome...")
            subprocess.run(["bwa", "index", str(self.reference_path)], check=True)

        # Run BWA alignment
        logger.info("Running BWA alignment...")
        with open(work_dir / "aligned.sam", "w") as sam_file:
            bwa_process = subprocess.Popen(
                [
                    "bwa",
                    "mem",
                    "-t",
                    str(self.max_threads),
                    "-M",  # Mark shorter split hits as secondary
                    str(self.reference_path),
                    str(fastq_path),
                ],
                stdout=sam_file,
                stderr=subprocess.PIPE,
            )

            _, stderr = bwa_process.communicate()
            if bwa_process.returncode != 0:
                raise RuntimeError("BWA alignment failed: {stderr.decode()}")

        # Convert SAM to sorted BAM
        logger.info("Converting to sorted BAM...")
        subprocess.run(
            [
                "samtools",
                "sort",
                "-@",
                str(self.max_threads),
                "-o",
                str(output_bam),
                str(work_dir / "aligned.sam"),
            ],
            check=True,
        )

        # Index BAM
        subprocess.run(["samtools", "index", str(output_bam)], check=True)

        # Remove intermediate SAM
        (work_dir / "aligned.sam").unlink()

        return output_bam

    def _mark_duplicates(self, bam_path: Path, work_dir: Path) -> Path:
        """Mark duplicate reads"""
        output_bam = work_dir / "dedup.bam"
        metrics_file = work_dir / "duplicate_metrics.txt"

        # Use samtools if GATK not available
        try:
            subprocess.run(
                [
                    "gatk",
                    "MarkDuplicates",
                    "-I",
                    str(bam_path),
                    "-O",
                    str(output_bam),
                    "-M",
                    str(metrics_file),
                    "--CREATE_INDEX",
                    "true",
                ],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            logger.warning("GATK not available, using samtools")
            subprocess.run(
                [
                    "samtools",
                    "markdup",
                    "-@",
                    str(self.max_threads),
                    str(bam_path),
                    str(output_bam),
                ],
                check=True,
            )
            subprocess.run(["samtools", "index", str(output_bam)], check=True)
            raise RuntimeError("Unspecified error")

        return output_bam

    def _calculate_coverage(self, bam_path: Path) -> dict[str, float]:
        """Calculate coverage statistics"""
        coverages = []

        with pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for contig in bam.references[:5]:  # Sample first 5 chromosomes
                coverage = bam.count_coverage(contig, 0, min(bam.lengths[0], 1000000))
                total_coverage = np.sum(coverage, axis=0)
                coverages.extend(total_coverage)

        if coverages:
            return {
                "mean": np.mean(coverages),
                "std": np.std(coverages),
                "uniformity": len([c for c in coverages if c > 0]) / len(coverages),
            }
        else:
            return {"mean": 0.0, "std": 0.0, "uniformity": 0.0}

    def _call_variants(self, bam_path: Path, work_dir: Path) -> list[Variant]:
        """Call variants using bcftools"""
        vcf_path = work_dir / "variants.vcf"

        # Run bcftools mpileup and call
        subprocess.run(
            [
                "bcftools",
                "mpileup",
                "-f",
                str(self.reference_path),
                "--threads",
                str(self.max_threads),
                str(bam_path),
                "|",
                "bcftools",
                "call",
                "-mv",
                "-Ov",
                "-o",
                str(vcf_path),
            ],
            shell=True,
            check=True,
        )

        # Parse VCF
        variants = []
        with open(vcf_path) as vcf:
            for line in vcf:
                if line.startswith("#"):
                    continue

                parts = line.strip().split("\t")
                if len(parts) < 10:
                    continue

                # Parse variant info
                chrom = parts[0]
                pos = int(parts[1])
                ref = parts[3]
                alt = parts[4]
                qual = float(parts[5]) if parts[5] != "." else 0

                # Parse genotype
                format_fields = parts[8].split(":")
                sample_fields = parts[9].split(":")

                gt_idx = format_fields.index("GT") if "GT" in format_fields else 0
                genotype = (
                    sample_fields[gt_idx] if gt_idx < len(sample_fields) else "0/0"
                )

                dp_idx = format_fields.index("DP") if "DP" in format_fields else -1
                depth = (
                    int(sample_fields[dp_idx])
                    if dp_idx >= 0 and dp_idx < len(sample_fields)
                    else 0
                )

                # Create variant
                variant = Variant(
                    chromosome=chrom,
                    position=pos,
                    reference=ref,
                    alternate=alt,
                    quality=qual,
                    genotype=genotype,
                    depth=depth,
                )

                variants.append(variant)

        return variants

    def _annotate_variants(self, variants: list[Variant]) -> list[Variant]:
        """Annotate variants with functional information"""
        # This is a simplified annotation
        # In production, would use tools like VEP or SnpEff

        for variant in variants:
            # Add basic annotations
            variant.annotations["variant_type"] = self._get_variant_type(
                variant.reference, variant.alternate
            )

            # Estimate allele frequency from genotype
            if variant.genotype == "0/1":
                variant.allele_frequency = 0.5
            elif variant.genotype == "1/1":
                variant.allele_frequency = 1.0
            else:
                variant.allele_frequency = 0.0

        return variants

    def _get_variant_type(self, ref: str, alt: str) -> str:
        """Determine variant type"""
        if len(ref) == 1 and len(alt) == 1:
            return "SNP"
        elif len(ref) > len(alt):
            return "DELETION"
        elif len(ref) < len(alt):
            return "INSERTION"
        else:
            return "COMPLEX"


class DifferentialStorage:
    """Handle reference-based differential storage of genomic data"""

    def __init__(self, reference_genome: str = "GRCh38"):
        """Initialize differential storage handler"""
        self.reference_genome = reference_genome
        self.chunk_size = 1000  # variants per chunk

    def compress_profile(self, profile: GenomicProfile) -> dict[str, Any]:
        """
        Compress genomic profile using differential storage

        Args:
            profile: GenomicProfile to compress

        Returns:
            Compressed representation
        """
        # Sort variants by position
        sorted_variants = sorted(
            profile.variants, key=lambda v: (v.chromosome, v.position)
        )

        # Chunk variants
        chunks = []
        for i in range(0, len(sorted_variants), self.chunk_size):
            chunk = sorted_variants[i : i + self.chunk_size]

            # Compress chunk
            compressed_chunk = {
                "start_pos": "{chunk[0].chromosome}:{chunk[0].position}",
                "end_pos": "{chunk[-1].chromosome}:{chunk[-1].position}",
                "variant_count": len(chunk),
                "variants": [self._compress_variant(v) for v in chunk],
            }
            chunks.append(compressed_chunk)

        return {
            "sample_id": profile.sample_id,
            "reference": profile.reference_genome,
            "chunks": chunks,
            "metrics": profile.quality_metrics.__dict__,
            "checksum": profile.checksum,
        }

    def _compress_variant(self, variant: Variant) -> dict[str, Any]:
        """Compress individual variant"""
        # Use short keys to save space
        compressed = {
            "p": variant.position,  # chromosome is in chunk header
            "r": variant.reference,
            "a": variant.alternate,
            "g": variant.genotype,
        }

        # Only include optional fields if they differ from defaults
        if variant.quality > 0:
            compressed["q"] = round(variant.quality, 1)
        if variant.depth > 0:
            compressed["d"] = variant.depth
        if variant.allele_frequency > 0:
            compressed["f"] = round(variant.allele_frequency, 3)

        return compressed

    def decompress_profile(self, compressed: dict[str, Any]) -> GenomicProfile:
        """Decompress genomic profile"""
        variants = []

        for chunk in compressed["chunks"]:
            # Parse chunk boundaries
            start_chr = chunk["start_pos"].split(":")[0]

            for v_data in chunk["variants"]:
                variant = Variant(
                    chromosome=start_chr,
                    position=v_data["p"],
                    reference=v_data["r"],
                    alternate=v_data["a"],
                    genotype=v_data["g"],
                    quality=v_data.get("q", 0),
                    depth=v_data.get("d", 0),
                    allele_frequency=v_data.get("f", 0),
                )
                variants.append(variant)

        # Reconstruct quality metrics
        metrics = QualityMetrics(**compressed["metrics"])

        return GenomicProfile(
            sample_id=compressed["sample_id"],
            reference_genome=compressed["reference"],
            variants=variants,
            quality_metrics=metrics,
            processing_metadata={},
            checksum=compressed["checksum"],
        )
