"""
Enhanced Differential Encoding Pipeline with FASTQ Support

Extends the base pipeline to support FASTQ input with automatic region detection.

Key Features:
- Auto-detects input format (FASTQ vs VCF)
- FASTQ → Alignment → Region Detection → Multi-Ref Extraction → Differential Encoding
- VCF → Direct differential encoding (existing path)
- Maintains k-anonymity by extracting from all references
"""

import logging
from pathlib import Path
from typing import Any, List, Optional, Union
from enum import Enum

from genomevault.differential_encoding.pipeline import (
    DifferentialGenomicEncoder,
    EncodingResult,
)
from genomevault.differential_encoding.fastq_processor import (
    FASTQProcessor,
    AlignmentResult,
    GenomicRegion,
)
from genomevault.differential_encoding.region_extractor import (
    MultiReferenceExtractor,
    MultiReferenceRegion,
)
from genomevault.differential_encoding.reference_management import (
    SecureReferenceGenomeManager,
)
from genomevault.differential_encoding.chunking import (
    Genome,
    AnalysisType,
)

logger = logging.getLogger(__name__)


class InputFormat(Enum):
    """Supported input formats."""
    FASTQ_SINGLE = "fastq_single"
    FASTQ_PAIRED = "fastq_paired"
    VCF = "vcf"
    BAM = "bam"
    UNKNOWN = "unknown"


class EnhancedDifferentialEncodingPipeline:
    """
    Enhanced pipeline with FASTQ support and automatic region detection.

    Workflow for FASTQ:
    1. Detect input format
    2. If FASTQ: Align to reference genome
    3. Identify covered genomic regions
    4. Extract those regions from ALL references in pool
    5. Run differential encoding on extracted regions
    6. Generate hypervectors with k-anonymity guarantee

    Workflow for VCF (backward compatible):
    1. Use existing pipeline (already has coordinates)
    """

    def __init__(
        self,
        reference_genome: Path,
        reference_manager: SecureReferenceGenomeManager,
        dimension: int = 8192,
        enable_fastq: bool = True,
        min_coverage: float = 5.0,
        min_confidence: float = 0.7,
        blockchain_enabled: bool = False,
        attestation_registry: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize enhanced pipeline.

        Args:
            reference_genome: Path to reference genome FASTA (for alignment)
            reference_manager: Manager with reference pool
            dimension: Hypervector dimension
            enable_fastq: Enable FASTQ processing (requires minimap2/samtools)
            min_coverage: Minimum coverage for region detection (default 5.0)
            min_confidence: Minimum confidence for region detection (default 0.7)
            blockchain_enabled: Enable blockchain attestation recording
            attestation_registry: Optional AttestationRegistry instance
            **kwargs: Additional arguments for base pipeline
        """
        self.reference_manager = reference_manager
        self.dimension = dimension
        self.reference_genome = Path(reference_genome)
        self.enable_fastq = enable_fastq
        self.min_coverage = min_coverage
        self.min_confidence = min_confidence
        self.blockchain_enabled = blockchain_enabled
        self.attestation = attestation_registry

        # Initialize FASTQ processor if enabled
        if enable_fastq:
            try:
                from genomevault.differential_encoding.fastq_processor import create_default_processor
                self.fastq_processor = create_default_processor(reference_genome)
                logger.info("FASTQ processing enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize FASTQ processor: {e}")
                logger.warning("FASTQ processing will be disabled")
                self.enable_fastq = False
                self.fastq_processor = None
        else:
            self.fastq_processor = None

        # Initialize region extractor (only if reference_manager provided)
        if reference_manager is not None:
            self.region_extractor = MultiReferenceExtractor(reference_manager)
        else:
            self.region_extractor = None
            logger.warning("No reference manager provided - multi-reference extraction disabled")

        logger.info("Initialized EnhancedDifferentialEncodingPipeline")

    def encode_file(
        self,
        input_file: Path,
        input_file_r2: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ) -> EncodingResult:
        """
        Encode input file with automatic format detection.

        Args:
            input_file: Path to input file (FASTQ, VCF, BAM)
            input_file_r2: Optional R2 file for paired-end FASTQ
            output_dir: Optional output directory for intermediate files

        Returns:
            EncodingResult with hypervectors and metadata
        """
        # Detect input format
        input_format = self._detect_format(input_file, input_file_r2)

        logger.info(f"Detected input format: {input_format.value}")

        if input_format in [InputFormat.FASTQ_SINGLE, InputFormat.FASTQ_PAIRED]:
            return self._encode_fastq(input_file, input_file_r2, output_dir)
        elif input_format == InputFormat.VCF:
            return self._encode_vcf(input_file)
        elif input_format == InputFormat.BAM:
            return self._encode_bam(input_file, output_dir)
        else:
            raise ValueError(f"Unsupported input format: {input_format}")

    def _detect_format(
        self,
        file1: Path,
        file2: Optional[Path] = None,
    ) -> InputFormat:
        """Detect input file format from extension."""
        ext = file1.suffix.lower()

        if ext in ['.fastq', '.fq'] or (ext == '.gz' and file1.stem.endswith(('.fastq', '.fq'))):
            if file2 is not None:
                return InputFormat.FASTQ_PAIRED
            else:
                return InputFormat.FASTQ_SINGLE
        elif ext in ['.vcf'] or (ext == '.gz' and file1.stem.endswith('.vcf')):
            return InputFormat.VCF
        elif ext in ['.bam', '.sam']:
            return InputFormat.BAM
        else:
            return InputFormat.UNKNOWN

    def _encode_fastq(
        self,
        fastq_r1: Path,
        fastq_r2: Optional[Path],
        output_dir: Optional[Path],
    ) -> EncodingResult:
        """
        Encode FASTQ file(s) with region detection and multi-reference extraction.

        Complete workflow:
        1. Align FASTQ to reference genome
        2. Identify covered genomic regions
        3. Extract those regions from ALL references
        4. Differential encode using randomly selected reference
        5. Generate hypervectors
        """
        if not self.enable_fastq or self.fastq_processor is None:
            raise RuntimeError(
                "FASTQ processing not available. Install minimap2 and samtools: "
                "conda install -c bioconda minimap2 samtools bcftools"
            )

        logger.info("=== FASTQ Processing Workflow ===")

        # Step 1: Process FASTQ (align + identify regions)
        logger.info("Step 1: Processing FASTQ and identifying regions...")
        alignment_result = self.fastq_processor.process_fastq(
            fastq_r1=fastq_r1,
            fastq_r2=fastq_r2,
            output_dir=output_dir,
        )

        if not alignment_result.regions:
            raise ValueError("No genomic regions detected from FASTQ input")

        # Get primary region (highest coverage)
        primary_region = alignment_result.get_primary_region()
        logger.info(f"Primary region identified: {primary_region}")

        # Step 2: Extract this region from ALL references
        logger.info("Step 2: Extracting region from all references...")
        multi_ref_region = self.region_extractor.extract_region(primary_region)

        logger.info(
            f"Extracted from {multi_ref_region.num_references} references: "
            f"{', '.join(multi_ref_region.get_reference_ids())}"
        )

        # Step 3: Run differential encoding
        logger.info("Step 3: Running differential encoding...")

        # Convert to format expected by base pipeline
        # Use VCF file if variants were called, otherwise use alignment coordinates
        if alignment_result.vcf_file and alignment_result.vcf_file.exists():
            logger.info("Using called variants for differential encoding")
            # Use base pipeline's VCF encoding path
            result = self._encode_vcf_with_regions(
                vcf_file=alignment_result.vcf_file,
                multi_ref_region=multi_ref_region,
            )
        else:
            logger.info("Using alignment-based differential encoding")
            # Use extracted reference sections directly
            result = self._encode_from_sections(multi_ref_region)

        logger.info("=== FASTQ Encoding Complete ===")
        logger.info(f"Generated {len(result.hypervectors)} hypervector(s)")
        logger.info(f"k-anonymity: k={multi_ref_region.num_references}")

        # Optional blockchain attestation
        if self.blockchain_enabled and self.attestation:
            try:
                import time
                encoding_id = f"fastq_{fastq_r1.stem}_{int(time.time())}"
                tx_hash = self.attestation.record_encoding(
                    encoding_id=encoding_id,
                    input_data=str(fastq_r1),  # Hash file path
                    output_data=result.hypervectors,
                    metadata={
                        "compression_ratio": self.dimension / len(result.hypervectors) if result.hypervectors else 1.0,
                        "k_anonymity": multi_ref_region.num_references,
                        "dimension": self.dimension,
                        "input_format": "fastq",
                        "num_regions": len(alignment_result.regions),
                    }
                )
                logger.info(f"Blockchain attestation recorded: {tx_hash}")
            except Exception as e:
                logger.warning(f"Failed to record blockchain attestation: {e}")

        return result

    def _encode_vcf(self, vcf_file: Path) -> EncodingResult:
        """
        Encode VCF file using existing pipeline (backward compatible).

        VCF already contains chromosome + coordinates, so we can use
        the base pipeline directly.
        """
        logger.info("Using existing VCF encoding path (backward compatible)")

        # Load VCF into Genome object
        genome = self._load_genome_from_vcf(vcf_file)

        # Use base pipeline's differential encoder
        from genomevault.differential_encoding.pipeline import DifferentialGenomicEncoder
        from genomevault.differential_encoding.hypervector_encoder import DifferentialHypervectorEncoder
        from genomevault.differential_encoding.crypto_primitives import CryptoRNG

        encoder = DifferentialGenomicEncoder(
            reference_manager=self.reference_manager,
            hypervector_encoder=DifferentialHypervectorEncoder(dimension=self.dimension),
            crypto_rng=CryptoRNG(),
        )

        # Encode with sliding window analysis (default)
        result = encoder.encode_experimental_genome(
            experimental_genome=genome,
            analysis_type=AnalysisType.SLIDING_WINDOW,
            bundle_chunks=True,
        )

        logger.info(f"VCF encoding complete: {len(result.hypervectors)} chunks")

        # Optional blockchain attestation
        if self.blockchain_enabled and self.attestation:
            try:
                import time
                encoding_id = f"vcf_{vcf_file.stem}_{int(time.time())}"
                num_variants = sum(len(v) for v in genome.chromosomes.values())
                tx_hash = self.attestation.record_encoding(
                    encoding_id=encoding_id,
                    input_data=str(vcf_file),
                    output_data=result.hypervectors,
                    metadata={
                        "compression_ratio": num_variants / len(result.hypervectors) if result.hypervectors else 1.0,
                        "dimension": self.dimension,
                        "input_format": "vcf",
                        "num_variants": num_variants,
                        "num_chunks": len(result.hypervectors),
                    }
                )
                logger.info(f"Blockchain attestation recorded: {tx_hash}")
            except Exception as e:
                logger.warning(f"Failed to record blockchain attestation: {e}")

        return result

    def _encode_vcf_with_regions(
        self,
        vcf_file: Path,
        multi_ref_region: MultiReferenceRegion,
    ) -> EncodingResult:
        """
        Encode VCF with explicit region constraints from multi-reference extraction.

        This method is used when we have both:
        1. Variant calls from FASTQ (VCF file)
        2. Pre-extracted regions from all references (for k-anonymity)

        The VCF provides the experimental variants, and the multi_ref_region
        ensures we only use references that have the same region extracted.
        """
        logger.info("Encoding VCF with multi-reference region constraints")

        # Load VCF into Genome object
        genome = self._load_genome_from_vcf(vcf_file)

        # Filter genome to only include variants in the extracted region
        filtered_genome = self._filter_genome_to_region(genome, multi_ref_region)

        # Use base pipeline's differential encoder
        from genomevault.differential_encoding.pipeline import DifferentialGenomicEncoder
        from genomevault.differential_encoding.hypervector_encoder import DifferentialHypervectorEncoder
        from genomevault.differential_encoding.crypto_primitives import CryptoRNG

        encoder = DifferentialGenomicEncoder(
            reference_manager=self.reference_manager,
            hypervector_encoder=DifferentialHypervectorEncoder(dimension=self.dimension),
            crypto_rng=CryptoRNG(),
        )

        # Encode with gene region analysis (appropriate for targeted regions)
        result = encoder.encode_experimental_genome(
            experimental_genome=filtered_genome,
            analysis_type=AnalysisType.GENE_REGION,
            bundle_chunks=True,
        )

        logger.info(
            f"Region-constrained encoding complete: "
            f"{len(result.hypervectors)} chunks, "
            f"k={multi_ref_region.num_references}"
        )
        return result

    def _encode_from_sections(
        self,
        multi_ref_region: MultiReferenceRegion,
    ) -> EncodingResult:
        """
        Encode directly from extracted reference sections.

        Uses the GenomeSection objects from all references to perform
        differential encoding.

        This method is used when we have reference sections but no VCF file
        (e.g., alignment-only mode without variant calling).

        Note: For full differential encoding, we need experimental variants.
        Without a VCF, we can only encode the reference sequences themselves.
        This is primarily useful for testing and reference pool validation.
        """
        logger.info("Encoding from extracted reference sections")

        # Create a simple Genome object from the first reference section
        # This represents the "experimental" genome for encoding purposes
        first_ref_id = list(multi_ref_region.reference_sections.keys())[0]
        first_section = multi_ref_region.reference_sections[first_ref_id]

        # Create Genome object with variants from first section
        genome = Genome(
            genome_id=f"extracted_{multi_ref_region.chromosome}",
            assembly="GRCh38",  # Default assembly
            chromosomes={
                multi_ref_region.chromosome: list(first_section.variants)
            },
            metadata={
                "source": "reference_section",
                "region": f"{multi_ref_region.chromosome}:{multi_ref_region.start}-{multi_ref_region.end}",
                "k_anonymity": multi_ref_region.num_references,
            }
        )

        # Use base pipeline's differential encoder
        from genomevault.differential_encoding.pipeline import DifferentialGenomicEncoder
        from genomevault.differential_encoding.hypervector_encoder import DifferentialHypervectorEncoder
        from genomevault.differential_encoding.crypto_primitives import CryptoRNG

        encoder = DifferentialGenomicEncoder(
            reference_manager=self.reference_manager,
            hypervector_encoder=DifferentialHypervectorEncoder(dimension=self.dimension),
            crypto_rng=CryptoRNG(),
        )

        # Encode with gene region analysis (single region)
        result = encoder.encode_experimental_genome(
            experimental_genome=genome,
            analysis_type=AnalysisType.GENE_REGION,
            bundle_chunks=True,
        )

        logger.info(
            f"Section-based encoding complete: "
            f"{len(result.hypervectors)} chunks, "
            f"k={multi_ref_region.num_references}"
        )
        return result

    def _encode_bam(
        self,
        bam_file: Path,
        output_dir: Optional[Path],
    ) -> EncodingResult:
        """
        Encode from existing BAM alignment file.

        Skips alignment step but still detects regions and extracts
        from all references.
        """
        logger.info("Processing existing BAM file...")

        # Create FASTQProcessor to reuse region detection logic
        processor = self.fastq_processor

        # Identify regions from BAM
        regions = processor._identify_regions(bam_file)

        if not regions:
            raise ValueError("No genomic regions detected from BAM input")

        # Get primary region
        primary_region = max(regions, key=lambda r: r.coverage)
        logger.info(f"Primary region: {primary_region}")

        # Extract from all references
        multi_ref_region = self.region_extractor.extract_region(primary_region)

        # Encode
        return self._encode_from_sections(multi_ref_region)

    def _load_genome_from_vcf(self, vcf_file: Path) -> Genome:
        """
        Load Genome object from VCF file.

        Args:
            vcf_file: Path to VCF file (can be gzipped)

        Returns:
            Genome object with variants from VCF
        """
        import gzip

        logger.info(f"Loading genome from VCF: {vcf_file}")

        variants_by_chr = {}

        # Open file (handle both gzipped and plain text)
        if vcf_file.suffix == '.gz':
            opener = lambda: gzip.open(vcf_file, 'rt')
        else:
            opener = lambda: open(vcf_file, 'r')

        with opener() as f:
            for line in f:
                # Skip headers
                if line.startswith('#'):
                    continue

                fields = line.strip().split('\t')
                if len(fields) < 5:
                    continue

                chrom = fields[0]
                pos = int(fields[1])
                ref = fields[3]
                alt = fields[4]

                # Extract genotype if available
                genotype = '0/1'  # Default
                if len(fields) >= 10:
                    # Parse FORMAT and sample columns
                    format_fields = fields[8].split(':')
                    sample_values = fields[9].split(':')
                    if 'GT' in format_fields:
                        gt_idx = format_fields.index('GT')
                        if gt_idx < len(sample_values):
                            genotype = sample_values[gt_idx]

                # Create Variant object
                if chrom not in variants_by_chr:
                    variants_by_chr[chrom] = []

                from genomevault.differential_encoding.reference_management import Variant

                variants_by_chr[chrom].append(Variant(
                    chromosome=chrom,
                    position=pos,
                    ref=ref,
                    alt=alt,
                    genotype=genotype,
                    quality=99.0,  # Default quality
                ))

        # Create Genome object
        genome_id = vcf_file.stem.replace('.vcf', '')
        if genome_id.endswith('.gz'):
            genome_id = genome_id[:-3]

        genome = Genome(
            genome_id=genome_id,
            assembly="GRCh38",  # Default assembly
            chromosomes=variants_by_chr,
            metadata={"source": str(vcf_file)}
        )

        logger.info(
            f"Loaded genome '{genome_id}': "
            f"{len(variants_by_chr)} chromosomes, "
            f"{sum(len(v) for v in variants_by_chr.values())} variants"
        )

        return genome

    def _filter_genome_to_region(
        self,
        genome: Genome,
        region: MultiReferenceRegion,
    ) -> Genome:
        """
        Filter genome to only include variants within specified region.

        Args:
            genome: Full genome object
            region: Region to filter to

        Returns:
            Filtered genome object
        """
        logger.debug(
            f"Filtering genome to region: "
            f"{region.chromosome}:{region.start}-{region.end}"
        )

        # Filter variants
        filtered_chromosomes = {}
        if region.chromosome in genome.chromosomes:
            filtered_variants = [
                v for v in genome.chromosomes[region.chromosome]
                if region.start <= v.position <= region.end
            ]
            filtered_chromosomes[region.chromosome] = filtered_variants

        # Create filtered genome
        filtered_genome = Genome(
            genome_id=genome.genome_id,
            assembly=genome.assembly,
            chromosomes=filtered_chromosomes,
            metadata={
                **genome.metadata,
                "filtered_region": f"{region.chromosome}:{region.start}-{region.end}",
            }
        )

        logger.debug(
            f"Filtered genome: {len(filtered_chromosomes.get(region.chromosome, []))} variants "
            f"in region"
        )

        return filtered_genome


def create_enhanced_pipeline(
    reference_genome: Path,
    reference_pool_dir: Path,
    dimension: int = 8192,
    blockchain_config: Optional[dict] = None,
    **kwargs
) -> EnhancedDifferentialEncodingPipeline:
    """
    Create enhanced pipeline with default configuration.

    Args:
        reference_genome: Path to reference genome FASTA (for alignment)
        reference_pool_dir: Directory containing reference genomes
        dimension: Hypervector dimension
        blockchain_config: Optional blockchain configuration dictionary
        **kwargs: Additional pipeline arguments

    Returns:
        Configured EnhancedDifferentialEncodingPipeline

    Example:
        >>> pipeline = create_enhanced_pipeline(
        ...     reference_genome=Path("data/reference/chr22.fa"),
        ...     reference_pool_dir=Path("data/references/"),
        ... )
        >>> result = pipeline.encode_file(Path("sample.fastq.gz"))

    Example with blockchain:
        >>> blockchain_config = {
        ...     "enabled": True,
        ...     "network": "polygon-mumbai",
        ...     "contract_address": "0x...",
        ... }
        >>> pipeline = create_enhanced_pipeline(
        ...     reference_genome=Path("data/reference/chr22.fa"),
        ...     reference_pool_dir=Path("data/references/"),
        ...     blockchain_config=blockchain_config,
        ... )
    """
    # Load reference manager
    ref_manager = SecureReferenceGenomeManager(reference_pool_dir)

    # Initialize blockchain if configured
    blockchain_enabled = False
    attestation_registry = None

    if blockchain_config and blockchain_config.get("enabled", False):
        try:
            from genomevault.blockchain.attestation_registry import create_attestation_registry

            attestation_registry = create_attestation_registry(blockchain_config)
            blockchain_enabled = attestation_registry.blockchain_enabled

            logger.info(f"Blockchain attestation {'enabled' if blockchain_enabled else 'disabled (offline mode)'}")
        except Exception as e:
            logger.warning(f"Failed to initialize blockchain: {e}")
            blockchain_enabled = False
            attestation_registry = None

    # Create pipeline
    pipeline = EnhancedDifferentialEncodingPipeline(
        reference_genome=reference_genome,
        reference_manager=ref_manager,
        dimension=dimension,
        enable_fastq=True,
        blockchain_enabled=blockchain_enabled,
        attestation_registry=attestation_registry,
        **kwargs
    )

    return pipeline
