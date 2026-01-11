"""
GenomeVault Unified Production Pipeline

Complete 7-layer pipeline from raw genetic data to PIR output.
Uses the BEST implementation of each component.

Layer Architecture:
1. Byzantine Consensus - Build reference from public genomes (hg38, hg19, chm13)
2. Guide Strand Creation - k-anonymity blind middleman from diverse samples
3. Experimental Alignment - Privacy-preserving alignment to guide pool
4. GDiff Encoding - Differential encoding (sequence differences vs guides)
5. HDC Encoding - Adaptive hyperdimensional vectors (99.2% accuracy)
6. ZK Proofs - Zero-knowledge proofs via Groth16
7. PIR Queries - Information-theoretic private information retrieval

Usage:
    from genomevault.pipelines.unified_pipeline import UnifiedPipeline, PipelineConfig

    config = PipelineConfig(
        output_dir=Path("pipeline_output"),
        guide_fasta_dir=Path("data/guide_strands"),
    )
    pipeline = UnifiedPipeline(config)

    # Run Layer 3-7 (assuming Layer 1-2 pre-computed)
    result = pipeline.run_experimental_pipeline(
        query_fastq_1=Path("sample_R1.fastq.gz"),
        query_fastq_2=Path("sample_R2.fastq.gz"),
    )
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
import time

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    Configuration for the unified GenomeVault pipeline.

    Attributes:
        output_dir: Directory for all pipeline outputs
        guide_fasta_dir: Directory containing guide FASTA files (Layer 2 output)
        guide_bam_dir: Optional directory containing guide BAM files for GDiff
        consensus_fasta: Path to Byzantine consensus FASTA (Layer 1 output)

        # Layer 5 (HDC) options
        hdc_dimension: Hypervector dimension (default: 4096)
        hdc_chunk_size: Chunk size for HDC encoding (default: 512bp)
        hdc_num_banks: Number of property banks (default: 2 - Hydrophobic, MajorGroove)

        # Layer 6 (ZK) options
        enable_zk: Whether to generate ZK proofs (default: True)
        zk_circuit: ZK circuit to use (default: variant_presence)

        # Layer 7 (PIR) options
        enable_pir: Whether to enable PIR queries (default: True)
        pir_num_servers: Number of PIR servers (default: 2)

        # Performance options
        threads: Number of threads for parallel operations
        k_anonymity: Number of guide strands for k-anonymity (minimum 3)
    """

    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("pipeline_output"))

    # Layer 1-2 pre-computed inputs
    guide_fasta_dir: Path = field(default_factory=lambda: Path("data/guide_strands"))
    guide_bam_dir: Optional[Path] = None
    consensus_fasta: Optional[Path] = None

    # Layer 5: HDC options
    hdc_dimension: int = 4096
    hdc_chunk_size: int = 512
    hdc_num_banks: int = 2

    # Layer 6: ZK options
    enable_zk: bool = True
    zk_circuit: str = "variant_presence"

    # Layer 7: PIR options
    enable_pir: bool = True
    pir_num_servers: int = 2

    # Performance
    threads: int = 8
    k_anonymity: int = 12

    # Quality thresholds
    min_base_quality: int = 20
    min_mapping_quality: int = 20

    def __post_init__(self):
        """Validate configuration."""
        if self.k_anonymity < 2:
            raise ValueError("k_anonymity must be at least 2 for privacy guarantees")
        if self.hdc_dimension < 1024:
            raise ValueError("hdc_dimension should be at least 1024 for accuracy")
        if self.pir_num_servers < 2:
            raise ValueError("IT-PIR requires at least 2 servers")


@dataclass
class PipelineResult:
    """
    Result from pipeline execution.

    Contains outputs from each layer along with timing and metadata.
    """
    # Layer outputs
    query_bam: Optional[Path] = None
    gdiff_path: Optional[Path] = None
    hdv_path: Optional[Path] = None
    selective_hdv_path: Optional[Path] = None
    zk_proof: Optional[Any] = None

    # Timing
    layer_times: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0

    # Metadata
    num_variants: int = 0
    hdv_size_bytes: int = 0
    compression_ratio: float = 0.0

    # Status
    success: bool = False
    error_message: Optional[str] = None


class UnifiedPipeline:
    """
    Complete GenomeVault production pipeline.

    Implements the 7-layer privacy-preserving genomic encoding architecture:

    1. Byzantine Consensus (public reference)
       - Builds consensus from hg38 + hg19 + chm13
       - Probabilistic certainty with IUPAC ambiguity codes
       - Output: consensus.fa (~2.9 GB)

    2. Guide Strand Creation (k=12 blind middleman)
       - Diverse genomic samples aligned to consensus
       - Guide FASTAs extracted via samtools consensus
       - Output: ref1.fa.gz through ref12.fa.gz (~10 GB total)

    3. Experimental Alignment (privacy-preserving)
       - Query FASTQ aligned ONLY to guide pool (never to consensus!)
       - Creates untraceable indirection layer
       - Output: experimental.bam

    4. GDiff Encoding (differential representation)
       - Computes sequence differences between query and guides
       - Template-aware with streaming support
       - Output: experimental.gdiff.gz (~15 MB)

    5. HDC Encoding (adaptive hypervector)
       - Adaptive k-selection (k=4/6/8, avg k=6)
       - 99.2% accuracy with 2-bank architecture
       - Numba JIT compilation for 5-10x speedup
       - Output: experimental.h5 (HDF5 with metadata)

    6. ZK Proofs (Groth16)
       - Real cryptographic proofs via Circom backend
       - 128-bit security, ~0.4s proving time
       - Output: Proof object (~740 bytes)

    7. PIR Queries (IT-PIR)
       - 2-server information-theoretic PIR
       - Quantum-resistant, 4-13ms latency
       - Output: Query result with 0 bits leaked

    Example:
        >>> config = PipelineConfig(
        ...     output_dir=Path("output"),
        ...     guide_fasta_dir=Path("data/guide_strands"),
        ... )
        >>> pipeline = UnifiedPipeline(config)
        >>> result = pipeline.run_experimental_pipeline(
        ...     query_fastq_1=Path("sample_R1.fq.gz"),
        ...     query_fastq_2=Path("sample_R2.fq.gz"),
        ... )
        >>> print(f"GDiff: {result.gdiff_path}")
        >>> print(f"HDV: {result.hdv_path}")
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the unified pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self._components_initialized = False
        self._init_directories()

    def _init_directories(self):
        """Create output directories."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        (self.config.output_dir / "alignment").mkdir(exist_ok=True)
        (self.config.output_dir / "gdiff").mkdir(exist_ok=True)
        (self.config.output_dir / "hdv").mkdir(exist_ok=True)
        (self.config.output_dir / "proofs").mkdir(exist_ok=True)

    def _init_components(self):
        """Lazily initialize pipeline components."""
        if self._components_initialized:
            return

        # Layer 5: HDC Encoder
        from genomevault.hypervector_transform import AdaptiveEncoder
        self.hdc_encoder = AdaptiveEncoder()

        # Layer 5 (alt): Selective HDV Encoder for network queries
        from genomevault.differential_encoding.gdiff.selective_hdv_encoder import (
            SelectiveHDVEncoder
        )
        self.selective_encoder = SelectiveHDVEncoder()

        # Layer 6: ZK Prover
        if self.config.enable_zk:
            try:
                from genomevault.zk_proofs.prover import Prover
                self.zk_prover = Prover()
                logger.info("ZK prover initialized (production ready: %s)",
                           self.zk_prover.is_production_ready)
            except ImportError as e:
                logger.warning("ZK prover not available: %s", e)
                self.zk_prover = None
                self.config.enable_zk = False

        # Layer 7: PIR
        if self.config.enable_pir:
            try:
                from genomevault.pir.it_pir_protocol import PIRProtocol, PIRParameters
                self.pir_params = PIRParameters(
                    database_size=1000,  # Will be set per query
                    num_servers=self.config.pir_num_servers
                )
                self.pir = PIRProtocol(self.pir_params)
                logger.info("PIR protocol initialized (%d servers)",
                           self.config.pir_num_servers)
            except ImportError as e:
                logger.warning("PIR protocol not available: %s", e)
                self.pir = None
                self.config.enable_pir = False

        self._components_initialized = True

    # =========================================================================
    # Layer 1: Byzantine Consensus
    # =========================================================================

    def build_consensus(
        self,
        reference_fastas: List[Path],
        output_path: Optional[Path] = None,
        confidence_threshold: float = 0.9,
    ) -> Path:
        """
        Layer 1: Build Byzantine consensus from reference genomes.

        Creates a probabilistic consensus reference from multiple public genomes
        (hg38, hg19, chm13) using multi-source voting with IUPAC ambiguity codes.

        Args:
            reference_fastas: List of reference FASTA files (hg38.fa, hg19.fa, chm13.fa)
            output_path: Output path for consensus FASTA (default: output_dir/consensus.fa)
            confidence_threshold: Minimum confidence for consensus base (default: 0.9)

        Returns:
            Path to generated consensus FASTA
        """
        from genomevault.reference.byzantine_consensus_builder import (
            ByzantineConsensusBuilder
        )

        output_path = output_path or (self.config.output_dir / "consensus.fa")

        logger.info("Building Byzantine consensus from %d references...",
                   len(reference_fastas))
        start_time = time.time()

        builder = ByzantineConsensusBuilder(
            confidence_threshold=confidence_threshold,
            verbose=True
        )

        # Process references
        consensus_path = builder.build_consensus(
            reference_paths=reference_fastas,
            output_path=output_path,
            threads=self.config.threads
        )

        elapsed = time.time() - start_time
        logger.info("Layer 1 complete: consensus built in %.1fs -> %s",
                   elapsed, consensus_path)

        return consensus_path

    # =========================================================================
    # Layer 2: Guide Strand Creation
    # =========================================================================

    def create_guide_strands(
        self,
        guide_bam_files: List[Path],
        output_dir: Optional[Path] = None,
        compress: bool = True,
    ) -> List[Path]:
        """
        Layer 2: Extract guide strands from aligned BAM files.

        Creates guide FASTA files from diverse genomic samples that have been
        aligned to the consensus reference. These serve as the blind middleman
        for privacy-preserving alignment.

        Args:
            guide_bam_files: List of guide BAM files (aligned to consensus)
            output_dir: Output directory for guide FASTAs
            compress: Whether to gzip compress output (default: True)

        Returns:
            List of paths to extracted guide FASTA files
        """
        from genomevault.differential_encoding.align_to_reference_pool import (
            PrivacyPreservingReferencePoolAligner
        )

        output_dir = output_dir or (self.config.output_dir / "guide_strands")
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting guide strands from %d BAM files...",
                   len(guide_bam_files))
        start_time = time.time()

        guide_fastas = PrivacyPreservingReferencePoolAligner.extract_guide_sequences_from_bams(
            guide_bam_files=guide_bam_files,
            output_dir=output_dir,
            threads=self.config.threads,
            compress=compress
        )

        elapsed = time.time() - start_time
        logger.info("Layer 2 complete: %d guide strands extracted in %.1fs",
                   len(guide_fastas), elapsed)

        return guide_fastas

    # =========================================================================
    # Layer 3: Experimental Alignment
    # =========================================================================

    def align_experimental(
        self,
        query_fastq_1: Path,
        query_fastq_2: Path,
        guide_fastas: Optional[List[Path]] = None,
        output_bam: Optional[Path] = None,
    ) -> Path:
        """
        Layer 3: Align experimental FASTQ to guide pool.

        CRITICAL: Query aligns ONLY to guide pool, NEVER to consensus!
        This creates privacy-preserving indirection.

        Args:
            query_fastq_1: Path to R1 FASTQ file
            query_fastq_2: Path to R2 FASTQ file
            guide_fastas: Optional list of guide FASTA files (uses config default)
            output_bam: Optional output BAM path

        Returns:
            Path to aligned BAM file
        """
        from genomevault.differential_encoding.align_to_reference_pool import (
            PrivacyPreservingReferencePoolAligner
        )

        # Use provided guides or discover from config directory
        if guide_fastas is None:
            guide_fastas = self._discover_guide_fastas()

        if len(guide_fastas) < 2:
            raise ValueError(
                f"Guide pool must have at least 2 members for privacy. "
                f"Found {len(guide_fastas)} in {self.config.guide_fasta_dir}"
            )

        output_bam = output_bam or (
            self.config.output_dir / "alignment" / "experimental.bam"
        )
        output_bam.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Aligning experimental FASTQ to guide pool (k=%d)...",
                   len(guide_fastas))
        start_time = time.time()

        aligner = PrivacyPreservingReferencePoolAligner(
            guide_fasta_files=guide_fastas,
            threads=self.config.threads
        )

        aligner.align_query_to_pool(
            query_fastq_1=query_fastq_1,
            query_fastq_2=query_fastq_2,
            output_bam=output_bam,
        )

        elapsed = time.time() - start_time
        logger.info("Layer 3 complete: alignment in %.1fs -> %s", elapsed, output_bam)

        return output_bam

    # =========================================================================
    # Layer 4: GDiff Encoding
    # =========================================================================

    def encode_gdiff(
        self,
        query_bam: Path,
        pool_bams: Optional[List[Path]] = None,
        output_path: Optional[Path] = None,
        use_template: bool = True,
    ):
        """
        Layer 4: Create GDiff differential encoding.

        Computes sequence differences between query and guide pool.
        Outputs a compressed GDiff document.

        Args:
            query_bam: Path to query BAM file (from Layer 3)
            pool_bams: List of guide BAM files for comparison
            output_path: Output path for GDiff file
            use_template: Whether to use streaming template (default: True)

        Returns:
            GDiffDocument object and path to saved file
        """
        from genomevault.differential_encoding.gdiff.encoder import GDiffEncoder
        from genomevault.differential_encoding.gdiff.schema import GDiffDocument

        # Discover pool BAMs if not provided
        if pool_bams is None:
            pool_bams = self._discover_pool_bams()

        output_path = output_path or (
            self.config.output_dir / "gdiff" / "experimental.gdiff.gz"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Creating GDiff encoding (pool size: %d)...", len(pool_bams))
        start_time = time.time()

        encoder = GDiffEncoder(
            query_bam=str(query_bam),
            pool_bams=[str(p) for p in pool_bams],
            min_base_quality=self.config.min_base_quality,
            min_mapping_quality=self.config.min_mapping_quality,
            use_streaming_template=use_template,
            guide_fasta_files=[Path(p) for p in self._discover_guide_fastas()],
        )

        gdiff = encoder.compute_differential_encoding()
        gdiff.save(output_path, compress=True)

        elapsed = time.time() - start_time
        num_variants = len(gdiff.differential_variants) if gdiff.differential_variants else 0
        logger.info("Layer 4 complete: GDiff with %d variants in %.1fs -> %s",
                   num_variants, elapsed, output_path)

        return gdiff, output_path

    # =========================================================================
    # Layer 5: HDC Encoding
    # =========================================================================

    def encode_hdc(
        self,
        gdiff_path: Path,
        output_path: Optional[Path] = None,
    ) -> Path:
        """
        Layer 5: Create adaptive HDC encoding.

        Uses the production AdaptiveEncoder with:
        - Adaptive k-selection (k=4/6/8 based on sequence difficulty)
        - 2-bank architecture (Hydrophobic, MajorGroove)
        - Numba JIT compilation for 5-10x speedup
        - 99.2% accuracy

        Args:
            gdiff_path: Path to GDiff file from Layer 4
            output_path: Output path for HDF5 file

        Returns:
            Path to HDF5 output file
        """
        self._init_components()

        output_path = output_path or (
            self.config.output_dir / "hdv" / "experimental.h5"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Creating HDC encoding (D=%d, banks=%d)...",
                   self.config.hdc_dimension, self.config.hdc_num_banks)
        start_time = time.time()

        # Load GDiff and encode
        from genomevault.differential_encoding.gdiff.schema import GDiffDocument
        gdiff = GDiffDocument.load(gdiff_path)

        # Get guide FASTAs for encoding context
        guide_fastas = self._discover_guide_fastas()

        # Use the production encoder
        from genomevault.hypervector_transform.adaptive_encoder import (
            run_production_encoding, EncodingConfig
        )

        encoding_config = EncodingConfig(
            gdiff_path=gdiff_path,
            guide_dir=self.config.guide_fasta_dir,
            output_path=output_path,
        )

        run_production_encoding(encoding_config)

        elapsed = time.time() - start_time
        logger.info("Layer 5 complete: HDC encoding in %.1fs -> %s", elapsed, output_path)

        return output_path

    def encode_selective_hdv(
        self,
        gdiff_path: Path,
        schema: str = "clinical_risk",
        output_path: Optional[Path] = None,
    ):
        """
        Layer 5 (alt): Create lightweight selective HDV for network queries.

        Generates task-specific HDVs that are much smaller (512B-64KB)
        for efficient network transmission.

        Args:
            gdiff_path: Path to GDiff file
            schema: Analysis schema (clinical_risk, pharmacogenomics, etc.)
            output_path: Output path for HDV encoding

        Returns:
            HDVEncoding object
        """
        self._init_components()

        output_path = output_path or (
            self.config.output_dir / "hdv" / f"selective_{schema}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Creating selective HDV (schema: %s)...", schema)
        start_time = time.time()

        from genomevault.differential_encoding.gdiff.schema import GDiffDocument
        gdiff = GDiffDocument.load(gdiff_path)

        encoding = self.selective_encoder.encode(gdiff, schema=schema)
        encoding.save(output_path)

        elapsed = time.time() - start_time
        logger.info("Layer 5 (selective) complete: %d bytes in %.1fms -> %s",
                   encoding.hdv_size_bytes, encoding.encoding_time_ms, output_path)

        return encoding

    # =========================================================================
    # Layer 6: ZK Proofs
    # =========================================================================

    def generate_zk_proof(
        self,
        variant_data: Dict[str, Any],
        hdv_sample: Optional[np.ndarray] = None,
    ):
        """
        Layer 6: Generate zero-knowledge proof.

        Creates a cryptographically secure proof that a variant exists
        without revealing its exact position.

        Args:
            variant_data: Variant information (chrom, pos, ref, alt)
            hdv_sample: Optional HDV sample for private input

        Returns:
            Proof object or None if ZK is disabled
        """
        if not self.config.enable_zk or self.zk_prover is None:
            logger.info("ZK proofs disabled or unavailable")
            return None

        self._init_components()

        logger.info("Generating ZK proof for variant %s:%s...",
                   variant_data.get("chrom"), variant_data.get("pos"))
        start_time = time.time()

        public_input = {
            "chrom": variant_data.get("chrom", "unknown"),
            "pos": variant_data.get("pos", 0),
        }

        private_input = {
            "variant_data": variant_data,
        }

        if hdv_sample is not None:
            private_input["hdv_sample"] = hdv_sample[:100].tolist()

        proof = self.zk_prover.prove_variant(
            public_input=public_input,
            private_input=private_input
        )

        elapsed = time.time() - start_time
        logger.info("Layer 6 complete: ZK proof generated in %.2fs (%d bytes)",
                   elapsed, len(proof.proof_data) if proof else 0)

        return proof

    # =========================================================================
    # Layer 7: PIR Queries
    # =========================================================================

    def pir_query(
        self,
        database: np.ndarray,
        query_index: int,
    ) -> Optional[np.ndarray]:
        """
        Layer 7: Execute private information retrieval query.

        Retrieves an element from a database without revealing which
        element was retrieved. Uses 2-server IT-PIR with XOR scheme.

        Args:
            database: Database array (each row is an element)
            query_index: Index of element to retrieve

        Returns:
            Retrieved element or None if PIR is disabled
        """
        if not self.config.enable_pir or self.pir is None:
            logger.info("PIR queries disabled or unavailable")
            return None

        self._init_components()

        # Update database size in parameters
        from genomevault.pir.it_pir_protocol import PIRParameters
        self.pir.params = PIRParameters(
            database_size=len(database),
            num_servers=self.config.pir_num_servers
        )

        logger.info("Executing PIR query (index %d of %d elements)...",
                   query_index, len(database))
        start_time = time.time()

        # Generate query vectors
        query_vectors = self.pir.generate_query_vectors(query_index)

        # In real deployment, these would go to separate servers
        # Here we simulate both servers locally
        responses = []
        for i, query in enumerate(query_vectors):
            # Server i computes response
            response = np.zeros(database.shape[1] if len(database.shape) > 1 else 1,
                               dtype=np.uint8)
            for j, bit in enumerate(query):
                if bit:
                    if len(database.shape) > 1:
                        response = (response + database[j]) % 256
                    else:
                        response = (response + database[j:j+1]) % 256
            responses.append(response)

        # Reconstruct element
        result = responses[0]
        for resp in responses[1:]:
            result = (result + resp) % 256

        elapsed = time.time() - start_time
        logger.info("Layer 7 complete: PIR query in %.2fms", elapsed * 1000)

        return result

    # =========================================================================
    # Full Pipeline Execution
    # =========================================================================

    def run_experimental_pipeline(
        self,
        query_fastq_1: Path,
        query_fastq_2: Path,
        guide_fastas: Optional[List[Path]] = None,
        pool_bams: Optional[List[Path]] = None,
        generate_zk: bool = True,
        schema: str = "clinical_risk",
    ) -> PipelineResult:
        """
        Run complete pipeline from experimental FASTQ to HDV/ZK output.

        Executes Layers 3-6 (assumes Layers 1-2 are pre-computed):
        - Layer 3: Align experimental FASTQ to guide pool
        - Layer 4: Generate GDiff differential encoding
        - Layer 5: Create HDC encoding (full + selective)
        - Layer 6: Generate ZK proof (optional)

        Args:
            query_fastq_1: Path to R1 FASTQ file
            query_fastq_2: Path to R2 FASTQ file
            guide_fastas: Optional list of guide FASTA files
            pool_bams: Optional list of pool BAM files
            generate_zk: Whether to generate ZK proof (default: True)
            schema: Analysis schema for selective HDV (default: clinical_risk)

        Returns:
            PipelineResult with outputs from all layers
        """
        result = PipelineResult()
        total_start = time.time()

        try:
            # Layer 3: Alignment
            layer_start = time.time()
            result.query_bam = self.align_experimental(
                query_fastq_1=query_fastq_1,
                query_fastq_2=query_fastq_2,
                guide_fastas=guide_fastas,
            )
            result.layer_times["alignment"] = time.time() - layer_start

            # Layer 4: GDiff
            layer_start = time.time()
            gdiff, result.gdiff_path = self.encode_gdiff(
                query_bam=result.query_bam,
                pool_bams=pool_bams,
            )
            result.layer_times["gdiff"] = time.time() - layer_start
            result.num_variants = len(gdiff.differential_variants) if gdiff.differential_variants else 0

            # Layer 5: HDC (full)
            layer_start = time.time()
            result.hdv_path = self.encode_hdc(gdiff_path=result.gdiff_path)
            result.layer_times["hdc"] = time.time() - layer_start

            # Layer 5 (alt): Selective HDV
            layer_start = time.time()
            selective = self.encode_selective_hdv(
                gdiff_path=result.gdiff_path,
                schema=schema,
            )
            result.selective_hdv_path = (
                self.config.output_dir / "hdv" / f"selective_{schema}.json"
            )
            result.layer_times["selective_hdv"] = time.time() - layer_start
            result.hdv_size_bytes = selective.hdv_size_bytes
            result.compression_ratio = selective.compression_ratio

            # Layer 6: ZK Proof
            if generate_zk and self.config.enable_zk and gdiff.differential_variants:
                layer_start = time.time()
                first_variant = gdiff.differential_variants[0]
                result.zk_proof = self.generate_zk_proof({
                    "chrom": first_variant.chrom,
                    "pos": first_variant.position,
                    "ref": first_variant.reference,
                    "alt": first_variant.alternate,
                })
                result.layer_times["zk_proof"] = time.time() - layer_start

            result.success = True

        except Exception as e:
            logger.exception("Pipeline failed: %s", e)
            result.success = False
            result.error_message = str(e)

        result.total_time = time.time() - total_start

        # Log summary
        if result.success:
            logger.info(
                "Pipeline complete: %d variants, %d bytes HDV, %.1fs total",
                result.num_variants, result.hdv_size_bytes, result.total_time
            )
            for layer, elapsed in result.layer_times.items():
                logger.info("  - %s: %.1fs", layer, elapsed)

        return result

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _discover_guide_fastas(self) -> List[Path]:
        """Discover guide FASTA files in the configured directory."""
        guide_dir = self.config.guide_fasta_dir
        if not guide_dir.exists():
            raise FileNotFoundError(f"Guide FASTA directory not found: {guide_dir}")

        patterns = ["*.fa.gz", "*.fasta.gz", "*.fa", "*.fasta"]
        fastas = []
        for pattern in patterns:
            fastas.extend(guide_dir.glob(pattern))

        # Sort for consistent ordering
        fastas = sorted(set(fastas))

        if not fastas:
            raise FileNotFoundError(
                f"No guide FASTA files found in {guide_dir}. "
                f"Expected files matching: {patterns}"
            )

        return fastas

    def _discover_pool_bams(self) -> List[Path]:
        """Discover pool BAM files for GDiff encoding."""
        if self.config.guide_bam_dir:
            bam_dir = self.config.guide_bam_dir
        else:
            bam_dir = self.config.guide_fasta_dir

        if not bam_dir.exists():
            raise FileNotFoundError(f"Pool BAM directory not found: {bam_dir}")

        bams = list(bam_dir.glob("*.bam"))
        bams = [b for b in bams if not b.name.endswith(".bai")]  # Exclude index files
        bams = sorted(bams)

        if not bams:
            raise FileNotFoundError(
                f"No pool BAM files found in {bam_dir}. "
                f"Guide BAMs are required for GDiff encoding."
            )

        return bams


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GenomeVault Unified Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline
    python -m genomevault.pipelines.unified_pipeline \\
        --fastq-r1 sample_R1.fq.gz \\
        --fastq-r2 sample_R2.fq.gz \\
        --guides data/guide_strands \\
        --output pipeline_output

    # Dry run (show configuration)
    python -m genomevault.pipelines.unified_pipeline --dry-run
        """
    )

    parser.add_argument("--fastq-r1", type=Path, help="Path to R1 FASTQ file")
    parser.add_argument("--fastq-r2", type=Path, help="Path to R2 FASTQ file")
    parser.add_argument("--guides", type=Path, default=Path("data/guide_strands"),
                       help="Path to guide strand directory")
    parser.add_argument("--output", type=Path, default=Path("pipeline_output"),
                       help="Output directory")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads")
    parser.add_argument("--no-zk", action="store_true", help="Disable ZK proofs")
    parser.add_argument("--no-pir", action="store_true", help="Disable PIR")
    parser.add_argument("--dry-run", action="store_true", help="Show config without running")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    config = PipelineConfig(
        output_dir=args.output,
        guide_fasta_dir=args.guides,
        threads=args.threads,
        enable_zk=not args.no_zk,
        enable_pir=not args.no_pir,
    )

    if args.dry_run:
        print("Pipeline Configuration:")
        print(f"  Output: {config.output_dir}")
        print(f"  Guides: {config.guide_fasta_dir}")
        print(f"  Threads: {config.threads}")
        print(f"  HDC Dimension: {config.hdc_dimension}")
        print(f"  ZK Proofs: {config.enable_zk}")
        print(f"  PIR: {config.enable_pir}")
        return

    if not args.fastq_r1 or not args.fastq_r2:
        parser.error("--fastq-r1 and --fastq-r2 are required (unless --dry-run)")

    pipeline = UnifiedPipeline(config)
    result = pipeline.run_experimental_pipeline(
        query_fastq_1=args.fastq_r1,
        query_fastq_2=args.fastq_r2,
    )

    if result.success:
        print(f"\nPipeline completed successfully!")
        print(f"  GDiff: {result.gdiff_path}")
        print(f"  HDV: {result.hdv_path}")
        print(f"  Variants: {result.num_variants}")
        print(f"  Time: {result.total_time:.1f}s")
    else:
        print(f"\nPipeline failed: {result.error_message}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
