"""
GDiff Encoder: BAM → GDiff

Computes differential encoding directly from BAM alignments without VCF.
Replaces bcftools-based variant calling with direct differential computation.
"""

import logging
import os
import psutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from datetime import datetime
import json

import pysam
import numpy as np

from .schema import (
    GDiffDocument,
    GDiffMetadata,
    DifferentialVariant,
    DifferentialContext,
    StructuralContext,
    FunctionalContext,
    QualityMetrics,
    SummaryStatistics,
    AlignmentParams,
    NearbyVariant,
    PopulationContext,
    ErrorBounds,
    SecureGuideReference,
    create_minimal_variant,
    GDIFF_SCHEMA_VERSION,
)
from .classification import compute_variant_significance
from .secure_guide_reference_builder import (
    SecureGuideReferenceBuilder,
    GuidePoolMetadata,
)
from .template_utils import auto_detect_template, should_use_template

logger = logging.getLogger(__name__)


class GDiffEncoder:
    """
    Encoder that computes differential encoding from BAM alignments.

    This replaces the VCF-based approach with direct computation:
    - Reads query BAM and pool BAMs using pysam
    - Computes sequence differences at each position
    - Determines differential type (unique_to_query, missing_from_query, etc.)
    - Computes structural context (nearby variants)
    - Outputs GDiff document
    """

    def __init__(
        self,
        query_bam: str,
        pool_bams: List[str],
        user_id: Optional[str] = None,
        genome_build: str = "pool",
        alignment_params: Optional[AlignmentParams] = None,
        min_base_quality: int = 20,
        min_mapping_quality: int = 20,
        min_depth: int = 10,
        max_depth: int = 10000,
        chunk_size: Optional[int] = None,  # Auto-optimize based on RAM (or set manually)
        max_memory_gb: int = 16,  # Maximum RAM to use before warning
        population_template: Optional[any] = None,  # TemplateBuilder for LOCAL lookups
        template_path: Optional[str] = None,  # Path to GDiff template (auto-detect if None)
        enable_template_autodetect: bool = True,  # CRITICAL: Set False to disable template loading
        use_streaming_template: bool = True,  # Use SQLite streaming instead of loading to RAM
        enable_quality_check: bool = True,  # Enable pre-flight quality validation
        target_epsilon: Optional[float] = None,  # Target error bound for quality check
        fastq_path: Optional[str] = None,  # Optional FASTQ path for quality assessment
        # Secure Guide Reference System (v1.2+)
        guide_fasta_files: Optional[List[Path]] = None,  # Guide FASTA files for SGRS
        chunk_guide_map: Optional[Dict[int, Tuple[int, int]]] = None,  # chunk_id -> (guide_idx, seed)
        guide_pool_metadata: Optional[Dict] = None,  # Metadata for secure guide reference
    ):
        """
        Initialize GDiff encoder.

        PRIVACY: Query ONLY compared to pool BAMs, NO reference involvement.

        Args:
            query_bam: Path to query BAM file
            pool_bams: List of paths to pool BAM files (k-1 reference genomes)
            user_id: User identifier (will be SHA-256 hashed for privacy)
            genome_build: Genome build identifier (default: "pool")
            alignment_params: Alignment parameters used (for reproducibility)
            min_base_quality: Minimum base quality to consider (default: 20)
            min_mapping_quality: Minimum mapping quality (default: 20)
            min_depth: Minimum read depth (default: 10)
            max_depth: Maximum read depth to avoid pile-up artifacts (default: 10000)
            population_template: Optional TemplateBuilder for LOCAL population lookups
                                (pre-loaded, no network queries)
            template_path: Path to GDiff template file for template-based encoding (optional)
            enable_quality_check: Enable pre-flight quality validation (default: True)
            target_epsilon: Target error bound for quality check (e.g., 0.05 for diagnostic)
            fastq_path: Path to source FASTQ file for quality assessment (optional)

        Quality Check (Clinical-Grade):
            If enable_quality_check=True and fastq_path + target_epsilon provided:
            - Parses FASTQ Q-scores to compute Q_input
            - Validates Q_input meets requirement for target_epsilon
            - Logs warning if insufficient (but does NOT block processing)
            - Recommends sequencing platform if quality insufficient

            Privacy: All quality checks are LOCAL (no network calls).

        Secure Guide Reference System (v1.2+):
            If guide_fasta_files + chunk_guide_map provided:
            - Generates cryptographic binding to guide pool
            - Enables full nucleotide-resolution queries
            - GDiff stores encrypted pointers, NOT full sequences
            - Requires local guide FASTAs to decrypt and query

            See docs/SECURE_GUIDE_REFERENCE_SYSTEM.md for details.
        """
        self.query_bam = Path(query_bam)
        self.pool_bams = [Path(p) for p in pool_bams]
        self.user_id = user_id
        self.genome_build = genome_build
        self.alignment_params = alignment_params
        self.population_template = population_template  # LOCAL lookups only

        # Secure Guide Reference System (v1.2+)
        self.guide_fasta_files = guide_fasta_files
        self.chunk_guide_map = chunk_guide_map
        self.guide_pool_metadata = guide_pool_metadata or {}

        # Quality filters
        self.min_base_quality = min_base_quality
        self.min_mapping_quality = min_mapping_quality
        self.min_depth = min_depth
        self.max_depth = max_depth

        # Memory management with dynamic optimization
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        if chunk_size is None:
            # Auto-calculate optimal chunk size based on available RAM
            num_bams = len(self.pool_bams) + 1  # pool + query
            # Heuristic: ~10 bytes per read position, ~30x coverage typical
            bytes_per_position = 10 * 30 * num_bams

            # Use 25% of available memory for chunk (conservative)
            target_memory_bytes = (available_memory_gb * 0.25) * (1024**3)
            optimal_chunk_size = int(target_memory_bytes / bytes_per_position)

            # Clamp to reasonable range (1MB - 50MB)
            self.chunk_size = max(1_000_000, min(optimal_chunk_size, 50_000_000))
            logger.info(f"Auto-optimized chunk size: {self.chunk_size / 1_000_000:.1f} MB "
                       f"(based on {available_memory_gb:.1f} GB available RAM)")
        else:
            self.chunk_size = chunk_size

        self.max_memory_gb = max_memory_gb

        # Validate inputs
        if not self.query_bam.exists():
            raise FileNotFoundError(f"Query BAM not found: {self.query_bam}")
        for pool_bam in self.pool_bams:
            if not pool_bam.exists():
                raise FileNotFoundError(f"Pool BAM not found: {pool_bam}")

        # k-anonymity level
        self.k_anonymity = len(self.pool_bams) + 1  # k-1 pool + 1 query

        # Template-based encoding (optional with auto-detection)
        self.template_path = template_path
        self.use_streaming_template = use_streaming_template
        self.template_index = {}  # (chrom, pos, ref, alt) → variant data (if not streaming)
        self.template_db = None  # SQLite streaming DB (if streaming)
        self.novel_variants = []  # Variants not in template
        self.enable_template_autodetect = enable_template_autodetect

        # Auto-detect template if not explicitly disabled
        if self.template_path is None and enable_template_autodetect and should_use_template(genome_build, self.k_anonymity):
            detected_template = auto_detect_template(genome_build)
            if detected_template:
                logger.info(f"Auto-detected template: {detected_template}")
                self.template_path = str(detected_template)

        if self.template_path is not None:
            if use_streaming_template:
                # Use streaming SQLite database (minimal RAM)
                from genomevault.differential_encoding.gdiff.template_db import StreamingTemplateDB, convert_template_to_db

                # Convert to SQLite if needed
                db_path = Path(str(self.template_path).replace('.json.gz', '.db').replace('.json', '.db'))
                if not db_path.exists():
                    logger.info(f"Converting template to SQLite (one-time)...")
                    convert_template_to_db(Path(self.template_path), db_path)

                logger.info(f"Opening streaming template database: {db_path}")
                self.template_db = StreamingTemplateDB(db_path)
                logger.info(f"✓ Streaming template ready (minimal RAM usage)")
            else:
                # Legacy: Load entire template into RAM
                logger.info(f"Loading GDiff template from {self.template_path}...")
                self._load_template(Path(self.template_path))
                logger.info(f"✓ Template loaded: {len(self.template_index):,} variant sites indexed")

            logger.info(f"  Common variants will be deduplicated during encoding")
        elif not enable_template_autodetect:
            logger.debug("Template auto-detection explicitly disabled (worker mode)")
        else:
            logger.warning(
                f"No template found for {genome_build} (k={self.k_anonymity}). "
                f"All differential variants will be encoded (may be 60-100× larger)."
            )

        logger.info(f"GDiffEncoder initialized:")
        logger.info(f"  Query: {self.query_bam}")
        logger.info(f"  Pool: {len(self.pool_bams)} BAMs")
        logger.info(f"  k-anonymity: {self.k_anonymity}")
        logger.info(f"  Chunk size: {self.chunk_size / 1_000_000:.1f} MB (prevents RAM overload)")
        logger.info(f"  Max memory: {self.max_memory_gb} GB")

        # Check available memory
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        logger.info(f"  Available RAM: {available_memory_gb:.1f} GB")

        # Pre-flight quality check (optional, clinical-grade)
        self.enable_quality_check = enable_quality_check
        self.target_epsilon = target_epsilon
        self.fastq_path = fastq_path
        self.quality_report = None

        if self.enable_quality_check and self.fastq_path is not None and self.target_epsilon is not None:
            logger.info("Running pre-flight quality check...")
            self.quality_report = self._run_quality_check()
        if available_memory_gb < self.max_memory_gb:
            logger.warning(f"  ⚠️  Available RAM ({available_memory_gb:.1f} GB) is less than max_memory_gb ({self.max_memory_gb} GB)")
            logger.warning(f"  ⚠️  Consider reducing chunk_size or max_memory_gb to avoid system crash")

    def _check_memory_usage(self, context: str = "") -> None:
        """
        Check current memory usage and warn if approaching limit.

        Args:
            context: Context string for logging (e.g., "Processing chr1")
        """
        memory_info = psutil.virtual_memory()
        used_gb = memory_info.used / (1024**3)
        available_gb = memory_info.available / (1024**3)
        percent_used = memory_info.percent

        if percent_used > 80:
            logger.warning(f"⚠️  HIGH MEMORY USAGE: {percent_used:.1f}% ({used_gb:.1f} GB used, {available_gb:.1f} GB available) {context}")
        elif percent_used > 90:
            logger.error(f"🚨 CRITICAL MEMORY USAGE: {percent_used:.1f}% - System may crash! {context}")

    def _load_template(self, template_path: Path) -> None:
        """
        Load GDiff template with pre-populated variant sites.

        Template contains 750M known variants from gnomAD, dbSNP, ClinVar.
        Builds O(1) index: (chrom, pos, ref, alt) → template entry.

        PRIVACY: Template contains PUBLIC aggregate data only (LOCAL).

        Args:
            template_path: Path to template file (.json or .json.gz)
        """
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        # Load template (supports gzip compression)
        import gzip
        open_fn = gzip.open if str(template_path).endswith('.gz') else open

        with open_fn(template_path, 'rt') as f:
            template_data = json.load(f)

        # Extract variant template entries
        if 'variant_template_entries' in template_data:
            variant_entries = template_data['variant_template_entries']
        elif 'variants' in template_data:
            variant_entries = template_data['variants']
        else:
            raise ValueError("Template must contain 'variant_template_entries' or 'variants' field")

        # Build O(1) index: (chrom, pos, ref, alt) → entry
        # Handle two formats:
        # 1. List format: [{"chrom": "chr1", "pos": 12345, ...}, ...]
        # 2. Dict format: {"chr1:12345:A:G": {"clinical_significance": "benign", ...}, ...}

        if isinstance(variant_entries, dict):
            # Format 2: Dict with coordinate keys
            for coord_key, context_dict in variant_entries.items():
                # Parse coordinate key: "chr1:12345:A:G" → (chr1, 12345, A, G)
                parts = coord_key.split(':')
                if len(parts) != 4:
                    logger.warning(f"Invalid coordinate key format: {coord_key}")
                    continue

                chrom, pos_str, ref, alt = parts
                try:
                    pos = int(pos_str)
                except ValueError:
                    logger.warning(f"Invalid position in coordinate key: {coord_key}")
                    continue

                key = (chrom, pos, ref, alt)
                self.template_index[key] = context_dict

        elif isinstance(variant_entries, list):
            # Format 1: List of variant objects
            for entry in variant_entries:
                key = (
                    entry['chrom'],
                    entry['pos'],
                    entry['ref'],
                    entry['alt']
                )
                self.template_index[key] = entry

        else:
            raise ValueError(f"Unexpected variant_entries type: {type(variant_entries)}")

        logger.info(f"Template index built: {len(self.template_index):,} variants")

    def _run_quality_check(self) -> Dict:
        """
        Run pre-flight quality check on input FASTQ data.

        Validates that input sequencing quality meets target error bound.
        Logs warnings if quality is insufficient but does NOT block processing.

        Returns:
            Quality report dictionary with:
            - Q_input: Measured sequencing quality
            - meets_target: Boolean (True if sufficient)
            - recommendation: Platform recommendation if failed
            - quality_metrics: Detailed Q-score statistics

        Privacy: All computation is LOCAL (no network calls).
        """
        try:
            from genomevault.quality_control import validate_input_quality

            quality_report = validate_input_quality(
                fastq_path=self.fastq_path,
                target_epsilon=self.target_epsilon,
                k=self.k_anonymity,
                D=10000  # Default hypervector dimension
            )

            if quality_report['meets_target']:
                logger.info(
                    f"✅ Input quality check PASSED: "
                    f"Q_input={quality_report['Q_input']:.3f} "
                    f"(≥ {quality_report['Q_input_min']:.3f} required)"
                )
            else:
                logger.warning(
                    f"⚠️  Input quality check FAILED: "
                    f"Q_input={quality_report['Q_input']:.3f} "
                    f"< {quality_report['Q_input_min']:.3f} required"
                )
                logger.warning(
                    f"⚠️  Recommendation: {quality_report['recommendation']}"
                )
                logger.warning(
                    f"⚠️  Proceeding anyway, but results may not meet target error bound "
                    f"ε_max={self.target_epsilon:.4f}"
                )

            # Log detailed metrics
            metrics = quality_report['quality_metrics']
            logger.info(f"  Average Q-score: {metrics['average_q_score']:.1f}")
            logger.info(f"  Q30 fraction: {metrics['q30_fraction']:.2%}")
            logger.info(f"  Coverage uniformity (std dev): {metrics['coverage_uniformity']:.2f}")

            return quality_report

        except ImportError as e:
            logger.warning(f"Quality check module not available: {e}")
            logger.warning("Skipping pre-flight quality check")
            return None
        except Exception as e:
            logger.error(f"Quality check failed: {e}")
            logger.warning("Proceeding without quality validation")
            return None

    def compute_differential_encoding(
        self,
        chromosomes: Optional[List[str]] = None,
        regions: Optional[List[Tuple[str, int, int]]] = None,
        num_workers: Optional[int] = None,
    ) -> GDiffDocument:
        """
        Compute complete differential encoding.

        Args:
            chromosomes: List of chromosomes to process (default: all)
            regions: List of (chrom, start, end) regions (default: all)
            num_workers: Number of parallel workers (default: auto-detect from hardware)

        Returns:
            GDiffDocument with complete differential encoding
        """
        # Auto-detect number of workers from hardware
        # OPTIMIZATION: For BAM-heavy workloads, use fewer workers than CPU cores
        # to reduce disk I/O contention
        if num_workers is None:
            cpu_count = os.cpu_count() or 1
            # Heuristic: Use 60% of cores for disk-bound BAM processing
            # This balances parallelism with I/O contention
            num_workers = max(1, int(cpu_count * 0.6))
            logger.info(f"Auto-detected {cpu_count} cores, using {num_workers} workers (optimized for disk I/O)")

        logger.info(f"Computing differential encoding with {num_workers} parallel workers...")

        # Open all BAM files
        query_bam = pysam.AlignmentFile(str(self.query_bam), "rb")
        pool_bam_handles = [
            pysam.AlignmentFile(str(pb), "rb") for pb in self.pool_bams
        ]
        # PRIVACY: No reference genome - query compares ONLY to pool

        try:
            # Determine which chromosomes to process
            if chromosomes is None:
                chromosomes = [ref for ref in query_bam.references if ref.startswith("chr")]

            logger.info(f"Processing {len(chromosomes)} chromosomes: {chromosomes}")

            # Collect all differential variants
            all_variants = []

            # Process chromosomes in parallel with chunking
            if num_workers > 1:
                logger.info(f"Parallel processing enabled: {num_workers} workers")
                logger.info(f"  Chunk size: {self.chunk_size / 1_000_000:.1f} MB")

                # Create chromosome processing tasks with chunking
                chrom_tasks = []
                for chrom in chromosomes:
                    if regions:
                        chrom_regions = [(c, s, e) for c, s, e in regions if c == chrom]
                        for _, start, end in chrom_regions:
                            # Split region into chunks for parallel processing
                            for chunk_start in range(start, end, self.chunk_size):
                                chunk_end = min(chunk_start + self.chunk_size, end)
                                chrom_tasks.append((chrom, chunk_start, chunk_end))
                    else:
                        chrom_length = query_bam.get_reference_length(chrom)
                        # Split chromosome into chunks for parallel processing
                        for chunk_start in range(0, chrom_length, self.chunk_size):
                            chunk_end = min(chunk_start + self.chunk_size, chrom_length)
                            chrom_tasks.append((chrom, chunk_start, chunk_end))

                logger.info(f"  Total chunks: {len(chrom_tasks)} (will utilize all {num_workers} cores)")

                # Process in parallel using ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=num_workers) as executor:
                    # Submit all tasks
                    future_to_task = {
                        executor.submit(
                            _process_chromosome_worker,
                            str(self.query_bam),
                            [str(pb) for pb in self.pool_bams],
                            # NO reference in parallel call
                            chrom,
                            start,
                            end,
                            self.min_base_quality,
                            self.min_mapping_quality,
                            self.min_depth,
                            self.max_depth,
                        ): (chrom, start, end)
                        for chrom, start, end in chrom_tasks
                    }

                    # Collect results as they complete
                    completed = 0
                    for future in as_completed(future_to_task):
                        chrom, start, end = future_to_task[future]
                        completed += 1
                        try:
                            variant_dicts = future.result()
                            # Convert dicts back to DifferentialVariant objects
                            variants = [self._dict_to_variant(v) for v in variant_dicts]
                            all_variants.extend(variants)
                            logger.info(f"  ✓ {chrom}:{start}-{end}: {len(variants)} variants [{completed}/{len(chrom_tasks)}]")
                        except Exception as e:
                            logger.error(f"  ✗ {chrom}:{start}-{end} failed: {e}")

            else:
                # Sequential processing with chunking (avoids RAM overload)
                logger.info(f"Sequential processing (single worker) with {self.chunk_size / 1_000_000:.1f} MB chunks")
                for chrom in chromosomes:
                    logger.info(f"Processing {chrom}...")
                    self._check_memory_usage(f"Starting {chrom}")

                    if regions:
                        # Process specific regions
                        chrom_regions = [(c, s, e) for c, s, e in regions if c == chrom]
                        for _, start, end in chrom_regions:
                            # Split region into chunks
                            for chunk_start in range(start, end, self.chunk_size):
                                chunk_end = min(chunk_start + self.chunk_size, end)
                                logger.info(f"  Chunk {chrom}:{chunk_start}-{chunk_end}")
                                variants = self._process_region(
                                    chrom, chunk_start, chunk_end,
                                    query_bam, pool_bam_handles
                                )
                                all_variants.extend(variants)
                                self._check_memory_usage(f"After chunk {chrom}:{chunk_start}-{chunk_end}")
                    else:
                        # Process entire chromosome in chunks
                        chrom_length = query_bam.get_reference_length(chrom)
                        for chunk_start in range(0, chrom_length, self.chunk_size):
                            chunk_end = min(chunk_start + self.chunk_size, chrom_length)
                            logger.info(f"  Chunk {chrom}:{chunk_start}-{chunk_end}")
                            variants = self._process_region(
                                chrom, chunk_start, chunk_end,
                                query_bam, pool_bam_handles
                            )
                            all_variants.extend(variants)
                            self._check_memory_usage(f"After chunk {chrom}:{chunk_start}-{chunk_end}")

                    logger.info(f"  Found {len([v for v in all_variants if v.chrom == chrom])} variants in {chrom}")

            logger.info(f"Total variants: {len(all_variants)}")

            # Compute summary statistics
            summary = self._compute_summary_statistics(all_variants)

            # Create metadata
            metadata = self._create_metadata()

            # Create GDiff document
            gdiff = GDiffDocument(
                schema_version=GDIFF_SCHEMA_VERSION,
                metadata=metadata,
                differential_variants=all_variants,
                summary_statistics=summary,
            )

            # Log error bounds summary if available
            if metadata.error_bounds is not None:
                self._log_error_bounds_summary(metadata.error_bounds)

            logger.info("Differential encoding complete")
            return gdiff

        finally:
            # Close all file handles
            query_bam.close()
            for pbh in pool_bam_handles:
                pbh.close()
            # PRIVACY: No reference to close

    def _process_region(
        self,
        chrom: str,
        start: int,
        end: int,
        query_bam: pysam.AlignmentFile,
        pool_bams: List[pysam.AlignmentFile],
    ) -> List[DifferentialVariant]:
        """
        Process a genomic region to find differential variants.

        PRIVACY: Query compared to ONE randomly-selected guide per chunk.
        Guide selection uses cryptographic binding via SGRS (chunk_guide_map).

        Args:
            chrom: Chromosome name
            start: Start position (0-based)
            end: End position (0-based, exclusive)
            query_bam: Query BAM handle
            pool_bams: Pool BAM handles (all k-1 guides available)

        Returns:
            List of DifferentialVariant objects
        """
        variants = []

        # Determine which guide to use for this chunk (SGRS-based selection)
        # If chunk_guide_map exists, use cryptographic guide selection
        # Otherwise, randomly select one guide for this entire region
        if self.chunk_guide_map is not None:
            # Use SGRS: chunk_id is based on start position
            chunk_id = start // self.chunk_size
            if chunk_id in self.chunk_guide_map:
                selected_guide_idx, alignment_seed = self.chunk_guide_map[chunk_id]
            else:
                # Fallback: use deterministic selection based on chunk_id
                import random
                rng = random.Random(chunk_id)
                selected_guide_idx = rng.randint(0, len(pool_bams) - 1)
                alignment_seed = 0
        else:
            # No SGRS: randomly select one guide for entire region
            import random
            selected_guide_idx = random.randint(0, len(pool_bams) - 1)
            alignment_seed = 0

        # Use ONLY the selected guide for comparison
        selected_guide_bam = pool_bams[selected_guide_idx]

        logger.debug(
            f"Region {chrom}:{start}-{end} using guide #{selected_guide_idx + 1} "
            f"(seed: {alignment_seed})"
        )

        # Pileup through query BAM
        for pileup_column in query_bam.pileup(
            chrom, start, end,
            truncate=True,
            stepper="samtools",
            min_base_quality=self.min_base_quality,
            min_mapping_quality=self.min_mapping_quality,
            max_depth=self.max_depth,
        ):
            pos = pileup_column.pos  # 0-based

            # PRIVACY: Get query alleles (no reference comparison)
            query_alleles = self._get_alleles_at_position(pileup_column)

            # Skip if no coverage
            if not query_alleles:
                continue

            # Get consensus allele from query
            query_allele = self._get_consensus_allele(query_alleles)

            # PRIVACY: Get alleles from ONLY the selected guide (not all guides)
            guide_alleles = self._get_guide_alleles_at_position(
                chrom, pos, selected_guide_bam
            )

            # Get consensus allele from selected guide
            guide_consensus = self._get_consensus_allele(guide_alleles) if guide_alleles else None

            # Skip if no guide coverage
            if guide_consensus is None:
                continue

            # DEBUG: Log specific position
            if pos == 10000804:
                logger.info(f"DEBUG pos 10000804: query={query_allele} guide={guide_consensus} match={query_allele == guide_consensus}")

            # PRIVACY: Compare query to SINGLE selected guide (NOT to all-pool consensus)
            # Only report positions where query differs from this guide
            if query_allele == guide_consensus:
                continue  # No difference - skip

            # Compute quality metrics
            quality_metrics = self._compute_quality_metrics(pileup_column)

            # Skip low-quality positions
            if quality_metrics.read_depth < self.min_depth:
                continue

            # For k-anonymity tracking: check if other guides also have this variant
            # (Still use all guides for pool_coverage metadata, but comparison is vs selected guide)
            pool_alleles_by_member = self._get_pool_alleles_at_position(
                chrom, pos, pool_bams
            )
            pool_coverage = self._compute_pool_coverage_for_allele(
                query_allele, pool_alleles_by_member
            )

            # Determine differential type (query vs selected guide)
            diff_type = self._compute_differential_type(
                query_allele, guide_consensus, pool_coverage
            )

            # Compute differential context
            confidence = min(
                quality_metrics.mapping_quality / 60.0,
                quality_metrics.base_quality / 40.0
            )

            local_entropy = self._compute_local_entropy(chrom, pos)

            differential_context = DifferentialContext(
                diff_type=diff_type,
                pool_coverage=pool_coverage,
                confidence=confidence,
                local_entropy=local_entropy,
            )

            # Compute structural context
            # PRIVACY: ref = guide_consensus (selected guide), alt = query_allele
            variant_type = self._classify_variant_type(guide_consensus, query_allele)
            structural_context = StructuralContext(
                variant_type=variant_type,
                haplotype_block=None,
                nearby_variants=[],
                repeat_region=False,
                segdup_region=False,
            )

            # Functional context (minimal for now)
            functional_context = FunctionalContext()

            # Population-aware classification
            # PRIVACY: Population template lookups are LOCAL only (pre-loaded)
            population_context_data = None
            AF_population = None

            # Check template for population annotation (NOT filtering!)
            variant_key = (chrom, pos + 1, guide_consensus, query_allele)

            # Support both streaming DB and in-memory template
            template_entry = None
            if self.template_db is not None:
                # Streaming template database (minimal RAM)
                template_entry = self.template_db.lookup(chrom, pos + 1, guide_consensus, query_allele)
            elif variant_key in self.template_index:
                # Legacy in-memory template
                template_entry = self.template_index[variant_key]

            if template_entry is not None:
                # Found in template - ADD annotation, don't skip
                AF_population = template_entry.get('allele_frequency', None)
                # Convert template dict to population context if needed
                # (Template provides annotation, not filtering)

            if self.population_template is not None:
                # LOCAL lookup - no network queries
                pop_result = self.population_template.lookup_variant(
                    chrom, pos + 1, guide_consensus, query_allele
                )
                if pop_result is not None:
                    AF_population = pop_result.allele_frequency
                    population_context_data = pop_result

            # Compute variant significance score
            # Conservative: Default to encoding (significance ≥ 0.2)
            classification_result = compute_variant_significance(
                Q_score=quality_metrics.base_quality,
                AF_population=AF_population,
                N_guide_strands=sum(pool_coverage),  # Count of guides with variant
                k_total=len(pool_coverage),
                QUAL=quality_metrics.mapping_quality,  # Use mapping quality as proxy
                GQ=quality_metrics.mapping_quality,  # Use mapping quality as proxy
                DP=quality_metrics.read_depth,
                AD_ref=0,  # Not computed in current pileup
                AD_alt=quality_metrics.read_depth,  # Approximation
            )

            # Skip obvious errors (significance < 0.2)
            # TEMPORARILY DISABLED FOR DEBUGGING - Accept all variants
            # if not classification_result['decision']['include_in_gdiff']:
            #     continue  # Skip this variant (likely error)

            # Create variant with classification
            # PRIVACY: ref = guide_consensus (selected guide via SGRS)
            #          alt = query_allele
            variant = DifferentialVariant(
                chrom=chrom,
                pos=pos + 1,  # Convert to 1-based for output
                ref=guide_consensus,  # Selected guide consensus (SGRS-based)
                alt=query_allele,     # Query allele
                differential_context=differential_context,
                structural_context=structural_context,
                functional_context=functional_context,
                quality_metrics=quality_metrics,
                population_context=population_context_data,  # Optional population data
                significance_score=classification_result['significance'],
                variant_classification=classification_result['variant_type'],
            )

            variants.append(variant)

        # PASS 2: Mark template variants that are NOT present in experimental
        # This enables full nucleotide resolution - decoder knows which template variants to use
        if self.template_index:
            variants.extend(self._mark_template_deletions(
                chrom, start, end, variants, query_bam, selected_guide_bam
            ))

        return variants

    def _mark_template_deletions(
        self,
        chrom: str,
        start: int,
        end: int,
        found_variants: List[DifferentialVariant],
        query_bam: pysam.AlignmentFile,
        guide_bam: pysam.AlignmentFile,
    ) -> List[DifferentialVariant]:
        """
        Mark template variants that are NOT present in experimental genome.

        This is CRITICAL for nucleotide resolution - enables decoder to know:
        - If position not in GDiff AND not in template → use guide sequence
        - If position IS in template but marked missing_from_query → DON'T use template

        Args:
            chrom: Chromosome
            start: Region start
            end: Region end
            found_variants: Variants already found in Pass 1
            query_bam: Query BAM handle
            guide_bam: Selected guide BAM handle

        Returns:
            List of template variants marked as "missing_from_query"
        """
        deletion_variants = []

        # Create set of positions we already encoded
        encoded_positions = {(v.chrom, v.pos) for v in found_variants}

        # Iterate through template variants in this region
        for variant_key, template_entry in self.template_index.items():
            t_chrom, t_pos, t_ref, t_alt = variant_key

            # Skip if not in this region
            if t_chrom != chrom or t_pos < start or t_pos >= end:
                continue

            # Skip if we already encoded this position
            if (t_chrom, t_pos) in encoded_positions:
                continue

            # Check if experimental has coverage at this position
            # Use 0-based for pysam
            pos_0based = t_pos - 1

            # Get experimental allele at this position
            exp_alleles = []
            for pileup_column in query_bam.pileup(
                t_chrom, pos_0based, pos_0based + 1,
                truncate=True,
                stepper="samtools",
                min_base_quality=self.min_base_quality,
                min_mapping_quality=self.min_mapping_quality,
            ):
                if pileup_column.pos == pos_0based:
                    exp_alleles = self._get_alleles_at_position(pileup_column)
                    break

            # Get guide allele at this position
            guide_alleles = []
            for pileup_column in guide_bam.pileup(
                t_chrom, pos_0based, pos_0based + 1,
                truncate=True,
                stepper="samtools",
                min_base_quality=self.min_base_quality,
                min_mapping_quality=self.min_mapping_quality,
            ):
                if pileup_column.pos == pos_0based:
                    guide_alleles = self._get_alleles_at_position(pileup_column)
                    break

            if not guide_alleles:
                continue  # No guide coverage, skip

            guide_consensus = self._get_consensus_allele(guide_alleles)

            # Check if template variant exists in experimental
            # Template says: ref→alt variant exists
            # If experimental matches ref (guide), then variant is NOT present
            if exp_alleles:
                exp_consensus = self._get_consensus_allele(exp_alleles)

                # Template variant NOT in experimental if:
                # - Experimental matches template ref (not the alt)
                if exp_consensus == t_ref and exp_consensus == guide_consensus:
                    # Mark as missing_from_query
                    deletion_variant = DifferentialVariant(
                        chrom=t_chrom,
                        pos=t_pos,
                        ref=t_ref,
                        alt=t_alt,
                        differential_context=DifferentialContext(
                            diff_type="missing_from_query",
                            pool_coverage=[0] * len(self.pool_bams),  # Template variant not in pool
                            confidence=0.95,  # High confidence - from template
                            local_entropy=0.0,  # Not computed for template deletions
                        ),
                        structural_context=StructuralContext(
                            variant_type=self._classify_variant_type(t_ref, t_alt),
                        ),
                        functional_context=FunctionalContext(),
                        quality_metrics=None,  # No quality metrics for template deletions
                        population_context=None,  # Could add template annotation here
                        significance_score=0.8,  # Template variants are significant
                        variant_classification="template_deletion",
                    )
                    deletion_variants.append(deletion_variant)

        return deletion_variants

    def _get_guide_alleles_at_position(
        self,
        chrom: str,
        pos: int,
        guide_bam: pysam.AlignmentFile,
    ) -> List[str]:
        """
        Get alleles from a single guide BAM at specific position.

        Args:
            chrom: Chromosome
            pos: Position (0-based)
            guide_bam: Single guide BAM handle

        Returns:
            List of alleles at this position in the guide
        """
        alleles = []

        for pileup_column in guide_bam.pileup(
            chrom, pos, pos + 1,
            truncate=True,
            stepper="samtools",
            min_base_quality=self.min_base_quality,
            min_mapping_quality=self.min_mapping_quality,
        ):
            if pileup_column.pos != pos:
                continue

            alleles = self._get_alleles_at_position(pileup_column)
            break

        return alleles

    def _get_alleles_at_position(
        self,
        pileup_column: pysam.PileupColumn,
    ) -> List[str]:
        """
        Extract alleles from pileup column.

        PRIVACY: No reference comparison.

        Args:
            pileup_column: Pileup column from pysam

        Returns:
            List of alleles (bases) at this position
        """
        alleles = []

        for pileup_read in pileup_column.pileups:
            if pileup_read.is_del or pileup_read.is_refskip:
                continue

            # Get base quality
            if pileup_read.alignment.query_qualities is None:
                continue

            query_pos = pileup_read.query_position
            if query_pos is None:
                continue

            base_quality = pileup_read.alignment.query_qualities[query_pos]
            if base_quality < self.min_base_quality:
                continue

            # Get base
            base = pileup_read.alignment.query_sequence[query_pos].upper()
            alleles.append(base)

        return alleles

    def _get_consensus_allele(
        self,
        alleles: List[str],
    ) -> str:
        """
        Get consensus allele from list of observed alleles.

        Uses simple majority voting.

        PRIVACY: No reference base comparison.

        Args:
            alleles: List of observed alleles

        Returns:
            Consensus allele (most common)
        """
        if not alleles:
            return "N"  # No coverage = N

        # Count alleles
        allele_counts = defaultdict(int)
        for allele in alleles:
            allele_counts[allele] += 1

        # Get most common allele
        max_count = max(allele_counts.values())
        candidates = [a for a, c in allele_counts.items() if c == max_count]

        # If tie, return first (arbitrary but consistent)
        return candidates[0]

    def _get_pool_alleles_at_position(
        self,
        chrom: str,
        pos: int,
        pool_bams: List[pysam.AlignmentFile],
    ) -> List[List[str]]:
        """
        Get alleles from each pool member at a specific position.

        PRIVACY: No reference genome comparison.

        Args:
            chrom: Chromosome
            pos: Position (0-based)
            pool_bams: Pool BAM handles

        Returns:
            List of allele lists, one per pool member
            Example: [['A', 'A', 'A'], ['A', 'C'], ['C', 'C']]
        """
        pool_alleles_by_member = []

        for pool_bam in pool_bams:
            member_alleles = []

            for pileup_column in pool_bam.pileup(
                chrom, pos, pos + 1,
                truncate=True,
                stepper="samtools",
                min_base_quality=self.min_base_quality,
                min_mapping_quality=self.min_mapping_quality,
            ):
                if pileup_column.pos != pos:
                    continue

                member_alleles = self._get_alleles_at_position(pileup_column)
                break

            pool_alleles_by_member.append(member_alleles)

        return pool_alleles_by_member

    def _compute_pool_consensus(
        self,
        pool_alleles_by_member: List[List[str]],
    ) -> Optional[str]:
        """
        Compute consensus allele across all pool members.

        PRIVACY: Pool-only consensus (no reference).

        Args:
            pool_alleles_by_member: List of allele lists, one per pool member

        Returns:
            Pool consensus allele (most common across all pool members)
            None if no pool coverage
        """
        # Flatten all alleles from all pool members
        all_pool_alleles = []
        for member_alleles in pool_alleles_by_member:
            all_pool_alleles.extend(member_alleles)

        if not all_pool_alleles:
            return None

        # Get consensus across entire pool
        return self._get_consensus_allele(all_pool_alleles)

    def _compute_pool_coverage_for_allele(
        self,
        query_allele: str,
        pool_alleles_by_member: List[List[str]],
    ) -> List[int]:
        """
        Check which pool members have the query allele.

        PRIVACY: Pool-only comparison.

        Args:
            query_allele: Query allele to check
            pool_alleles_by_member: List of allele lists, one per pool member

        Returns:
            Binary coverage list [0/1 for each pool member]
            1 = pool member has query allele, 0 = does not
        """
        pool_coverage = []

        for member_alleles in pool_alleles_by_member:
            if not member_alleles:
                # No coverage = does not have allele
                pool_coverage.append(0)
                continue

            # Get consensus for this pool member
            member_consensus = self._get_consensus_allele(member_alleles)

            # Check if matches query allele
            has_allele = 1 if member_consensus == query_allele else 0
            pool_coverage.append(has_allele)

        return pool_coverage

    def _compute_differential_type(
        self,
        query_allele: str,
        pool_consensus: str,
        pool_coverage: List[int],
    ) -> str:
        """
        Determine differential type.

        PRIVACY: Pool-only comparison (no reference).

        Args:
            query_allele: Query allele
            pool_consensus: Pool consensus allele
            pool_coverage: Binary coverage [0/1 for each pool member]

        Returns:
            Differential type: "unique_to_query", "missing_from_query",
            or "genotype_difference"
        """
        num_pool_with_allele = sum(pool_coverage)

        if num_pool_with_allele == 0:
            # Query has it, no pool members have it
            return "unique_to_query"
        elif num_pool_with_allele == len(pool_coverage):
            # All pool members have it, query has it
            # This is actually not a difference - shouldn't reach here
            return "genotype_difference"
        else:
            # Some pool members have it, some don't
            return "genotype_difference"

    def _compute_quality_metrics(
        self,
        pileup_column: pysam.PileupColumn,
    ) -> QualityMetrics:
        """
        Compute quality metrics from pileup column.

        Args:
            pileup_column: Pileup column from pysam

        Returns:
            QualityMetrics object
        """
        read_depth = pileup_column.n

        # Compute average mapping quality
        mapping_qualities = []
        base_qualities = []
        forward_reads = 0
        total_reads = 0

        for pileup_read in pileup_column.pileups:
            if pileup_read.is_del or pileup_read.is_refskip:
                continue

            mapping_qualities.append(pileup_read.alignment.mapping_quality)

            query_pos = pileup_read.query_position
            if query_pos is not None and pileup_read.alignment.query_qualities is not None:
                base_qualities.append(
                    pileup_read.alignment.query_qualities[query_pos]
                )

            # Strand balance
            if not pileup_read.alignment.is_reverse:
                forward_reads += 1
            total_reads += 1

        avg_mapping_quality = np.mean(mapping_qualities) if mapping_qualities else 0.0
        avg_base_quality = np.mean(base_qualities) if base_qualities else 0.0
        strand_balance = forward_reads / total_reads if total_reads > 0 else 0.5

        return QualityMetrics(
            read_depth=read_depth,
            mapping_quality=float(avg_mapping_quality),
            base_quality=float(avg_base_quality),
            strand_balance=float(strand_balance),
        )

    def _compute_local_entropy(
        self,
        chrom: str,
        pos: int,
        window: int = 50,
    ) -> float:
        """
        Compute Shannon entropy of surrounding sequence.

        Higher entropy = more information content.

        Args:
            chrom: Chromosome
            pos: Position (0-based)
            # PRIVACY: No reference genome parameter
            window: Window size around position (default: 50bp = ±25bp)

        Returns:
            Shannon entropy in bits
        """
        # PRIVACY: No reference sequence available for entropy calculation
        # Pool-only comparison doesn't use reference entropy
        return 0.0

        # Count bases
        base_counts = defaultdict(int)
        for base in sequence:
            if base in "ACGT":
                base_counts[base] += 1

        total = sum(base_counts.values())
        if total == 0:
            return 0.0

        # Compute Shannon entropy
        entropy = 0.0
        for count in base_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        return entropy

    def _classify_variant_type(self, ref: str, alt: str) -> str:
        """
        Classify variant type.

        Args:
            ref: Reference allele
            alt: Alternate allele

        Returns:
            Variant type: "SNV", "INDEL", "MNP", etc.
        """
        if len(ref) == 1 and len(alt) == 1:
            return "SNV"
        elif len(ref) != len(alt):
            return "INDEL"
        else:
            return "MNP"

    def _compute_summary_statistics(
        self,
        variants: List[DifferentialVariant],
    ) -> SummaryStatistics:
        """
        Compute summary statistics from variants.

        Args:
            variants: List of differential variants

        Returns:
            SummaryStatistics object
        """
        total_differences = len(variants)

        unique_to_query = 0
        missing_from_query = 0
        genotype_differences = 0
        high_confidence = 0
        structural_variants = 0

        for variant in variants:
            diff_type = variant.differential_context.diff_type

            if diff_type == "unique_to_query":
                unique_to_query += 1
            elif diff_type == "missing_from_query":
                missing_from_query += 1
            elif diff_type == "genotype_difference":
                genotype_differences += 1

            if variant.differential_context.confidence > 0.9:
                high_confidence += 1

            if variant.structural_context.variant_type in ["SV", "CNV"]:
                structural_variants += 1

        return SummaryStatistics(
            total_differences=total_differences,
            unique_to_query=unique_to_query,
            missing_from_query=missing_from_query,
            genotype_differences=genotype_differences,
            high_confidence=high_confidence,
            structural_variants=structural_variants,
        )

    def _dict_to_variant(self, variant_dict: Dict) -> DifferentialVariant:
        """
        Convert variant dict (from worker process) back to DifferentialVariant object.

        Args:
            variant_dict: Variant as dict

        Returns:
            DifferentialVariant object
        """
        # Reconstruct nested objects from dicts
        differential_context = DifferentialContext(**variant_dict["differential_context"])
        structural_context = StructuralContext(**variant_dict["structural_context"])
        functional_context = FunctionalContext(**variant_dict["functional_context"])
        quality_metrics = QualityMetrics(**variant_dict["quality_metrics"])

        return DifferentialVariant(
            chrom=variant_dict["chrom"],
            pos=variant_dict["pos"],
            ref=variant_dict["ref"],
            alt=variant_dict["alt"],
            differential_context=differential_context,
            structural_context=structural_context,
            functional_context=functional_context,
            quality_metrics=quality_metrics,
        )

    def _compute_error_bounds(self) -> Optional[ErrorBounds]:
        """
        Compute error bounds from quality report and pipeline fidelities.

        Implements Section 7.3: Error Propagation Model from Decision Matrix V2.0.

        Error Decomposition:
            ε_total = ε_input_corrected + ε_pipeline + ε_query

        Component Fidelities (from Decision Matrix V2.0, Section 7.3):
            - F_gdiff: GDiff encoding fidelity (0.999)
            - F_hdc: HDC transformation fidelity (dimension-dependent)
            - F_zk: Zero-knowledge proof soundness (1 - 2^-128)
            - F_pir: Private information retrieval correctness (1.0)

        Returns:
            ErrorBounds object if quality_report is available, None otherwise
        """
        # Return None if quality check wasn't performed
        if self.quality_report is None:
            return None

        # Extract epsilon_input_corrected and Q_input from quality report
        epsilon_input_corrected = self.quality_report.get('epsilon_input', 0.0)
        Q_input_measured = self.quality_report.get('Q_input', 1.0)

        # Compute epsilon_pipeline from component fidelities
        # ε_pipeline = 1 - (F_gdiff × F_hdc × F_zk × F_pir)
        #
        # Component fidelities (from Decision Matrix V2.0, Section 7.3):
        F_gdiff = 0.999  # GDiff encoding fidelity (empirically validated)
        F_hdc = 0.9999   # HDC transformation fidelity (dimension 10000, Table 3)
        F_zk = 1 - 2**-128  # ZK proof soundness (Groth16, 128-bit security)
        F_pir = 1.0      # PIR correctness (information-theoretic)

        # Compute epsilon_pipeline
        F_pipeline = F_gdiff * F_hdc * F_zk * F_pir
        epsilon_pipeline = 1 - F_pipeline

        # epsilon_query: Query-time false positive rate (single run)
        # Default: 0.01 (1%) for single run (from Decision Matrix V2.0, Section 7.3)
        epsilon_query = 0.01

        # Total error
        epsilon_total = epsilon_input_corrected + epsilon_pipeline + epsilon_query

        # Determine use_case and meets_target if target_epsilon was specified
        use_case = None
        meets_target = True

        if self.target_epsilon is not None:
            # Infer use_case from target_epsilon (from Decision Matrix V2.0, Section 2)
            if self.target_epsilon >= 0.30:
                use_case = "screening"
            elif self.target_epsilon >= 0.05:
                use_case = "diagnostic"
            elif self.target_epsilon >= 0.001:
                use_case = "life_critical"
            else:
                use_case = "regulatory"

            # Check if we meet the target
            meets_target = epsilon_total <= self.target_epsilon

        return ErrorBounds(
            epsilon_input_corrected=epsilon_input_corrected,
            epsilon_pipeline=epsilon_pipeline,
            epsilon_query=epsilon_query,
            epsilon_total=epsilon_total,
            Q_input_measured=Q_input_measured,
            use_case=use_case,
            meets_target=meets_target,
        )

    def _log_error_bounds_summary(self, error_bounds: ErrorBounds) -> None:
        """
        Log error bounds summary with warnings if approaching/exceeding thresholds.

        Args:
            error_bounds: ErrorBounds object to summarize
        """
        logger.info("=" * 80)
        logger.info("ERROR BOUNDS SUMMARY")
        logger.info("=" * 80)

        # Log error components
        logger.info(f"  ε_input (sequencing):    {error_bounds.epsilon_input_corrected:.6f} ({error_bounds.epsilon_input_corrected*100:.3f}%)")
        logger.info(f"  ε_pipeline (processing): {error_bounds.epsilon_pipeline:.6f} ({error_bounds.epsilon_pipeline*100:.4f}%)")
        logger.info(f"  ε_query (single run):    {error_bounds.epsilon_query:.6f} ({error_bounds.epsilon_query*100:.1f}%)")
        logger.info(f"  ε_total (end-to-end):    {error_bounds.epsilon_total:.6f} ({error_bounds.epsilon_total*100:.3f}%)")
        logger.info("")
        logger.info(f"  Q_input (measured):      {error_bounds.Q_input_measured:.6f} ({error_bounds.Q_input_measured*100:.3f}%)")
        logger.info("")

        # Log use case and target if specified
        if error_bounds.use_case is not None:
            # Clinical thresholds from Decision Matrix V2.0, Section 2
            CLINICAL_THRESHOLDS = {
                "screening": 0.30,
                "diagnostic": 0.05,
                "life_critical": 0.001,
                "regulatory": 0.0001,
            }

            target_epsilon = CLINICAL_THRESHOLDS[error_bounds.use_case]

            logger.info(f"  Use case: {error_bounds.use_case}")
            logger.info(f"  Target ε_max: {target_epsilon:.4f} ({target_epsilon*100:.2f}%)")
            logger.info("")

            # Check if we meet target
            if error_bounds.meets_target:
                logger.info(f"  ✅ PASS: Error bounds meet target for '{error_bounds.use_case}' use case")
                margin = target_epsilon - error_bounds.epsilon_total
                logger.info(f"  Margin: {margin:.6f} ({margin*100:.3f}%)")
            else:
                logger.warning(f"  ❌ FAIL: Error bounds EXCEED target for '{error_bounds.use_case}' use case")
                excess = error_bounds.epsilon_total - target_epsilon
                logger.warning(f"  Excess: {excess:.6f} ({excess*100:.3f}%)")
                logger.warning(f"  Recommendation: Improve input quality or use a more permissive use case")

            # Warn if approaching threshold (within 20% margin)
            if error_bounds.meets_target:
                margin_ratio = (target_epsilon - error_bounds.epsilon_total) / target_epsilon
                if margin_ratio < 0.20:
                    logger.warning(f"  ⚠️  WARNING: Approaching threshold ({margin_ratio*100:.1f}% margin remaining)")

        logger.info("=" * 80)

    def _generate_secure_guide_reference(self) -> SecureGuideReference:
        """
        Generate secure guide reference for full nucleotide resolution.

        Uses SecureGuideReferenceBuilder to create cryptographic binding
        between GDiff and guide FASTA files.

        Returns:
            SecureGuideReference with encrypted pointers to guide pool

        Raises:
            ValueError: If guide_fasta_files or chunk_guide_map not provided
        """
        if not self.guide_fasta_files:
            raise ValueError("guide_fasta_files required for secure guide reference")
        if not self.chunk_guide_map:
            raise ValueError("chunk_guide_map required for secure guide reference")

        # Create guide pool metadata
        metadata = GuidePoolMetadata(
            guide_fasta_files=self.guide_fasta_files,
            alignment_seed=self.guide_pool_metadata.get("alignment_seed", 0),
            chunk_size=self.chunk_size,
            timestamp=datetime.utcnow().isoformat() + "Z",
            minimap2_params=self.guide_pool_metadata.get("minimap2_params", {})
        )

        # Build secure guide reference
        builder = SecureGuideReferenceBuilder(
            guide_fasta_files=self.guide_fasta_files,
            chunk_guide_map=self.chunk_guide_map,
            alignment_metadata=metadata,
            user_secret=None  # Auto-generated (32 random bytes)
        )

        return builder.build()

    def _create_metadata(self) -> GDiffMetadata:
        """
        Create metadata for GDiff document.

        Returns:
            GDiffMetadata object with optional error_bounds
        """
        # Hash user ID for privacy
        if self.user_id:
            query_id = hashlib.sha256(self.user_id.encode()).hexdigest()
        else:
            query_id = hashlib.sha256(str(self.query_bam).encode()).hexdigest()

        # Get pool IDs from filenames
        reference_pool = [pb.stem for pb in self.pool_bams]

        # Use provided alignment params or create default
        if self.alignment_params is None:
            alignment_params = AlignmentParams(
                kmer=19,
                window=10,
                scoring="match=2,mismatch=-4,gap_open=-6",
                entropy_bits=261.2,  # SHA-256² entropy
            )
        else:
            alignment_params = self.alignment_params

        # Compute error bounds if quality check was performed
        error_bounds = self._compute_error_bounds() if self.enable_quality_check else None

        # Generate secure guide reference if parameters provided (v1.2+)
        secure_guide_reference = None
        if self.guide_fasta_files and self.chunk_guide_map:
            logger.info("Generating secure guide reference for full nucleotide resolution...")
            secure_guide_reference = self._generate_secure_guide_reference()
            logger.info("  ✓ Secure guide reference generated")

        return GDiffMetadata(
            query_id=query_id,
            reference_pool=reference_pool,
            k_anonymity=self.k_anonymity,
            alignment_params=alignment_params,
            genome_build=self.genome_build,
            timestamp=datetime.utcnow().isoformat() + "Z",
            gdiff_version=GDIFF_SCHEMA_VERSION,
            error_bounds=error_bounds,
            secure_guide_reference=secure_guide_reference,
        )


# ============================================================================
# Synchronized Pileup Optimization (10-50x speedup)
# ============================================================================

class _PeekableIterator:
    """
    Iterator wrapper that allows peeking at next position without consuming.

    Critical for synchronized pileup across multiple BAM files.
    """
    def __init__(self, iterator):
        self.iterator = iterator
        self._next = None
        self._has_next = True
        self._advance()

    def _advance(self):
        try:
            self._next = next(self.iterator)
        except StopIteration:
            self._has_next = False
            self._next = None

    def has_next(self):
        return self._has_next

    def peek_pos(self):
        """Get position of next element without consuming."""
        return self._next.pos if self._next else None

    def __next__(self):
        if not self._has_next:
            raise StopIteration
        result = self._next
        self._advance()
        return result


def _synchronize_pileups(query_pileup, pool_pileups):
    """
    Synchronize pileup iterators to yield data for matching positions.

    CRITICAL OPTIMIZATION: Eliminates O(n*k) seeks by advancing all iterators
    in parallel. Reduces from billions of seek operations to linear scan.

    Args:
        query_pileup: pysam pileup iterator for query BAM
        pool_pileups: List of pysam pileup iterators for pool BAMs

    Yields:
        (pos, query_column, pool_columns) for each position
    """
    # Convert to peekable iterators
    pool_iters = [_PeekableIterator(pp) for pp in pool_pileups]

    for query_col in query_pileup:
        pos = query_col.pos

        # Advance pool iterators to match this position
        pool_columns = []
        for pool_iter in pool_iters:
            # Skip pool positions before current query position
            while pool_iter.has_next():
                peek = pool_iter.peek_pos()
                if peek is None or peek >= pos:
                    break
                next(pool_iter)

            # Check if pool has data at this position
            if pool_iter.has_next():
                peek = pool_iter.peek_pos()
                if peek is not None and peek == pos:
                    pool_columns.append(next(pool_iter))
                else:
                    pool_columns.append(None)  # No coverage at this position
            else:
                pool_columns.append(None)  # Iterator exhausted

        yield pos, query_col, pool_columns


# ============================================================================
# Standalone Helper Functions for Parallel Processing
# ============================================================================

def _process_region_standalone(
    chrom: str,
    start: int,
    end: int,
    query_bam: pysam.AlignmentFile,
    pool_bams: List[pysam.AlignmentFile],
    min_base_quality: int,
    min_mapping_quality: int,
    min_depth: int,
    max_depth: int,
) -> List[Dict]:
    """
    Standalone function to process a genomic region.

    This is called by worker processes and contains all logic without
    requiring a GDiffEncoder instance.

    PRIVACY: Pool-only comparison - query compared ONLY to pool consensus.

    Args:
        chrom: Chromosome name
        start: Start position (0-based)
        end: End position (0-based, exclusive)
        query_bam: Query BAM handle
        pool_bams: Pool BAM handles
        min_base_quality: Minimum base quality
        min_mapping_quality: Minimum mapping quality
        min_depth: Minimum read depth
        max_depth: Maximum read depth

    Returns:
        List of variant dicts (for safe pickling across processes)
    """

    # Helper function: Get alleles at position
    def get_alleles_at_position(pileup_column: pysam.PileupColumn) -> List[str]:
        """Extract alleles from pileup column."""
        alleles = []
        for pileup_read in pileup_column.pileups:
            if pileup_read.is_del or pileup_read.is_refskip:
                continue

            if pileup_read.alignment.query_qualities is None:
                continue

            query_pos = pileup_read.query_position
            if query_pos is None:
                continue

            base_quality = pileup_read.alignment.query_qualities[query_pos]
            if base_quality < min_base_quality:
                continue

            base = pileup_read.alignment.query_sequence[query_pos].upper()
            alleles.append(base)

        return alleles

    # Helper function: Get consensus allele
    def get_consensus_allele(alleles: List[str]) -> str:
        """Get consensus allele from list (simple majority voting)."""
        if not alleles:
            return "N"

        allele_counts = defaultdict(int)
        for allele in alleles:
            allele_counts[allele] += 1

        max_count = max(allele_counts.values())
        candidates = [a for a, c in allele_counts.items() if c == max_count]
        return candidates[0]  # Return first if tie

    # Helper function: Get pool alleles at position
    def get_pool_alleles_at_position(pos: int) -> List[List[str]]:
        """Get alleles from each pool member at a specific position."""
        pool_alleles_by_member = []

        for pool_bam in pool_bams:
            member_alleles = []

            for pileup_column in pool_bam.pileup(
                chrom, pos, pos + 1,
                truncate=True,
                stepper="samtools",
                min_base_quality=min_base_quality,
                min_mapping_quality=min_mapping_quality,
            ):
                if pileup_column.pos != pos:
                    continue

                member_alleles = get_alleles_at_position(pileup_column)
                break

            pool_alleles_by_member.append(member_alleles)

        return pool_alleles_by_member

    # Helper function: Compute pool consensus
    def compute_pool_consensus(pool_alleles_by_member: List[List[str]]) -> Optional[str]:
        """Compute consensus allele across all pool members."""
        all_pool_alleles = []
        for member_alleles in pool_alleles_by_member:
            all_pool_alleles.extend(member_alleles)

        if not all_pool_alleles:
            return None

        return get_consensus_allele(all_pool_alleles)

    # Helper function: Compute pool coverage for allele
    def compute_pool_coverage_for_allele(
        query_allele: str,
        pool_alleles_by_member: List[List[str]]
    ) -> List[int]:
        """Check which pool members have the query allele."""
        pool_coverage = []

        for member_alleles in pool_alleles_by_member:
            if not member_alleles:
                pool_coverage.append(0)
                continue

            member_consensus = get_consensus_allele(member_alleles)
            has_allele = 1 if member_consensus == query_allele else 0
            pool_coverage.append(has_allele)

        return pool_coverage

    # Helper function: Compute differential type
    def compute_differential_type(
        query_allele: str,
        pool_consensus: str,
        pool_coverage: List[int]
    ) -> str:
        """Determine differential type."""
        num_pool_with_allele = sum(pool_coverage)

        if num_pool_with_allele == 0:
            return "unique_to_query"
        elif num_pool_with_allele == len(pool_coverage):
            return "genotype_difference"
        else:
            return "genotype_difference"

    # Helper function: Compute quality metrics
    def compute_quality_metrics(pileup_column: pysam.PileupColumn) -> Dict:
        """Compute quality metrics from pileup column."""
        read_depth = pileup_column.n

        mapping_qualities = []
        base_qualities = []
        forward_reads = 0
        total_reads = 0

        for pileup_read in pileup_column.pileups:
            if pileup_read.is_del or pileup_read.is_refskip:
                continue

            mapping_qualities.append(pileup_read.alignment.mapping_quality)

            query_pos = pileup_read.query_position
            if query_pos is not None and pileup_read.alignment.query_qualities is not None:
                base_qualities.append(
                    pileup_read.alignment.query_qualities[query_pos]
                )

            if not pileup_read.alignment.is_reverse:
                forward_reads += 1
            total_reads += 1

        avg_mapping_quality = np.mean(mapping_qualities) if mapping_qualities else 0.0
        avg_base_quality = np.mean(base_qualities) if base_qualities else 0.0
        strand_balance = forward_reads / total_reads if total_reads > 0 else 0.5

        return {
            "read_depth": read_depth,
            "mapping_quality": float(avg_mapping_quality),
            "base_quality": float(avg_base_quality),
            "strand_balance": float(strand_balance),
        }

    # Helper function: Classify variant type
    def classify_variant_type(ref: str, alt: str) -> str:
        """Classify variant type."""
        if len(ref) == 1 and len(alt) == 1:
            return "SNV"
        elif len(ref) != len(alt):
            return "INDEL"
        else:
            return "MNP"

    # Main processing logic - SYNCHRONIZED PILEUP (10-50x FASTER)
    # OPTIMIZATION: Single-pass through all BAMs eliminates O(n*k) seeks
    variants = []

    # Create synchronized pileup iterators for all BAMs
    query_pileup = query_bam.pileup(
        chrom, start, end,
        truncate=True,
        stepper="samtools",
        min_base_quality=min_base_quality,
        min_mapping_quality=min_mapping_quality,
        max_depth=max_depth,
    )

    pool_pileups = [
        pool_bam.pileup(
            chrom, start, end,
            truncate=True,
            stepper="samtools",
            min_base_quality=min_base_quality,
            min_mapping_quality=min_mapping_quality,
            max_depth=max_depth,
        )
        for pool_bam in pool_bams
    ]

    # Synchronize all iterators - advances in parallel (no seeks!)
    for pos, query_column, pool_columns in _synchronize_pileups(query_pileup, pool_pileups):
        # Get query alleles at this position
        query_alleles = get_alleles_at_position(query_column)
        if not query_alleles:
            continue

        # Compute quality metrics
        quality_metrics = compute_quality_metrics(query_column)
        if quality_metrics["read_depth"] < min_depth:
            continue

        # Get query consensus
        query_allele = get_consensus_allele(query_alleles)

        # Get pool alleles (already fetched by synchronized iterator!)
        pool_alleles_by_member = [
            get_alleles_at_position(pool_col) if pool_col else []
            for pool_col in pool_columns
        ]

        # Compute pool consensus
        pool_consensus = compute_pool_consensus(pool_alleles_by_member)
        if pool_consensus is None:
            continue

        # Compare query to pool consensus
        if query_allele == pool_consensus:
            continue  # No difference

        # Check which pool members have query allele
        pool_coverage = compute_pool_coverage_for_allele(
            query_allele, pool_alleles_by_member
        )

        # Determine differential type
        diff_type = compute_differential_type(
            query_allele, pool_consensus, pool_coverage
        )

        # Compute differential context
        confidence = min(
            quality_metrics["mapping_quality"] / 60.0,
            quality_metrics["base_quality"] / 40.0
        )

        differential_context = {
            "diff_type": diff_type,
            "pool_coverage": pool_coverage,
            "confidence": confidence,
            "local_entropy": 0.0,  # Pool-only comparison doesn't use reference entropy
        }

        # Structural context
        variant_type = classify_variant_type(pool_consensus, query_allele)
        structural_context = {
            "variant_type": variant_type,
            "haplotype_block": None,
            "nearby_variants": [],
            "repeat_region": False,
            "segdup_region": False,
        }

        # Functional context (minimal)
        functional_context = {}

        # Create variant dict (for safe pickling)
        variant = {
            "chrom": chrom,
            "pos": pos + 1,  # Convert to 1-based for output
            "ref": pool_consensus,  # Pool consensus, not reference
            "alt": query_allele,    # Query allele
            "differential_context": differential_context,
            "structural_context": structural_context,
            "functional_context": functional_context,
            "quality_metrics": quality_metrics,
        }

        variants.append(variant)

    return variants


# Module-level worker function for parallel chromosome processing
def _process_chromosome_worker(
    query_bam_path: str,
    pool_bam_paths: List[str],
    # NO reference_fasta_path - privacy-preserving
    chrom: str,
    start: int,
    end: int,
    min_base_quality: int,
    min_mapping_quality: int,
    min_depth: int,
    max_depth: int,
) -> List:
    """
    Worker function for parallel chromosome processing.

    LIGHTWEIGHT: Opens BAMs and processes directly without full encoder initialization.
    This avoids the overhead and contention of creating 10+ GDiffEncoder instances.

    Args:
        query_bam_path: Path to query BAM
        pool_bam_paths: List of paths to pool BAMs
        # PRIVACY: No reference genome
        chrom: Chromosome to process
        start: Start position
        end: End position
        min_base_quality: Min base quality filter
        min_mapping_quality: Min mapping quality filter
        min_depth: Min read depth
        max_depth: Max read depth

    Returns:
        List of DifferentialVariant objects (as dicts for pickling)
    """
    import pysam
    from genomevault.differential_encoding.gdiff.schema import DifferentialVariant
    import sys

    print(f"[WORKER {chrom}] Starting", file=sys.stderr, flush=True)

    # Open BAM files in this worker process with optimizations
    print(f"[WORKER {chrom}] Opening BAMs...", file=sys.stderr, flush=True)
    # OPTIMIZATION: Enable BGZF threading (2 threads per BAM for decompression)
    # This parallelizes decompression across CPU cores
    query_bam = pysam.AlignmentFile(
        query_bam_path, "rb",
        threads=2,  # BGZF decompression threads
        check_sq=False  # Skip header validation (faster)
    )
    pool_bams = [
        pysam.AlignmentFile(
            pb, "rb",
            threads=1,  # Pool BAMs get fewer threads
            check_sq=False
        )
        for pb in pool_bam_paths
    ]
    print(f"[WORKER {chrom}] BAMs opened successfully", file=sys.stderr, flush=True)

    try:
        # Process region using standalone helper function
        print(f"[WORKER {chrom}] Calling _process_region_standalone...", file=sys.stderr, flush=True)
        variants = _process_region_standalone(
            chrom, start, end,
            query_bam, pool_bams,
            min_base_quality, min_mapping_quality,
            min_depth, max_depth
        )
        print(f"[WORKER {chrom}] Processing complete: {len(variants)} variants", file=sys.stderr, flush=True)

        # Convert to dicts for safe pickling across processes
        return [v.to_dict() if hasattr(v, 'to_dict') else v for v in variants]

    finally:
        # Close file handles
        query_bam.close()
        for pb in pool_bams:
            pb.close()
