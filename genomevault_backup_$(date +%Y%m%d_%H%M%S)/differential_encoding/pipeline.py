"""
Differential Genomic Encoder Pipeline

This module implements the main encoding pipeline that integrates all differential
encoding components into a complete end-to-end system for encoding experimental
genomes as cryptographically verified differences from random reference genomes.

Sections 7.1 and 7.2 of the specification.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np

from genomevault.differential_encoding.reference_management import (
    SecureReferenceGenomeManager,
    GenomeSection,
)
from genomevault.differential_encoding.hypervector_encoder import (
    DifferentialHypervectorEncoder,
)
from genomevault.differential_encoding.crypto_primitives import (
    CryptoRNG,
    compute_chunk_id,
    compute_reference_hash,
    compute_chunk_reference_binding,
)
from genomevault.differential_encoding.chunking import (
    CryptographicChunker,
    Genome,
    GenomeChunk,
    AnalysisType,
    STRATEGY_CONFIGS,
    GenomicFeature,
)
from genomevault.differential_encoding.differences import (
    compute_variant_differences,
    VariantDifference,
)
from genomevault.differential_encoding.metadata import (
    DifferentialEncodingMetadata,
    create_metadata_from_chunk,
)

logger = logging.getLogger(__name__)


class EncodingResult:
    """
    Result of encoding an experimental genome.

    Contains hypervectors, metadata, and summary statistics for the encoding.
    """

    def __init__(
        self,
        hypervectors: List[np.ndarray],
        metadata: List[DifferentialEncodingMetadata],
        bundled_hypervector: Optional[np.ndarray] = None,
        statistics: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize encoding result.

        Args:
            hypervectors: List of chunk hypervectors
            metadata: List of chunk metadata
            bundled_hypervector: Optional bundled genome hypervector
            statistics: Optional encoding statistics
        """
        self.hypervectors = hypervectors
        self.metadata = metadata
        self.bundled_hypervector = bundled_hypervector
        self.statistics = statistics or {}

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EncodingResult("
            f"chunks={len(self.hypervectors)}, "
            f"bundled={'Yes' if self.bundled_hypervector is not None else 'No'}, "
            f"total_differences={self.statistics.get('total_differences', 'N/A')})"
        )


class DifferentialGenomicEncoder:
    """
    Main differential genomic encoder pipeline.

    Integrates all differential encoding components to encode experimental genomes
    as cryptographically verified differences from randomly selected reference genomes.

    The pipeline:
    1. Chunks the experimental genome into analysis-appropriate segments
    2. For each chunk, selects a random reference genome section
    3. Computes variant differences between experimental and reference
    4. Encodes differences into high-dimensional hypervectors
    5. Generates cryptographically bound metadata
    6. Optionally bundles chunk hypervectors into a single genome representation

    Example:
        >>> from pathlib import Path
        >>> import tempfile
        >>>
        >>> # Initialize components
        >>> tmpdir = Path(tempfile.mkdtemp())
        >>> ref_manager = SecureReferenceGenomeManager(tmpdir)
        >>> encoder = DifferentialHypervectorEncoder(dimension=10000)
        >>> rng = CryptoRNG()
        >>>
        >>> # Create pipeline
        >>> pipeline = DifferentialGenomicEncoder(
        ...     reference_manager=ref_manager,
        ...     hypervector_encoder=encoder,
        ...     crypto_rng=rng,
        ... )
        >>>
        >>> # Encode genome
        >>> result = pipeline.encode_experimental_genome(
        ...     experimental_genome=genome,
        ...     analysis_type=AnalysisType.SLIDING_WINDOW,
        ... )
        >>>
        >>> print(f"Encoded {len(result.hypervectors)} chunks")
        >>> print(f"Bundled hypervector: {result.bundled_hypervector.shape}")
    """

    def __init__(
        self,
        reference_manager: SecureReferenceGenomeManager,
        hypervector_encoder: DifferentialHypervectorEncoder,
        crypto_rng: Optional[CryptoRNG] = None,
    ):
        """
        Initialize differential genomic encoder.

        Args:
            reference_manager: Manager for reference genomes
            hypervector_encoder: Encoder for converting differences to hypervectors
            crypto_rng: Optional cryptographic RNG (creates new one if not provided)
        """
        self.reference_manager = reference_manager
        self.hypervector_encoder = hypervector_encoder
        self.crypto_rng = crypto_rng or CryptoRNG()

        logger.info(
            f"Initialized DifferentialGenomicEncoder: "
            f"references={reference_manager.reference_count}, "
            f"hv_dimension={hypervector_encoder.dimension}"
        )

    def encode_experimental_genome(
        self,
        experimental_genome: Genome,
        analysis_type: AnalysisType,
        master_seed: Optional[bytes] = None,
        feature_annotations: Optional[List[GenomicFeature]] = None,
        bundle_chunks: bool = True,
        progress_callback: Optional[callable] = None,
    ) -> EncodingResult:
        """
        Encode experimental genome as differential hypervectors.

        This is the main pipeline method that orchestrates the complete encoding process:
        1. Initialize CryptoRNG with master seed
        2. Create chunking strategy for analysis type
        3. Chunk experimental genome
        4. For each chunk:
           - Select random reference genome
           - Extract matching reference section
           - Compute variant differences
           - Encode into hypervector
           - Generate cryptographically bound metadata
        5. Optionally bundle chunks into single genome hypervector

        Args:
            experimental_genome: Genome to encode
            analysis_type: Type of analysis (determines chunking strategy)
            master_seed: Optional master seed for reproducibility
            feature_annotations: Optional genomic features for feature-based chunking
            bundle_chunks: Whether to bundle chunks into single genome hypervector
            progress_callback: Optional callback(chunk_idx, total_chunks, chunk)

        Returns:
            EncodingResult containing hypervectors, metadata, and statistics

        Raises:
            ValueError: If reference manager has no references
            RuntimeError: If encoding fails
        """
        logger.info(
            f"Starting encoding: genome={experimental_genome.genome_id}, "
            f"analysis={analysis_type.value}, bundle={bundle_chunks}"
        )

        # Validate reference manager
        if self.reference_manager.reference_count == 0:
            raise ValueError("Reference manager has no reference genomes")

        # 1. Initialize CryptoRNG with master seed
        if master_seed is not None:
            encoding_rng = CryptoRNG(master_seed=master_seed)
            logger.debug(f"Using provided master seed (length={len(master_seed)})")
        else:
            # Derive seed from crypto_rng
            master_seed = self.crypto_rng.derive_seed(
                experimental_genome.genome_id.encode()
            )
            encoding_rng = CryptoRNG(master_seed=master_seed)
            logger.debug("Generated master seed from genome ID")

        # 2. Create CryptographicChunker with appropriate strategy
        strategy = STRATEGY_CONFIGS[analysis_type]
        chunker = CryptographicChunker(strategy=strategy, crypto_rng=encoding_rng)

        logger.info(f"Created chunker with strategy: {strategy}")

        # 3. Chunk experimental genome
        chunks = chunker.chunk_genome(
            genome=experimental_genome,
            analysis_type=analysis_type,
            master_seed=master_seed,
            feature_annotations=feature_annotations,
        )

        logger.info(f"Chunked genome into {len(chunks)} chunks")

        if len(chunks) == 0:
            logger.warning("No chunks generated, returning empty result")
            return EncodingResult(
                hypervectors=[],
                metadata=[],
                statistics={"total_chunks": 0, "total_differences": 0},
            )

        # 4. Encode each chunk
        hypervectors = []
        metadata_list = []
        statistics = {
            "total_chunks": len(chunks),
            "total_differences": 0,
            "new_mutations": 0,
            "missing_variants": 0,
            "genotype_differences": 0,
            "chromosomes": set(),
        }

        for chunk_idx, chunk in enumerate(chunks):
            try:
                # Encode chunk
                hv, meta = self._encode_chunk(
                    chunk=chunk,
                    experimental_genome=experimental_genome,
                    analysis_type=analysis_type,
                    master_seed=master_seed,
                )

                hypervectors.append(hv)
                metadata_list.append(meta)

                # Update statistics
                statistics["total_differences"] += meta.difference_counts["total"]
                statistics["new_mutations"] += meta.difference_counts["new_mutations"]
                statistics["missing_variants"] += meta.difference_counts["missing_variants"]
                statistics["genotype_differences"] += meta.difference_counts["genotype_differences"]
                statistics["chromosomes"].add(chunk.chromosome)

                # Progress callback
                if progress_callback is not None:
                    progress_callback(chunk_idx, len(chunks), chunk)

                logger.debug(
                    f"Encoded chunk {chunk_idx + 1}/{len(chunks)}: "
                    f"{chunk.chromosome}:{chunk.start_position}-{chunk.end_position}, "
                    f"differences={meta.difference_counts['total']}"
                )

            except Exception as e:
                logger.error(
                    f"Failed to encode chunk {chunk_idx}: {chunk.chromosome}:{chunk.start_position}-{chunk.end_position}",
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Chunk encoding failed at {chunk_idx}/{len(chunks)}: {e}"
                ) from e

        # Convert chromosomes set to list for serialization
        statistics["chromosomes"] = sorted(list(statistics["chromosomes"]))

        logger.info(
            f"Encoding complete: {len(hypervectors)} chunks, "
            f"{statistics['total_differences']} total differences"
        )

        # 5. Bundle chunks if requested
        bundled_hypervector = None
        if bundle_chunks and len(hypervectors) > 0:
            bundled_hypervector = self.bundle_hypervectors(hypervectors)
            logger.info(f"Bundled {len(hypervectors)} chunks into genome hypervector")

        return EncodingResult(
            hypervectors=hypervectors,
            metadata=metadata_list,
            bundled_hypervector=bundled_hypervector,
            statistics=statistics,
        )

    def _encode_chunk(
        self,
        chunk: GenomeChunk,
        experimental_genome: Genome,
        analysis_type: AnalysisType,
        master_seed: bytes,
    ) -> Tuple[np.ndarray, DifferentialEncodingMetadata]:
        """
        Encode a single genomic chunk.

        Args:
            chunk: Chunk to encode
            experimental_genome: Experimental genome
            analysis_type: Analysis type
            master_seed: Master seed for reproducibility

        Returns:
            Tuple of (hypervector, metadata)
        """
        # a. Select random reference genome
        # Derive a seed for reference selection
        ref_selection_seed = self.crypto_rng.derive_seed(
            chunk.chunk_id + experimental_genome.genome_id.encode()
        )
        reference_genome = self.reference_manager.get_random_reference(
            seed=ref_selection_seed,
            exclude=[experimental_genome.genome_id],
        )

        logger.debug(
            f"Selected reference: {reference_genome.genome_id} "
            f"for chunk {chunk.chromosome}:{chunk.start_position}-{chunk.end_position}"
        )

        # b. Extract matching reference section
        reference_section = reference_genome.get_section(
            chromosome=chunk.chromosome,
            start=chunk.start_position,
            end=chunk.end_position,
        )

        # c. Extract experimental section
        experimental_section = experimental_genome.get_chromosome_section(
            chromosome=chunk.chromosome,
            start=chunk.start_position,
            end=chunk.end_position,
        )

        # d. Compute variant differences
        differences = compute_variant_differences(
            experimental_section,
            reference_section,
        )

        logger.debug(
            f"Computed {len(differences)} differences "
            f"({sum(1 for d in differences if d.is_new_mutation)} new, "
            f"{sum(1 for d in differences if d.is_missing)} missing, "
            f"{sum(1 for d in differences if d.is_genotype_diff)} genotype)"
        )

        # e. Generate chunk data and reference data for binding
        chunk_data = self._serialize_chunk_data(chunk, experimental_section)
        reference_data = self._serialize_reference_data(reference_section)

        # f. Compute reference hash
        reference_hash_hex = compute_reference_hash(reference_genome)
        reference_hash = bytes.fromhex(reference_hash_hex)

        # g. Derive reference seed
        reference_seed = self.crypto_rng.derive_seed(
            reference_genome.genome_id.encode()
        )

        # h. Create metadata
        metadata = create_metadata_from_chunk(
            chunk_id=chunk.chunk_id,
            chromosome=chunk.chromosome,
            start_position=chunk.start_position,
            end_position=chunk.end_position,
            reference_genome_id=reference_genome.genome_id,
            reference_seed=reference_seed,
            reference_hash=reference_hash,
            chunking_strategy=analysis_type.value,
            chunking_seed=master_seed,
            analysis_type=analysis_type,
            new_mutations=sum(1 for d in differences if d.is_new_mutation),
            missing_variants=sum(1 for d in differences if d.is_missing),
            genotype_differences=sum(1 for d in differences if d.is_genotype_diff),
            chunk_data=chunk_data,
            reference_data=reference_data,
        )

        # i. Encode into hypervector
        hypervector = self.hypervector_encoder.encode_difference_vector(
            differences=differences,
            metadata=metadata,
        )

        return hypervector, metadata

    def bundle_hypervectors(
        self,
        hypervectors: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """
        Bundle multiple chunk hypervectors into a single genome representation.

        Uses superposition (element-wise sum + normalization) to combine chunk
        hypervectors into a single hypervector representing the entire genome.

        This allows:
        - Efficient storage (single vector instead of many chunks)
        - Fast similarity comparison between genomes
        - Preservation of distributed information across chunks

        Args:
            hypervectors: List of chunk hypervectors to bundle
            weights: Optional weights for each hypervector (default: equal weights)

        Returns:
            Bundled hypervector (normalized to unit length)

        Raises:
            ValueError: If hypervectors list is empty or vectors have different dimensions
        """
        if not hypervectors:
            raise ValueError("Cannot bundle empty list of hypervectors")

        # Validate dimensions
        dimension = len(hypervectors[0])
        for i, hv in enumerate(hypervectors):
            if len(hv) != dimension:
                raise ValueError(
                    f"Dimension mismatch: hypervector {i} has dimension {len(hv)}, "
                    f"expected {dimension}"
                )

        # Use hypervector encoder's bundling method
        bundled = self.hypervector_encoder._bundle(hypervectors, weights)

        logger.debug(
            f"Bundled {len(hypervectors)} hypervectors into "
            f"{dimension}D representation (norm={np.linalg.norm(bundled):.4f})"
        )

        return bundled

    def _compute_binding(self, chunk_id: bytes, reference_id: bytes) -> bytes:
        """
        Compute cryptographic binding between chunk and reference.

        Uses HMAC-SHA256 to bind chunk_id to reference_id, ensuring that
        the chunk data cannot be used with a different reference genome.

        Args:
            chunk_id: Chunk identifier (32 bytes)
            reference_id: Reference genome identifier (bytes)

        Returns:
            Binding hash (32 bytes)
        """
        return compute_chunk_reference_binding(chunk_id, reference_id.encode())

    def _serialize_chunk_data(
        self,
        chunk: GenomeChunk,
        section: GenomeSection,
    ) -> bytes:
        """
        Serialize chunk data for cryptographic binding.

        Args:
            chunk: Genome chunk
            section: Genome section with variants

        Returns:
            Serialized chunk data
        """
        # Combine chunk metadata and variant data
        data_parts = [
            chunk.chromosome.encode(),
            chunk.start_position.to_bytes(8, byteorder='big'),
            chunk.end_position.to_bytes(8, byteorder='big'),
            str(len(section.variants)).encode(),
        ]

        # Add variant positions
        for variant in section.variants:
            data_parts.append(variant.position.to_bytes(8, byteorder='big'))

        return b''.join(data_parts)

    def _serialize_reference_data(self, section: GenomeSection) -> bytes:
        """
        Serialize reference section data for cryptographic binding.

        Args:
            section: Reference genome section

        Returns:
            Serialized reference data
        """
        # Combine reference section metadata
        data_parts = [
            section.chromosome.encode(),
            section.start_position.to_bytes(8, byteorder='big'),
            section.end_position.to_bytes(8, byteorder='big'),
            str(len(section.variants)).encode(),
        ]

        # Add variant data
        for variant in section.variants:
            data_parts.extend([
                variant.position.to_bytes(8, byteorder='big'),
                variant.ref.encode(),
                variant.alt.encode(),
            ])

        return b''.join(data_parts)

    def decode_and_verify(
        self,
        metadata: DifferentialEncodingMetadata,
    ) -> bool:
        """
        Verify cryptographic binding of encoded chunk.

        Args:
            metadata: Chunk metadata to verify

        Returns:
            True if binding is valid, False otherwise
        """
        # Verify that the chunk was bound to the claimed reference genome
        chunk_data = metadata.chunk_id
        reference_data = metadata.reference_hash

        return metadata.verify_binding(chunk_data, reference_data)

    def get_statistics(self, result: EncodingResult) -> Dict[str, Any]:
        """
        Get detailed statistics about encoding result.

        Args:
            result: Encoding result

        Returns:
            Dictionary of statistics
        """
        stats = dict(result.statistics)

        # Add per-chunk statistics
        if result.metadata:
            chunk_sizes = [
                meta.end_position - meta.start_position
                for meta in result.metadata
            ]
            difference_counts = [
                meta.difference_counts["total"]
                for meta in result.metadata
            ]

            stats.update({
                "chunk_size_mean": np.mean(chunk_sizes),
                "chunk_size_std": np.std(chunk_sizes),
                "chunk_size_min": np.min(chunk_sizes),
                "chunk_size_max": np.max(chunk_sizes),
                "differences_per_chunk_mean": np.mean(difference_counts),
                "differences_per_chunk_std": np.std(difference_counts),
                "differences_per_chunk_min": np.min(difference_counts),
                "differences_per_chunk_max": np.max(difference_counts),
            })

        return stats
