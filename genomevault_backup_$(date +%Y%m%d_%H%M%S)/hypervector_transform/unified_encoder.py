"""
Unified Encoding Interface for GenomeVault

This module provides a unified interface supporting both:
- Legacy direct variant encoding (HypervectorEncoder)
- New differential encoding (DifferentialGenomicEncoder)

The encoder automatically selects the appropriate backend based on:
- Feature flags
- Encoding mode parameter
- Data type and analysis requirements
"""

from __future__ import annotations

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional, Dict, List
from pathlib import Path
import tempfile

import numpy as np
import torch

from genomevault.core.constants import OmicsType
from genomevault.utils.config import CompressionTier
from genomevault.differential_encoding import (
    DifferentialGenomicEncoder,
    DifferentialHypervectorEncoder,
    SecureReferenceGenomeManager,
    AnalysisType,
    Genome,
    Variant,
    EncodedGenome,
    CryptoRNG,
)
from .encoding import HypervectorEncoder, HypervectorConfig, ProjectionType

logger = logging.getLogger(__name__)


class EncodingMode(str, Enum):
    """Encoding mode selection."""

    LEGACY = "legacy"  # Original direct encoding
    DIFFERENTIAL = "differential"  # New differential encoding
    AUTO = "auto"  # Automatic selection based on data type


@dataclass
class EncodingFeatureFlags:
    """
    Feature flags for controlling encoding behavior.

    Allows gradual rollout and A/B testing of differential encoding.
    """

    enable_differential: bool = True
    differential_by_default: bool = False
    legacy_fallback: bool = True
    enable_hybrid_mode: bool = False

    # Performance flags
    enable_caching: bool = True
    enable_batching: bool = True

    # Compatibility flags
    strict_compatibility_mode: bool = False

    @classmethod
    def from_env(cls) -> "EncodingFeatureFlags":
        """Load feature flags from environment variables."""
        import os

        return cls(
            enable_differential=os.getenv(
                "GENOMEVAULT_ENABLE_DIFFERENTIAL", "true"
            ).lower() == "true",
            differential_by_default=os.getenv(
                "GENOMEVAULT_DIFFERENTIAL_DEFAULT", "false"
            ).lower() == "true",
            legacy_fallback=os.getenv(
                "GENOMEVAULT_LEGACY_FALLBACK", "true"
            ).lower() == "true",
            enable_hybrid_mode=os.getenv(
                "GENOMEVAULT_HYBRID_MODE", "false"
            ).lower() == "true",
            enable_caching=os.getenv(
                "GENOMEVAULT_ENABLE_CACHING", "true"
            ).lower() == "true",
            enable_batching=os.getenv(
                "GENOMEVAULT_ENABLE_BATCHING", "true"
            ).lower() == "true",
            strict_compatibility_mode=os.getenv(
                "GENOMEVAULT_STRICT_COMPATIBILITY", "false"
            ).lower() == "true",
        )


class UnifiedGenomicEncoder:
    """
    Unified encoder supporting both legacy and differential encoding.

    This class provides a single interface that routes to the appropriate
    encoding backend based on the mode, feature flags, and data characteristics.

    Attributes:
        mode: Encoding mode (legacy, differential, or auto)
        feature_flags: Feature flags controlling behavior
        legacy_encoder: Legacy HypervectorEncoder instance
        differential_encoder: DifferentialGenomicEncoder instance
        reference_manager: Reference genome manager for differential encoding

    Example:
        >>> # Use differential encoding explicitly
        >>> encoder = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL)
        >>> result = encoder.encode_genome(genome, AnalysisType.SLIDING_WINDOW)
        >>>
        >>> # Use auto mode (selects best based on data)
        >>> encoder = UnifiedGenomicEncoder(mode=EncodingMode.AUTO)
        >>> result = encoder.encode(features, OmicsType.GENOMIC)
    """

    def __init__(
        self,
        mode: EncodingMode = EncodingMode.AUTO,
        feature_flags: Optional[EncodingFeatureFlags] = None,
        legacy_config: Optional[HypervectorConfig] = None,
        reference_dir: Optional[Path] = None,
        dimension: int = 10000,
        seed: int = 42,
    ):
        """
        Initialize unified encoder.

        Args:
            mode: Encoding mode selection
            feature_flags: Feature flags (loads from env if None)
            legacy_config: Config for legacy encoder
            reference_dir: Directory containing reference genomes
            dimension: Hypervector dimension
            seed: Random seed for reproducibility
        """
        self.mode = mode
        self.feature_flags = feature_flags or EncodingFeatureFlags.from_env()
        self.dimension = dimension
        self.seed = seed

        # Initialize legacy encoder
        if legacy_config is None:
            legacy_config = HypervectorConfig(
                dimension=dimension,
                projection_type=ProjectionType.RANDOM_GAUSSIAN,
            )

        self.legacy_encoder = HypervectorEncoder(config=legacy_config)

        # Initialize differential encoder components if enabled
        self.differential_encoder: Optional[DifferentialGenomicEncoder] = None
        self.reference_manager: Optional[SecureReferenceGenomeManager] = None

        if self.feature_flags.enable_differential:
            self._initialize_differential_encoder(reference_dir)

        logger.info(
            f"Initialized UnifiedGenomicEncoder: mode={mode}, "
            f"differential_enabled={self.feature_flags.enable_differential}, "
            f"dimension={dimension}"
        )

    def _initialize_differential_encoder(
        self, reference_dir: Optional[Path]
    ):
        """Initialize differential encoding components."""
        try:
            # Create reference manager
            if reference_dir is None:
                # Use temporary directory if none provided
                reference_dir = Path(tempfile.mkdtemp(prefix="genomevault_refs_"))
                logger.warning(
                    f"No reference directory provided, using temp: {reference_dir}"
                )

            self.reference_manager = SecureReferenceGenomeManager(
                reference_dir=reference_dir
            )

            # Create hypervector encoder for differential encoding
            hv_encoder = DifferentialHypervectorEncoder(
                dimension=self.dimension,
                seed=self.seed
            )

            # Create differential encoder
            self.differential_encoder = DifferentialGenomicEncoder(
                reference_manager=self.reference_manager,
                hypervector_encoder=hv_encoder,
            )

            logger.info(
                f"Initialized differential encoder: references={self.reference_manager.reference_count}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize differential encoder: {e}", exc_info=True)

            if not self.feature_flags.legacy_fallback:
                raise

            logger.warning("Falling back to legacy encoder only")
            self.differential_encoder = None

    def _select_encoding_mode(
        self,
        data_type: Optional[str] = None,
        explicit_mode: Optional[EncodingMode] = None,
    ) -> EncodingMode:
        """
        Select encoding mode based on configuration and data.

        Args:
            data_type: Type of data being encoded
            explicit_mode: Explicitly requested mode

        Returns:
            Selected encoding mode
        """
        # Explicit mode takes precedence
        if explicit_mode is not None:
            return explicit_mode

        # Use configured mode
        if self.mode != EncodingMode.AUTO:
            return self.mode

        # AUTO mode: decide based on data type and flags
        if self.feature_flags.differential_by_default:
            if self.differential_encoder is not None:
                return EncodingMode.DIFFERENTIAL

        # Default to legacy for backward compatibility
        return EncodingMode.LEGACY

    def encode(
        self,
        features: Any,
        omics_type: OmicsType,
        compression_tier: Optional[CompressionTier] = None,
        mode: Optional[EncodingMode] = None,
    ) -> torch.Tensor | np.ndarray:
        """
        Encode features using selected backend.

        This method provides backward-compatible interface with the legacy
        encoder while supporting differential encoding when appropriate.

        Args:
            features: Feature data to encode
            omics_type: Type of omics data
            compression_tier: Optional compression tier
            mode: Explicit encoding mode override

        Returns:
            Encoded hypervector

        Example:
            >>> encoder = UnifiedGenomicEncoder()
            >>> features = {...}  # Feature dict
            >>> vector = encoder.encode(features, OmicsType.GENOMIC)
        """
        selected_mode = self._select_encoding_mode(
            data_type=omics_type.value,
            explicit_mode=mode
        )

        logger.debug(
            f"Encoding with mode={selected_mode}, omics_type={omics_type}"
        )

        if selected_mode == EncodingMode.LEGACY:
            return self.legacy_encoder.encode(features, omics_type, compression_tier)

        elif selected_mode == EncodingMode.DIFFERENTIAL:
            if self.differential_encoder is None:
                if self.feature_flags.legacy_fallback:
                    logger.warning("Differential encoder not available, falling back to legacy")
                    return self.legacy_encoder.encode(features, omics_type, compression_tier)
                else:
                    raise RuntimeError("Differential encoder not initialized")

            # Convert features to differential encoding format
            # This is a simplified conversion - in production you'd have
            # more sophisticated feature extraction
            logger.warning(
                "Direct feature encoding with differential mode not yet fully "
                "implemented. Use encode_genome() for full differential encoding."
            )
            return self.legacy_encoder.encode(features, omics_type, compression_tier)

        else:
            raise ValueError(f"Unknown encoding mode: {selected_mode}")

    def encode_genome(
        self,
        genome: Genome,
        analysis_type: AnalysisType,
        master_seed: Optional[bytes] = None,
        bundle_chunks: bool = True,
        mode: Optional[EncodingMode] = None,
    ) -> EncodedGenome:
        """
        Encode complete genome using differential encoding.

        This is the primary method for differential encoding, providing full
        cryptographic security and compression benefits.

        Args:
            genome: Genome to encode
            analysis_type: Type of analysis/chunking strategy
            master_seed: Optional master seed for reproducibility
            bundle_chunks: Whether to create bundled hypervector
            mode: Explicit encoding mode override

        Returns:
            EncodedGenome with hypervectors and metadata

        Raises:
            RuntimeError: If differential encoder not available

        Example:
            >>> genome = Genome(genome_id="patient_001", assembly="GRCh38", ...)
            >>> encoder = UnifiedGenomicEncoder(mode=EncodingMode.DIFFERENTIAL)
            >>> encoded = encoder.encode_genome(genome, AnalysisType.SLIDING_WINDOW)
            >>> encoded.save("patient_001.enc.gz")
        """
        selected_mode = self._select_encoding_mode(
            data_type="genome",
            explicit_mode=mode
        )

        if selected_mode != EncodingMode.DIFFERENTIAL:
            raise ValueError(
                f"encode_genome() requires differential mode, got {selected_mode}. "
                f"Use mode=EncodingMode.DIFFERENTIAL explicitly."
            )

        if self.differential_encoder is None:
            raise RuntimeError(
                "Differential encoder not initialized. "
                "Ensure references are loaded and feature flags are enabled."
            )

        logger.info(
            f"Encoding genome {genome.genome_id} with differential mode, "
            f"analysis_type={analysis_type.value}"
        )

        # Encode using differential encoder
        result = self.differential_encoder.encode_experimental_genome(
            experimental_genome=genome,
            analysis_type=analysis_type,
            master_seed=master_seed,
            bundle_chunks=bundle_chunks,
        )

        # Create EncodedGenome
        if master_seed is None:
            # Generate deterministic seed from genome ID
            crypto_rng = CryptoRNG()
            master_seed = crypto_rng.derive_seed(genome.genome_id.encode())

        encoded = EncodedGenome.from_encoding_result(
            genome_id=genome.genome_id,
            assembly=genome.assembly,
            result=result,
            master_seed=master_seed,
        )

        logger.info(
            f"Genome encoding complete: {len(result.hypervectors)} chunks, "
            f"bundled={result.bundled_hypervector is not None}"
        )

        return encoded

    def get_encoding_info(self) -> Dict[str, Any]:
        """
        Get information about current encoding configuration.

        Returns:
            Dictionary with encoding configuration details
        """
        return {
            "mode": self.mode.value,
            "dimension": self.dimension,
            "seed": self.seed,
            "feature_flags": {
                "differential_enabled": self.feature_flags.enable_differential,
                "differential_by_default": self.feature_flags.differential_by_default,
                "legacy_fallback": self.feature_flags.legacy_fallback,
                "hybrid_mode": self.feature_flags.enable_hybrid_mode,
            },
            "encoders": {
                "legacy_available": self.legacy_encoder is not None,
                "differential_available": self.differential_encoder is not None,
                "reference_count": (
                    self.reference_manager.reference_count
                    if self.reference_manager
                    else 0
                ),
            },
        }

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"UnifiedGenomicEncoder("
            f"mode={self.mode.value}, "
            f"differential={'✓' if self.differential_encoder else '✗'}, "
            f"dimension={self.dimension})"
        )
