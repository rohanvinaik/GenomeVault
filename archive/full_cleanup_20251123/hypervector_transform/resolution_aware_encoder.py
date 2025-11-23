"""
Resolution-Aware HDV Encoder

Configurable granularity for different analysis types with Secure Guide Reference System.

Resolution Presets:
- snp_only: 2K dimensions, variants only (~512 bytes)
- clinical_risk: 10K dimensions, clinical variants + reference (~2 KB)
- pharmacogenomics: 15K dimensions, drug response variants + reference (~3 KB)
- full_nucleotide: 50K dimensions, full nucleotide representation (~10 KB)

Architecture:
- GDiff: Local database (~150 MB compressed)
- HDV: On-demand encoding (10-300ms)
- Network: HDV only (512 B - 10 KB transmitted)

See docs/SECURE_GUIDE_REFERENCE_SYSTEM.md for details.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Set
from pathlib import Path
import logging

import numpy as np

from genomevault.query.nucleotide_resolver import NucleotideResolver, NucleotideQuery
from genomevault.differential_encoding.gdiff.schema import GDiffDocument

logger = logging.getLogger(__name__)


class ResolutionPreset(Enum):
    """Pre-configured resolution presets for common analysis types"""

    SNP_ONLY = "snp_only"
    """Variants only, no reference nucleotides. Fastest, minimal size (512 B)."""

    CLINICAL_RISK = "clinical_risk"
    """Clinical variants + sparse reference points. Balanced (2 KB)."""

    PHARMACOGENOMICS = "pharmacogenomics"
    """Drug response variants + metabolism genes. Targeted (3 KB)."""

    ANCESTRY_INFERENCE = "ancestry_inference"
    """Population-specific variants. Specialized (5 KB)."""

    FULL_NUCLEOTIDE = "full_nucleotide"
    """Complete nucleotide representation. Maximum resolution (10 KB)."""


@dataclass
class ResolutionConfig:
    """Configuration for resolution-aware encoding"""

    preset: ResolutionPreset
    dimension: int
    """HDV dimension"""

    include_variants: bool = True
    """Include encoded variants from GDiff"""

    include_reference: bool = False
    """Include reference nucleotides (requires SGRS)"""

    reference_sampling_rate: float = 0.0
    """Fraction of reference genome to sample (0.0 = none, 1.0 = all)"""

    target_regions: Optional[List[str]] = None
    """Specific regions to include (e.g., ["BRCA1", "BRCA2"] for clinical)"""

    def __post_init__(self):
        """Validate configuration"""
        if self.include_reference and self.reference_sampling_rate == 0.0:
            raise ValueError("include_reference=True requires reference_sampling_rate > 0")


# Pre-configured resolution presets
RESOLUTION_PRESETS: Dict[ResolutionPreset, ResolutionConfig] = {
    ResolutionPreset.SNP_ONLY: ResolutionConfig(
        preset=ResolutionPreset.SNP_ONLY,
        dimension=2_000,
        include_variants=True,
        include_reference=False,
        reference_sampling_rate=0.0,
    ),
    ResolutionPreset.CLINICAL_RISK: ResolutionConfig(
        preset=ResolutionPreset.CLINICAL_RISK,
        dimension=10_000,
        include_variants=True,
        include_reference=True,
        reference_sampling_rate=0.001,  # 0.1% of reference genome
    ),
    ResolutionPreset.PHARMACOGENOMICS: ResolutionConfig(
        preset=ResolutionPreset.PHARMACOGENOMICS,
        dimension=15_000,
        include_variants=True,
        include_reference=True,
        reference_sampling_rate=0.005,  # 0.5% (metabolism genes)
    ),
    ResolutionPreset.ANCESTRY_INFERENCE: ResolutionConfig(
        preset=ResolutionPreset.ANCESTRY_INFERENCE,
        dimension=20_000,
        include_variants=True,
        include_reference=True,
        reference_sampling_rate=0.01,  # 1% (population markers)
    ),
    ResolutionPreset.FULL_NUCLEOTIDE: ResolutionConfig(
        preset=ResolutionPreset.FULL_NUCLEOTIDE,
        dimension=50_000,
        include_variants=True,
        include_reference=True,
        reference_sampling_rate=1.0,  # Complete genome
    ),
}


class ResolutionAwareEncoder:
    """
    HDV encoder with configurable resolution for different analysis types.

    Example:
        # Simple SNP-only encoding (minimal size)
        encoder = ResolutionAwareEncoder(
            gdiff_path=Path("experimental.gdiff.gz"),
            preset=ResolutionPreset.SNP_ONLY
        )
        hdv = encoder.encode()  # ~512 bytes

        # Clinical risk assessment (with reference context)
        encoder = ResolutionAwareEncoder(
            gdiff_path=Path("experimental.gdiff.gz"),
            local_guide_dir=Path("data/guides"),
            preset=ResolutionPreset.CLINICAL_RISK
        )
        hdv = encoder.encode()  # ~2 KB

        # Full nucleotide representation
        encoder = ResolutionAwareEncoder(
            gdiff_path=Path("experimental.gdiff.gz"),
            local_guide_dir=Path("data/guides"),
            preset=ResolutionPreset.FULL_NUCLEOTIDE
        )
        hdv = encoder.encode()  # ~10 KB
    """

    def __init__(
        self,
        gdiff_path: Path,
        local_guide_dir: Optional[Path] = None,
        preset: ResolutionPreset = ResolutionPreset.SNP_ONLY,
        custom_config: Optional[ResolutionConfig] = None,
    ):
        """
        Initialize resolution-aware encoder.

        Args:
            gdiff_path: Path to GDiff file
            local_guide_dir: Directory with guide FASTAs (required for reference encoding)
            preset: Resolution preset (ignored if custom_config provided)
            custom_config: Custom resolution configuration (overrides preset)

        Raises:
            ValueError: If reference encoding requested but local_guide_dir not provided
        """
        self.gdiff_path = gdiff_path
        self.local_guide_dir = local_guide_dir

        # Load configuration
        if custom_config:
            self.config = custom_config
        else:
            self.config = RESOLUTION_PRESETS[preset]

        logger.info(f"Resolution-aware encoder initialized: {self.config.preset.value}")
        logger.info(f"  Dimension: {self.config.dimension:,}D")
        logger.info(f"  Include variants: {self.config.include_variants}")
        logger.info(f"  Include reference: {self.config.include_reference}")
        if self.config.include_reference:
            logger.info(f"  Reference sampling: {self.config.reference_sampling_rate*100:.2f}%")

        # Validate
        if self.config.include_reference and not self.local_guide_dir:
            raise ValueError(
                "local_guide_dir required for reference encoding "
                f"(preset: {self.config.preset.value})"
            )

        # Load GDiff
        self.gdiff = self._load_gdiff()

        # Initialize nucleotide resolver if needed
        self.resolver = None
        if self.config.include_reference:
            logger.info("Initializing nucleotide resolver...")
            self.resolver = NucleotideResolver(
                gdiff_path=self.gdiff_path,
                local_guide_dir=self.local_guide_dir,
                cache_guide_fastas=True
            )
            logger.info("  ✓ Resolver ready")

    def encode(self) -> np.ndarray:
        """
        Encode GDiff to hypervector with configured resolution.

        Returns:
            HDV (dimension × 1 array of +1/-1 values)
        """
        logger.info(f"Encoding with resolution: {self.config.preset.value}")

        # Initialize hypervector
        hdv = np.ones(self.config.dimension, dtype=np.int8)

        # Step 1: Encode variants if requested
        if self.config.include_variants:
            logger.info(f"Encoding {len(self.gdiff.differential_variants)} variants...")
            self._encode_variants(hdv)

        # Step 2: Encode reference nucleotides if requested
        if self.config.include_reference:
            logger.info(f"Encoding reference nucleotides (sampling rate: {self.config.reference_sampling_rate*100:.2f}%)...")
            self._encode_reference(hdv)

        logger.info(f"✓ Encoding complete: {hdv.nbytes:,} bytes")
        return hdv

    def _encode_variants(self, hdv: np.ndarray):
        """
        Encode variants into hypervector.

        Uses position-based hashing to map variants to HDV dimensions.

        Args:
            hdv: Hypervector to modify in-place
        """
        for variant in self.gdiff.differential_variants:
            # Hash variant position to HDV dimension
            variant_hash = hash(f"{variant.chrom}:{variant.pos}:{variant.ref}:{variant.alt}")
            dim_idx = abs(variant_hash) % self.config.dimension

            # Flip bit based on alt allele
            nucleotide_value = self._nucleotide_to_value(variant.alt)
            hdv[dim_idx] *= nucleotide_value

    def _encode_reference(self, hdv: np.ndarray):
        """
        Encode reference nucleotides into hypervector.

        Samples reference genome at configured rate using nucleotide resolver.

        Args:
            hdv: Hypervector to modify in-place
        """
        if not self.resolver:
            raise ValueError("Resolver not initialized (should not happen)")

        # Determine sampling positions
        sampling_positions = self._generate_sampling_positions()

        logger.info(f"Sampling {len(sampling_positions)} reference positions...")

        # Query nucleotides and encode
        for chrom, pos in sampling_positions:
            result = self.resolver.query(chrom=chrom, pos=pos)

            # Hash position to HDV dimension
            pos_hash = hash(f"{chrom}:{pos}")
            dim_idx = abs(pos_hash) % self.config.dimension

            # Flip bit based on nucleotide
            nucleotide_value = self._nucleotide_to_value(result.nucleotide)
            hdv[dim_idx] *= nucleotide_value

    def _generate_sampling_positions(self) -> List[Tuple[str, int]]:
        """
        Generate reference genome sampling positions based on configuration.

        Returns:
            List of (chrom, pos) tuples
        """
        # Simple uniform sampling strategy
        # In production, this would be more sophisticated (e.g., clinically relevant regions)

        positions = []

        # Get chromosomes from GDiff variants
        chromosomes = list(set(v.chrom for v in self.gdiff.differential_variants))

        # Estimate genome size (approximate)
        genome_size_estimate = 3_000_000_000  # 3 Gbp
        num_samples = int(genome_size_estimate * self.config.reference_sampling_rate)

        # Uniform sampling across chromosomes
        samples_per_chrom = num_samples // len(chromosomes) if chromosomes else 0

        for chrom in chromosomes:
            # Sample positions uniformly across chromosome
            # (Simplified - in production, would use actual chromosome lengths)
            chrom_length_estimate = genome_size_estimate // len(chromosomes)
            step = chrom_length_estimate // samples_per_chrom if samples_per_chrom > 0 else chrom_length_estimate

            for i in range(samples_per_chrom):
                pos = i * step
                positions.append((chrom, pos))

        return positions[:num_samples]  # Trim to exact count

    def _nucleotide_to_value(self, nucleotide: str) -> int:
        """
        Convert nucleotide to HDV value (+1 or -1).

        Args:
            nucleotide: A, T, G, C, or N

        Returns:
            +1 or -1
        """
        # Simple mapping: A/G → +1, T/C → -1, N → +1 (default)
        if nucleotide in "AGN":
            return 1
        else:
            return -1

    def _load_gdiff(self) -> GDiffDocument:
        """Load GDiff file"""
        import json
        import gzip

        if not self.gdiff_path.exists():
            raise FileNotFoundError(f"GDiff not found: {self.gdiff_path}")

        open_func = gzip.open if str(self.gdiff_path).endswith('.gz') else open
        with open_func(self.gdiff_path, 'rt') as f:
            gdiff_dict = json.load(f)

        return GDiffDocument(**gdiff_dict)

    def close(self):
        """Close resources"""
        if self.resolver:
            self.resolver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
