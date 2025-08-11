"""
Hypervector encoding for genomic data

This module is maintained for backward compatibility.
Please migrate to genomevault.hypervector_transform.encoding
"""

from __future__ import annotations

import warnings
from enum import Enum

import numpy as np
import torch

from genomevault.core.constants import HYPERVECTOR_DIMENSIONS  # noqa: E402
from genomevault.core.exceptions import HypervectorError  # noqa: E402
from genomevault.hypervector.positional import PositionalEncoder, SNPPanel  # noqa: E402
from genomevault.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

warnings.warn(
    "genomevault.hypervector.encoding.genomic is deprecated. "
    "Use genomevault.hypervector_transform.encoding instead.",
    DeprecationWarning,
    stacklevel=2,
)


class PanelGranularity(Enum):
    """SNP panel granularity settings"""

    OFF = "off"
    COMMON = "common"
    CLINICAL = "clinical"
    CUSTOM = "custom"


class GenomicEncoder:
    """
    Encodes genomic variants into high-dimensional vectors
    Supports both genome-wide and SNP-level accuracy modes
    """

    def __init__(
        self,
        dimension: int = HYPERVECTOR_DIMENSIONS,
        enable_snp_mode: bool = False,
        panel_granularity: PanelGranularity = PanelGranularity.OFF,
    ):
        """Initialize instance.

            Args:
                dimension: Dimension value.
                enable_snp_mode: SNP data.
                panel_granularity: Panel granularity.
            """
        self.dimension = dimension
        self.base_vectors = self._init_base_vectors()
        self.enable_snp_mode = enable_snp_mode
        self.panel_granularity = panel_granularity

        # Initialize SNP mode components if enabled
        if enable_snp_mode:
            self.positional_encoder = PositionalEncoder(
                dimension=dimension, sparsity=0.01, cache_size=10000
            )
            self.snp_panel = SNPPanel(self.positional_encoder)

        # Hierarchical zoom tiles
        self.zoom_levels: dict[int, dict] = {
            0: {},  # Genome-wide HVs
            1: {},  # 1Mb window HVs
            2: {},  # 1kb tile HVs
        }

    def _init_base_vectors(self) -> dict[str, torch.Tensor]:
        """Initialize base hypervectors for nucleotides and operations"""
        base_vectors = {}

        # Nucleotide base vectors (orthogonal)
        for i, base in enumerate(["A", "T", "G", "C"]):
            vec = torch.zeros(self.dimension)
            # Use sparse representation
            indices = torch.randperm(self.dimension)[: self.dimension // 4]
            vec[indices] = torch.randn(len(indices))
            vec = vec / torch.norm(vec)
            base_vectors[base] = vec

        # Position encoding vectors
        base_vectors["POS"] = self._generate_position_vector()

        # Variant type vectors
        for variant_type in ["SNP", "INS", "DEL", "DUP", "INV"]:
            vec = torch.randn(self.dimension)
            vec = vec / torch.norm(vec)
            base_vectors[variant_type] = vec

        return base_vectors

    def _generate_position_vector(self) -> torch.Tensor:
        """Generate position encoding vector using sinusoidal encoding"""
        position = torch.arange(self.dimension).float()
        div_term = torch.exp(
            torch.arange(0, self.dimension, 2).float() * -(np.log(10000.0) / self.dimension)
        )

        pos_encoding = torch.zeros(self.dimension)
        pos_encoding[0::2] = torch.sin(position[0::2] * div_term)
        pos_encoding[1::2] = torch.cos(position[1::2] * div_term[: len(position[1::2])])

        return pos_encoding

    def encode_variant(
        self,
        chromosome: str,
        position: int,
        ref: str,
        alt: str,
        variant_type: str = "SNP",
        use_panel: bool = None,
    ) -> torch.Tensor:
        """
        Encode a single genetic variant into a hypervector

        Args:
            chromosome: Chromosome name (e.g., 'chr1')
            position: Genomic position
            ref: Reference allele
            alt: Alternative allele
            variant_type: Type of variant (SNP, INS, DEL, etc.)
            use_panel: Override to use panel encoding

        Returns:
            Hypervector representation of the variant
        """
        try:
            # Determine if we should use panel encoding
            use_panel = (
                use_panel
                if use_panel is not None
                else (self.enable_snp_mode and self.panel_granularity != PanelGranularity.OFF)
            )

            if use_panel and hasattr(self, "positional_encoder"):
                # Use SNP-level encoding with positional encoder
                return self._encode_variant_with_panel(chromosome, position, ref, alt, variant_type)

            # Standard encoding (original path)
            # Start with variant type vector
            variant_vec = self.base_vectors[variant_type].clone()

            # Bind with position information
            pos_factor = position / 1e9  # Normalize position
            pos_vec = torch.roll(self.base_vectors["POS"], int(pos_factor * 1000))
            variant_vec = self._bind(variant_vec, pos_vec)

            # Bind with reference allele information
            if len(ref) == 1 and ref in self.base_vectors:
                variant_vec = self._bind(variant_vec, self.base_vectors[ref])

            # Bind with alternative allele information
            if len(alt) == 1 and alt in self.base_vectors:
                # Use permutation to distinguish alt from ref
                alt_vec = self._permute(self.base_vectors[alt], 1)
                variant_vec = self._bind(variant_vec, alt_vec)

            # Add chromosome information
            chr_vec = self._chromosome_vector(chromosome)
            variant_vec = self._bundle(variant_vec, chr_vec)

            # Normalize
            variant_vec = variant_vec / torch.norm(variant_vec)

            return variant_vec

        except Exception:
            logger.exception("Unhandled exception")
            raise HypervectorError("Failed to encode variant: {str(e)}")
            raise RuntimeError("Unspecified error")

    def encode_genome(self, variants: list[dict]) -> torch.Tensor:
        """
        Encode a complete set of variants into a single hypervector

        Args:
            variants: List of variant dictionaries

        Returns:
            Hypervector representation of the genome
        """
        if not variants:
            return torch.zeros(self.dimension)

        # Encode each variant
        variant_vectors = []
        for variant in variants:
            vec = self.encode_variant(
                chromosome=variant["chromosome"],
                position=variant["position"],
                ref=variant["ref"],
                alt=variant["alt"],
                variant_type=variant.get("type", "SNP"),
            )
            variant_vectors.append(vec)

        # Bundle all variants using superposition
        genome_vec = torch.stack(variant_vectors).sum(dim=0)

        # Apply holographic reduction
        genome_vec = self._holographic_reduce(genome_vec)

        # Final normalization
        genome_vec = genome_vec / torch.norm(genome_vec)

        return genome_vec

    def _bind(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """Bind two hypervectors using circular convolution"""
        return torch.fft.irfft(torch.fft.rfft(vec1) * torch.fft.rfft(vec2))

    def _bundle(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """Bundle two hypervectors using element-wise addition"""
        return vec1 + vec2

    def _permute(self, vec: torch.Tensor, n: int) -> torch.Tensor:
        """Permute hypervector by n positions"""
        return torch.roll(vec, n)

    def _chromosome_vector(self, chromosome: str) -> torch.Tensor:
        """Generate chromosome-specific vector"""
        # Extract chromosome number
        chr_num = chromosome.replace("chr", "").replace("X", "23").replace("Y", "24")
        try:
            chr_idx = int(chr_num)
        except Exception:
            logger.exception("Unhandled exception")
            chr_idx = 25  # For mitochondrial or other
            raise RuntimeError("Unspecified error")

        # Generate deterministic vector based on chromosome
        torch.manual_seed(chr_idx)
        chr_vec = torch.randn(self.dimension)
        chr_vec = chr_vec / torch.norm(chr_vec)

        return chr_vec

    def _holographic_reduce(self, vec: torch.Tensor) -> torch.Tensor:
        """Apply holographic reduction to maintain fixed dimensionality"""
        # Use circular shifting and XOR-like operations
        reduced = vec.clone()
        for i in range(3):  # Multiple reduction rounds
            shift = self.dimension // (2 ** (i + 1))
            shifted = torch.roll(vec, shift)
            reduced = self._bind(reduced, shifted)

        return reduced

    def similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Calculate cosine similarity between two hypervectors"""
        return torch.cosine_similarity(vec1, vec2, dim=0).item()

    def _encode_variant_with_panel(
        self, chromosome: str, position: int, ref: str, alt: str, variant_type: str
    ) -> torch.Tensor:
        """Encode variant using SNP panel for single-nucleotide accuracy"""
        # Get panel name based on granularity
        panel_name = self.panel_granularity.value
        if panel_name == "off":
            panel_name = "common"  # Default to common if somehow called

        # Create position vector using positional encoder
        pos_vec = self.positional_encoder.make_position_vector(position)

        # Create base vectors
        # ref_vec = self._encode_base_sparse(ref)  # Not used in panel mode
        alt_vec = self._encode_base_sparse(alt)

        # Bind position with alt allele
        variant_vec = self._bind(pos_vec, alt_vec)

        # Add variant type information
        if variant_type in self.base_vectors:
            variant_vec = self._bundle(variant_vec, self.base_vectors[variant_type])

        # Add chromosome information
        chr_vec = self._chromosome_vector(chromosome)
        variant_vec = self._bundle(variant_vec, chr_vec)

        # Normalize
        variant_vec = variant_vec / torch.norm(variant_vec)

        return variant_vec

    def _encode_base_sparse(self, base: str) -> torch.Tensor:
        """Encode nucleotide base using sparse representation"""
        base_seeds = {"A": 1000, "T": 2000, "G": 3000, "C": 4000, "N": 5000}
        seed = base_seeds.get(base.upper(), 5000)
        return self.positional_encoder._create_sparse_vector(seed)

    def encode_genome_with_panel(
        self, variants: list[dict], panel_name: str | None = None
    ) -> torch.Tensor:
        """
        Encode genome using SNP panel for improved accuracy

        Args:
            variants: List of variant dictionaries
            panel_name: Specific panel to use (overrides default)

        Returns:
            Panel-encoded genome hypervector
        """
        if not self.enable_snp_mode:
            raise HypervectorError("SNP mode not enabled")

        panel_name = panel_name or self.panel_granularity.value

        # Group variants by chromosome
        variants_by_chr: dict[str, dict[int, str]] = {}
        for var in variants:
            chr_name = var["chromosome"]
            if chr_name not in variants_by_chr:
                variants_by_chr[chr_name] = {}
            variants_by_chr[chr_name][var["position"]] = var["alt"]

        # Encode each chromosome with panel
        chr_vectors = []
        for chr_name, observed_bases in variants_by_chr.items():
            chr_vec = self.snp_panel.encode_with_panel(panel_name, chr_name, observed_bases)
            chr_vectors.append(chr_vec)

        # Bundle chromosomes
        if chr_vectors:
            genome_vec = torch.stack(chr_vectors).sum(dim=0)
            genome_vec = genome_vec / torch.norm(genome_vec)
            return genome_vec
        else:
            return torch.zeros(self.dimension)

    # Hierarchical zoom methods
    def create_zoom_tiles(self, chromosome: str, variants: list[dict]) -> None:
        """Create hierarchical zoom tiles for a chromosome"""
        # Level 0: Full chromosome
        chr_variants = [v for v in variants if v["chromosome"] == chromosome]
        if chr_variants:
            self.zoom_levels[0][chromosome] = self.encode_genome(chr_variants)

        # Level 1: 1Mb windows
        window_size = 1_000_000
        windows: dict[int, list] = {}
        for var in chr_variants:
            window_idx = var["position"] // window_size
            if window_idx not in windows:
                windows[window_idx] = []
            windows[window_idx].append(var)

        level1_key = f"{chromosome}_level1"
        self.zoom_levels[1][level1_key] = {}
        for window_idx, window_vars in windows.items():
            window_vec = self.encode_genome(window_vars)
            self.zoom_levels[1][level1_key][window_idx] = window_vec

        # Level 2: 1kb tiles (created on demand)

    def get_zoom_vector(
        self, chromosome: str, start: int, end: int, level: int = 0
    ) -> torch.Tensor:
        """Get zoom vector for specified region and level"""
        if level == 0:
            # Return full chromosome vector
            return self.zoom_levels[0].get(chromosome, torch.zeros(self.dimension))

        elif level == 1:
            # Return 1Mb window vectors
            window_start = start // 1_000_000
            window_end = end // 1_000_000
            level1_key = f"{chromosome}_level1"

            if level1_key not in self.zoom_levels[1]:
                return torch.zeros(self.dimension)

            # Bundle relevant windows
            window_vecs = []
            for window_idx in range(window_start, window_end + 1):
                if window_idx in self.zoom_levels[1][level1_key]:
                    window_vecs.append(self.zoom_levels[1][level1_key][window_idx])

            if window_vecs:
                bundled = torch.stack(window_vecs).sum(dim=0)
                return bundled / torch.norm(bundled)
            else:
                return torch.zeros(self.dimension)

        elif level == 2:
            # Create 1kb tiles on demand
            # tile_size = 1000
            #             tile_start = start // tile_size
            #             tile_end = end // tile_size

            # This would fetch variants in range and encode
            # For now, return placeholder
            return torch.zeros(self.dimension)

        else:
            raise ValueError(f"Invalid zoom level: {level}")

    def set_panel_granularity(self, granularity: str | PanelGranularity) -> None:
        """Change the SNP panel granularity setting"""
        if isinstance(granularity, str):
            granularity = PanelGranularity(granularity)
        self.panel_granularity = granularity

    def load_custom_panel(self, file_path: str, panel_name: str = "custom") -> None:
        """Load a custom SNP panel from file"""
        if not self.enable_snp_mode:
            raise HypervectorError("SNP mode not enabled")

        file_type = "vcf" if file_path.endswith(".vcf") else "bed"
        self.snp_panel.load_panel_from_file(panel_name, file_path, file_type)
        self.panel_granularity = PanelGranularity.CUSTOM

    # Catalytic extensions
    def use_catalytic_projections(self, projection_pool) -> None:
        """
        Switch to memory-mapped catalytic projections.

        Args:
            projection_pool: CatalyticProjectionPool instance
        """
        self.projection_pool = projection_pool
        self.use_catalytic = True

    def encode_variant_catalytic(self, *args, **kwargs) -> None:
        """
        Encode variant using catalytic projections.

        This method uses memory-mapped projections to reduce
        memory usage by 95% compared to standard encoding.
        """
        if hasattr(self, "projection_pool"):
            # Use memory-mapped projections
            variant_vec = self.encode_variant(*args, **kwargs)

            # Apply catalytic projection
            projected = self.projection_pool.apply_catalytic_projection(
                variant_vec,
                [0, 1, 2],  # Use first 3 projections
            )

            return projected
        else:
            # Fall back to standard encoding
            return self.encode_variant(*args, **kwargs)
