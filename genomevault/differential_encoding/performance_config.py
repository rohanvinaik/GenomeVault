"""
Performance Configuration for GenomeVault

Provides configurable performance vs accuracy trade-offs while maintaining
all security guarantees.

Configuration Presets:
1. FAST (1K dimension): Development/testing, ~1ms encoding
2. PRODUCTION (10K dimension): Balanced performance, ~5-10ms encoding (default)
3. RESEARCH (100K dimension): Highest accuracy, ~50-100ms encoding

All presets maintain:
- k-anonymity guarantees
- Cryptographic security (SHA-256)
- Zero-knowledge proof correctness
- PIR privacy guarantees

Only affects performance/accuracy trade-offs, NOT security.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class PerformancePreset(Enum):
    """
    Performance preset configurations.

    FAST: Optimized for speed (development/testing)
    PRODUCTION: Balanced performance and accuracy (default)
    RESEARCH: Maximum accuracy (research workflows)
    CUSTOM: User-defined configuration
    """

    FAST = "fast"
    PRODUCTION = "production"
    RESEARCH = "research"
    CUSTOM = "custom"


@dataclass
class PerformanceConfig:
    """
    Performance configuration for GenomeVault pipeline.

    Attributes:
        preset: Performance preset (FAST, PRODUCTION, RESEARCH, CUSTOM)
        hypervector_dimension: Dimension of hypervectors (1K, 10K, 100K)
        enable_parallel: Enable parallel chunk processing
        num_workers: Number of parallel workers (None = auto-detect)
        enable_gpu: Enable GPU acceleration for batch operations
        enable_cache: Enable reference pool caching
        cache_size: Maximum cache size (number of entries)
        min_chunks_for_parallel: Minimum chunks to trigger parallelism

    Performance Impact by Preset:
        FAST (1K):
            - Encoding: ~1ms per chunk
            - Accuracy: Lower (sufficient for testing)
            - Memory: ~4KB per hypervector
            - Use case: Development, quick tests

        PRODUCTION (10K):
            - Encoding: ~5-10ms per chunk
            - Accuracy: Good (recommended for deployment)
            - Memory: ~40KB per hypervector
            - Use case: Production deployments, fingerprinting

        RESEARCH (100K):
            - Encoding: ~50-100ms per chunk
            - Accuracy: Highest (maximum discrimination)
            - Memory: ~400KB per hypervector
            - Use case: Research, maximum accuracy requirements
    """

    preset: PerformancePreset = PerformancePreset.PRODUCTION
    hypervector_dimension: int = 10000
    enable_parallel: bool = True
    num_workers: Optional[int] = None  # None = auto-detect
    enable_gpu: bool = False  # GPU for batch HDC only, not ZK/PIR
    enable_cache: bool = True
    cache_size: int = 1000
    min_chunks_for_parallel: int = 4

    def __post_init__(self):
        """Validate configuration."""
        # Validate dimension
        if self.hypervector_dimension < 100:
            raise ValueError(
                f"hypervector_dimension must be >= 100, "
                f"got {self.hypervector_dimension}"
            )

        if self.hypervector_dimension > 1000000:
            logger.warning(
                f"Very large hypervector dimension: {self.hypervector_dimension}. "
                f"This may cause memory issues."
            )

        # Validate cache size
        if self.cache_size < 10:
            logger.warning(
                f"Small cache size: {self.cache_size}. "
                f"Cache may not be effective."
            )

        logger.info(
            f"PerformanceConfig initialized: "
            f"preset={self.preset.value}, "
            f"dimension={self.hypervector_dimension}, "
            f"parallel={self.enable_parallel}, "
            f"gpu={self.enable_gpu}, "
            f"cache={self.enable_cache}"
        )

    @classmethod
    def fast(cls) -> PerformanceConfig:
        """
        FAST preset: Optimized for speed.

        - 1K hypervector dimension
        - Parallel processing enabled
        - GPU disabled (overhead not worth it)
        - Caching enabled

        Use for: Development, testing, quick validation
        Expected performance: ~1ms per chunk encoding
        """
        return cls(
            preset=PerformancePreset.FAST,
            hypervector_dimension=1000,
            enable_parallel=True,
            num_workers=None,  # Auto-detect
            enable_gpu=False,
            enable_cache=True,
            cache_size=500,
            min_chunks_for_parallel=4
        )

    @classmethod
    def production(cls) -> PerformanceConfig:
        """
        PRODUCTION preset: Balanced performance and accuracy (default).

        - 10K hypervector dimension
        - Parallel processing enabled
        - GPU disabled (use for batch only if available)
        - Caching enabled

        Use for: Production deployments, genomic fingerprinting, clinical workflows
        Expected performance: ~5-10ms per chunk encoding
        """
        return cls(
            preset=PerformancePreset.PRODUCTION,
            hypervector_dimension=10000,
            enable_parallel=True,
            num_workers=None,  # Auto-detect
            enable_gpu=False,
            enable_cache=True,
            cache_size=1000,
            min_chunks_for_parallel=4
        )

    @classmethod
    def research(cls) -> PerformanceConfig:
        """
        RESEARCH preset: Maximum accuracy.

        - 100K hypervector dimension
        - Parallel processing enabled (essential for performance)
        - GPU recommended for batch operations
        - Large cache

        Use for: Research workflows, maximum accuracy requirements
        Expected performance: ~50-100ms per chunk encoding
        """
        return cls(
            preset=PerformancePreset.RESEARCH,
            hypervector_dimension=100000,
            enable_parallel=True,
            num_workers=None,  # Auto-detect (use all cores)
            enable_gpu=True,  # Recommended for 100K dimension
            enable_cache=True,
            cache_size=2000,
            min_chunks_for_parallel=2  # Lower threshold for parallelism
        )

    @classmethod
    def custom(
        cls,
        dimension: int,
        enable_parallel: bool = True,
        enable_gpu: bool = False,
        **kwargs
    ) -> PerformanceConfig:
        """
        CUSTOM preset: User-defined configuration.

        Args:
            dimension: Hypervector dimension
            enable_parallel: Enable parallel processing
            enable_gpu: Enable GPU acceleration
            **kwargs: Additional configuration parameters

        Returns:
            PerformanceConfig with custom settings
        """
        return cls(
            preset=PerformancePreset.CUSTOM,
            hypervector_dimension=dimension,
            enable_parallel=enable_parallel,
            enable_gpu=enable_gpu,
            **kwargs
        )

    def get_estimated_encoding_time_ms(self, num_variants: int) -> float:
        """
        Estimate encoding time for given number of variants.

        Args:
            num_variants: Number of variants to encode

        Returns:
            Estimated encoding time in milliseconds
        """
        # Base time per variant (empirical measurements)
        if self.hypervector_dimension <= 1000:
            time_per_variant = 0.01  # 1K: ~10μs per variant
        elif self.hypervector_dimension <= 10000:
            time_per_variant = 0.05  # 10K: ~50μs per variant
        else:
            time_per_variant = 0.5  # 100K: ~500μs per variant

        # Parallel speedup (if enabled)
        if self.enable_parallel and num_variants > 100:
            import multiprocessing as mp
            num_cores = self.num_workers or (mp.cpu_count() - 1)
            parallel_speedup = min(num_cores, 8)  # Diminishing returns after 8 cores
        else:
            parallel_speedup = 1

        # GPU speedup (if enabled and beneficial)
        if self.enable_gpu and self.hypervector_dimension >= 10000 and num_variants > 1000:
            gpu_speedup = 10  # ~10× for large batches
        else:
            gpu_speedup = 1

        total_time = (num_variants * time_per_variant) / (parallel_speedup * gpu_speedup)

        return total_time

    def get_memory_estimate_mb(self, num_genomes: int = 1) -> float:
        """
        Estimate memory usage for configuration.

        Args:
            num_genomes: Number of genomes to encode

        Returns:
            Estimated memory usage in MB
        """
        # Hypervector size (float32)
        hypervector_size_bytes = self.hypervector_dimension * 4

        # Per genome
        per_genome_mb = hypervector_size_bytes / (1024 * 1024)

        # Cache overhead (~10% of hypervectors)
        cache_overhead_mb = per_genome_mb * 0.1 if self.enable_cache else 0

        total_mb = (per_genome_mb * num_genomes) + cache_overhead_mb

        return total_mb

    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"PerformanceConfig("
            f"preset={self.preset.value}, "
            f"dim={self.hypervector_dimension}, "
            f"parallel={self.enable_parallel}, "
            f"gpu={self.enable_gpu}, "
            f"cache={self.enable_cache})"
        )


# Default configuration
DEFAULT_CONFIG = PerformanceConfig.production()


def get_config(preset: str = "production") -> PerformanceConfig:
    """
    Get performance configuration by preset name.

    Args:
        preset: Preset name ("fast", "production", "research")

    Returns:
        PerformanceConfig instance

    Raises:
        ValueError: If preset is invalid
    """
    preset_lower = preset.lower()

    if preset_lower == "fast":
        return PerformanceConfig.fast()
    elif preset_lower == "production":
        return PerformanceConfig.production()
    elif preset_lower == "research":
        return PerformanceConfig.research()
    else:
        raise ValueError(
            f"Invalid preset: {preset}. "
            f"Choose from: fast, production, research"
        )


def print_preset_comparison():
    """Print comparison of all presets."""
    presets = {
        "FAST": PerformanceConfig.fast(),
        "PRODUCTION": PerformanceConfig.production(),
        "RESEARCH": PerformanceConfig.research(),
    }

    print("\n=== GenomeVault Performance Presets ===\n")

    for name, config in presets.items():
        print(f"{name}:")
        print(f"  Dimension: {config.hypervector_dimension:,}")
        print(f"  Encoding time (1K variants): {config.get_estimated_encoding_time_ms(1000):.1f}ms")
        print(f"  Memory (1 genome): {config.get_memory_estimate_mb(1):.1f}MB")
        print(f"  Parallel: {config.enable_parallel}")
        print(f"  GPU: {config.enable_gpu}")
        print()


if __name__ == "__main__":
    # Print preset comparison
    print_preset_comparison()
