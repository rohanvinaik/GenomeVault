"""
GenomeVault Pipelines Module

Primary production pipeline:
- UnifiedPipeline: Complete 7-layer privacy-preserving genomic pipeline

Usage:
    from genomevault.pipelines import UnifiedPipeline, PipelineConfig

    config = PipelineConfig(output_dir=Path("output"))
    pipeline = UnifiedPipeline(config)
    result = pipeline.run_experimental_pipeline(...)
"""

# Primary production pipeline (new unified architecture)
from .unified_pipeline import (
    UnifiedPipeline,
    PipelineConfig,
    PipelineResult,
)

# Legacy profile utility (still functional)
from .profile import profile_dataframe

__all__ = [
    # Primary pipeline
    "UnifiedPipeline",
    "PipelineConfig",
    "PipelineResult",
    # Utilities
    "profile_dataframe",
]
