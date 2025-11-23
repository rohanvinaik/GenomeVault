"""
Differential encoding module for GenomeVault.

This module implements cryptographically secure differential encoding of genomic data,
storing experimental genomes as cryptographically verified differences from randomly
selected reference genome sections.
"""

from genomevault.differential_encoding.crypto_primitives import (
    CryptoRNG,
    compute_chunk_id,
    compute_reference_hash,
    compute_chunk_reference_binding,
)
from genomevault.differential_encoding.reference_management import (
    Variant,
    GenomeSection,
    ReferenceGenome,
    ReferencePool,
    SecureReferenceGenomeManager,
    IntervalTree,
    Interval,
)
from genomevault.differential_encoding.chunking import (
    AnalysisType,
    ChunkingStrategy,
    STRATEGY_CONFIGS,
    GenomicFeature,
    Genome,
    GenomeChunk,
    CryptographicChunker,
    get_strategy_for_analysis,
)
from genomevault.differential_encoding.differences import (
    DifferenceType,
    FunctionalImpact,
    VariantDifference,
    compute_variant_differences,
    variant_key,
    get_functional_impact,
)
from genomevault.differential_encoding.metadata import (
    DifferentialEncodingMetadata,
    METADATA_SCHEMA,
    validate_metadata_schema,
    create_metadata_from_chunk,
)
from genomevault.differential_encoding.feature_vectors import (
    differences_to_feature_vector,
    sinusoidal_position_encoding,
    compute_functional_impact_vector,
    compute_allele_composition,
    compute_genotype_distribution,
    compute_quality_metrics,
    compute_difference_type_distribution,
    get_feature_names,
    describe_feature_vector,
    TOTAL_FEATURE_DIM,
    DIM_DIFFERENCE_TYPES,
    DIM_POSITION_ENCODING,
    DIM_ALLELE_COMPOSITION,
    DIM_GENOTYPE_DIST,
    DIM_FUNCTIONAL_IMPACT,
    DIM_QUALITY_METRICS,
)
from genomevault.differential_encoding.hypervector_encoder import (
    DifferentialHypervectorEncoder,
)
from genomevault.differential_encoding.pipeline import (
    DifferentialGenomicEncoder,
    EncodingResult,
)
from genomevault.differential_encoding.storage import (
    EncodedGenome,
)
from genomevault.differential_encoding.query import (
    DifferentialGenomeQuery,
    QueryResult,
    SimilarityMatch,
)
from genomevault.differential_encoding.reference_setup import (
    ReferenceSource,
    ValidationResult,
    STANDARD_REFERENCES,
    RECOMMENDED_POOLS,
    download_reference_genomes,
    validate_reference_pool,
    setup_default_references,
    get_reference_info,
)

__all__ = [
    # Cryptographic primitives
    "CryptoRNG",
    "compute_chunk_id",
    "compute_reference_hash",
    "compute_chunk_reference_binding",
    # Reference management
    "Variant",
    "GenomeSection",
    "ReferenceGenome",
    "ReferencePool",
    "SecureReferenceGenomeManager",
    "IntervalTree",
    "Interval",
    # Chunking
    "AnalysisType",
    "ChunkingStrategy",
    "STRATEGY_CONFIGS",
    "GenomicFeature",
    "Genome",
    "GenomeChunk",
    "CryptographicChunker",
    "get_strategy_for_analysis",
    # Variant differences
    "DifferenceType",
    "FunctionalImpact",
    "VariantDifference",
    "compute_variant_differences",
    "variant_key",
    "get_functional_impact",
    # Metadata
    "DifferentialEncodingMetadata",
    "METADATA_SCHEMA",
    "validate_metadata_schema",
    "create_metadata_from_chunk",
    # Feature vectors
    "differences_to_feature_vector",
    "sinusoidal_position_encoding",
    "compute_functional_impact_vector",
    "compute_allele_composition",
    "compute_genotype_distribution",
    "compute_quality_metrics",
    "compute_difference_type_distribution",
    "get_feature_names",
    "describe_feature_vector",
    "TOTAL_FEATURE_DIM",
    "DIM_DIFFERENCE_TYPES",
    "DIM_POSITION_ENCODING",
    "DIM_ALLELE_COMPOSITION",
    "DIM_GENOTYPE_DIST",
    "DIM_FUNCTIONAL_IMPACT",
    "DIM_QUALITY_METRICS",
    # Hypervector encoder
    "DifferentialHypervectorEncoder",
    # Pipeline
    "DifferentialGenomicEncoder",
    "EncodingResult",
    # Storage
    "EncodedGenome",
    # Query
    "DifferentialGenomeQuery",
    "QueryResult",
    "SimilarityMatch",
    # Reference Setup
    "ReferenceSource",
    "ValidationResult",
    "STANDARD_REFERENCES",
    "RECOMMENDED_POOLS",
    "download_reference_genomes",
    "validate_reference_pool",
    "setup_default_references",
    "get_reference_info",
]
