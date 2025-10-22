"""
Differential Encoding API Endpoints

Provides RESTful API endpoints for differential genomic encoding with:
- Cryptographic security
- Reference genome management
- Analysis type selection
- Metadata and statistics
"""

from __future__ import annotations

from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict
import logging
from pathlib import Path

import numpy as np

from genomevault.differential_encoding import (
    AnalysisType,
    Genome,
    Variant,
    EncodedGenome,
    DifferentialGenomeQuery,
    QueryResult,
    SimilarityMatch,
)
from .unified_encoder import UnifiedGenomicEncoder, EncodingMode, EncodingFeatureFlags

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1/differential", tags=["Differential Encoding"])

# Global encoder instance
_encoder: Optional[UnifiedGenomicEncoder] = None


def get_differential_encoder() -> UnifiedGenomicEncoder:
    """Get or create unified encoder instance."""
    global _encoder

    if _encoder is None:
        # Initialize with differential mode
        _encoder = UnifiedGenomicEncoder(
            mode=EncodingMode.DIFFERENTIAL,
            feature_flags=EncodingFeatureFlags.from_env(),
        )
        logger.info("Initialized differential encoder for API")

    return _encoder


# Request/Response Models

class VariantModel(BaseModel):
    """Variant data model for API."""

    chromosome: str = Field(..., description="Chromosome identifier")
    position: int = Field(..., description="Position on chromosome")
    ref: str = Field(..., description="Reference allele")
    alt: str = Field(..., description="Alternate allele")
    genotype: Optional[str] = Field(None, description="Genotype (e.g., '0/1')")
    quality: Optional[float] = Field(None, description="Quality score")
    info: Optional[Dict[str, Any]] = Field(None, description="Additional info")


class GenomeModel(BaseModel):
    """Genome data model for API."""

    genome_id: str = Field(..., description="Unique genome identifier")
    assembly: str = Field(..., description="Reference assembly (e.g., 'GRCh38')")
    chromosomes: Dict[str, List[VariantModel]] = Field(
        ..., description="Variants by chromosome"
    )


class DifferentialEncodingRequest(BaseModel):
    """Request for differential encoding."""

    genome: GenomeModel = Field(..., description="Genome to encode")
    analysis_type: str = Field(
        "sliding_window", description="Analysis type/chunking strategy"
    )
    bundle_chunks: bool = Field(True, description="Create bundled hypervector")
    master_seed: Optional[str] = Field(None, description="Hex-encoded master seed (64 chars)")


class DifferentialEncodingResponse(BaseModel):
    """Response for differential encoding."""

    genome_id: str = Field(..., description="Genome identifier")
    assembly: str = Field(..., description="Reference assembly")
    total_chunks: int = Field(..., description="Number of chunks created")
    hypervector_dimension: int = Field(..., description="Hypervector dimension")
    bundled_hypervector: Optional[List[float]] = Field(
        None, description="Bundled genome-level hypervector"
    )
    statistics: Dict[str, Any] = Field(..., description="Encoding statistics")
    encoding_time_ms: float = Field(..., description="Encoding time in milliseconds")
    storage_size_kb: float = Field(..., description="Estimated storage size in KB")


class RegionQueryRequest(BaseModel):
    """Request for region query."""

    genome_id: str = Field(..., description="Genome identifier")
    chromosome: str = Field(..., description="Chromosome to query")
    start: int = Field(..., description="Start position (inclusive)")
    end: int = Field(..., description="End position (exclusive)")


class RegionQueryResponse(BaseModel):
    """Response for region query."""

    chromosome: str
    start: int
    end: int
    variants: List[VariantModel]
    chunks_used: int
    variant_count: int


class SimilarityQueryRequest(BaseModel):
    """Request for similarity search."""

    genome_id: str = Field(..., description="Genome identifier")
    query_hypervector: List[float] = Field(..., description="Query hypervector")
    threshold: float = Field(0.7, description="Similarity threshold [0, 1]")
    top_k: Optional[int] = Field(None, description="Return top k matches")


class SimilarityMatchModel(BaseModel):
    """Similarity match model."""

    chunk_index: int
    similarity: float
    chromosome: str
    start_position: int
    end_position: int
    difference_counts: Dict[str, int]


class SimilarityQueryResponse(BaseModel):
    """Response for similarity search."""

    matches: List[SimilarityMatchModel]
    query_dimension: int
    threshold: float
    total_matches: int


# API Endpoints

@router.post("/encode", response_model=DifferentialEncodingResponse)
async def encode_genome(
    request: DifferentialEncodingRequest,
    encoder: UnifiedGenomicEncoder = Depends(get_differential_encoder),
) -> DifferentialEncodingResponse:
    """
    Encode a genome using differential encoding.

    This endpoint provides cryptographically secure differential encoding with:
    - Reference genome-based compression
    - Cryptographic binding and verification
    - Hyperdimensional vector representation
    - Complete metadata and statistics

    Example:
        POST /api/v1/differential/encode
        {
            "genome": {
                "genome_id": "patient_001",
                "assembly": "GRCh38",
                "chromosomes": {
                    "chr1": [
                        {"chromosome": "chr1", "position": 100000, "ref": "A", "alt": "G"}
                    ]
                }
            },
            "analysis_type": "sliding_window",
            "bundle_chunks": true
        }
    """
    try:
        start_time = datetime.now().timestamp()

        # Parse analysis type
        try:
            analysis_type = AnalysisType(request.analysis_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid analysis type: {request.analysis_type}. "
                       f"Valid types: {[t.value for t in AnalysisType]}"
            )

        # Convert API model to Genome
        genome_variants = {}
        for chr_name, variants in request.genome.chromosomes.items():
            genome_variants[chr_name] = [
                Variant(
                    chromosome=v.chromosome,
                    position=v.position,
                    ref=v.ref,
                    alt=v.alt,
                    genotype=v.genotype,
                    quality=v.quality,
                    info=v.info or {},
                )
                for v in variants
            ]

        genome = Genome(
            genome_id=request.genome.genome_id,
            assembly=request.genome.assembly,
            chromosomes=genome_variants,
        )

        # Parse master seed if provided
        master_seed = None
        if request.master_seed:
            try:
                master_seed = bytes.fromhex(request.master_seed)
                if len(master_seed) != 32:
                    raise ValueError("Master seed must be 32 bytes (64 hex chars)")
            except ValueError as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid master seed: {e}"
                )

        # Encode the genome
        logger.info(
            f"Encoding genome {genome.genome_id} with analysis_type={analysis_type.value}"
        )

        encoded = encoder.encode_genome(
            genome=genome,
            analysis_type=analysis_type,
            master_seed=master_seed,
            bundle_chunks=request.bundle_chunks,
            mode=EncodingMode.DIFFERENTIAL,  # Explicit mode
        )

        # Calculate encoding time
        encoding_time_ms = (datetime.now().timestamp() - start_time) * 1000

        # Prepare response
        bundled_hv = None
        if encoded.bundled_hypervector is not None:
            bundled_hv = encoded.bundled_hypervector.tolist()

        return DifferentialEncodingResponse(
            genome_id=encoded.genome_id,
            assembly=encoded.assembly,
            total_chunks=len(encoded.chunk_hypervectors),
            hypervector_dimension=len(encoded.bundled_hypervector),
            bundled_hypervector=bundled_hv,
            statistics=encoded.statistics,
            encoding_time_ms=encoding_time_ms,
            storage_size_kb=encoded.storage_size_kb(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Encoding error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analysis_types")
async def get_analysis_types() -> Dict[str, List[str]]:
    """
    Get list of available analysis types.

    Returns:
        Dictionary with analysis type names and descriptions
    """
    return {
        "analysis_types": [t.value for t in AnalysisType],
        "descriptions": {
            "sliding_window": "Sliding window chunking for general analysis",
            "gene_region": "Gene-based chunking for functional analysis",
            "single_snp_query": "Single SNP queries",
            "exome_only": "Exome-only chunking",
            "regulatory_regions": "Regulatory region chunking",
            "structural_variants": "Structural variant detection",
            "custom_intervals": "Custom interval specification",
        }
    }


@router.get("/encoder_info")
async def get_encoder_info(
    encoder: UnifiedGenomicEncoder = Depends(get_differential_encoder),
) -> Dict[str, Any]:
    """
    Get information about the current encoder configuration.

    Returns:
        Dictionary with encoder configuration and capabilities
    """
    return encoder.get_encoding_info()


@router.get("/health")
async def health_check(
    encoder: UnifiedGenomicEncoder = Depends(get_differential_encoder),
) -> Dict[str, str]:
    """
    Health check endpoint for differential encoding service.

    Returns:
        Status of the differential encoding service
    """
    info = encoder.get_encoding_info()

    status = "healthy" if info["encoders"]["differential_available"] else "degraded"

    return {
        "status": status,
        "mode": info["mode"],
        "differential_enabled": str(info["feature_flags"]["differential_enabled"]),
        "references_loaded": str(info["encoders"]["reference_count"]),
    }


# Include router setup function
def include_differential_routes(app):
    """
    Include differential encoding routes in FastAPI app.

    Args:
        app: FastAPI application instance

    Example:
        >>> from fastapi import FastAPI
        >>> app = FastAPI()
        >>> include_differential_routes(app)
    """
    app.include_router(router)
    logger.info("Differential encoding routes included")
