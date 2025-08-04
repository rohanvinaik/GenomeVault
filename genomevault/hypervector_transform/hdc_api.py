"""
HDC API endpoints for GenomeVault

Provides RESTful API endpoints for encoding genomic data using
Hierarchical Hyperdimensional Computing (HDC).
"""

import logging
from datetime import datetime
from typing import Any

import torch
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .binding_operations import BindingType, HypervectorBinder
from .hdc_encoder import (CompressionTier, HypervectorEncoder, OmicsType,
                          ProjectionType)
from .registry import HypervectorRegistry

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1/hdc", tags=["HDC Encoding"])

# Global registry instance
registry = HypervectorRegistry()


class EncodingRequest(BaseModel):
    """Request model for encoding genomic data"""

    features: dict[str, Any] = Field(..., description="Feature dictionary or array")
    omics_type: str = Field(..., description="Type of omics data")
    compression_tier: str | None = Field("full", description="Compression tier")
    version: str | None = Field(None, description="Encoding version to use")


class EncodingResponse(BaseModel):
    """Response model for encoded data"""

    vector: list[float] = Field(..., description="Encoded hypervector")
    dimension: int = Field(..., description="Vector dimension")
    version: str = Field(..., description="Encoding version used")
    encoding_time_ms: float = Field(..., description="Encoding time in milliseconds")
    metadata: dict[str, Any] = Field(default_factory=dict)


class MultiModalEncodingRequest(BaseModel):
    """Request for encoding multiple modalities"""

    modalities: dict[str, dict[str, Any]] = Field(
        ..., description="Dict of modality data"
    )
    compression_tier: str | None = Field("full", description="Compression tier")
    binding_type: str | None = Field(
        "fourier", description="Binding type for cross-modal"
    )


class SimilarityRequest(BaseModel):
    """Request for computing similarity between vectors"""

    vector1: list[float] = Field(..., description="First hypervector")
    vector2: list[float] = Field(..., description="Second hypervector")
    metric: str | None = Field("cosine", description="Similarity metric")


class DecodeRequest(BaseModel):
    """Request for decoding/querying a hypervector"""

    vector: list[float] = Field(..., description="Hypervector to decode")
    query_type: str = Field(..., description="Type of query")
    role: str | None = Field(None, description="Role to query for composite vectors")


class VersionInfo(BaseModel):
    """Version information response"""

    current_version: str
    available_versions: list[dict[str, Any]]
    hdc_seed: str
    encoder_version: str


class PerformanceMetrics(BaseModel):
    """Performance metrics response"""

    average_encoding_time_ms: float
    encodings_per_second: float
    test_feature_size: int
    output_dimension: int
    projection_type: str


# Dependency to get encoder
def get_encoder(version: str | None = None) -> HypervectorEncoder:
    """Get encoder instance with specified version"""
    try:
        return registry.get_encoder(version)
    except (KeyError, ValueError, RuntimeError) as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        logger.error("Failed to get encoder: %se")
        raise HTTPException(status_code=500, detail=str(e))
        raise RuntimeError("Unspecified error")


@router.post("/encode", response_model=EncodingResponse)
async def encode_genome(
    request: EncodingRequest, encoder: HypervectorEncoder = Depends(get_encoder)
) -> EncodingResponse:
    """
    Encode genomic data into a hypervector

    This endpoint accepts various types of genomic data and returns
    a privacy-preserving hyperdimensional vector representation.
    """
    try:
        start_time = datetime.now().timestamp()

        # Parse omics type
        try:
            omics_type = OmicsType(request.omics_type)
        except ValueError:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid omics type: {request.omics_type}. Valid types: {[t.value for t in OmicsType]}",
            )
            raise RuntimeError("Unspecified error")

        # Parse compression tier if provided
        compression_tier = None
        if request.compression_tier:
            try:
                compression_tier = CompressionTier(request.compression_tier)
            except ValueError:
                from genomevault.observability.logging import configure_logging

                logger = configure_logging()
                logger.exception("Unhandled exception")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid compression tier: {request.compression_tier}",
                )
                raise RuntimeError("Unspecified error")

        # Encode the data
        hypervector = encoder.encode(request.features, omics_type, compression_tier)

        # Calculate encoding time
        encoding_time_ms = (datetime.now().timestamp() - start_time) * 1000

        # Get encoding metrics
        metrics = encoder.get_encoding_metrics(start_time, hypervector)

        # Prepare response
        return EncodingResponse(
            vector=hypervector.tolist(),
            dimension=hypervector.shape[0],
            version=getattr(encoder, "version", "unknown"),
            encoding_time_ms=encoding_time_ms,
            metadata={
                "omics_type": omics_type.value,
                "compression_tier": (
                    compression_tier.value if compression_tier else "default"
                ),
                "fingerprint": getattr(encoder, "fingerprint", "unknown"),
                "sparsity": metrics.sparsity,
                "compression_ratio": metrics.compression_ratio,
            },
        )

    except HTTPException:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        raise RuntimeError("Unspecified error")
        raise RuntimeError("Unspecified error")
    except (ValueError, RuntimeError, TypeError, AttributeError) as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        logger.error("Encoding error: %se")
        raise HTTPException(status_code=500, detail=str(e))
        raise RuntimeError("Unspecified error")


@router.post("/encode_multimodal", response_model=EncodingResponse)
async def encode_multimodal(
    request: MultiModalEncodingRequest,
    encoder: HypervectorEncoder = Depends(get_encoder),
) -> EncodingResponse:
    """
    Encode multiple modalities and bind them together

    This creates a cross-modal representation that captures
    relationships between different types of biological data.
    """
    try:
        start_time = datetime.now().timestamp()

        # Parse compression tier and binding type
        compression_tier = CompressionTier(request.compression_tier)
        binding_type = BindingType(request.binding_type)

        # Create binder
        binder = HypervectorBinder(encoder.config.dimension)

        # Encode each modality
        encoded_modalities = {}
        for modality_name, modality_data in request.modalities.items():
            # Infer omics type from modality name
            if "genom" in modality_name.lower():
                omics_type = OmicsType.GENOMIC
            elif "transcript" in modality_name.lower():
                omics_type = OmicsType.TRANSCRIPTOMIC
            elif "epigen" in modality_name.lower():
                omics_type = OmicsType.EPIGENOMIC
            elif "proteo" in modality_name.lower():
                omics_type = OmicsType.PROTEOMIC
            else:
                omics_type = OmicsType.CLINICAL

            # Encode
            encoded = encoder.encode(modality_data, omics_type, compression_tier)
            encoded_modalities[modality_name] = encoded

        # Bind modalities together
        if len(encoded_modalities) == 1:
            # Single modality, return as is
            combined = list(encoded_modalities.values())[0]
        else:
            # Multi-modal binding
            combined = binder.bind(list(encoded_modalities.values()), binding_type)

        # Calculate encoding time
        encoding_time_ms = (datetime.now().timestamp() - start_time) * 1000

        return EncodingResponse(
            vector=combined.tolist(),
            dimension=combined.shape[0],
            version=getattr(encoder, "version", "unknown"),
            encoding_time_ms=encoding_time_ms,
            metadata={
                "modalities": list(request.modalities.keys()),
                "binding_type": binding_type.value,
                "compression_tier": compression_tier.value,
                "num_modalities": len(encoded_modalities),
            },
        )

    except (ValueError, RuntimeError, TypeError, KeyError, AttributeError) as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        logger.error("Multi-modal encoding error: %se")
        raise HTTPException(status_code=500, detail=str(e))
        raise RuntimeError("Unspecified error")


@router.post("/decode")
async def decode_vector(request: DecodeRequest):
    """
    Decode or query a hypervector

    Note: Full reconstruction is computationally infeasible by design.
    This endpoint provides approximate querying capabilities.
    """
    try:
        # Convert to tensor
        vector = torch.tensor(request.vector)

        # Validate vector
        if not torch.isfinite(vector).all():
            raise HTTPException(
                status_code=400, detail="Vector contains invalid values (inf/nan)"
            )

        # Get encoder for decoding operations
        registry.get_encoder()

        if request.query_type == "non_invertible_proof":
            # Demonstrate non-invertibility
            return {
                "message": "HDC encoding is non-invertible by design",
                "vector_dimension": len(request.vector),
                "information_theoretic_bound": (
                    "Cannot recover original data from hypervector alone"
                ),
                "security_guarantee": "Computationally infeasible reconstruction",
                "privacy_preservation": "Original genomic data cannot be recovered",
            }

        elif request.query_type == "vector_stats":
            # Return statistical information about the vector
            return {
                "dimension": vector.shape[0],
                "mean": float(vector.mean()),
                "std": float(vector.std()),
                "min": float(vector.min()),
                "max": float(vector.max()),
                "sparsity": float((vector == 0).float().mean()),
                "norm": float(torch.norm(vector)),
            }

        elif request.query_type == "composite_query" and request.role:
            # Query composite vector for a role
            # This requires the role vector which should be stored
            return {
                "message": "Composite querying requires role vectors",
                "note": "Implementation depends on stored role-filler mappings",
                "role_requested": request.role,
            }

        else:
            raise HTTPException(
                status_code=400, detail=f"Unknown query type: {request.query_type}"
            )

    except HTTPException:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        raise RuntimeError("Unspecified error")
        raise RuntimeError("Unspecified error")
    except (ValueError, RuntimeError, TypeError, AttributeError) as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        logger.error("Decoding error: %se")
        raise HTTPException(status_code=500, detail=str(e))
        raise RuntimeError("Unspecified error")


@router.post("/similarity")
async def compute_similarity(request: SimilarityRequest):
    """
    Compute similarity between two hypervectors

    This preserves the similarity relationships from the original space.
    """
    try:
        # Convert to tensors
        v1 = torch.tensor(request.vector1)
        v2 = torch.tensor(request.vector2)

        # Validate dimensions match
        if v1.shape != v2.shape:
            raise HTTPException(
                status_code=400,
                detail=f"Vectors must have the same dimension: {v1.shape} != {v2.shape}",
            )

        # Validate vectors
        if not (torch.isfinite(v1).all() and torch.isfinite(v2).all()):
            raise HTTPException(
                status_code=400, detail="Vectors contain invalid values (inf/nan)"
            )

        # Get encoder for similarity computation
        encoder = registry.get_encoder()

        # Compute similarity
        similarity = encoder.similarity(v1, v2, request.metric)

        # Additional metrics
        euclidean_distance = torch.dist(v1, v2, p=2).item()
        manhattan_distance = torch.dist(v1, v2, p=1).item()

        return {
            "similarity": float(similarity),
            "metric": request.metric,
            "dimension": v1.shape[0],
            "additional_metrics": {
                "euclidean_distance": euclidean_distance,
                "manhattan_distance": manhattan_distance,
                "dot_product": float(torch.dot(v1, v2)),
            },
        }

    except HTTPException:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        raise RuntimeError("Unspecified error")
        raise RuntimeError("Unspecified error")
    except (ValueError, RuntimeError, TypeError, AttributeError) as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        logger.error("Similarity computation error: %se")
        raise HTTPException(status_code=500, detail=str(e))
        raise RuntimeError("Unspecified error")


@router.get("/version", response_model=VersionInfo)
async def get_version_info():
    """Get HDC encoding version information"""
    try:
        from genomevault.version import HDC_ENCODER_VERSION, HDC_SEED

        versions = registry.list_versions()

        return VersionInfo(
            current_version=registry.current_version,
            available_versions=versions,
            hdc_seed=HDC_SEED,
            encoder_version=HDC_ENCODER_VERSION,
        )

    except (ImportError, AttributeError, RuntimeError) as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        logger.error("Version info error: %se")
        raise HTTPException(status_code=500, detail=str(e))
        raise RuntimeError("Unspecified error")


@router.post("/register_version")
async def register_new_version(
    version: str,
    dimension: int,
    projection_type: str,
    description: str | None = None,
    sparsity: float | None = 0.1,
):
    """Register a new encoding version"""
    try:
        # Validate projection type
        try:
            ProjectionType(projection_type)
        except ValueError:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            raise HTTPException(
                status_code=400, detail=f"Invalid projection type: {projection_type}"
            )
            raise RuntimeError("Unspecified error")

        params = {
            "dimension": dimension,
            "projection_type": projection_type,
            "sparsity": sparsity,
        }

        registry.register_version(
            version=version,
            params=params,
            description=description or f"Registered via API at {datetime.now()}",
        )

        return {
            "message": "Version registered successfully",
            "version": version,
            "params": params,
        }

    except HTTPException:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        raise RuntimeError("Unspecified error")
        raise RuntimeError("Unspecified error")
    except (ValueError, RuntimeError, TypeError) as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        logger.error("Version registration error: %se")
        raise HTTPException(status_code=500, detail=str(e))
        raise RuntimeError("Unspecified error")


@router.post("/encode_file")
async def encode_genomic_file(
    file: UploadFile = File(...),
    omics_type: str = "genomic",
    compression_tier: str = "full",
):
    """
    Encode a genomic data file directly

    Supports various file formats (VCF, BAM, FASTQ, etc.)
    This is a placeholder for full implementation.
    """
    try:
        # Validate file size (max 100MB)
        max_size = 100 * 1024 * 1024  # 100MB
        content = await file.read()

        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size: {max_size / 1024 / 1024}MB",
            )

        # This is a placeholder - actual implementation would
        # parse the file and extract features based on file type

        return {
            "message": "File encoding endpoint (placeholder)",
            "filename": file.filename,
            "size_bytes": len(content),
            "omics_type": omics_type,
            "compression_tier": compression_tier,
            "note": "Full implementation requires file parsing modules",
            "supported_formats": ["VCF", "BAM", "FASTQ", "CSV", "TSV"],
        }

    except HTTPException:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        raise RuntimeError("Unspecified error")
        raise RuntimeError("Unspecified error")
    except (ValueError, RuntimeError, OSError) as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        logger.error("File encoding error: %se")
        raise HTTPException(status_code=500, detail=str(e))
        raise RuntimeError("Unspecified error")


@router.get("/performance_metrics", response_model=PerformanceMetrics)
async def get_performance_metrics():
    """Get HDC encoding performance metrics"""
    try:
        # Get encoder
        encoder = registry.get_encoder()

        # Generate test data
        test_features = torch.randn(1000)

        # Measure encoding performance
        import time

        # Warm-up
        for _ in range(10):
            _ = encoder.encode(test_features, OmicsType.GENOMIC)

        # Time multiple encodings
        num_trials = 100
        start = time.time()

        for _ in range(num_trials):
            _ = encoder.encode(test_features, OmicsType.GENOMIC)

        elapsed = time.time() - start
        avg_time_ms = (elapsed / num_trials) * 1000

        # Calculate throughput
        encodings_per_second = num_trials / elapsed

        return PerformanceMetrics(
            average_encoding_time_ms=avg_time_ms,
            encodings_per_second=encodings_per_second,
            test_feature_size=1000,
            output_dimension=encoder.config.dimension,
            projection_type=encoder.config.projection_type.value,
        )

    except (RuntimeError, AttributeError, TypeError) as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        logger.error("Performance metrics error: %se")
        raise HTTPException(status_code=500, detail=str(e))
        raise RuntimeError("Unspecified error")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test encoder creation
        encoder = registry.get_encoder()

        # Test encoding
        test_vector = encoder.encode({"test": [1, 2, 3]}, OmicsType.GENOMIC)

        return {
            "status": "healthy",
            "registry_versions": len(registry.versions),
            "current_version": registry.current_version,
            "test_encoding_dimension": test_vector.shape[0],
        }
    except (RuntimeError, ValueError, AttributeError, Exception) as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        return {"status": "unhealthy", "error": str(e)}
        raise RuntimeError("Unspecified error")


# Include router in main app
def include_routes(app):
    """Include HDC routes in the main FastAPI app"""
    app.include_router(router)
