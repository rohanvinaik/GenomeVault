"""
HDC (Hyperdimensional Computing) router for genomic data encoding.

This module provides REST API endpoints for encoding genomic variants
using hyperdimensional computing, storing encodings, and comparing them.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.postgresql import UUID

from genomevault.hypervector import HypervectorEncoder, HypervectorConfig
from genomevault.hypervector.featurizers.variants import VariantFeaturizer

logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()

# Get database URL from environment or use default
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://genomevault:genomevault@localhost:5432/genomevault"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

router = APIRouter(
    prefix="/api/hdc",
    tags=["HDC", "Hyperdimensional Computing"],
    responses={404: {"description": "Not found"}},
)


# Database Models
class HDCEncoding(Base):
    """Database model for storing HDC encodings."""

    __tablename__ = "hdc_encodings"

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    encoding_id = Column(String(64), unique=True, index=True, nullable=False)
    encoding_data = Column(Text, nullable=False)  # Base64 encoded
    dimension = Column(Integer, nullable=False)
    variant_count = Column(Integer, nullable=False)
    checksum = Column(String(64), nullable=False)
    metadata = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=datetime.utcnow)
    source_type = Column(String(32), nullable=False)  # 'json' or 'vcf'


# Pydantic Models for Request/Response
class Variant(BaseModel):
    """Single genomic variant."""

    chromosome: str = Field(..., description="Chromosome (e.g., '1', '2', 'X')")
    position: int = Field(..., gt=0, description="Genomic position")
    ref: str = Field(..., min_length=1, description="Reference allele")
    alt: str = Field(..., min_length=1, description="Alternate allele")
    quality: Optional[float] = Field(None, ge=0, description="Variant quality score")

    @validator("chromosome")
    def validate_chromosome(cls, v):
        """Validate chromosome format."""
        # Remove 'chr' prefix if present
        if v.lower().startswith("chr"):
            v = v[3:]

        # Validate chromosome value
        valid_chromosomes = [str(i) for i in range(1, 23)] + ["X", "Y", "M", "MT"]
        if v.upper() not in valid_chromosomes:
            raise ValueError(f"Invalid chromosome: {v}")

        return v.upper()


class EncodeRequest(BaseModel):
    """Request model for encoding variants."""

    variants: List[Variant] = Field(..., min_items=1, max_items=10000)
    dimension: int = Field(10000, ge=1000, le=100000, description="Hypervector dimension")
    normalize: bool = Field(True, description="Whether to normalize the hypervector")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class EncodeResponse(BaseModel):
    """Response model for encoding operation."""

    encoding_id: str = Field(..., description="Unique ID for the encoding")
    dimension: int = Field(..., description="Dimension of the hypervector")
    variant_count: int = Field(..., description="Number of variants encoded")
    encoding: str = Field(..., description="Base64 encoded hypervector")
    checksum: str = Field(..., description="SHA256 checksum of the encoding")
    created_at: datetime = Field(..., description="Timestamp of creation")


class CompareRequest(BaseModel):
    """Request model for comparing two encodings."""

    encoding_id_1: str = Field(..., description="First encoding ID")
    encoding_id_2: str = Field(..., description="Second encoding ID")
    metric: str = Field("hamming", description="Distance metric to use")

    @validator("metric")
    def validate_metric(cls, v):
        """Validate distance metric."""
        valid_metrics = ["hamming", "cosine", "euclidean"]
        if v.lower() not in valid_metrics:
            raise ValueError(f"Invalid metric. Must be one of: {valid_metrics}")
        return v.lower()


class CompareResponse(BaseModel):
    """Response model for comparison operation."""

    encoding_id_1: str = Field(..., description="First encoding ID")
    encoding_id_2: str = Field(..., description="Second encoding ID")
    similarity: float = Field(..., ge=0, le=1, description="Similarity score (0-1)")
    distance: float = Field(..., ge=0, description="Distance value")
    metric: str = Field(..., description="Metric used for comparison")


class EncodingInfo(BaseModel):
    """Model for encoding information."""

    encoding_id: str
    dimension: int
    variant_count: int
    checksum: str
    created_at: datetime
    source_type: str
    metadata: Optional[Dict[str, Any]] = None


# Dependency to get database session
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Helper functions
def encode_variants_to_hypervector(
    variants: List[Dict[str, Any]], dimension: int = 10000, normalize: bool = True
) -> np.ndarray:
    """
    Encode a list of variants into a hypervector.

    Args:
        variants: List of variant dictionaries
        dimension: Hypervector dimension
        normalize: Whether to normalize the result

    Returns:
        Encoded hypervector as numpy array
    """
    try:
        # Initialize encoder with specified dimension
        config = HypervectorConfig(dimension=dimension, normalize=normalize)
        encoder = HypervectorEncoder(config=config)

        # Initialize variant featurizer
        featurizer = VariantFeaturizer()

        # Accumulate hypervectors for all variants
        accumulated_hv = np.zeros(dimension)

        for variant in variants:
            # Convert variant to feature vector
            features = featurizer.featurize_variant(variant)

            # Encode to hypervector
            hv = encoder.encode_value(features.sum())  # Simple aggregation
            accumulated_hv += hv

        # Normalize if requested
        if normalize:
            norm = np.linalg.norm(accumulated_hv)
            if norm > 0:
                accumulated_hv = accumulated_hv / norm

        return accumulated_hv

    except Exception as e:
        logger.error(f"Error encoding variants: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to encode variants: {str(e)}",
        )


def parse_vcf_file(file_content: bytes) -> List[Dict[str, Any]]:
    """
    Parse VCF file content and extract variants.

    Args:
        file_content: VCF file content as bytes

    Returns:
        List of variant dictionaries
    """
    variants = []

    try:
        lines = file_content.decode("utf-8").split("\n")

        for line in lines:
            # Skip headers and empty lines
            if line.startswith("#") or not line.strip():
                continue

            # Parse VCF line
            parts = line.split("\t")
            if len(parts) >= 5:
                variant = {
                    "chromosome": parts[0],
                    "position": int(parts[1]),
                    "ref": parts[3],
                    "alt": parts[4].split(",")[0],  # Take first alt allele
                    "quality": float(parts[5]) if parts[5] != "." else None,
                }
                variants.append(variant)

        if not variants:
            raise ValueError("No valid variants found in VCF file")

        return variants

    except Exception as e:
        logger.error(f"Error parsing VCF file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid VCF file format: {str(e)}"
        )


def calculate_similarity(
    hv1: np.ndarray, hv2: np.ndarray, metric: str = "hamming"
) -> tuple[float, float]:
    """
    Calculate similarity between two hypervectors.

    Args:
        hv1: First hypervector
        hv2: Second hypervector
        metric: Distance metric to use

    Returns:
        Tuple of (similarity, distance)
    """
    if metric == "hamming":
        # For binary hypervectors
        binary_hv1 = (hv1 > 0).astype(int)
        binary_hv2 = (hv2 > 0).astype(int)
        distance = np.sum(binary_hv1 != binary_hv2)
        max_distance = len(hv1)
        similarity = 1.0 - (distance / max_distance)

    elif metric == "cosine":
        # Cosine similarity
        dot_product = np.dot(hv1, hv2)
        norm_product = np.linalg.norm(hv1) * np.linalg.norm(hv2)
        if norm_product == 0:
            similarity = 0.0
        else:
            similarity = dot_product / norm_product
        distance = 1.0 - similarity

    elif metric == "euclidean":
        # Euclidean distance
        distance = np.linalg.norm(hv1 - hv2)
        # Normalize to 0-1 range (approximate)
        max_distance = np.sqrt(2 * len(hv1))  # Maximum possible distance
        similarity = 1.0 - min(distance / max_distance, 1.0)

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return similarity, distance


# API Endpoints
@router.post("/encode", response_model=EncodeResponse)
async def encode_variants(request: EncodeRequest, db: Session = Depends(get_db)):
    """
    Encode genomic variants into a hypervector.

    This endpoint accepts a list of genomic variants and encodes them
    into a high-dimensional vector using hyperdimensional computing.
    The resulting encoding is stored in the database for later retrieval.
    """
    try:
        logger.info(f"Encoding {len(request.variants)} variants with dimension {request.dimension}")

        # Convert Pydantic models to dictionaries
        variants_dict = [v.dict() for v in request.variants]

        # Encode variants
        hypervector = encode_variants_to_hypervector(
            variants_dict, dimension=request.dimension, normalize=request.normalize
        )

        # Convert to base64
        hv_bytes = hypervector.astype(np.float32).tobytes()
        hv_base64 = base64.b64encode(hv_bytes).decode("utf-8")

        # Calculate checksum
        checksum = hashlib.sha256(hv_bytes).hexdigest()

        # Generate encoding ID
        encoding_id = f"hdc_{uuid.uuid4().hex[:12]}"

        # Store in database
        db_encoding = HDCEncoding(
            encoding_id=encoding_id,
            encoding_data=hv_base64,
            dimension=request.dimension,
            variant_count=len(request.variants),
            checksum=checksum,
            metadata=json.dumps(request.metadata) if request.metadata else None,
            source_type="json",
        )

        db.add(db_encoding)
        db.commit()

        logger.info(f"Successfully created encoding {encoding_id}")

        return EncodeResponse(
            encoding_id=encoding_id,
            dimension=request.dimension,
            variant_count=len(request.variants),
            encoding=hv_base64,
            checksum=checksum,
            created_at=db_encoding.created_at,
        )

    except Exception as e:
        logger.error(f"Error in encode_variants: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to encode variants: {str(e)}",
        )


@router.post("/encode-vcf", response_model=EncodeResponse)
async def encode_vcf_file(
    file: UploadFile = File(..., description="VCF file to encode"),
    dimension: int = Form(10000, ge=1000, le=100000),
    normalize: bool = Form(True),
    db: Session = Depends(get_db),
):
    """
    Encode variants from a VCF file into a hypervector.

    This endpoint accepts a VCF file upload and encodes all variants
    into a hypervector representation.
    """
    try:
        # Validate file type
        if not file.filename.endswith((".vcf", ".vcf.gz")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be a VCF file (.vcf or .vcf.gz)",
            )

        # Read file content
        content = await file.read()

        # Handle gzipped files
        if file.filename.endswith(".gz"):
            import gzip

            content = gzip.decompress(content)

        # Parse VCF file
        variants = parse_vcf_file(content)

        logger.info(f"Encoding {len(variants)} variants from VCF file {file.filename}")

        # Encode variants
        hypervector = encode_variants_to_hypervector(
            variants, dimension=dimension, normalize=normalize
        )

        # Convert to base64
        hv_bytes = hypervector.astype(np.float32).tobytes()
        hv_base64 = base64.b64encode(hv_bytes).decode("utf-8")

        # Calculate checksum
        checksum = hashlib.sha256(hv_bytes).hexdigest()

        # Generate encoding ID
        encoding_id = f"hdc_{uuid.uuid4().hex[:12]}"

        # Store in database
        db_encoding = HDCEncoding(
            encoding_id=encoding_id,
            encoding_data=hv_base64,
            dimension=dimension,
            variant_count=len(variants),
            checksum=checksum,
            metadata=json.dumps({"filename": file.filename}),
            source_type="vcf",
        )

        db.add(db_encoding)
        db.commit()

        logger.info(f"Successfully created encoding {encoding_id} from VCF file")

        return EncodeResponse(
            encoding_id=encoding_id,
            dimension=dimension,
            variant_count=len(variants),
            encoding=hv_base64,
            checksum=checksum,
            created_at=db_encoding.created_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in encode_vcf_file: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to encode VCF file: {str(e)}",
        )


@router.get("/{encoding_id}", response_model=EncodingInfo)
async def get_encoding(encoding_id: str, db: Session = Depends(get_db)):
    """
    Retrieve information about a stored encoding.

    This endpoint returns metadata about a previously created encoding,
    including its dimension, variant count, and creation timestamp.
    """
    try:
        # Query database
        encoding = db.query(HDCEncoding).filter(HDCEncoding.encoding_id == encoding_id).first()

        if not encoding:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Encoding {encoding_id} not found"
            )

        # Parse metadata if present
        metadata = None
        if encoding.metadata:
            try:
                metadata = json.loads(encoding.metadata)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse metadata for encoding {encoding_id}")

        return EncodingInfo(
            encoding_id=encoding.encoding_id,
            dimension=encoding.dimension,
            variant_count=encoding.variant_count,
            checksum=encoding.checksum,
            created_at=encoding.created_at,
            source_type=encoding.source_type,
            metadata=metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_encoding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve encoding: {str(e)}",
        )


@router.get("/{encoding_id}/download")
async def download_encoding(encoding_id: str, db: Session = Depends(get_db)):
    """
    Download the raw hypervector data for an encoding.

    Returns the hypervector as a base64-encoded string that can be
    decoded and used for further processing.
    """
    try:
        # Query database
        encoding = db.query(HDCEncoding).filter(HDCEncoding.encoding_id == encoding_id).first()

        if not encoding:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Encoding {encoding_id} not found"
            )

        return JSONResponse(
            content={
                "encoding_id": encoding.encoding_id,
                "encoding": encoding.encoding_data,
                "dimension": encoding.dimension,
                "checksum": encoding.checksum,
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in download_encoding: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download encoding: {str(e)}",
        )


@router.post("/compare", response_model=CompareResponse)
async def compare_encodings(request: CompareRequest, db: Session = Depends(get_db)):
    """
    Compare two hypervector encodings.

    This endpoint computes the similarity between two previously stored
    encodings using the specified distance metric.
    """
    try:
        logger.info(f"Comparing encodings {request.encoding_id_1} and {request.encoding_id_2}")

        # Retrieve both encodings
        encoding1 = (
            db.query(HDCEncoding).filter(HDCEncoding.encoding_id == request.encoding_id_1).first()
        )

        if not encoding1:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Encoding {request.encoding_id_1} not found",
            )

        encoding2 = (
            db.query(HDCEncoding).filter(HDCEncoding.encoding_id == request.encoding_id_2).first()
        )

        if not encoding2:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Encoding {request.encoding_id_2} not found",
            )

        # Check dimension compatibility
        if encoding1.dimension != encoding2.dimension:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Encodings have different dimensions: {encoding1.dimension} vs {encoding2.dimension}",
            )

        # Decode hypervectors
        hv1_bytes = base64.b64decode(encoding1.encoding_data)
        hv1 = np.frombuffer(hv1_bytes, dtype=np.float32)

        hv2_bytes = base64.b64decode(encoding2.encoding_data)
        hv2 = np.frombuffer(hv2_bytes, dtype=np.float32)

        # Calculate similarity
        similarity, distance = calculate_similarity(hv1, hv2, request.metric)

        logger.info(f"Comparison complete: similarity={similarity:.4f}, distance={distance:.4f}")

        return CompareResponse(
            encoding_id_1=request.encoding_id_1,
            encoding_id_2=request.encoding_id_2,
            similarity=similarity,
            distance=distance,
            metric=request.metric,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in compare_encodings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare encodings: {str(e)}",
        )


@router.post("/batch-compare")
async def batch_compare_encodings(
    encoding_id: str,
    comparison_ids: List[str],
    metric: str = "hamming",
    db: Session = Depends(get_db),
):
    """
    Compare one encoding against multiple others.

    This endpoint efficiently compares a single encoding against a list
    of other encodings, returning similarity scores for each comparison.
    """
    try:
        # Validate metric
        valid_metrics = ["hamming", "cosine", "euclidean"]
        if metric.lower() not in valid_metrics:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metric. Must be one of: {valid_metrics}",
            )

        # Retrieve base encoding
        base_encoding = db.query(HDCEncoding).filter(HDCEncoding.encoding_id == encoding_id).first()

        if not base_encoding:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Encoding {encoding_id} not found"
            )

        # Decode base hypervector
        base_hv_bytes = base64.b64decode(base_encoding.encoding_data)
        base_hv = np.frombuffer(base_hv_bytes, dtype=np.float32)

        results = []

        for comp_id in comparison_ids:
            # Retrieve comparison encoding
            comp_encoding = db.query(HDCEncoding).filter(HDCEncoding.encoding_id == comp_id).first()

            if not comp_encoding:
                results.append({"encoding_id": comp_id, "error": "Not found"})
                continue

            # Check dimension compatibility
            if comp_encoding.dimension != base_encoding.dimension:
                results.append(
                    {
                        "encoding_id": comp_id,
                        "error": f"Dimension mismatch: {comp_encoding.dimension} vs {base_encoding.dimension}",
                    }
                )
                continue

            # Decode and compare
            comp_hv_bytes = base64.b64decode(comp_encoding.encoding_data)
            comp_hv = np.frombuffer(comp_hv_bytes, dtype=np.float32)

            similarity, distance = calculate_similarity(base_hv, comp_hv, metric)

            results.append({"encoding_id": comp_id, "similarity": similarity, "distance": distance})

        return JSONResponse(
            content={"base_encoding_id": encoding_id, "metric": metric, "comparisons": results}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch_compare_encodings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to perform batch comparison: {str(e)}",
        )


# Create database tables on module import
try:
    Base.metadata.create_all(bind=engine)
    logger.info("HDC database tables created successfully")
except Exception as e:
    logger.error(f"Failed to create HDC database tables: {e}")
