"""
HDC (Hyperdimensional Computing) router for genomic data encoding - FIXED VERSION

This module provides REST API endpoints for encoding genomic variants
using the REAL working DifferentialHypervectorEncoder from the genomevault pipeline.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import uuid
import gzip
import os
from datetime import datetime
from typing import Any, Dict, List, Optional
from multiprocessing import Pool, cpu_count

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.postgresql import UUID

# Import the REAL working encoder
from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig

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
    metadata_json = Column(Text, nullable=True)  # JSON string (renamed from metadata to avoid SQLAlchemy conflict)
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
    """Response model for encoding."""

    encoding_id: str = Field(..., description="Unique ID for this encoding")
    dimension: int = Field(..., description="Hypervector dimension")
    variant_count: int = Field(..., description="Number of variants encoded")
    encoding: str = Field(..., description="Base64-encoded hypervector")
    checksum: str = Field(..., description="SHA256 checksum of encoding")
    created_at: datetime = Field(..., description="Creation timestamp")


# Dependency for database session
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Helper functions
def _encode_variant_batch(args):
    """
    Encode a batch of variants in parallel (worker function).
    This function is called by multiprocessing.Pool.
    """
    variants_batch, dimension, seed_offset = args

    # Initialize encoder with offset seed for this batch
    encoder = DifferentialHypervectorEncoder(dimension=dimension, seed=42 + seed_offset)
    batch_hv = np.zeros(dimension)

    for variant in variants_batch:
        diff = VariantDifference(
            difference_type=DifferenceType.NEW_MUTATION,
            chromosome=variant['chromosome'],
            position=variant['position'],
            exp_ref=variant['ref'],
            exp_alt=variant['alt'],
            exp_quality=variant.get('quality', 1.0)
        )

        metadata = DifferentialEncodingMetadata(
            reference_genomes=['hg38'],
            chromosome=variant['chromosome'],
            total_differences=1,
            substitutions=1,
            insertions=0,
            deletions=0
        )

        hv = encoder.encode_difference_vector([diff], metadata)
        batch_hv += hv

    return batch_hv


def encode_variants_to_hypervector(
    variants: List[Dict[str, Any]], dimension: int = 10000, normalize: bool = True
) -> np.ndarray:
    """
    Encode a list of variants into a hypervector using the REAL encoder with PARALLEL processing.

    Args:
        variants: List of variant dictionaries
        dimension: Hypervector dimension
        normalize: Whether to normalize the result

    Returns:
        Encoded hypervector as numpy array
    """
    try:
        num_cores = cpu_count()
        num_variants = len(variants)

        # Use parallel processing for large variant lists
        if num_variants > 1000:
            logger.info(f"Parallel encoding {num_variants} variants using {num_cores} cores")

            # Split variants into batches
            batch_size = max(100, num_variants // (num_cores * 4))  # 4 batches per core
            batches = []
            for i in range(0, num_variants, batch_size):
                batch = variants[i:i + batch_size]
                batches.append((batch, dimension, i // batch_size))

            # Process batches in parallel
            with Pool(processes=num_cores) as pool:
                batch_results = pool.map(_encode_variant_batch, batches)

            # Sum all batch results
            accumulated_hv = np.sum(batch_results, axis=0)

        else:
            # Sequential for small lists
            logger.info(f"Sequential encoding {num_variants} variants")
            encoder = DifferentialHypervectorEncoder(dimension=dimension, seed=42)
            accumulated_hv = np.zeros(dimension)

            for variant in variants:
                diff = VariantDifference(
                    difference_type=DifferenceType.NEW_MUTATION,
                    chromosome=variant['chromosome'],
                    position=variant['position'],
                    exp_ref=variant['ref'],
                    exp_alt=variant['alt'],
                    exp_quality=variant.get('quality', 1.0)
                )

                metadata = DifferentialEncodingMetadata(
                    reference_genomes=['hg38'],
                    chromosome=variant['chromosome'],
                    total_differences=1,
                    substitutions=1,
                    insertions=0,
                    deletions=0
                )

                hv = encoder.encode_difference_vector([diff], metadata)
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
        # Handle gzip compression
        if file_content[:2] == b'\x1f\x8b':
            file_content = gzip.decompress(file_content)

        # Parse VCF
        for line in file_content.decode('utf-8').split('\n'):
            line = line.strip()

            # Skip header lines and empty lines
            if not line or line.startswith('#'):
                continue

            # Parse variant line
            parts = line.split('\t')
            if len(parts) >= 5:
                chrom, pos, _, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
                quality = float(parts[5]) if len(parts) > 5 and parts[5] != '.' else None

                variants.append({
                    'chromosome': chrom.replace('chr', ''),
                    'position': int(pos),
                    'ref': ref,
                    'alt': alt,
                    'quality': quality
                })

    except Exception as e:
        logger.error(f"Error parsing VCF: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse VCF file: {str(e)}"
        )

    return variants


@router.post("/encode", response_model=EncodeResponse)
async def encode_variants(request: EncodeRequest, db: Session = Depends(get_db)):
    """
    Encode genomic variants into a hypervector using the REAL encoder.

    This endpoint accepts a list of genomic variants and encodes them
    into a high-dimensional vector using the actual DifferentialHypervectorEncoder.
    The resulting encoding is stored in the database for later retrieval.
    """
    try:
        logger.info(f"Encoding {len(request.variants)} variants with dimension {request.dimension}")

        # Convert Pydantic models to dictionaries
        variants_dict = [v.dict() for v in request.variants]

        # Encode variants using REAL encoder
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
            metadata_json=json.dumps(request.metadata) if request.metadata else None,
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
    Encode a VCF file into a hypervector using the REAL encoder.

    This endpoint accepts a VCF file (plain or gzipped), parses it,
    and encodes all variants into a high-dimensional vector.
    """
    try:
        # Read file content
        file_content = await file.read()

        logger.info(f"Encoding VCF file: {file.filename} ({len(file_content)} bytes)")

        # Parse VCF
        variants = parse_vcf_file(file_content)

        if not variants:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No variants found in VCF file"
            )

        logger.info(f"Parsed {len(variants)} variants from VCF")

        # Encode variants using REAL encoder
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
            metadata_json=json.dumps({"filename": file.filename}),
            source_type="vcf",
        )

        db.add(db_encoding)
        db.commit()

        logger.info(f"Successfully created encoding {encoding_id}")

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


@router.get("/healthz")
async def healthz():
    """Health check endpoint."""
    return {"status": "healthy", "service": "hdc-encoding"}
