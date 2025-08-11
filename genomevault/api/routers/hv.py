"""
Hypervector encoding API endpoints.
"""

import time
from typing import Any, Dict
from pathlib import Path
import numpy as np

from fastapi import APIRouter, HTTPException, status
from prometheus_client import Counter, Histogram, Gauge

from genomevault.api.models.hv import (
    HVEncodeRequest,
    HVEncodeResponse,
    HVBatchEncodeRequest,
    HVBatchEncodeResponse,
    HVSimilarityRequest,
    HVSimilarityResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from genomevault.hypervector_transform.hdc_encoder import (
    HypervectorEncoder,
    HypervectorConfig,
)
from genomevault.hypervector import index as hv_index
from genomevault.core.constants import OmicsType
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

# Prometheus metrics
request_count = Counter(
    "genomevault_hv_requests_total",
    "Total HV encoding requests",
    ["method", "endpoint", "status"],
)

request_duration = Histogram(
    "genomevault_hv_request_duration_seconds",
    "HV request duration in seconds",
    ["endpoint"],
)

encoding_size = Histogram(
    "genomevault_hv_encoding_size_bytes",
    "Size of data being encoded",
    buckets=[100, 1000, 10000, 100000, 1000000, 10000000],
)

active_encodings = Gauge("genomevault_hv_active_encodings", "Number of active encoding operations")

router = APIRouter(
    prefix="/api/v1/hv",
    tags=["hypervector"],
    responses={
        404: {"description": "Not found"},
        503: {"description": "Service temporarily unavailable"},
    },
)


def parse_genomic_data(data: str) -> Dict[str, Any]:
    """
    Parse raw genomic data string into structured format.

    Supports:
    - Simple variant format: "chr1:123456 A>G"
    - VCF-like format
    - JSON format
    """
    variants = []

    # Try to parse as simple variant format
    lines = data.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Parse simple format: "chr1:123456 A>G"
        if ":" in line and ">" in line:
            try:
                location, mutation = line.split(maxsplit=1)
                chrom, pos = location.split(":")
                ref, alt = mutation.split(">")

                variants.append({"chromosome": chrom, "position": int(pos), "ref": ref, "alt": alt})
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse variant line: {line}, error: {e}")
                continue

    return {"variants": variants}


@router.post("/encode", response_model=HVEncodeResponse)
async def encode_hypervector(request: HVEncodeRequest) -> Any:
    """
    Encode genomic data into a privacy-preserving hypervector.

    This endpoint transforms raw genomic data into a high-dimensional
    vector representation that preserves biological relationships while
    ensuring privacy through irreversible transformation.
    """
    start_time = time.time()
    active_encodings.inc()

    try:
        # Track request
        request_count.labels(method="POST", endpoint="/encode", status="started").inc()
        encoding_size.observe(len(request.data.encode("utf-8")))

        # Parse input data
        parsed_data = parse_genomic_data(request.data)
        if not parsed_data.get("variants"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid variants found in input data",
            )

        # Configure encoder
        config = HypervectorConfig(
            dimension=request.dimension,
            seed=42,  # Use deterministic seed for reproducibility
        )
        encoder = HypervectorEncoder(config)

        # Perform encoding
        logger.info(
            f"Encoding {len(parsed_data['variants'])} variants to {request.dimension}D vector"
        )
        encoded_vector = encoder.encode(parsed_data, OmicsType.GENOMIC)

        # Convert tensor to list
        vector_list = (
            encoded_vector.tolist() if hasattr(encoded_vector, "tolist") else list(encoded_vector)
        )

        # Calculate metadata
        encoding_time_ms = (time.time() - start_time) * 1000
        compression_ratio = len(request.data) / (len(vector_list) * 4)  # Assuming float32

        # Build response
        response = HVEncodeResponse(
            vector=vector_list[:100],  # Return first 100 elements for API response
            dimension=request.dimension,
            version=request.version.value,
            compression_tier=request.compression_tier.value,
            metadata={
                "encoding_time_ms": round(encoding_time_ms, 2),
                "input_size_bytes": len(request.data.encode("utf-8")),
                "compression_ratio": round(compression_ratio, 2),
                "variants_encoded": len(parsed_data["variants"]),
                "full_vector_available": True,
                "truncated_for_response": True,
            },
        )

        # Track successful request
        request_count.labels(method="POST", endpoint="/encode", status="success").inc()
        request_duration.labels(endpoint="/encode").observe(time.time() - start_time)

        logger.info(f"Successfully encoded data in {encoding_time_ms:.2f}ms")
        return response

    except HTTPException:
        request_count.labels(method="POST", endpoint="/encode", status="client_error").inc()
        raise
    except Exception as e:
        request_count.labels(method="POST", endpoint="/encode", status="server_error").inc()
        logger.error(f"Encoding failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Encoding failed: {str(e)}",
        )
    finally:
        active_encodings.dec()


@router.post("/encode/batch", response_model=HVBatchEncodeResponse)
async def batch_encode_hypervector(
    request: HVBatchEncodeRequest,
) -> Any:
    """
    Batch encode multiple genomic samples into hypervectors.

    Efficiently processes multiple samples with optional parallel processing.
    """
    start_time = time.time()
    request_count.labels(method="POST", endpoint="/encode/batch", status="started").inc()

    results = []
    successful = 0
    failed = 0

    try:
        config = HypervectorConfig(dimension=request.dimension, seed=42)
        encoder = HypervectorEncoder(config)

        for i, sample in enumerate(request.samples):
            try:
                # Process each sample
                sample_data = sample.get("data", "")
                parsed_data = parse_genomic_data(sample_data)

                if parsed_data.get("variants"):
                    encoded = encoder.encode(parsed_data, OmicsType.GENOMIC)

                    results.append(
                        {
                            "sample_id": sample.get("id", f"sample_{i}"),
                            "status": "success",
                            "dimension": request.dimension,
                            "vector_summary": {
                                "mean": float(encoded.mean()),
                                "std": float(encoded.std()),
                                "min": float(encoded.min()),
                                "max": float(encoded.max()),
                            },
                        }
                    )
                    successful += 1
                else:
                    results.append(
                        {
                            "sample_id": sample.get("id", f"sample_{i}"),
                            "status": "failed",
                            "error": "No valid variants found",
                        }
                    )
                    failed += 1

            except Exception as e:
                results.append(
                    {
                        "sample_id": sample.get("id", f"sample_{i}"),
                        "status": "failed",
                        "error": str(e),
                    }
                )
                failed += 1

        processing_time_ms = (time.time() - start_time) * 1000

        response = HVBatchEncodeResponse(
            results=results,
            total_samples=len(request.samples),
            successful=successful,
            failed=failed,
            metadata={
                "processing_time_ms": round(processing_time_ms, 2),
                "parallel_processing": request.parallel,
                "encoding_version": request.version.value,
            },
        )

        request_count.labels(method="POST", endpoint="/encode/batch", status="success").inc()
        request_duration.labels(endpoint="/encode/batch").observe(time.time() - start_time)

        return response

    except Exception as e:
        request_count.labels(method="POST", endpoint="/encode/batch", status="server_error").inc()
        logger.error(f"Batch encoding failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch encoding failed: {str(e)}",
        )


@router.post("/similarity", response_model=HVSimilarityResponse)
async def compute_similarity(request: HVSimilarityRequest) -> Any:
    """
    Compute similarity between two hypervectors.

    Supports multiple similarity metrics for comparing encoded genomic data.
    """
    start_time = time.time()
    request_count.labels(method="POST", endpoint="/similarity", status="started").inc()

    try:
        import numpy as np

        # Convert to numpy arrays
        v1 = np.array(request.vector1)
        v2 = np.array(request.vector2)

        # Check dimensions match
        if len(v1) != len(v2):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Vector dimensions must match: {len(v1)} != {len(v2)}",
            )

        # Compute similarity based on metric
        if request.metric == "cosine":
            # Cosine similarity
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if norm_v1 == 0 or norm_v2 == 0:
                similarity = 0.0
            else:
                similarity = float(dot_product / (norm_v1 * norm_v2))

        elif request.metric == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(v1 - v2)
            # Convert to similarity score (0 to 1)
            similarity = float(1.0 / (1.0 + distance))

        elif request.metric == "hamming":
            # Hamming distance for binary vectors
            # Binarize vectors first
            v1_binary = (v1 > 0).astype(int)
            v2_binary = (v2 > 0).astype(int)
            hamming_dist = np.sum(v1_binary != v2_binary)
            similarity = float(1.0 - (hamming_dist / len(v1)))
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported metric: {request.metric}",
            )

        processing_time_ms = (time.time() - start_time) * 1000

        response = HVSimilarityResponse(
            similarity=similarity,
            metric=request.metric,
            metadata={
                "processing_time_ms": round(processing_time_ms, 2),
                "vector_dimension": len(v1),
            },
        )

        request_count.labels(method="POST", endpoint="/similarity", status="success").inc()
        request_duration.labels(endpoint="/similarity").observe(time.time() - start_time)

        return response

    except HTTPException:
        request_count.labels(method="POST", endpoint="/similarity", status="client_error").inc()
        raise
    except Exception as e:
        request_count.labels(method="POST", endpoint="/similarity", status="server_error").inc()
        logger.error(f"Similarity computation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Similarity computation failed: {str(e)}",
        )


@router.post("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search_hypervectors(request: SearchRequest) -> Any:
    """
    Search for similar hypervectors in an index.

    This endpoint performs k-nearest neighbor search on a pre-built index
    of hypervectors using the specified distance metric.

    Args:
        request: Search request with query vector and parameters

    Returns:
        SearchResponse with top-k results ordered by distance

    Raises:
        HTTPException: If index not found or search fails
    """
    start_time = time.time()

    try:
        # Validate and resolve index path
        index_path = Path(request.index_path)

        # If relative path, resolve relative to a safe base directory
        if not index_path.is_absolute():
            # Use a safe base directory for indices
            base_dir = Path("data/indices")
            index_path = base_dir / index_path

        # Check if index exists
        if not index_path.exists():
            logger.warning(f"Index not found at: {index_path}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Index not found at: {request.index_path}",
            )

        manifest_file = index_path / "manifest.json"
        if not manifest_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Index manifest not found at: {request.index_path}",
            )

        # Load index metadata
        try:
            metadata = hv_index.load_index_metadata(index_path)
        except Exception as e:
            logger.error(f"Failed to load index metadata: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to load index metadata: {str(e)}",
            )

        # Validate query vector dimension
        query_dimension = len(request.query_vector)
        if query_dimension != metadata.get("dimension", 0):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Query vector dimension ({query_dimension}) doesn't match "
                f"index dimension ({metadata.get('dimension', 0)})",
            )

        # Convert query vector to numpy array
        query_vector = np.array(request.query_vector, dtype=np.float32)

        # Perform search
        try:
            search_results = hv_index.search(
                query=query_vector, path=index_path, k=request.k, metric=request.metric
            )
        except ValueError as e:
            logger.error(f"Search validation error: {e}")
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {str(e)}",
            )

        # Calculate search time
        search_time_ms = (time.time() - start_time) * 1000

        # Convert results to response format
        results = [
            SearchResult(id=result["id"], distance=result["distance"]) for result in search_results
        ]

        # Build response
        response = SearchResponse(
            results=results,
            query_dimension=query_dimension,
            index_size=metadata.get("n_vectors", 0),
            metric=request.metric,
            search_time_ms=round(search_time_ms, 2),
        )

        # Update metrics
        request_count.labels(method="POST", endpoint="/search", status="success").inc()
        request_duration.labels(endpoint="/search").observe(time.time() - start_time)

        logger.info(
            f"Search completed: found {len(results)} results in {search_time_ms:.2f}ms "
            f"from index with {metadata.get('n_vectors', 0)} vectors"
        )

        return response

    except HTTPException:
        request_count.labels(method="POST", endpoint="/search", status="client_error").inc()
        raise
    except Exception as e:
        request_count.labels(method="POST", endpoint="/search", status="server_error").inc()
        logger.error(f"Search endpoint failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search operation failed: {str(e)}",
        )
