"""
GDiff/HDV API router.

Provides REST API endpoints for GDiff-based HDV encoding with caching,
schema selection, and k-anonymity configuration.
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from genomevault.api.models.gdiff import (
    GDiffHDVGenerateRequest,
    GDiffHDVGenerateResponse,
    GDiffBatchGenerateRequest,
    GDiffBatchGenerateResponse,
    SchemaInfo,
    SchemasListResponse,
    CacheStatsResponse,
    CachedHDVInfo,
    ListCachedHDVsResponse,
)
from genomevault.differential_encoding.hdv_cache import HDVCacheManager
from genomevault.differential_encoding.gdiff.selective_hdv_encoder import SelectiveHDVEncoder
from genomevault.differential_encoding.gdiff.schema import GDiffDocument
from genomevault.differential_encoding.gdiff.analysis_schemas import (
    get_schema,
    list_schemas,
    get_schema_summary,
    validate_schema_compatibility,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/gdiff",
    tags=["GDiff", "HDV Encoding"],
    responses={404: {"description": "Not found"}},
)


@router.post("/generate-hdv", response_model=GDiffHDVGenerateResponse)
async def generate_hdv(request: GDiffHDVGenerateRequest):
    """
    Generate HDV encoding from GDiff document with caching.

    This endpoint:
    1. Checks if HDV already exists in cache
    2. Auto-detects reference pool if not specified
    3. Auto-selects k-anonymity based on available references
    4. Validates schema compatibility with GDiff
    5. Generates HDV using SelectiveHDVEncoder
    6. Stores result in cache with metadata

    Returns cached HDV if already exists (unless force=True).
    """
    try:
        logger.info(
            f"HDV generation request: vcf={request.vcf_path}, "
            f"schema={request.schema}, k={request.k_anonymity}"
        )

        # Initialize cache manager with encryption if enabled
        cache = HDVCacheManager(
            cache_root=Path(request.cache_dir),
            enable_encryption=request.enable_encryption,
            encryption_password=request.encryption_password
        )

        # Auto-detect reference pool
        if request.reference_pool:
            ref_pool_path = Path(request.reference_pool)
        else:
            # Try common locations
            possible_pools = [
                Path("benchmark_results/layer2_reference_pool"),
                Path("data/reference_pool"),
                Path("reference_pool"),
            ]
            ref_pool_path = None
            for pool_path in possible_pools:
                if pool_path.exists():
                    ref_pool_path = pool_path
                    break

            if not ref_pool_path:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Reference pool not found. Please specify --reference-pool",
                )

        # Get available references
        available_refs = list(ref_pool_path.glob("*.vcf.gz"))
        num_refs = len(available_refs)

        if num_refs == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No VCF files found in reference pool: {ref_pool_path}",
            )

        # Auto-select k-anonymity
        if request.k_anonymity is None:
            k_anonymity = min(num_refs + 1, 11)  # Max k=11 for k=10+query
            logger.info(f"Auto-selected k-anonymity: k={k_anonymity} (using {num_refs} references)")
        else:
            k_anonymity = request.k_anonymity
            if k_anonymity > num_refs + 1:
                logger.warning(
                    f"k={k_anonymity} requires {k_anonymity-1} references, "
                    f"but only {num_refs} available. Using k={num_refs+1} instead"
                )
                k_anonymity = num_refs + 1

        # Generate query ID
        reference_ids = [ref.stem for ref in available_refs]
        query_id = cache.generate_query_id(request.vcf_path, reference_ids)

        logger.info(f"Query ID: {query_id[:16]}...")

        # Check if already cached
        if not request.force and cache.hdv_exists(query_id, k_anonymity, request.schema):
            logger.info(f"HDV already cached for k={k_anonymity}, schema={request.schema}")

            hdv_path = cache.get_hdv(query_id, k_anonymity, request.schema)
            stats = cache.get_cache_stats(query_id)

            # Load cached metadata
            metadata = cache._load_metadata(query_id)
            cached_entry = next(
                (
                    e
                    for e in metadata.get("hdv_encodings", [])
                    if e["k_anonymity"] == k_anonymity and e["schema_name"] == request.schema
                ),
                None,
            )

            if cached_entry:
                return GDiffHDVGenerateResponse(
                    status="cached",
                    query_id=query_id,
                    k_anonymity=k_anonymity,
                    schema=request.schema,
                    dimension=cached_entry["dimension"],
                    hdv_size_kb=cached_entry["hdv_size_bytes"] / 1024,
                    encoding_time_ms=None,  # Not re-encoded
                    num_variants=cached_entry["num_variants"],
                    features_used=[],  # Not available in cache metadata
                    hdv_path=str(hdv_path),
                    cache_stats=stats,
                )

        # Validate schema
        try:
            schema_obj = get_schema(request.schema)
        except KeyError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown schema '{request.schema}'. "
                f"Available: {', '.join(list_schemas())}",
            )

        # Load GDiff (from cache or provided path)
        if request.gdiff_path and Path(request.gdiff_path).exists():
            logger.info(f"Loading GDiff from: {request.gdiff_path}")
            try:
                gdiff = GDiffDocument.load(Path(request.gdiff_path))
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to load GDiff: {str(e)}",
                )
        else:
            gdiff_path_cached = cache.get_gdiff_path(query_id)
            if gdiff_path_cached.exists():
                logger.info(f"Loading cached GDiff: {gdiff_path_cached}")
                try:
                    gdiff = GDiffDocument.load(gdiff_path_cached)
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to load cached GDiff: {str(e)}",
                    )
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"GDiff not found. Generate GDiff first using differential encoding pipeline. "
                    f"Expected location: {gdiff_path_cached}",
                )

        # Validate schema compatibility
        try:
            validate_schema_compatibility(schema_obj, gdiff)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e),
            )

        # Generate HDV encoding
        logger.info(f"Generating HDV encoding with {schema_obj.schema_name}")

        encoder = SelectiveHDVEncoder(seed=42)

        start_time = time.time()
        try:
            hdv_encoding = encoder.encode(gdiff, schema_obj)
            encoding_duration = (time.time() - start_time) * 1000  # ms

            logger.info(
                f"HDV generated in {encoding_duration:.2f} ms "
                f"(dimension={hdv_encoding.dimension}D, "
                f"variants={hdv_encoding.num_variants_encoded})"
            )

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate HDV: {str(e)}",
            )

        # Store in cache
        try:
            stored_path = cache.store_hdv(
                query_id=query_id,
                k_anonymity=k_anonymity,
                schema_name=request.schema,
                hdv_encoding=hdv_encoding,
                gdiff_path=gdiff_path_cached if not request.gdiff_path else Path(request.gdiff_path),
            )

            logger.info(f"HDV cached at: {stored_path}")

        except Exception as e:
            logger.error(f"Failed to cache HDV: {e}")
            # Continue even if caching fails

        # Get updated cache stats
        stats = cache.get_cache_stats(query_id)

        return GDiffHDVGenerateResponse(
            status="generated",
            query_id=query_id,
            k_anonymity=k_anonymity,
            schema=request.schema,
            dimension=hdv_encoding.dimension,
            hdv_size_kb=hdv_encoding.hdv_size_bytes / 1024,
            encoding_time_ms=encoding_duration,
            num_variants=hdv_encoding.num_variants_encoded,
            features_used=hdv_encoding.features_used,
            hdv_path=str(stored_path),
            cache_stats=stats,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate_hdv: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate HDV: {str(e)}",
        )


@router.post("/batch-generate-hdv", response_model=GDiffBatchGenerateResponse)
async def batch_generate_hdv(request: GDiffBatchGenerateRequest):
    """
    Generate multiple HDV encodings in batch mode.

    Generates HDVs for multiple schemas and k-anonymity levels in one request.
    Useful for pre-populating the cache with common configurations.
    """
    try:
        logger.info(
            f"Batch HDV generation: vcf={request.vcf_path}, "
            f"schemas={request.schemas}, k-levels={request.k_levels}"
        )

        total = len(request.schemas) * len(request.k_levels)
        success_count = 0
        skip_count = 0
        error_count = 0
        results = []

        # Generate all combinations
        for schema in request.schemas:
            for k in request.k_levels:
                result_entry: Dict[str, Any] = {
                    "schema": schema,
                    "k": k,
                }

                try:
                    # Create individual request
                    individual_request = GDiffHDVGenerateRequest(
                        vcf_path=request.vcf_path,
                        reference_pool=request.reference_pool,
                        schema=schema,
                        k_anonymity=k,
                        cache_dir=request.cache_dir,
                        gdiff_path=request.gdiff_path,
                        force=False,  # Don't force in batch mode
                    )

                    # Generate HDV
                    response = await generate_hdv(individual_request)

                    if response.status == "cached":
                        skip_count += 1
                        result_entry["status"] = "cached"
                    else:
                        success_count += 1
                        result_entry["status"] = "generated"
                        result_entry["encoding_time_ms"] = response.encoding_time_ms

                    result_entry["dimension"] = response.dimension
                    result_entry["hdv_size_kb"] = response.hdv_size_kb
                    result_entry["num_variants"] = response.num_variants

                except Exception as e:
                    error_count += 1
                    result_entry["status"] = "error"
                    result_entry["error"] = str(e)
                    logger.error(f"Error generating {schema} k={k}: {e}")

                results.append(result_entry)

        # Get final cache stats
        cache = HDVCacheManager(cache_root=Path(request.cache_dir))
        ref_pool_path = Path(request.reference_pool) if request.reference_pool else Path("benchmark_results/layer2_reference_pool")
        available_refs = list(ref_pool_path.glob("*.vcf.gz"))
        reference_ids = [ref.stem for ref in available_refs]
        query_id = cache.generate_query_id(request.vcf_path, reference_ids)
        stats = cache.get_cache_stats(query_id)

        logger.info(
            f"Batch generation complete: total={total}, "
            f"success={success_count}, skipped={skip_count}, errors={error_count}"
        )

        return GDiffBatchGenerateResponse(
            total=total,
            success=success_count,
            skipped=skip_count,
            errors=error_count,
            results=results,
            cache_stats=stats,
        )

    except Exception as e:
        logger.error(f"Error in batch_generate_hdv: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to batch generate HDVs: {str(e)}",
        )


@router.get("/list-schemas", response_model=SchemasListResponse)
async def list_available_schemas():
    """
    List all available analysis schemas with details.

    Returns information about each schema including dimension,
    encoding time, features, privacy level, and use cases.
    """
    try:
        summary = get_schema_summary()
        schema_infos = []

        for schema_name in list_schemas():
            schema_obj = get_schema(schema_name)
            info = summary[schema_name]

            schema_info = SchemaInfo(
                schema_name=schema_name,
                dimension=info["dimension"],
                encoding_time_ms=info["encoding_time_ms"],
                hdv_size_kb=info["hdv_size_bytes"] / 1024,
                num_features=info["num_features"],
                privacy_level=info["privacy_level"],
                description=info["description"],
                use_cases=schema_obj.use_cases,
                requires_nanopore=schema_obj.requires_nanopore,
                requires_epigenetic=schema_obj.requires_epigenetic,
            )
            schema_infos.append(schema_info)

        return SchemasListResponse(
            schemas=schema_infos,
            total=len(schema_infos),
        )

    except Exception as e:
        logger.error(f"Error in list_available_schemas: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list schemas: {str(e)}",
        )


@router.get("/cache-stats/{query_id}", response_model=CacheStatsResponse)
async def get_cache_stats(query_id: str, cache_dir: str = "data/hdv_cache"):
    """
    Get cache statistics for a specific query.

    Returns information about cached HDV encodings, available k-levels,
    schemas, and total cache size.
    """
    try:
        cache = HDVCacheManager(cache_root=Path(cache_dir))
        stats = cache.get_cache_stats(query_id)

        total_size_bytes = stats.get("total_hdv_size_bytes", 0)

        return CacheStatsResponse(
            query_id=query_id,
            num_encodings=stats["num_encodings"],
            k_levels_available=stats["k_levels_available"],
            schemas_available=stats["schemas_available"],
            gdiff_exists=stats["gdiff_exists"],
            total_hdv_size_mb=total_size_bytes / (1024 * 1024),
        )

    except Exception as e:
        logger.error(f"Error in get_cache_stats: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}",
        )


@router.get("/list-cached/{query_id}", response_model=ListCachedHDVsResponse)
async def list_cached_hdvs(query_id: str, cache_dir: str = "data/hdv_cache"):
    """
    List all cached HDV encodings for a specific query.

    Returns detailed information about each cached HDV including
    k-anonymity level, schema, dimension, size, and creation time.
    """
    try:
        cache = HDVCacheManager(cache_root=Path(cache_dir))
        encodings = cache.list_available_hdvs(query_id)

        cached_hdvs = []
        for encoding in encodings:
            hdv_info = CachedHDVInfo(
                k_anonymity=encoding["k_anonymity"],
                schema_name=encoding["schema_name"],
                dimension=encoding["dimension"],
                hdv_size_kb=encoding["hdv_size_bytes"] / 1024,
                num_variants=encoding["num_variants"],
                encoding_time_ms=encoding["encoding_time_ms"],
                created_timestamp=encoding["created_timestamp"],
                hdv_path=encoding["hdv_path"],
            )
            cached_hdvs.append(hdv_info)

        return ListCachedHDVsResponse(
            query_id=query_id,
            cached_hdvs=cached_hdvs,
            total=len(cached_hdvs),
        )

    except Exception as e:
        logger.error(f"Error in list_cached_hdvs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list cached HDVs: {str(e)}",
        )


@router.get("/healthz")
async def healthz():
    """Health check endpoint."""
    return {"status": "healthy", "service": "gdiff-hdv-encoding"}
