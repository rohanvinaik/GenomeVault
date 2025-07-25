"""
API Router for SNP-tuned and Zoom Queries
Extends the tuned query endpoint with panel granularity and hierarchical zoom
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from genomevault.hypervector.encoding.genomic import GenomicEncoder, PanelGranularity
from genomevault.hypervector.error_handling import ErrorBudget, ErrorBudgetAllocator
from genomevault.pir.client import BatchedPIRQueryBuilder, GenomicQuery, QueryType
from genomevault.utils.logging import get_logger
from genomevault.zk.proof import ProofGenerator

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["snp-tuned-queries"])


class SNPPanelQueryRequest(BaseModel):
    """Request for SNP panel-based query"""

    cohort_id: str
    query_type: str = Field("variant_lookup", description="Type of query")
    query_params: Dict[str, Any]
    panel: str = Field("off", description="SNP panel granularity: off/common/clinical/custom")
    custom_panel_path: Optional[str] = Field(None, description="Path to custom BED/VCF file")
    epsilon: float = Field(0.01, description="Allowed relative error", gt=0, le=1)
    delta_exp: int = Field(15, description="Target failure probability exponent", ge=5, le=30)


class ZoomQueryRequest(BaseModel):
    """Request for hierarchical zoom query"""

    cohort_id: str
    chromosome: str
    start_position: int
    end_position: int
    initial_level: int = Field(0, description="Initial zoom level (0-2)", ge=0, le=2)
    auto_zoom: bool = Field(True, description="Automatically zoom to hotspots")
    epsilon: float = Field(0.01, description="Allowed relative error")
    delta_exp: int = Field(15, description="Target failure probability exponent")


class PanelQueryResponse(BaseModel):
    """Response from panel-based query"""

    result: Any
    panel_used: str
    positions_encoded: int
    confidence: float
    encoding_time_ms: float
    extra_encode_time_ms: float
    extra_ram_mb: float
    proof_uri: str


class ZoomQueryResponse(BaseModel):
    """Response from hierarchical zoom query"""

    chromosome: str
    region: List[int]
    levels_fetched: List[int]
    hotspots: List[Dict[str, int]]
    aggregated_result: Any
    proof_uri: str
    performance: Dict[str, float]


# Dependencies
async def get_genomic_encoder(enable_snp: bool = True) -> GenomicEncoder:
    """Get genomic encoder with SNP mode support"""
    return GenomicEncoder(
        dimension=100000,  # 100k dimensions for SNP accuracy
        enable_snp_mode=enable_snp,
        panel_granularity=PanelGranularity.OFF,
    )


@router.post("/panel", response_model=PanelQueryResponse)
async def query_with_panel(
    request: SNPPanelQueryRequest,
    encoder: GenomicEncoder = Depends(get_genomic_encoder),
    query_builder: BatchedPIRQueryBuilder = Depends(),
    proof_generator: ProofGenerator = Depends(),
):
    """
    Execute query with SNP panel encoding for single-nucleotide accuracy
    """
    try:
        start_time = time.time()

        # Configure encoder with requested panel
        panel_granularity = PanelGranularity(request.panel)
        encoder.set_panel_granularity(panel_granularity)

        # Load custom panel if provided
        if panel_granularity == PanelGranularity.CUSTOM and request.custom_panel_path:
            encoder.load_custom_panel(request.custom_panel_path)
            logger.info(f"Loaded custom panel from {request.custom_panel_path}")

        # Get panel info
        panel_info = encoder.snp_panel.get_panel_info(panel_granularity.value)

        # Plan error budget with panel considerations
        allocator = ErrorBudgetAllocator()
        budget = allocator.plan_budget(
            epsilon=request.epsilon,
            delta_exp=request.delta_exp,
            ecc_enabled=True,
            dimension=encoder.dimension,  # Use encoder's dimension
        )

        # Build genomic query
        genomic_query = _build_genomic_query_from_params(request.query_type, request.query_params)

        # Measure encoding overhead
        encode_start = time.time()

        # Execute query with panel encoding
        if request.query_type == "variant_lookup":
            # Single variant with panel
            variant = request.query_params
            encoded_variant = encoder.encode_variant(
                chromosome=variant["chromosome"],
                position=variant["position"],
                ref=variant.get("ref", "N"),
                alt=variant.get("alt", "N"),
                use_panel=True,
            )

            # PIR query for encoded variant
            batched_result = await query_builder.query_with_error_budget(genomic_query, budget)
            result = batched_result.aggregated_result

        elif request.query_type == "genome_scan":
            # Full genome with panel
            variants = request.query_params.get("variants", [])
            encoded_genome = encoder.encode_genome_with_panel(
                variants, panel_name=panel_granularity.value
            )

            # Create special query for encoded genome
            genome_query = GenomicQuery(
                query_type=QueryType.GENOME_SIMILARITY,
                parameters={"encoded_genome": encoded_genome},
            )

            batched_result = await query_builder.query_with_error_budget(genome_query, budget)
            result = batched_result.aggregated_result

        else:
            raise ValueError(f"Unsupported query type: {request.query_type}")

        encode_time = (time.time() - encode_start) * 1000

        # Generate proof
        proof = await proof_generator.generate_median_proof(
            results=batched_result.results,
            median=result,
            budget=budget,
            metadata={"panel": panel_granularity.value, "positions": panel_info["size"]},
        )

        # Calculate overhead
        baseline_encode_time = 10  # ms for standard encoding
        extra_encode_time = max(0, encode_time - baseline_encode_time)

        # RAM overhead estimation
        positions = panel_info["size"]
        bytes_per_position = 4  # 32-bit hash seed
        extra_ram_mb = (positions * bytes_per_position) / (1024**2)

        total_time = (time.time() - start_time) * 1000

        return PanelQueryResponse(
            result=result,
            panel_used=panel_info["name"],
            positions_encoded=positions,
            confidence=batched_result.metadata["error_within_bound"],
            encoding_time_ms=encode_time,
            extra_encode_time_ms=extra_encode_time,
            extra_ram_mb=extra_ram_mb,
            proof_uri=f"ipfs://Qm{proof.hash[:32]}...",
        )

    except Exception as e:
        logger.error(f"Panel query failed: {e}")
        raise HTTPException(500, f"Panel query failed: {str(e)}")


@router.post("/zoom", response_model=ZoomQueryResponse)
async def query_with_zoom(
    request: ZoomQueryRequest,
    encoder: GenomicEncoder = Depends(get_genomic_encoder),
    query_builder: BatchedPIRQueryBuilder = Depends(),
    proof_generator: ProofGenerator = Depends(),
):
    """
    Execute hierarchical zoom query for genomic regions
    """
    try:
        start_time = time.time()

        # Plan error budget
        allocator = ErrorBudgetAllocator()
        budget = allocator.plan_budget(
            epsilon=request.epsilon, delta_exp=request.delta_exp, ecc_enabled=True
        )

        # Execute hierarchical zoom
        if request.auto_zoom:
            # Automatic zoom with hotspot detection
            zoom_results = await query_builder.execute_hierarchical_zoom(
                chromosome=request.chromosome,
                region_start=request.start_position,
                region_end=request.end_position,
                budget=budget,
            )

            levels_fetched = [0, 1]
            hotspots = zoom_results["hotspots"]

        else:
            # Manual single-level zoom
            zoom_results = await query_builder.execute_zoom_query(
                chromosome=request.chromosome,
                start=request.start_position,
                end=request.end_position,
                zoom_level=request.initial_level,
            )

            levels_fetched = [request.initial_level]
            hotspots = []

        # Generate aggregated proof
        proof = await proof_generator.generate_zoom_proof(
            zoom_results=zoom_results,
            budget=budget,
            metadata={
                "chromosome": request.chromosome,
                "region_size": request.end_position - request.start_position,
                "levels": levels_fetched,
            },
        )

        total_time = (time.time() - start_time) * 1000

        # Calculate performance metrics
        performance = {
            "total_time_ms": total_time,
            "level0_time_ms": zoom_results.get("level0_time", 0),
            "level1_time_ms": zoom_results.get("level1_time", 0),
            "proof_time_ms": proof.generation_time_ms,
            "tiles_fetched": zoom_results.get("tiles_fetched", 0),
        }

        return ZoomQueryResponse(
            chromosome=request.chromosome,
            region=[request.start_position, request.end_position],
            levels_fetched=levels_fetched,
            hotspots=hotspots,
            aggregated_result=zoom_results.get("aggregated_result", {}),
            proof_uri=f"ipfs://Qm{proof.hash[:32]}...",
            performance=performance,
        )

    except Exception as e:
        logger.error(f"Zoom query failed: {e}")
        raise HTTPException(500, f"Zoom query failed: {str(e)}")


@router.get("/panel/info")
async def get_panel_info(
    panel_name: str = Query(..., description="Panel name"),
    encoder: GenomicEncoder = Depends(get_genomic_encoder),
):
    """Get information about available SNP panels"""
    try:
        if panel_name == "all":
            # List all panels
            panels = encoder.snp_panel.list_panels()
            panel_infos = {}
            for panel in panels:
                panel_infos[panel] = encoder.snp_panel.get_panel_info(panel)
            return {"panels": panel_infos}
        else:
            # Get specific panel info
            info = encoder.snp_panel.get_panel_info(panel_name)
            return {"panel": panel_name, "info": info}

    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Failed to get panel info: {e}")
        raise HTTPException(500, f"Failed to get panel info: {str(e)}")


@router.post("/panel/estimate")
async def estimate_panel_overhead(panel: str = "common", positions: Optional[int] = None):
    """Estimate overhead for using a specific SNP panel"""
    # Panel position counts
    panel_sizes = {
        "off": 0,
        "common": 1_000_000,
        "clinical": 10_000_000,
        "custom": positions or 5_000_000,
    }

    panel_size = panel_sizes.get(panel, 1_000_000)

    # Encoding time estimates (GPU)
    if panel == "off":
        encode_time_ms = 0
    elif panel == "common":
        encode_time_ms = 8000  # 8s
    elif panel == "clinical":
        encode_time_ms = 45000  # 45s
    else:
        encode_time_ms = panel_size * 0.0045  # ~4.5ms per 1k positions

    # RAM estimates
    bytes_per_position = 4  # 32-bit hash seeds
    disk_mb = (panel_size * bytes_per_position) / (1024**2)

    return {
        "panel": panel,
        "positions": panel_size,
        "extra_encode_time_ms": encode_time_ms,
        "extra_disk_mb": disk_mb,
        "description": f"Panel with {panel_size:,} SNP positions",
    }


# Helper functions
def _build_genomic_query_from_params(query_type: str, params: Dict) -> GenomicQuery:
    """Build genomic query from type and parameters"""
    if query_type == "variant_lookup":
        return GenomicQuery(query_type=QueryType.VARIANT_LOOKUP, parameters=params)
    elif query_type == "region_scan":
        return GenomicQuery(query_type=QueryType.REGION_SCAN, parameters=params)
    elif query_type == "genome_scan":
        return GenomicQuery(query_type=QueryType.GENOME_SIMILARITY, parameters=params)
    else:
        raise ValueError(f"Unknown query type: {query_type}")
