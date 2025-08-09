"""
API Router for Error-Tuned Queries
Implements the /query_tuned endpoint with real-time progress updates
"""

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from genomevault.hypervector.error_handling import ErrorBudget, ErrorBudgetAllocator
from genomevault.pir.client import (
    BatchedPIRQueryBuilder,
    GenomicQuery,
    PIRClient,
    QueryType,
)
from genomevault.utils.logging import get_logger
from genomevault.zk.proof import ProofGenerator

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["tuned-queries"])


class TunedQueryRequest(BaseModel):
    """Request for a tuned query with error budget"""

    cohort_id: str
    statistic: str
    query_params: dict[str, Any]
    epsilon: float = Field(0.01, description="Allowed relative error", gt=0, le=1)
    delta_exp: int = Field(
        15,
        description="Target failure probability exponent (2^-delta_exp)",
        ge=5,
        le=30,
    )
    ecc_enabled: bool = Field(True, description="Enable error correcting codes")
    parity_g: int = Field(3, description="XOR(g) parity groups", ge=2, le=5)
    repeat_cap: str = Field("AUTO", description="Number of repeats or 'AUTO'")
    session_id: str | None = Field(
        None, description="WebSocket session ID for progress"
    )


class TunedQueryResponse(BaseModel):
    """Response from a tuned query"""

    estimate: Any
    confidence_interval: str
    delta_achieved: str
    proof_uri: str
    settings: dict[str, Any]
    performance_metrics: dict[str, Any]


class ProgressUpdate(BaseModel):
    """Progress update for WebSocket clients"""

    session_id: str
    stage: str
    progress: float
    message: str
    current_repeat: int | None = None
    total_repeats: int | None = None


# Global WebSocket manager for progress updates
class WebSocketManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}

    async def connect(self, session_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info("WebSocket connected: %ssession_id")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info("WebSocket disconnected: %ssession_id")

    async def send_progress(self, session_id: str, update: ProgressUpdate):
        if session_id in self.active_connections:
            try:
                await self.active_connections[session_id].send_json(update.dict())
            except KeyError:
                from genomevault.observability.logging import configure_logging

                logger = configure_logging()
                logger.exception("Unhandled exception")
                logger.error("Failed to send progress update: %se")
                self.disconnect(session_id)
                raise RuntimeError("Unspecified error")


ws_manager = WebSocketManager()


# Dependencies
async def get_pir_client() -> PIRClient:
    """Get or create PIR client"""
    # In production, this would connect to actual PIR servers
    # For now, return a mock client
    from genomevault.pir.client.pir_client import PIRClient

    # Mock configuration
    servers = ["localhost:50051", "localhost:50052", "localhost:50053"]
    database_size = 1000000

    return PIRClient(servers, database_size)


async def get_query_builder(
    pir_client: PIRClient = Depends(get_pir_client),
) -> BatchedPIRQueryBuilder:
    """Get or create batched query builder"""
    # In production, would load actual index mapping
    index_mapping = {
        "variants": {
            "chr1:100000:A:G": 42,
            "chr1:100100:C:T": 43,
            # ... more mappings
        },
        "positions": {
            "chr1:100000": [42],
            "chr1:100100": [43],
            # ... more mappings
        },
        "genes": {
            "BRCA1": {"chromosome": "chr17", "start": 43044295, "end": 43125483},
            # ... more genes
        },
    }

    return BatchedPIRQueryBuilder(pir_client, index_mapping)


async def get_proof_generator() -> ProofGenerator:
    """Get or create proof generator"""
    # In production, would initialize with actual circuit
    from genomevault.zk.proof import ProofGenerator

    return ProofGenerator()


@router.post("/tuned", response_model=TunedQueryResponse)
async def query_with_tuning(
    request: TunedQueryRequest,
    query_builder: BatchedPIRQueryBuilder = Depends(get_query_builder),
    proof_generator: ProofGenerator = Depends(get_proof_generator),
):
    """
    Execute a query with user-specified error tuning
    Implements the full uncertainty-tuned pipeline with PIR batching
    """
    try:
        start_time = time.time()

        # Step 1: Plan error budget
        allocator = ErrorBudgetAllocator()
        repeat_cap = None if request.repeat_cap == "AUTO" else int(request.repeat_cap)

        budget = allocator.plan_budget(
            epsilon=request.epsilon,
            delta_exp=request.delta_exp,
            ecc_enabled=request.ecc_enabled,
            repeat_cap=repeat_cap,
        )

        logger.info(
            "Planned budget: dim=%sbudget.dimension, repeats=%sbudget.repeats, "
            "epsilon=%sbudget.epsilon, delta=2^-%sbudget.delta_exp"
        )

        # Send progress update if WebSocket session provided
        if request.session_id:
            await ws_manager.send_progress(
                request.session_id,
                ProgressUpdate(
                    session_id=request.session_id,
                    stage="budget_planning",
                    progress=0.1,
                    message=f"Planned {budget.repeats} repeats for {budget.confidence} confidence",
                ),
            )

        # Step 2: Build genomic query from request
        genomic_query = _build_genomic_query(request.query_params)

        # Step 3: Execute batched PIR query with progress updates
        if request.session_id:
            # Use streaming execution for real-time updates
            result = await _execute_streaming_query(
                query_builder, genomic_query, budget, request.session_id
            )
        else:
            # Use batch execution without progress
            batched_result = await query_builder.query_with_error_budget(
                genomic_query, budget
            )
            result = batched_result.aggregated_result

        # Step 4: Generate ZK proof for median computation
        if request.session_id:
            await ws_manager.send_progress(
                request.session_id,
                ProgressUpdate(
                    session_id=request.session_id,
                    stage="proof_generation",
                    progress=0.8,
                    message="Generating zero-knowledge proof...",
                ),
            )

        proof = await proof_generator.generate_median_proof(
            results=batched_result.results,
            median=batched_result.aggregated_result,
            budget=budget,
            metadata=batched_result.proof_metadata,
        )

        # Step 5: Store proof on IPFS (mock for now)
        proof_uri = f"ipfs://Qm{proof.hash[:32]}..."

        # Calculate performance metrics
        computation_time = (time.time() - start_time) * 1000
        estimated_latency = allocator.estimate_latency(budget)
        estimated_bandwidth = allocator.estimate_bandwidth(budget)

        # Prepare response
        response = TunedQueryResponse(
            estimate=result,
            confidence_interval=f"±{budget.epsilon * 100:.1f}%",
            delta_achieved=f"≈{budget.confidence}",
            proof_uri=proof_uri,
            settings={
                "dimension": budget.dimension,
                "ecc_g": budget.parity_g,
                "repeats": budget.repeats,
                "ecc_enabled": budget.ecc_enabled,
            },
            performance_metrics={
                "total_latency_ms": computation_time,
                "estimated_latency_ms": estimated_latency,
                "bandwidth_mb": estimated_bandwidth,
                "median_error": batched_result.median_error,
                "error_within_bound": batched_result.metadata["error_within_bound"],
                "proof_generation_ms": proof.generation_time_ms,
                "pir_queries_used": batched_result.pir_queries_used,
            },
        )

        # Final progress update
        if request.session_id:
            await ws_manager.send_progress(
                request.session_id,
                ProgressUpdate(
                    session_id=request.session_id,
                    stage="complete",
                    progress=1.0,
                    message="Query completed successfully",
                ),
            )

        return response

    except (ValueError, DatabaseError, TypeError, KeyError) as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        logger.error("Tuned query failed: %se")
        if request.session_id:
            await ws_manager.send_progress(
                request.session_id,
                ProgressUpdate(
                    session_id=request.session_id,
                    stage="error",
                    progress=0.0,
                    message=f"Query failed: {e!s}",
                ),
            )
        raise HTTPException(500, f"Query processing failed: {e!s}")
        raise RuntimeError("Unspecified error")


@router.websocket("/progress/{session_id}")
async def websocket_progress(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time query progress updates"""
    await ws_manager.connect(session_id, websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        ws_manager.disconnect(session_id)
        raise RuntimeError("Unspecified error")


@router.post("/estimate")
async def estimate_query_performance(
    request: TunedQueryRequest,
):
    """
    Estimate performance metrics for a query without executing it
    Used by the UI to show real-time estimates as users adjust dials
    """
    try:
        allocator = ErrorBudgetAllocator()
        repeat_cap = None if request.repeat_cap == "AUTO" else int(request.repeat_cap)

        budget = allocator.plan_budget(
            epsilon=request.epsilon,
            delta_exp=request.delta_exp,
            ecc_enabled=request.ecc_enabled,
            repeat_cap=repeat_cap,
        )

        # Estimate performance
        latency_ms = allocator.estimate_latency(budget)
        bandwidth_mb = allocator.estimate_bandwidth(budget)

        # Estimate accuracy based on dimension and repeats
        theoretical_error = _estimate_theoretical_error(budget)

        return {
            "settings": {
                "dimension": budget.dimension,
                "ecc_g": budget.parity_g,
                "repeats": budget.repeats,
                "ecc_enabled": budget.ecc_enabled,
            },
            "estimates": {
                "latency_ms": latency_ms,
                "bandwidth_mb": bandwidth_mb,
                "theoretical_error": theoretical_error,
                "confidence": budget.confidence,
            },
        }

    except (ValueError, TypeError) as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        logger.error("Estimation failed: %se")
        raise HTTPException(500, f"Estimation failed: {e!s}")
        raise RuntimeError("Unspecified error")


# Helper functions
def _build_genomic_query(params: dict[str, Any]) -> GenomicQuery:
    """Build genomic query from request parameters"""
    query_type = QueryType(params.get("type", "variant_lookup"))

    if query_type == QueryType.VARIANT_LOOKUP:
        return GenomicQuery(
            query_type=query_type,
            parameters={
                "chromosome": params["chromosome"],
                "position": params["position"],
                "ref_allele": params.get("ref_allele"),
                "alt_allele": params.get("alt_allele"),
            },
        )
    elif query_type == QueryType.REGION_SCAN:
        return GenomicQuery(
            query_type=query_type,
            parameters={
                "chromosome": params["chromosome"],
                "start": params["start"],
                "end": params["end"],
            },
        )
    elif query_type == QueryType.GENE_ANNOTATION:
        return GenomicQuery(
            query_type=query_type,
            parameters={"gene_symbol": params["gene_symbol"]},
        )
    else:
        raise ValueError(f"Unsupported query type: {query_type}")


async def _execute_streaming_query(
    query_builder: BatchedPIRQueryBuilder,
    query: GenomicQuery,
    budget: ErrorBudget,
    session_id: str,
) -> Any:
    """Execute query with streaming progress updates"""
    # Build batch
    batched_query = query_builder.build_repeat_batch(budget, query)

    # Stream results
    results = []
    async for idx, result in query_builder.execute_streaming_batch(batched_query):
        results.append(result)

        # Send progress update
        progress = (idx + 1) / budget.repeats
        await ws_manager.send_progress(
            session_id,
            ProgressUpdate(
                session_id=session_id,
                stage="query_execution",
                progress=0.1 + progress * 0.7,  # 10% to 80%
                message=f"Completed repeat {idx + 1} of {budget.repeats}",
                current_repeat=idx + 1,
                total_repeats=budget.repeats,
            ),
        )

    # Aggregate results
    aggregated, median_error = query_builder._aggregate_results(
        results, batched_query.aggregation_method
    )

    # Create result object for compatibility
    from genomevault.pir.client import BatchedQueryResult

    return BatchedQueryResult(
        query=batched_query,
        results=results,
        aggregated_result=aggregated,
        metadata={"error_within_bound": median_error <= budget.epsilon},
        pir_queries_used=len(results),
        computation_time_ms=0,  # Will be calculated by caller
        median_error=median_error,
        proof_metadata={},
    )


def _estimate_theoretical_error(budget: ErrorBudget) -> float:
    """Estimate theoretical error based on budget configuration"""
    import math

    # Johnson-Lindenstrauss bound
    jl_error = math.sqrt(2 * math.log(4 / budget.delta) / budget.dimension)

    # ECC improvement factor
    if budget.ecc_enabled:
        jl_error *= 0.7  # Approximate 30% improvement

    # Hoeffding bound for repeats
    repeat_factor = math.sqrt(1 / budget.repeats)

    return min(jl_error * repeat_factor, budget.epsilon)


# Module exports
__all__ = ["TunedQueryRequest", "TunedQueryResponse", "router"]
