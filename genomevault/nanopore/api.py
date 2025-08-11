"""
FastAPI endpoints for nanopore streaming analysis.

Provides REST API for real-time nanopore data processing
with biological signal detection.

"""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    UploadFile,
)
from pydantic import BaseModel, Field

from genomevault.hypervector.encoding import HypervectorEncoder
from genomevault.nanopore.biological_signals import (
    BiologicalSignalDetector,
    BiologicalSignalType,
)
from genomevault.nanopore.streaming import NanoporeStreamProcessor
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

# Create router
router = APIRouter(prefix="/nanopore", tags=["nanopore"])

# Global processor instances (in production, use dependency injection)
_processors = {}
_results_cache = {}


class NanoporeStreamConfig(BaseModel):
    """Configuration for nanopore streaming."""

    slice_size: int = Field(50000, description="Events per slice")
    overlap: int = Field(1000, description="Event overlap between slices")
    catalytic_space_mb: int = Field(100, description="Catalytic memory size (MB)")
    clean_space_mb: int = Field(1, description="Clean memory limit (MB)")
    enable_gpu: bool = Field(True, description="Enable GPU acceleration")
    anomaly_threshold: float = Field(3.0, description="Anomaly detection threshold")


class StreamingResult(BaseModel):
    """Result from streaming analysis."""

    stream_id: str
    status: str
    total_events: int
    total_slices: int
    processing_time: float
    anomalies_detected: int
    biological_signals: list[dict]


class BiologicalSignalResult(BaseModel):
    """Detected biological signal."""

    signal_type: str
    genomic_position: int
    confidence: float
    variance_score: float
    context: str
    metadata: dict


@router.post("/stream/start")
async def start_streaming(
    config: NanoporeStreamConfig,
    background_tasks: BackgroundTasks = Depends(),
) -> dict[str, str]:
    """
    Start nanopore streaming analysis.

    Returns stream ID for tracking.
    """
    # Generate stream ID
    stream_id = hashlib.sha256(f"{time.time()}_{config.json()}".encode()).hexdigest()[:16]

    # Initialize processor
    encoder = HypervectorEncoder(dimension=10000)
    processor = NanoporeStreamProcessor(
        hv_encoder=encoder,
        catalytic_space_mb=config.catalytic_space_mb,
        clean_space_mb=config.clean_space_mb,
        enable_gpu=config.enable_gpu,
    )

    _processors[stream_id] = processor
    _results_cache[stream_id] = {
        "status": "initializing",
        "results": [],
        "stats": None,
    }

    logger.info(f"Started nanopore stream: {stream_id}")

    return {
        "stream_id": stream_id,
        "status": "started",
        "config": config.dict(),
    }


@router.post("/stream/{stream_id}/upload")
async def upload_fast5(
    stream_id: str,
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = Depends(),
) -> dict[str, Any]:
    """
    Upload Fast5 file for streaming analysis.
    """
    if stream_id not in _processors:
        raise HTTPException(404, f"Stream {stream_id} not found")

    # Save uploaded file
    upload_path = Path(f"/tmp/nanopore_{stream_id}_{file.filename}")

    with open(upload_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Start processing in background
    background_tasks.add_task(
        _process_fast5_async,
        stream_id,
        upload_path,
    )

    _results_cache[stream_id]["status"] = "processing"

    return {
        "stream_id": stream_id,
        "filename": file.filename,
        "size_mb": len(content) / (1024 * 1024),
        "status": "processing",
    }


async def _process_fast5_async(stream_id: str, fast5_path: Path):
    """Background task for Fast5 processing."""
    processor = _processors[stream_id]
    results = []

    # Biological signal detector
    detector = BiologicalSignalDetector()

    async def collect_results(result):
        """Async operation to Collect results.

        Args:
            result: Operation result.
        """
        results.append(result)

        # Detect biological signals
        if result["anomalies"]:
            # In real implementation, would have actual sequence context
            signals = detector.detect_signals(
                variance_array=np.array([a[1] for a in result["anomalies"]]),
                genomic_positions=np.array([a[0] for a in result["anomalies"]]),
            )

            result["biological_signals"] = [
                {
                    "type": sig.signal_type.value,
                    "position": sig.genomic_position,
                    "confidence": sig.confidence,
                }
                for sig in signals
            ]

        _results_cache[stream_id]["results"] = results[-100:]  # Keep last 100

    try:
        # Process file
        stats = await processor.process_fast5(
            fast5_path,
            output_callback=collect_results,
        )

        _results_cache[stream_id]["stats"] = stats
        _results_cache[stream_id]["status"] = "completed"

    except Exception as e:
        logger.exception("Unhandled exception")
        logger.error(f"Error processing stream {stream_id:} %se")
        _results_cache[stream_id]["status"] = "error"
        _results_cache[stream_id]["error"] = str(e)
        raise RuntimeError("Unspecified error")

    finally:
        # Cleanup
        if fast5_path.exists():
            fast5_path.unlink()


@router.get("/stream/{stream_id}/status")
async def get_stream_status(stream_id: str) -> StreamingResult:
    """Get status of streaming analysis."""
    if stream_id not in _results_cache:
        raise HTTPException(404, f"Stream {stream_id} not found")

    cache = _results_cache[stream_id]
    stats = cache.get("stats")

    # Aggregate biological signals
    all_signals = []
    for result in cache["results"]:
        if "biological_signals" in result:
            all_signals.extend(result["biological_signals"])

    return StreamingResult(
        stream_id=stream_id,
        status=cache["status"],
        total_events=stats.total_events if stats else 0,
        total_slices=stats.total_slices if stats else 0,
        processing_time=stats.processing_time if stats else 0,
        anomalies_detected=len(stats.variance_peaks) if stats else 0,
        biological_signals=all_signals,
    )


@router.get("/stream/{stream_id}/signals")
async def get_biological_signals(
    stream_id: str,
    signal_type: str | None = None,
    min_confidence: float = Query(0.5, ge=0, le=1),
) -> list[BiologicalSignalResult]:
    """Get detected biological signals."""
    if stream_id not in _results_cache:
        raise HTTPException(404, f"Stream {stream_id} not found")

    cache = _results_cache[stream_id]

    # Collect all signals
    all_signals = []

    for result in cache["results"]:
        if "biological_signals" not in result:
            continue

        for sig in result["biological_signals"]:
            # Filter by type if specified
            if signal_type and sig["type"] != signal_type:
                continue

            # Filter by confidence
            if sig["confidence"] < min_confidence:
                continue

            all_signals.append(
                BiologicalSignalResult(
                    signal_type=sig["type"],
                    genomic_position=sig["position"],
                    confidence=sig["confidence"],
                    variance_score=sig.get("variance_score", 0),
                    context=sig.get("context", ""),
                    metadata=sig.get("metadata", {}),
                )
            )

    return all_signals


@router.get("/stream/{stream_id}/export")
async def export_results(
    stream_id: str,
    format: str = Query("bedgraph", regex="^(bedgraph|bed|json)$"),
) -> dict[str, Any]:
    """Export analysis results in various formats."""
    if stream_id not in _results_cache:
        raise HTTPException(404, f"Stream {stream_id} not found")

    cache = _results_cache[stream_id]

    if cache["status"] != "completed":
        raise HTTPException(400, f"Stream {stream_id} not completed")

    # Get all biological signals
    detector = BiologicalSignalDetector()
    all_signals = []

    for result in cache["results"]:
        if "anomalies" in result:
            signals = detector.detect_signals(
                variance_array=np.array([a[1] for a in result["anomalies"]]),
                genomic_positions=np.array([a[0] for a in result["anomalies"]]),
            )
            all_signals.extend(signals)

    if format in ["bedgraph", "bed"]:
        # Export as genome browser track
        track_data = detector.export_signal_track(all_signals, format)

        return {
            "format": format,
            "content": track_data,
            "filename": f"nanopore_signals_{stream_id}.{format}",
        }

    else:  # JSON
        return {
            "format": "json",
            "stream_id": stream_id,
            "stats": {
                "total_events": cache["stats"].total_events,
                "total_reads": cache["stats"].total_reads,
                "processing_time": cache["stats"].processing_time,
            },
            "signals": [
                {
                    "type": sig.signal_type.value,
                    "position": sig.genomic_position,
                    "confidence": sig.confidence,
                    "variance_score": sig.variance_score,
                    "context": sig.context,
                    "metadata": sig.metadata,
                }
                for sig in all_signals
            ],
        }


@router.post("/stream/{stream_id}/proof")
async def generate_proof(
    stream_id: str,
    max_slices: int = Query(10, description="Maximum slices to include in proof"),
) -> dict[str, Any]:
    """Generate zero-knowledge proof of analysis."""
    if stream_id not in _processors:
        raise HTTPException(404, f"Stream {stream_id} not found")

    processor = _processors[stream_id]
    cache = _results_cache[stream_id]

    if not cache["results"]:
        raise HTTPException(400, "No results available for proof")

    # Generate proof
    proof_data = await processor.generate_streaming_proof(
        cache["results"][:max_slices],
        proof_type="anomaly_detection",
    )

    return {
        "stream_id": stream_id,
        "proof_size": len(proof_data),
        "proof_hash": hashlib.sha256(proof_data).hexdigest(),
        "slices_included": min(len(cache["results"]), max_slices),
        "anomalies_proven": sum(len(r.get("anomalies", [])) for r in cache["results"][:max_slices]),
    }


@router.delete("/stream/{stream_id}")
async def stop_streaming(stream_id: str) -> dict[str, str]:
    """Stop and cleanup streaming analysis."""
    if stream_id not in _processors:
        raise HTTPException(404, f"Stream {stream_id} not found")

    # Cleanup
    del _processors[stream_id]
    del _results_cache[stream_id]

    return {
        "stream_id": stream_id,
        "status": "stopped",
    }


@router.get("/signal-types")
async def get_signal_types() -> list[dict[str, str]]:
    """Get available biological signal types."""
    return [
        {
            "type": sig_type.value,
            "name": sig_type.name,
            "description": f"Detection of {sig_type.value} modifications",
        }
        for sig_type in BiologicalSignalType
    ]


# WebSocket endpoint for real-time streaming
from fastapi import WebSocket, WebSocketDisconnect


@router.websocket("/stream/{stream_id}/ws")
async def websocket_stream(
    websocket: WebSocket,
    stream_id: str,
):
    """WebSocket for real-time streaming updates."""
    await websocket.accept()

    if stream_id not in _results_cache:
        await websocket.send_json({"error": f"Stream {stream_id} not found"})
        await websocket.close()
        return

    try:
        last_sent = 0

        while True:
            cache = _results_cache[stream_id]

            # Send new results
            if len(cache["results"]) > last_sent:
                new_results = cache["results"][last_sent:]

                for result in new_results:
                    await websocket.send_json(
                        {
                            "type": "slice_result",
                            "data": result,
                        }
                    )

                last_sent = len(cache["results"])

            # Send status update
            await websocket.send_json(
                {
                    "type": "status",
                    "status": cache["status"],
                    "total_results": len(cache["results"]),
                }
            )

            # Check if completed
            if cache["status"] in ["completed", "error"]:
                await websocket.send_json(
                    {
                        "type": "completed",
                        "status": cache["status"],
                        "stats": cache.get("stats"),
                    }
                )
                break

            await asyncio.sleep(1)  # Poll interval

    except WebSocketDisconnect:
        logger.exception("Unhandled exception")
        logger.info(f"WebSocket disconnected for stream {stream_id}")
        raise RuntimeError("Unspecified error")
    except Exception as e:
        logger.exception("Unhandled exception")
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json(
            {
                "type": "error",
                "error": str(e),
            }
        )
        raise RuntimeError("Unspecified error")
    finally:
        await websocket.close()
