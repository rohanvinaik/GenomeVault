"""
Private Information Retrieval (PIR) router for genomic data.

This module provides REST API endpoints for private information retrieval
with multi-server XOR aggregation and Byzantine fault tolerance.
"""

from __future__ import annotations

import base64
import hashlib
import time
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, validator

from genomevault.pir.engine import PIREngine
from genomevault.pir.byzantine_handler import ByzantineHandler
from genomevault.pir.xor_scheme import XORPIRScheme
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/pir",
    tags=["PIR", "Private Information Retrieval"],
    responses={404: {"description": "Not found"}},
)


# Global state for demo
_demo_dataset: Optional[List[bytes]] = None
_pir_engine: Optional[PIREngine] = None
_byzantine_handler: Optional[ByzantineHandler] = None
_server_status = {
    "servers": [],
    "initialized": False,
    "dataset_size": 0,
    "dataset_hash": "",
    "queries_processed": 0,
    "byzantine_incidents": 0,
    "start_time": None,
}


# Pydantic Models
class PIRQueryRequest(BaseModel):
    """Request model for PIR query."""

    index: int = Field(..., ge=0, description="Index to query privately")
    use_byzantine_protection: bool = Field(True, description="Enable Byzantine fault tolerance")
    num_servers: int = Field(3, ge=2, le=10, description="Number of servers to use")

    @validator("index")
    def validate_index(cls, v):
        """Validate index is non-negative."""
        if v < 0:
            raise ValueError("Index must be non-negative")
        return v


class PIRQueryResponse(BaseModel):
    """Response model for PIR query."""

    index: int = Field(..., description="Index that was queried")
    value: str = Field(..., description="Retrieved value (base64 encoded)")
    byzantine_detected: bool = Field(..., description="Whether Byzantine behavior was detected")
    servers_used: int = Field(..., description="Number of servers used")
    query_time_ms: float = Field(..., description="Query execution time in milliseconds")


class PIRSetupRequest(BaseModel):
    """Request model for PIR setup."""

    dataset_type: str = Field("genomic", description="Type of dataset to initialize")
    dataset_size: int = Field(1000, ge=10, le=100000, description="Size of dataset")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    @validator("dataset_type")
    def validate_dataset_type(cls, v):
        """Validate dataset type."""
        valid_types = ["genomic", "random", "sequential", "test"]
        if v not in valid_types:
            raise ValueError(f"Dataset type must be one of: {valid_types}")
        return v


class PIRSetupResponse(BaseModel):
    """Response model for PIR setup."""

    success: bool = Field(..., description="Whether setup was successful")
    dataset_size: int = Field(..., description="Size of initialized dataset")
    dataset_hash: str = Field(..., description="SHA256 hash of dataset")
    num_servers: int = Field(..., description="Number of servers initialized")
    message: str = Field(..., description="Setup status message")


class ServerInfo(BaseModel):
    """Information about a single PIR server."""

    server_id: int = Field(..., description="Server ID")
    is_healthy: bool = Field(..., description="Server health status")
    queries_handled: int = Field(..., description="Number of queries handled")
    last_response_time_ms: Optional[float] = Field(None, description="Last response time")


class PIRStatusResponse(BaseModel):
    """Response model for PIR status."""

    system_healthy: bool = Field(..., description="Overall system health")
    servers: List[ServerInfo] = Field(..., description="Individual server statuses")
    dataset_initialized: bool = Field(..., description="Whether dataset is initialized")
    dataset_size: int = Field(..., description="Current dataset size")
    dataset_hash: str = Field(..., description="Dataset hash")
    total_queries: int = Field(..., description="Total queries processed")
    byzantine_incidents: int = Field(..., description="Number of Byzantine incidents")
    uptime_seconds: Optional[float] = Field(None, description="System uptime in seconds")


# Helper functions
def generate_demo_dataset(dataset_type: str, size: int, seed: Optional[int] = None) -> List[bytes]:
    """Generate a demo dataset based on type."""
    if seed is not None:
        np.random.seed(seed)

    if dataset_type == "test":
        # Simple test dataset
        return [f"item_{i}".encode() for i in range(size)]

    elif dataset_type == "genomic":
        # Simulated genomic data
        dataset = []
        chromosomes = ["chr" + str(i) for i in range(1, 23)] + ["chrX", "chrY"]
        for i in range(size):
            chrom = np.random.choice(chromosomes)
            position = np.random.randint(1, 250000000)
            ref = np.random.choice(["A", "C", "G", "T"])
            alt = np.random.choice(["A", "C", "G", "T"])
            variant = f"{chrom}:{position}:{ref}>{alt}"
            # Hash to fixed size (32 bytes)
            variant_hash = hashlib.sha256(variant.encode()).digest()
            dataset.append(variant_hash)
        return dataset

    elif dataset_type == "random":
        # Random binary data
        return [np.random.bytes(32) for _ in range(size)]

    elif dataset_type == "sequential":
        # Sequential numeric data
        dataset = []
        for i in range(size):
            data = i.to_bytes(32, byteorder="big")
            dataset.append(data)
        return dataset

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_byzantine_handler() -> ByzantineHandler:
    """Get or create Byzantine handler."""
    global _byzantine_handler
    if _byzantine_handler is None:
        _byzantine_handler = ByzantineHandler(
            total_servers=3, threshold=2, enable_error_correction=True
        )
    return _byzantine_handler


def execute_pir_query_with_byzantine(
    index: int, dataset: List[bytes], num_servers: int = 3, use_byzantine: bool = True
) -> Tuple[bytes, bool, float]:
    """
    Execute PIR query with optional Byzantine fault tolerance.

    Returns: (result, byzantine_detected, query_time_ms)
    """
    start_time = time.perf_counter()

    # Create XOR-based PIR scheme
    from genomevault.pir.xor_scheme import XORSchemeParams

    params = XORSchemeParams(database_size=len(dataset))
    pir_scheme = XORPIRScheme(params)

    # Generate queries for each server
    queries = []
    for server_id in range(num_servers):
        query = pir_scheme.generate_query(index, server_id)
        queries.append(query)

    # Process queries on each server
    responses = []
    for server_id, query in enumerate(queries):
        response = pir_scheme.process_query(dataset, query)

        # Simulate Byzantine behavior (for testing)
        if use_byzantine and server_id == 1 and np.random.random() < 0.05:  # 5% Byzantine rate
            # Corrupt the response
            response = bytes([(b + 1) % 256 for b in response])
            logger.warning(f"Simulated Byzantine behavior on server {server_id}")

        responses.append(response)

    # Aggregate responses
    byzantine_detected = False
    if use_byzantine and len(responses) >= 3:
        handler = get_byzantine_handler()
        try:
            result = handler.aggregate_responses(responses)
            byzantine_detected = handler.last_byzantine_detected
            if byzantine_detected:
                _server_status["byzantine_incidents"] += 1
        except Exception as e:
            logger.error(f"Byzantine aggregation failed: {e}")
            # Fall back to simple XOR
            result = pir_scheme.combine_responses(*responses)
    else:
        # Simple XOR aggregation
        result = pir_scheme.combine_responses(*responses)

    query_time_ms = (time.perf_counter() - start_time) * 1000

    # Update server stats
    for i in range(num_servers):
        if i < len(_server_status["servers"]):
            _server_status["servers"][i]["queries_handled"] += 1
            _server_status["servers"][i]["last_response_time_ms"] = query_time_ms / num_servers

    return result, byzantine_detected, query_time_ms


# API Endpoints
@router.post("/query", response_model=PIRQueryResponse)
async def pir_query(request: PIRQueryRequest):
    """
    Execute a private information retrieval query.

    This endpoint allows querying a specific index from the dataset
    without revealing which index was accessed. Uses multi-server
    XOR aggregation with optional Byzantine fault tolerance.
    """
    global _demo_dataset, _server_status

    if _demo_dataset is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dataset not initialized. Call /api/pir/setup first.",
        )

    if request.index >= len(_demo_dataset):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Index {request.index} out of range. Dataset size: {len(_demo_dataset)}",
        )

    try:
        # Execute PIR query
        result, byzantine_detected, query_time = execute_pir_query_with_byzantine(
            index=request.index,
            dataset=_demo_dataset,
            num_servers=request.num_servers,
            use_byzantine=request.use_byzantine_protection,
        )

        # Update global stats
        _server_status["queries_processed"] += 1

        # Encode result as base64
        value_b64 = base64.b64encode(result).decode("ascii")

        logger.info(f"PIR query for index {request.index} completed in {query_time:.2f}ms")

        return PIRQueryResponse(
            index=request.index,
            value=value_b64,
            byzantine_detected=byzantine_detected,
            servers_used=request.num_servers,
            query_time_ms=query_time,
        )

    except Exception as e:
        logger.error(f"PIR query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Query failed: {str(e)}"
        )


@router.post("/setup", response_model=PIRSetupResponse)
async def setup_dataset(request: PIRSetupRequest):
    """
    Initialize a demo dataset for PIR queries.

    This endpoint sets up a dataset that can be queried using the
    PIR protocol. Various dataset types are available.
    """
    global _demo_dataset, _pir_engine, _server_status

    try:
        # Generate dataset
        _demo_dataset = generate_demo_dataset(
            dataset_type=request.dataset_type, size=request.dataset_size, seed=request.seed
        )

        # Calculate dataset hash
        dataset_bytes = b"".join(_demo_dataset)
        dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()

        # Initialize PIR engine
        _pir_engine = PIREngine(_demo_dataset, n_servers=3)

        # Initialize server status
        num_servers = 3
        _server_status = {
            "servers": [
                {
                    "server_id": i,
                    "is_healthy": True,
                    "queries_handled": 0,
                    "last_response_time_ms": None,
                }
                for i in range(num_servers)
            ],
            "initialized": True,
            "dataset_size": len(_demo_dataset),
            "dataset_hash": dataset_hash,
            "queries_processed": 0,
            "byzantine_incidents": 0,
            "start_time": datetime.utcnow(),
        }

        logger.info(
            f"PIR dataset initialized: {request.dataset_size} elements, type: {request.dataset_type}"
        )

        return PIRSetupResponse(
            success=True,
            dataset_size=len(_demo_dataset),
            dataset_hash=dataset_hash,
            num_servers=num_servers,
            message=f"Successfully initialized {request.dataset_type} dataset with {request.dataset_size} elements",
        )

    except Exception as e:
        logger.error(f"Failed to setup dataset: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Setup failed: {str(e)}"
        )


@router.get("/status", response_model=PIRStatusResponse)
async def get_status():
    """
    Get PIR system status and server health.

    Returns the current status of all PIR servers and system health metrics.
    """
    global _server_status, _demo_dataset

    # Calculate uptime
    uptime = None
    if _server_status.get("start_time"):
        uptime = (datetime.utcnow() - _server_status["start_time"]).total_seconds()

    # Convert server info to response models
    servers = [
        ServerInfo(
            server_id=s["server_id"],
            is_healthy=s["is_healthy"],
            queries_handled=s["queries_handled"],
            last_response_time_ms=s.get("last_response_time_ms"),
        )
        for s in _server_status.get("servers", [])
    ]

    # Determine system health
    healthy_servers = sum(1 for s in servers if s.is_healthy)
    system_healthy = (
        _server_status.get("initialized", False)
        and healthy_servers >= 2  # Need at least 2 servers for PIR
    )

    return PIRStatusResponse(
        system_healthy=system_healthy,
        servers=servers,
        dataset_initialized=_server_status.get("initialized", False),
        dataset_size=_server_status.get("dataset_size", 0),
        dataset_hash=_server_status.get("dataset_hash", ""),
        total_queries=_server_status.get("queries_processed", 0),
        byzantine_incidents=_server_status.get("byzantine_incidents", 0),
        uptime_seconds=uptime,
    )


@router.delete("/reset")
async def reset_system():
    """
    Reset the PIR system.

    Clears the dataset and resets all statistics.
    """
    global _demo_dataset, _pir_engine, _byzantine_handler, _server_status

    _demo_dataset = None
    _pir_engine = None
    _byzantine_handler = None
    _server_status = {
        "servers": [],
        "initialized": False,
        "dataset_size": 0,
        "dataset_hash": "",
        "queries_processed": 0,
        "byzantine_incidents": 0,
        "start_time": None,
    }

    logger.info("PIR system reset")

    return {"success": True, "message": "PIR system reset successfully"}
