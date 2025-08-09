"""
Topology API endpoints
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from genomevault.core.config import get_config

router = APIRouter()
config = get_config()


class NodeInfo(BaseModel):
    """Information about a network node"""

    node_id: str
    node_type: str  # light, full, archive
    is_signatory: bool
    voting_power: int
    endpoint: str
    latency: float | None = None
    available: bool = True


class TopologyRequest(BaseModel):
    """Request for network topology"""

    client_location: str | None = None
    required_signatories: int = 2
    max_nodes: int = 10


class TopologyResponse(BaseModel):
    """Network topology response"""

    nearest_lns: list[NodeInfo]
    ts_nodes: list[NodeInfo]
    total_nodes: int
    network_health: float


# Simulated network topology (in production, this would be from a discovery service)
NETWORK_NODES = {
    "ln_node_1": NodeInfo(
        node_id="ln_node_1",
        node_type="light",
        is_signatory=False,
        voting_power=1,
        endpoint="https://ln1.genomevault.org",
        latency=45.2,
    ),
    "ln_node_2": NodeInfo(
        node_id="ln_node_2",
        node_type="full",
        is_signatory=False,
        voting_power=4,
        endpoint="https://ln2.genomevault.org",
        latency=62.8,
    ),
    "ln_node_3": NodeInfo(
        node_id="ln_node_3",
        node_type="archive",
        is_signatory=False,
        voting_power=8,
        endpoint="https://ln3.genomevault.org",
        latency=38.9,
    ),
    "ts_node_1": NodeInfo(
        node_id="ts_node_1",
        node_type="full",
        is_signatory=True,
        voting_power=14,  # 4 + 10
        endpoint="https://ts1.genomevault.org",
        latency=55.3,
    ),
    "ts_node_2": NodeInfo(
        node_id="ts_node_2",
        node_type="light",
        is_signatory=True,
        voting_power=11,  # 1 + 10
        endpoint="https://ts2.genomevault.org",
        latency=41.7,
    ),
}


@router.post("/", response_model=TopologyResponse)
async def get_network_topology(request: TopologyRequest):
    """
    Get optimal network topology for PIR queries

    Returns nearest light nodes and trusted signatories based on:
    - Network latency
    - Node availability
    - Required number of trusted signatories
    """
    try:
        # Separate nodes by type
        light_nodes = []
        signatory_nodes = []

        for node in NETWORK_NODES.values():
            if node.available:
                if node.is_signatory:
                    signatory_nodes.append(node)
                else:
                    light_nodes.append(node)

        # Sort by latency
        light_nodes.sort(key=lambda x: x.latency or float("inf"))
        signatory_nodes.sort(key=lambda x: x.latency or float("inf"))

        # Select nodes based on requirements
        selected_lns = light_nodes[: max(3, request.max_nodes - request.required_signatories)]
        selected_ts = signatory_nodes[: request.required_signatories]

        # Validate we have enough nodes
        if len(selected_ts) < request.required_signatories:
            raise HTTPException(
                status_code=503,
                detail="Insufficient trusted signatories: {len(selected_ts)} < {request.required_signatories}",
            )

        # Calculate network health
        total_available = len(light_nodes) + len(signatory_nodes)
        total_nodes = len(NETWORK_NODES)
        network_health = total_available / total_nodes if total_nodes > 0 else 0

        return TopologyResponse(
            nearest_lns=selected_lns,
            ts_nodes=selected_ts,
            total_nodes=total_nodes,
            network_health=network_health,
        )

    except Exception as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        raise HTTPException(status_code=500, detail=str(e))
        raise RuntimeError("Unspecified error")


@router.get("/nodes")
async def list_all_nodes():
    """List all nodes in the network"""
    return {
        "nodes": list(NETWORK_NODES.values()),
        "total": len(NETWORK_NODES),
        "summary": {
            "light_nodes": sum(1 for n in NETWORK_NODES.values() if n.node_type == "light"),
            "full_nodes": sum(1 for n in NETWORK_NODES.values() if n.node_type == "full"),
            "archive_nodes": sum(1 for n in NETWORK_NODES.values() if n.node_type == "archive"),
            "trusted_signatories": sum(1 for n in NETWORK_NODES.values() if n.is_signatory),
        },
    }


@router.get("/node/{node_id}")
async def get_node_info(node_id: str):
    """Get information about a specific node"""
    if node_id not in NETWORK_NODES:
        raise HTTPException(status_code=404, detail="Node {node_id} not found")

    return NETWORK_NODES[node_id]


@router.post("/ping/{node_id}")
async def ping_node(node_id: str):
    """Ping a node to check availability and latency"""
    if node_id not in NETWORK_NODES:
        raise HTTPException(status_code=404, detail="Node {node_id} not found")

    # Simulate ping
    import random

    latency = random.uniform(20, 100)
    available = random.random() > 0.1  # 90% availability

    # Update node info
    NETWORK_NODES[node_id].latency = latency
    NETWORK_NODES[node_id].available = available

    return {
        "node_id": node_id,
        "latency": latency,
        "available": available,
        "timestamp": asyncio.get_event_loop().time(),
    }
