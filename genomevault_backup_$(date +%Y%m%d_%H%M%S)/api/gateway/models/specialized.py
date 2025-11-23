"""
Specialized endpoint models for GenomeVault API Gateway.

Models for specialized endpoints from Section 5.2.4 including:
- Topology management
- Credit/vault redemption
- Audit challenges
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, validator

from genomevault.api.gateway.models.base import BaseModel


class NodeType(str, Enum):
    """Types of network nodes."""

    LIGHT_NODE = "lightNode"
    TRUSTED_SERVER = "trustedServer"
    VALIDATOR = "validator"
    STORAGE_NODE = "storageNode"
    COMPUTE_NODE = "computeNode"


class NodeStatus(str, Enum):
    """Node status values."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    DEGRADED = "degraded"


class NetworkNode(BaseModel):
    """Network node information."""

    node_id: str = Field(..., description="Unique node identifier")
    node_type: NodeType = Field(..., description="Type of node")
    status: NodeStatus = Field(..., description="Current node status")

    # Network information
    address: str = Field(..., description="Network address")
    port: int = Field(..., ge=1, le=65535, description="Network port")
    public_key: str = Field(..., description="Node public key")

    # Capabilities
    capabilities: List[str] = Field(..., description="Node capabilities")
    max_connections: int = Field(..., ge=1, description="Maximum concurrent connections")

    # Performance metrics
    latency_ms: Optional[float] = Field(None, description="Average latency in milliseconds")
    uptime_percent: Optional[float] = Field(None, ge=0, le=100, description="Uptime percentage")
    load_percent: Optional[float] = Field(None, ge=0, le=100, description="Current load percentage")

    # Metadata
    last_seen: datetime = Field(..., description="Last contact timestamp")
    version: str = Field(..., description="Node software version")
    region: Optional[str] = Field(None, description="Geographic region")


class TopologyRequest(BaseModel):
    """Request model for topology discovery."""

    # Client information
    client_location: Optional[Dict[str, float]] = Field(
        None,
        description="Client location (lat/lng) for proximity optimization"
    )
    client_capabilities: Optional[List[str]] = Field(
        None,
        description="Client capabilities for compatibility matching"
    )

    # Discovery parameters
    max_nodes: int = Field(10, ge=1, le=100, description="Maximum nodes to return")
    node_types: Optional[List[NodeType]] = Field(None, description="Filter by node types")
    required_capabilities: Optional[List[str]] = Field(None, description="Required node capabilities")

    # Optimization preferences
    optimize_for: str = Field(
        "latency",
        pattern=r"^(latency|bandwidth|reliability|cost)$",
        description="Optimization preference"
    )
    exclude_nodes: Optional[List[str]] = Field(None, description="Node IDs to exclude")

    # Privacy preferences
    privacy_level: str = Field(
        "standard",
        pattern=r"^(minimal|standard|high|maximum)$",
        description="Required privacy level"
    )

    model_config = {
        "json_schema_extra": {
            "examples": {
                "basic_topology": {
                    "summary": "Basic topology request",
                    "value": {
                        "max_nodes": 5,
                        "node_types": ["lightNode", "trustedServer"],
                        "optimize_for": "latency"
                    }
                },
                "location_based": {
                    "summary": "Location-based topology request",
                    "value": {
                        "client_location": {"lat": 37.7749, "lng": -122.4194},
                        "max_nodes": 10,
                        "required_capabilities": ["pir", "zk_proofs"],
                        "optimize_for": "reliability"
                    }
                }
            }
        }


class TopologyResponse(BaseModel):
    """Response model for topology discovery."""

    # Primary topology nodes
    nearestLNs: List[NetworkNode] = Field(..., description="Nearest light nodes")
    tsNodes: List[NetworkNode] = Field(..., description="Trusted server nodes")

    # Additional topology information
    total_nodes_available: int = Field(..., description="Total nodes available in network")
    selection_criteria: Dict[str, Any] = Field(..., description="Criteria used for node selection")

    # Network statistics
    network_health: str = Field(
        ...,
        pattern=r"^(excellent|good|fair|poor)$",
        description="Overall network health"
    )
    average_latency_ms: float = Field(..., description="Average network latency")
    coverage_score: float = Field(..., ge=0, le=1, description="Geographic coverage score")

    # Optimization results
    optimization_score: float = Field(..., ge=0, le=1, description="Optimization effectiveness score")
    failover_nodes: Optional[List[NetworkNode]] = Field(None, description="Backup/failover nodes")

    model_config = {
        "json_schema_extra": {
            "example": {
                "nearestLNs": [
                    {
                        "node_id": "ln_abc123",
                        "node_type": "lightNode",
                        "status": "active",
                        "address": "192.168.1.100",
                        "port": 8080,
                        "public_key": "0x1234567890abcdef",
                        "capabilities": ["pir", "basic_compute"],
                        "max_connections": 100,
                        "latency_ms": 15.2,
                        "uptime_percent": 99.8,
                        "last_seen": "2024-01-15T10:30:00Z",
                        "version": "1.0.0",
                        "region": "us-west"
                    }
                ],
                "tsNodes": [
                    {
                        "node_id": "ts_def456",
                        "node_type": "trustedServer",
                        "status": "active",
                        "address": "10.0.0.50",
                        "port": 8443,
                        "public_key": "0xfedcba0987654321",
                        "capabilities": ["pir", "zk_proofs", "federated_learning"],
                        "max_connections": 500,
                        "latency_ms": 25.7,
                        "uptime_percent": 99.9,
                        "last_seen": "2024-01-15T10:29:45Z",
                        "version": "1.0.0",
                        "region": "us-west"
                    }
                ],
                "total_nodes_available": 150,
                "selection_criteria": {
                    "optimize_for": "latency",
                    "max_nodes": 10,
                    "required_capabilities": ["pir"]
                },
                "network_health": "excellent",
                "average_latency_ms": 20.4,
                "coverage_score": 0.85,
                "optimization_score": 0.92
            }
        }


class CreditType(str, Enum):
    """Types of credits in the system."""

    COMPUTE_CREDITS = "compute"
    STORAGE_CREDITS = "storage"
    NETWORK_CREDITS = "network"
    PREMIUM_CREDITS = "premium"
    RESEARCH_CREDITS = "research"


class CreditRedemptionRequest(BaseModel):
    """Request model for credit/vault redemption."""

    # Credit information
    vault_id: str = Field(..., description="Vault identifier containing credits")
    credit_type: CreditType = Field(..., description="Type of credits to redeem")
    amount: int = Field(..., ge=1, description="Number of credits to redeem")

    # Redemption details
    service_type: str = Field(..., description="Service for which credits are being redeemed")
    service_params: Optional[Dict[str, Any]] = Field(None, description="Service-specific parameters")

    # Authentication
    vault_signature: str = Field(..., description="Cryptographic signature proving vault ownership")
    nonce: str = Field(..., description="Unique nonce to prevent replay attacks")

    # Optional preferences
    priority: int = Field(5, ge=1, le=10, description="Redemption priority (1=highest)")
    expiry_time: Optional[datetime] = Field(None, description="Redemption request expiry")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

    @field_validator("vault_signature")
    def validate_signature(cls, v):
        """Validate signature format."""
        if not v.startswith("0x") or len(v) < 10:
            raise ValueError("Invalid signature format")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": {
                "compute_redemption": {
                    "summary": "Redeem compute credits",
                    "value": {
                        "vault_id": "vault_abc123456789",
                        "credit_type": "compute",
                        "amount": 100,
                        "service_type": "zk_proof_generation",
                        "service_params": {
                            "circuit_type": "variant_frequency",
                            "proof_system": "groth16"
                        },
                        "vault_signature": "0x1234567890abcdef...",
                        "nonce": "nonce_987654321",
                        "priority": 7
                    }
                },
                "storage_redemption": {
                    "summary": "Redeem storage credits",
                    "value": {
                        "vault_id": "vault_def456789012",
                        "credit_type": "storage",
                        "amount": 50,
                        "service_type": "vector_storage",
                        "vault_signature": "0xfedcba0987654321...",
                        "nonce": "nonce_123456789"
                    }
                }
            }
        }


class CreditRedemptionResponse(BaseModel):
    """Response model for credit/vault redemption."""

    # Redemption result
    invoiceId: str = Field(..., description="Unique invoice identifier")
    creditsBurned: int = Field(..., description="Number of credits actually consumed")

    # Transaction details
    transaction_hash: Optional[str] = Field(None, description="Blockchain transaction hash")
    redemption_timestamp: datetime = Field(..., description="Redemption completion timestamp")

    # Service allocation
    service_allocation: Dict[str, Any] = Field(..., description="Allocated service resources")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated service completion")

    # Credit balance after redemption
    remaining_credits: Dict[CreditType, int] = Field(..., description="Remaining credits by type")

    # Receipt information
    receipt_url: Optional[str] = Field(None, description="URL to detailed receipt")
    audit_trail: str = Field(..., description="Audit trail hash for the transaction")

    model_config = {
        "json_schema_extra": {
            "example": {
                "invoiceId": "inv_abc123456789def",
                "creditsBurned": 95,
                "transaction_hash": "0xabcdef123456789...",
                "redemption_timestamp": "2024-01-15T10:30:00Z",
                "service_allocation": {
                    "compute_units": 95,
                    "estimated_runtime_minutes": 30,
                    "allocated_nodes": ["node_1", "node_2"]
                },
                "estimated_completion": "2024-01-15T11:00:00Z",
                "remaining_credits": {
                    "compute": 405,
                    "storage": 1000,
                    "network": 250
                },
                "audit_trail": "audit_hash_def456789012"
            }
        }


class ChallengeType(str, Enum):
    """Types of audit challenges."""

    PROOF_VERIFICATION = "proof_verification"
    DATA_INTEGRITY = "data_integrity"
    COMPUTATION_CORRECTNESS = "computation_correctness"
    PRIVACY_COMPLIANCE = "privacy_compliance"
    PERFORMANCE_AUDIT = "performance_audit"


class AuditChallengeRequest(BaseModel):
    """Request model for audit challenge."""

    # Challenge details
    challenge_type: ChallengeType = Field(..., description="Type of audit challenge")
    target_node: str = Field(..., description="Node ID being challenged")
    challenge_data: Dict[str, Any] = Field(..., description="Challenge-specific data")

    # Challenger information
    challenger_id: str = Field(..., description="Challenger node/entity ID")
    challenger_signature: str = Field(..., description="Challenger's cryptographic signature")

    # Challenge parameters
    epoch: int = Field(..., ge=0, description="Epoch number for the challenge")
    deadline: datetime = Field(..., description="Challenge response deadline")
    stake_amount: Optional[int] = Field(None, ge=0, description="Stake amount for challenge")

    # Evidence
    evidence_hash: Optional[str] = Field(None, description="Hash of supporting evidence")
    witness_nodes: Optional[List[str]] = Field(None, description="Witness node IDs")

    # Challenge specifics
    expected_result: Optional[str] = Field(None, description="Expected result hash")
    verification_method: str = Field(..., description="Method for verification")

    @field_validator("challenge_data")
    def validate_challenge_data(cls, v, values):
        """Validate challenge data based on challenge type."""
        challenge_type = values.get("challenge_type")
        if challenge_type == ChallengeType.PROOF_VERIFICATION:
            required_fields = ["proof_id", "verification_key"]
            for field in required_fields:
                if field not in v:
                    raise ValueError(f"Proof verification challenges require '{field}' in challenge_data")
        elif challenge_type == ChallengeType.DATA_INTEGRITY:
            if "data_hash" not in v:
                raise ValueError("Data integrity challenges require 'data_hash' in challenge_data")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": {
                "proof_verification": {
                    "summary": "Proof verification challenge",
                    "value": {
                        "challenge_type": "proof_verification",
                        "target_node": "node_abc123",
                        "challenge_data": {
                            "proof_id": "proof_def456",
                            "verification_key": "0xabcdef123456...",
                            "public_inputs": ["42", "0", "1"]
                        },
                        "challenger_id": "challenger_xyz789",
                        "challenger_signature": "0x987654321fedcba...",
                        "epoch": 1005,
                        "deadline": "2024-01-15T12:00:00Z",
                        "expected_result": "verification_success_hash",
                        "verification_method": "groth16_verify"
                    }
                },
                "data_integrity": {
                    "summary": "Data integrity challenge",
                    "value": {
                        "challenge_type": "data_integrity",
                        "target_node": "storage_node_456",
                        "challenge_data": {
                            "data_hash": "0xfedcba987654321...",
                            "chunk_indices": [10, 25, 73],
                            "merkle_root": "0x123456789abcdef..."
                        },
                        "challenger_id": "auditor_123",
                        "challenger_signature": "0xabcdef123456789...",
                        "epoch": 1005,
                        "deadline": "2024-01-15T11:30:00Z",
                        "verification_method": "merkle_proof"
                    }
                }
            }
        }


class AuditChallengeResponse(BaseModel):
    """Response model for audit challenge."""

    # Challenge result
    challenge_id: str = Field(..., description="Unique challenge identifier")
    challenger: str = Field(..., description="Challenger node/entity ID")
    target: str = Field(..., description="Target node ID")
    epoch: int = Field(..., description="Challenge epoch")
    resultHash: str = Field(..., description="Result hash for verification")

    # Challenge status
    status: str = Field(
        ...,
        pattern=r"^(accepted|rejected|pending_response)$",
        description="Challenge acceptance status"
    )
    acceptance_timestamp: datetime = Field(..., description="Challenge acceptance timestamp")

    # Response details
    response_deadline: datetime = Field(..., description="Deadline for target node response")
    verification_nodes: List[str] = Field(..., description="Nodes assigned to verify response")

    # Economic details
    stake_locked: Optional[int] = Field(None, description="Stake amount locked for challenge")
    reward_pool: Optional[int] = Field(None, description="Reward pool for successful challenge")
    penalty_amount: Optional[int] = Field(None, description="Penalty for failed challenge")

    # Metadata
    challenge_complexity: int = Field(..., ge=1, le=10, description="Challenge complexity score")
    estimated_verification_time: int = Field(..., description="Estimated verification time in minutes")

    model_config = {
        "json_schema_extra": {
            "example": {
                "challenge_id": "challenge_abc123456789",
                "challenger": "challenger_xyz789",
                "target": "node_abc123",
                "epoch": 1005,
                "resultHash": "0xabc123def456789...",
                "status": "accepted",
                "acceptance_timestamp": "2024-01-15T10:30:00Z",
                "response_deadline": "2024-01-15T12:00:00Z",
                "verification_nodes": ["verifier_001", "verifier_002", "verifier_003"],
                "stake_locked": 1000,
                "reward_pool": 500,
                "challenge_complexity": 7,
                "estimated_verification_time": 45
            }
        }


class ChallengeStatusRequest(BaseModel):
    """Request model for challenge status inquiry."""

    challenge_id: str = Field(..., description="Challenge identifier")
    include_response_data: bool = Field(False, description="Include target node response data")
    include_verification_details: bool = Field(False, description="Include verification process details")


class ChallengeStatusResponse(BaseModel):
    """Response model for challenge status."""

    challenge_id: str = Field(..., description="Challenge identifier")
    current_status: str = Field(..., description="Current challenge status")
    phase: str = Field(..., description="Current phase of challenge process")

    # Timeline
    created_at: datetime = Field(..., description="Challenge creation time")
    response_received_at: Optional[datetime] = Field(None, description="Target response time")
    verification_completed_at: Optional[datetime] = Field(None, description="Verification completion time")

    # Results (if completed)
    challenge_result: Optional[str] = Field(None, description="Final challenge result")
    verification_outcome: Optional[bool] = Field(None, description="Verification success/failure")

    # Response data (if requested and available)
    target_response: Optional[Dict[str, Any]] = Field(None, description="Target node response")
    verification_details: Optional[Dict[str, Any]] = Field(None, description="Verification process details")

    model_config = {
        "json_schema_extra": {
            "example": {
                "challenge_id": "challenge_abc123456789",
                "current_status": "verification_in_progress",
                "phase": "response_verification",
                "created_at": "2024-01-15T10:30:00Z",
                "response_received_at": "2024-01-15T11:15:00Z",
                "challenge_result": "pending",
                "verification_outcome": None
            }
        }
                            }
                        }
                    }
                }
            }
        }
    }
