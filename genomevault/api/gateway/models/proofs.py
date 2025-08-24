"""
Zero-knowledge proof models for GenomeVault API Gateway.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from genomevault.api.gateway.models.base import BaseModel


class ProofType(str, Enum):
    """Types of zero-knowledge proofs."""
    
    GENOMIC = "genomic"
    CLINICAL = "clinical"
    RESEARCH = "research"
    TRAINING = "training"
    VARIANT_PRESENCE = "variant_presence"
    STATISTICAL = "statistical"


class ProofSystem(str, Enum):
    """Zero-knowledge proof systems."""
    
    GROTH16 = "groth16"
    PLONK = "plonk"
    STARK = "stark"
    BULLETPROOFS = "bulletproofs"
    NOVA = "nova"


class CircuitType(str, Enum):
    """Circuit types for proof generation."""
    
    VARIANT_FREQUENCY = "variant_frequency"
    POPULATION_STATS = "population_stats"
    CLINICAL_DECISION = "clinical_decision"
    TRAINING_ATTESTATION = "training_attestation"
    DIABETES_RISK = "diabetes_risk"
    MULTI_OMICS = "multi_omics"


class ProofGenerationRequest(BaseModel):
    """Request model for zero-knowledge proof generation."""
    
    proof_type: ProofType = Field(..., description="Type of proof to generate")
    circuit_type: CircuitType = Field(..., description="Circuit type for the proof")
    proof_system: ProofSystem = Field(ProofSystem.GROTH16, description="Proof system to use")
    
    # Inputs
    public_inputs: Dict[str, Any] = Field(..., description="Public inputs visible to verifiers")
    private_inputs_hash: str = Field(
        ...,
        pattern=r"^[a-f0-9]{64}$",
        description="SHA-256 hash of private inputs"
    )
    
    # Circuit parameters
    circuit_params: Optional[Dict[str, Any]] = Field(None, description="Circuit-specific parameters")
    
    # Generation options
    use_cached_setup: bool = Field(True, description="Use cached trusted setup if available")
    compress_proof: bool = Field(False, description="Compress proof data")
    include_public_signals: bool = Field(True, description="Include public signals in response")
    
    # Metadata
    description: Optional[str] = Field(None, description="Human-readable proof description")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    
    @field_validator("public_inputs")
    def validate_public_inputs(cls, v):
        """Validate public inputs."""
        if not v:
            raise ValueError("Public inputs cannot be empty")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": {
                "genomic_proof": {
                    "summary": "Generate genomic variant proof",
                    "value": {
                        "proof_type": "genomic",
                        "circuit_type": "variant_frequency",
                        "proof_system": "groth16",
                        "public_inputs": {
                            "variant_count": 42,
                            "population": "EUR",
                            "threshold": 0.05
                        },
                        "private_inputs_hash": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
                        "circuit_params": {
                            "precision": 32,
                            "max_variants": 1000
                        }
                    }
                },
                "clinical_proof": {
                    "summary": "Generate clinical analysis proof",
                    "value": {
                        "proof_type": "clinical",
                        "circuit_type": "clinical_decision",
                        "proof_system": "plonk",
                        "public_inputs": {
                            "risk_category": "high",
                            "analysis_type": "pharmacogenomics"
                        },
                        "private_inputs_hash": "fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321"
                    }
                }
            }
        }


class ProofGenerationResponse(BaseModel):
    """Response model for zero-knowledge proof generation."""
    
    proof_id: str = Field(..., description="Unique proof identifier")
    proof_type: ProofType = Field(..., description="Type of proof generated")
    proof_system: ProofSystem = Field(..., description="Proof system used")
    
    # Proof data
    proof_data: str = Field(..., description="Hex-encoded proof data")
    verification_key: str = Field(..., description="Verification key for proof validation")
    public_signals: Optional[List[str]] = Field(None, description="Public signals from the proof")
    
    # Metadata
    generation_time_ms: float = Field(..., description="Proof generation time in milliseconds")
    proof_size_bytes: int = Field(..., description="Proof size in bytes")
    validity_period_hours: int = Field(..., description="Proof validity period in hours")
    
    # Verification info
    verification_endpoint: str = Field(..., description="Endpoint for proof verification")
    verification_instructions: Optional[str] = Field(None, description="Verification instructions")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "proof_id": "proof_1234567890abcdef",
                "proof_type": "genomic",
                "proof_system": "groth16",
                "proof_data": "0x1a2b3c4d5e6f...",
                "verification_key": "0xabcdef123456...",
                "public_signals": ["42", "0", "1"],
                "generation_time_ms": 2534.7,
                "proof_size_bytes": 256,
                "validity_period_hours": 24,
                "verification_endpoint": "/proofs/verify"
            }
        }


class ProofVerificationRequest(BaseModel):
    """Request model for zero-knowledge proof verification."""
    
    proof_id: Optional[str] = Field(None, description="Proof ID (if verifying stored proof)")
    proof_data: Optional[str] = Field(None, description="Hex-encoded proof data")
    verification_key: Optional[str] = Field(None, description="Verification key")
    public_signals: Optional[List[str]] = Field(None, description="Public signals")
    
    # Verification options
    check_validity_period: bool = Field(True, description="Check if proof is still valid")
    verify_public_inputs: bool = Field(True, description="Verify public inputs match expectations")
    expected_public_inputs: Optional[Dict[str, Any]] = Field(None, description="Expected public inputs")
    
    @field_validator("proof_data", "verification_key", "public_signals", mode="before")
    @classmethod
    def validate_proof_components(cls, v, info):
        """Ensure proof components are provided if not using proof_id."""
        if info.data.get("proof_id") is None and v is None:
            field_name = info.field_name
            raise ValueError(f"Must provide {field_name} when not using proof_id")
        return v
    
    model_config = {
        "json_schema_extra": {
            "examples": {
                "verify_by_id": {
                    "summary": "Verify stored proof by ID",
                    "value": {
                        "proof_id": "proof_1234567890abcdef",
                        "check_validity_period": True
                    }
                },
                "verify_by_data": {
                    "summary": "Verify proof by providing data",
                    "value": {
                        "proof_data": "0x1a2b3c4d5e6f...",
                        "verification_key": "0xabcdef123456...",
                        "public_signals": ["42", "0", "1"],
                        "verify_public_inputs": True,
                        "expected_public_inputs": {
                            "variant_count": 42,
                            "population": "EUR"
                        }
                    }
                }
            }
        }


class ProofVerificationResponse(BaseModel):
    """Response model for zero-knowledge proof verification."""
    
    valid: bool = Field(..., description="Whether the proof is valid")
    proof_id: Optional[str] = Field(None, description="Proof identifier (if applicable)")
    verification_time_ms: float = Field(..., description="Verification time in milliseconds")
    
    # Verification details
    checks_performed: List[str] = Field(..., description="List of verification checks performed")
    public_inputs_valid: Optional[bool] = Field(None, description="Whether public inputs are valid")
    validity_period_check: Optional[bool] = Field(None, description="Whether proof is within validity period")
    
    # Error information (if invalid)
    error_message: Optional[str] = Field(None, description="Error message if verification failed")
    failure_details: Optional[Dict[str, str]] = Field(None, description="Detailed failure information")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "valid": True,
                "proof_id": "proof_1234567890abcdef",
                "verification_time_ms": 45.2,
                "checks_performed": [
                    "proof_verification",
                    "public_inputs_validation",
                    "validity_period_check"
                ],
                "public_inputs_valid": True,
                "validity_period_check": True
            }
        }


class ProofBatchRequest(BaseModel):
    """Request model for batch proof operations."""
    
    operation: str = Field(..., pattern=r"^(generate|verify)$", description="Batch operation type")
    requests: List[Dict[str, Any]] = Field(..., description="List of individual requests")
    parallel_execution: bool = Field(True, description="Execute requests in parallel")
    fail_fast: bool = Field(False, description="Stop on first failure")
    
    @field_validator("requests")
    def validate_requests(cls, v):
        """Validate batch requests."""
        if len(v) == 0:
            raise ValueError("Batch requests cannot be empty")
        if len(v) > 100:
            raise ValueError("Batch size cannot exceed 100 requests")
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "operation": "generate",
                "requests": [
                    {
                        "proof_type": "genomic",
                        "circuit_type": "variant_frequency",
                        "public_inputs": {"variant_count": 42},
                        "private_inputs_hash": "a1b2c3..."
                    }
                ],
                "parallel_execution": True,
                "fail_fast": False
            }
        }


class ProofBatchResponse(BaseModel):
    """Response model for batch proof operations."""
    
    total_requests: int = Field(..., description="Total number of requests processed")
    successful: int = Field(..., description="Number of successful operations")
    failed: int = Field(..., description="Number of failed operations")
    execution_time_ms: float = Field(..., description="Total execution time in milliseconds")
    
    # Results
    results: List[Dict[str, Any]] = Field(..., description="Individual operation results")
    errors: List[Dict[str, str]] = Field(..., description="Error details for failed operations")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "total_requests": 5,
                "successful": 4,
                "failed": 1,
                "execution_time_ms": 12456.8,
                "results": [
                    {
                        "index": 0,
                        "status": "success",
                        "proof_id": "proof_abc123"
                    }
                ],
                "errors": [
                    {
                        "index": 2,
                        "error": "Invalid circuit parameters"
                    }
                ]
            }
        }


class ProofListRequest(BaseModel):
    """Request model for listing proofs."""
    
    proof_type: Optional[ProofType] = Field(None, description="Filter by proof type")
    circuit_type: Optional[CircuitType] = Field(None, description="Filter by circuit type") 
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    created_before: Optional[datetime] = Field(None, description="Filter by creation date")
    valid_only: bool = Field(True, description="Only return valid (non-expired) proofs")
    
    # Pagination
    page: int = Field(1, ge=1, description="Page number")
    per_page: int = Field(20, ge=1, le=100, description="Items per page")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "proof_type": "genomic",
                "created_after": "2024-01-01T00:00:00Z",
                "valid_only": True,
                "page": 1,
                "per_page": 20
            }
        }


class ProofSummary(BaseModel):
    """Summary information for a proof."""
    
    proof_id: str = Field(..., description="Proof identifier")
    proof_type: ProofType = Field(..., description="Proof type")
    circuit_type: CircuitType = Field(..., description="Circuit type")
    created_at: datetime = Field(..., description="Proof creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Proof expiration timestamp")
    valid: bool = Field(..., description="Whether proof is currently valid")
    size_bytes: int = Field(..., description="Proof size in bytes")


class ProofListResponse(BaseModel):
    """Response model for listing proofs."""
    
    proofs: List[ProofSummary] = Field(..., description="List of proof summaries")
    total: int = Field(..., description="Total number of matching proofs")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "proofs": [
                    {
                        "proof_id": "proof_abc123",
                        "proof_type": "genomic",
                        "circuit_type": "variant_frequency",
                        "created_at": "2024-01-15T10:30:00Z",
                        "expires_at": "2024-01-16T10:30:00Z",
                        "valid": True,
                        "size_bytes": 256
                    }
                ],
                "total": 42,
                "page": 1,
                "per_page": 20,
                "total_pages": 3
            }
        }
                                }
                            }
                        }
                    }
                }
            }
        }
    }