"""
Zero-Knowledge Proof router for genomic computations.

This module provides REST API endpoints for generating and verifying
zero-knowledge proofs using the RealZKEngine.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, DateTime, Integer, String, Text, Float, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.dialects.postgresql import UUID
import uuid

from genomevault.zk.real_engine import RealZKEngine, RealProof
from genomevault.zk.models import CircuitType

logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()

# Get database URL from environment or use default
import os
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://genomevault:genomevault@localhost:5432/genomevault"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

router = APIRouter(
    prefix="/api/zk",
    tags=["ZK", "Zero-Knowledge Proofs"],
    responses={404: {"description": "Not found"}},
)


# Initialize ZK Engine (singleton)
_zk_engine = None

def get_zk_engine() -> RealZKEngine:
    """Get or create the ZK engine singleton."""
    global _zk_engine
    if _zk_engine is None:
        _zk_engine = RealZKEngine()
        logger.info("ZK Engine initialized")
    return _zk_engine


# Database Models
class ZKProofRecord(Base):
    """Database model for storing ZK proof records."""
    
    __tablename__ = "zk_proofs"
    
    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    proof_id = Column(String(64), unique=True, index=True, nullable=False)
    circuit_name = Column(String(128), nullable=False, index=True)
    circuit_type = Column(String(64), nullable=True)
    proof_data = Column(Text, nullable=False)  # JSON string
    public_inputs = Column(Text, nullable=False)  # JSON string
    private_inputs_hash = Column(String(64), nullable=True)  # SHA256 hash
    verified = Column(Boolean, default=False)
    verification_count = Column(Integer, default=0)
    generation_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(Text, nullable=True)  # JSON string


# Pydantic Models for Request/Response
class CircuitInput(BaseModel):
    """Input data for a circuit."""
    
    name: str = Field(..., description="Input parameter name")
    value: Any = Field(..., description="Input value")
    is_public: bool = Field(True, description="Whether this is a public input")


class ProveRequest(BaseModel):
    """Request model for generating a proof."""
    
    circuit_name: str = Field(..., description="Name of the circuit to use")
    inputs: List[CircuitInput] = Field(..., min_items=1, description="Circuit inputs")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    store_proof: bool = Field(True, description="Whether to store the proof in database")
    
    @validator('circuit_name')
    def validate_circuit_name(cls, v):
        """Validate circuit name."""
        if not v or not v.strip():
            raise ValueError("Circuit name cannot be empty")
        return v.strip()
    
    def get_public_inputs(self) -> Dict[str, Any]:
        """Extract public inputs."""
        return {inp.name: inp.value for inp in self.inputs if inp.is_public}
    
    def get_private_inputs(self) -> Dict[str, Any]:
        """Extract private inputs."""
        return {inp.name: inp.value for inp in self.inputs if not inp.is_public}


class ProveResponse(BaseModel):
    """Response model for proof generation."""
    
    proof_id: str = Field(..., description="Unique ID for the proof")
    circuit_name: str = Field(..., description="Circuit used")
    proof: Dict[str, Any] = Field(..., description="The generated proof")
    public_inputs: Dict[str, Any] = Field(..., description="Public inputs used")
    generation_time_ms: float = Field(..., description="Proof generation time in milliseconds")
    stored: bool = Field(..., description="Whether proof was stored in database")
    created_at: datetime = Field(..., description="Timestamp of creation")


class VerifyRequest(BaseModel):
    """Request model for verifying a proof."""
    
    proof: Dict[str, Any] = Field(..., description="The proof to verify")
    public_inputs: Dict[str, Any] = Field(..., description="Public inputs for verification")
    circuit_name: Optional[str] = Field(None, description="Circuit name (if known)")
    proof_id: Optional[str] = Field(None, description="Proof ID (if stored)")


class VerifyResponse(BaseModel):
    """Response model for proof verification."""
    
    valid: bool = Field(..., description="Whether the proof is valid")
    circuit_name: Optional[str] = Field(None, description="Circuit used (if known)")
    verification_time_ms: float = Field(..., description="Verification time in milliseconds")
    proof_id: Optional[str] = Field(None, description="Proof ID (if provided)")
    message: str = Field(..., description="Verification result message")


class CircuitInfo(BaseModel):
    """Information about an available circuit."""
    
    name: str = Field(..., description="Circuit name")
    type: Optional[str] = Field(None, description="Circuit type/category")
    description: str = Field(..., description="Circuit description")
    required_inputs: List[Dict[str, Any]] = Field(..., description="Required inputs specification")
    constraints: Optional[int] = Field(None, description="Number of constraints")
    supported: bool = Field(True, description="Whether circuit is currently supported")


class ProofInfo(BaseModel):
    """Information about a stored proof."""
    
    proof_id: str
    circuit_name: str
    verified: bool
    verification_count: int
    generation_time_ms: Optional[float]
    created_at: datetime
    metadata: Optional[Dict[str, Any]]


# Available circuits configuration
AVAILABLE_CIRCUITS = {
    "sum64": {
        "type": "arithmetic",
        "description": "Proves knowledge of two numbers that sum to a public value",
        "required_inputs": [
            {"name": "a", "type": "integer", "visibility": "private", "description": "First addend"},
            {"name": "b", "type": "integer", "visibility": "private", "description": "Second addend"},
            {"name": "c", "type": "integer", "visibility": "public", "description": "Sum (a + b)"}
        ],
        "constraints": 65,
        "supported": True
    },
    "variant_proof": {
        "type": "genomic",
        "description": "Proves possession of a genomic variant without revealing the full genome",
        "required_inputs": [
            {"name": "genome_hash", "type": "string", "visibility": "public", "description": "Hash of the genome"},
            {"name": "variant_position", "type": "integer", "visibility": "public", "description": "Position of variant"},
            {"name": "variant_allele", "type": "string", "visibility": "public", "description": "Variant allele"},
            {"name": "genome_data", "type": "bytes", "visibility": "private", "description": "Full genome data"},
            {"name": "salt", "type": "string", "visibility": "private", "description": "Random salt"}
        ],
        "constraints": 10000,
        "supported": True
    },
    "polygenic_risk_score": {
        "type": "genomic",
        "description": "Proves a polygenic risk score is above/below threshold without revealing genotypes",
        "required_inputs": [
            {"name": "score_threshold", "type": "float", "visibility": "public", "description": "Risk score threshold"},
            {"name": "score_comparison", "type": "string", "visibility": "public", "description": "Comparison operator (>, <, >=, <=)"},
            {"name": "genotypes", "type": "array", "visibility": "private", "description": "Individual genotypes"},
            {"name": "weights", "type": "array", "visibility": "private", "description": "SNP weights"}
        ],
        "constraints": 50000,
        "supported": True
    },
    "ancestry_proof": {
        "type": "genomic",
        "description": "Proves ancestry composition within ranges without revealing exact percentages",
        "required_inputs": [
            {"name": "ancestry_ranges", "type": "object", "visibility": "public", "description": "Acceptable ancestry ranges"},
            {"name": "ancestry_composition", "type": "object", "visibility": "private", "description": "Actual ancestry percentages"}
        ],
        "constraints": 25000,
        "supported": True
    },
    "clinical_criteria": {
        "type": "clinical",
        "description": "Proves clinical trial eligibility without revealing medical history",
        "required_inputs": [
            {"name": "criteria_hash", "type": "string", "visibility": "public", "description": "Hash of eligibility criteria"},
            {"name": "eligible", "type": "boolean", "visibility": "public", "description": "Eligibility status"},
            {"name": "medical_records", "type": "object", "visibility": "private", "description": "Medical history data"},
            {"name": "genomic_data", "type": "object", "visibility": "private", "description": "Genomic test results"}
        ],
        "constraints": 100000,
        "supported": True
    },
    "computation_result": {
        "type": "generic",
        "description": "Proves correct execution of a computation without revealing inputs",
        "required_inputs": [
            {"name": "result", "type": "any", "visibility": "public", "description": "Computation result"},
            {"name": "computation_hash", "type": "string", "visibility": "public", "description": "Hash of computation"},
            {"name": "inputs", "type": "array", "visibility": "private", "description": "Computation inputs"}
        ],
        "constraints": 5000,
        "supported": True
    }
}


# Dependency to get database session
def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Helper functions
def store_proof_record(
    db: Session,
    proof_id: str,
    circuit_name: str,
    proof_data: Dict[str, Any],
    public_inputs: Dict[str, Any],
    generation_time_ms: float,
    metadata: Optional[Dict[str, Any]] = None
) -> ZKProofRecord:
    """Store a proof record in the database."""
    import hashlib
    import json
    
    # Create private inputs hash if available
    private_hash = None
    if metadata and "private_inputs_hash" in metadata:
        private_hash = metadata["private_inputs_hash"]
    
    record = ZKProofRecord(
        proof_id=proof_id,
        circuit_name=circuit_name,
        circuit_type=AVAILABLE_CIRCUITS.get(circuit_name, {}).get("type"),
        proof_data=json.dumps(proof_data),
        public_inputs=json.dumps(public_inputs),
        private_inputs_hash=private_hash,
        generation_time_ms=generation_time_ms,
        metadata=json.dumps(metadata) if metadata else None
    )
    
    db.add(record)
    db.commit()
    db.refresh(record)
    
    return record


# API Endpoints
@router.post("/prove", response_model=ProveResponse)
async def generate_proof(
    request: ProveRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    zk_engine: RealZKEngine = Depends(get_zk_engine)
):
    """
    Generate a zero-knowledge proof.
    
    This endpoint accepts a circuit name and inputs, generates a ZK proof
    using the RealZKEngine, and optionally stores it in the database.
    
    The proof can later be verified using the /verify endpoint.
    """
    try:
        logger.info(f"Generating proof for circuit: {request.circuit_name}")
        
        # Check if circuit is available
        if request.circuit_name not in AVAILABLE_CIRCUITS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown circuit: {request.circuit_name}. Use /api/zk/circuits to list available circuits."
            )
        
        circuit_info = AVAILABLE_CIRCUITS[request.circuit_name]
        if not circuit_info.get("supported", False):
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail=f"Circuit {request.circuit_name} is not currently supported"
            )
        
        # Extract inputs
        public_inputs = request.get_public_inputs()
        private_inputs = request.get_private_inputs()
        
        # Generate proof
        start_time = time.perf_counter()
        
        try:
            # Map circuit name to CircuitType if needed
            circuit_type = None
            if request.circuit_name == "sum64":
                circuit_type = CircuitType.SUM64
            
            # Generate proof using RealZKEngine
            proof = zk_engine.prove(
                circuit_type=circuit_type or request.circuit_name,
                public_inputs=public_inputs,
                private_inputs=private_inputs
            )
            
            generation_time_ms = (time.perf_counter() - start_time) * 1000
            
            # Convert proof to wire format
            if isinstance(proof, RealProof):
                proof_data = proof.to_wire()
            else:
                proof_data = {"proof": proof, "public_inputs": public_inputs}
            
        except Exception as e:
            logger.error(f"Proof generation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate proof: {str(e)}"
            )
        
        # Generate proof ID
        proof_id = f"zk_{uuid.uuid4().hex[:12]}"
        
        # Store proof if requested
        stored = False
        created_at = datetime.utcnow()
        
        if request.store_proof:
            try:
                # Calculate private inputs hash for record
                import hashlib
                private_hash = None
                if private_inputs:
                    private_str = json.dumps(private_inputs, sort_keys=True)
                    private_hash = hashlib.sha256(private_str.encode()).hexdigest()
                
                metadata = request.metadata or {}
                if private_hash:
                    metadata["private_inputs_hash"] = private_hash
                
                record = store_proof_record(
                    db=db,
                    proof_id=proof_id,
                    circuit_name=request.circuit_name,
                    proof_data=proof_data,
                    public_inputs=public_inputs,
                    generation_time_ms=generation_time_ms,
                    metadata=metadata
                )
                stored = True
                created_at = record.created_at
                
                logger.info(f"Proof {proof_id} stored in database")
                
            except Exception as e:
                logger.error(f"Failed to store proof: {e}")
                # Continue even if storage fails
        
        logger.info(f"Successfully generated proof {proof_id} in {generation_time_ms:.2f}ms")
        
        return ProveResponse(
            proof_id=proof_id,
            circuit_name=request.circuit_name,
            proof=proof_data.get("proof", proof_data),
            public_inputs=public_inputs,
            generation_time_ms=generation_time_ms,
            stored=stored,
            created_at=created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in generate_proof: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate proof: {str(e)}"
        )


@router.post("/verify", response_model=VerifyResponse)
async def verify_proof(
    request: VerifyRequest,
    db: Session = Depends(get_db),
    zk_engine: RealZKEngine = Depends(get_zk_engine)
):
    """
    Verify a zero-knowledge proof.
    
    This endpoint accepts a proof and public inputs, verifies the proof
    using the RealZKEngine, and returns whether the proof is valid.
    
    If a proof_id is provided, it will also update the verification count
    in the database.
    """
    try:
        logger.info(f"Verifying proof{' ' + request.proof_id if request.proof_id else ''}")
        
        # If proof_id is provided, try to get circuit name from database
        circuit_name = request.circuit_name
        if request.proof_id and not circuit_name:
            record = db.query(ZKProofRecord).filter(
                ZKProofRecord.proof_id == request.proof_id
            ).first()
            if record:
                circuit_name = record.circuit_name
        
        # Verify proof
        start_time = time.perf_counter()
        
        try:
            # Map circuit name to CircuitType if needed
            circuit_type = None
            if circuit_name == "sum64":
                circuit_type = CircuitType.SUM64
            
            # Create RealProof object if needed
            if not isinstance(request.proof, RealProof):
                proof_obj = RealProof(
                    proof=request.proof,
                    public=request.public_inputs
                )
            else:
                proof_obj = request.proof
            
            # Verify using RealZKEngine
            is_valid = zk_engine.verify(
                circuit_type=circuit_type or circuit_name,
                proof=proof_obj,
                public_inputs=request.public_inputs
            )
            
            verification_time_ms = (time.perf_counter() - start_time) * 1000
            
        except Exception as e:
            logger.error(f"Proof verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to verify proof: {str(e)}"
            )
        
        # Update verification count if proof_id provided
        if request.proof_id:
            try:
                record = db.query(ZKProofRecord).filter(
                    ZKProofRecord.proof_id == request.proof_id
                ).first()
                if record:
                    record.verified = is_valid
                    record.verification_count += 1
                    db.commit()
                    logger.info(f"Updated verification count for proof {request.proof_id}")
            except Exception as e:
                logger.error(f"Failed to update verification count: {e}")
                # Continue even if update fails
        
        message = "Proof is valid" if is_valid else "Proof is invalid"
        logger.info(f"Verification complete in {verification_time_ms:.2f}ms: {message}")
        
        return VerifyResponse(
            valid=is_valid,
            circuit_name=circuit_name,
            verification_time_ms=verification_time_ms,
            proof_id=request.proof_id,
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in verify_proof: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to verify proof: {str(e)}"
        )


@router.get("/circuits", response_model=List[CircuitInfo])
async def list_circuits():
    """
    List available ZK circuits.
    
    This endpoint returns information about all available circuits,
    including their required inputs and constraints.
    """
    try:
        circuits = []
        
        for name, info in AVAILABLE_CIRCUITS.items():
            circuits.append(CircuitInfo(
                name=name,
                type=info.get("type"),
                description=info.get("description", ""),
                required_inputs=info.get("required_inputs", []),
                constraints=info.get("constraints"),
                supported=info.get("supported", True)
            ))
        
        # Sort by type and name
        circuits.sort(key=lambda x: (x.type or "", x.name))
        
        logger.info(f"Listed {len(circuits)} available circuits")
        return circuits
        
    except Exception as e:
        logger.error(f"Error in list_circuits: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list circuits: {str(e)}"
        )


@router.get("/circuits/{circuit_name}", response_model=CircuitInfo)
async def get_circuit_info(circuit_name: str):
    """
    Get detailed information about a specific circuit.
    
    Returns detailed information including required inputs,
    constraints, and usage examples.
    """
    try:
        if circuit_name not in AVAILABLE_CIRCUITS:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Circuit {circuit_name} not found"
            )
        
        info = AVAILABLE_CIRCUITS[circuit_name]
        
        return CircuitInfo(
            name=circuit_name,
            type=info.get("type"),
            description=info.get("description", ""),
            required_inputs=info.get("required_inputs", []),
            constraints=info.get("constraints"),
            supported=info.get("supported", True)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_circuit_info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get circuit info: {str(e)}"
        )


@router.get("/proofs/{proof_id}", response_model=ProofInfo)
async def get_proof_info(
    proof_id: str,
    db: Session = Depends(get_db)
):
    """
    Get information about a stored proof.
    
    Returns metadata about a previously generated proof,
    including verification status and generation time.
    """
    try:
        record = db.query(ZKProofRecord).filter(
            ZKProofRecord.proof_id == proof_id
        ).first()
        
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Proof {proof_id} not found"
            )
        
        # Parse metadata if present
        metadata = None
        if record.metadata:
            try:
                metadata = json.loads(record.metadata)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse metadata for proof {proof_id}")
        
        return ProofInfo(
            proof_id=record.proof_id,
            circuit_name=record.circuit_name,
            verified=record.verified,
            verification_count=record.verification_count,
            generation_time_ms=record.generation_time_ms,
            created_at=record.created_at,
            metadata=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_proof_info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get proof info: {str(e)}"
        )


@router.post("/batch-verify")
async def batch_verify_proofs(
    proofs: List[VerifyRequest],
    db: Session = Depends(get_db),
    zk_engine: RealZKEngine = Depends(get_zk_engine)
):
    """
    Verify multiple proofs in batch.
    
    This endpoint efficiently verifies multiple proofs and returns
    the verification results for each.
    """
    try:
        if not proofs:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No proofs provided for verification"
            )
        
        if len(proofs) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 100 proofs can be verified in a single batch"
            )
        
        logger.info(f"Batch verifying {len(proofs)} proofs")
        
        results = []
        total_time = 0
        
        for i, proof_request in enumerate(proofs):
            start_time = time.perf_counter()
            
            try:
                # Get circuit name if proof_id provided
                circuit_name = proof_request.circuit_name
                if proof_request.proof_id and not circuit_name:
                    record = db.query(ZKProofRecord).filter(
                        ZKProofRecord.proof_id == proof_request.proof_id
                    ).first()
                    if record:
                        circuit_name = record.circuit_name
                
                # Map circuit name to CircuitType if needed
                circuit_type = None
                if circuit_name == "sum64":
                    circuit_type = CircuitType.SUM64
                
                # Create RealProof object
                proof_obj = RealProof(
                    proof=proof_request.proof,
                    public=proof_request.public_inputs
                )
                
                # Verify
                is_valid = zk_engine.verify(
                    circuit_type=circuit_type or circuit_name,
                    proof=proof_obj,
                    public_inputs=proof_request.public_inputs
                )
                
                verification_time = (time.perf_counter() - start_time) * 1000
                total_time += verification_time
                
                results.append({
                    "index": i,
                    "proof_id": proof_request.proof_id,
                    "valid": is_valid,
                    "verification_time_ms": verification_time,
                    "circuit_name": circuit_name
                })
                
            except Exception as e:
                logger.error(f"Failed to verify proof {i}: {e}")
                results.append({
                    "index": i,
                    "proof_id": proof_request.proof_id,
                    "valid": False,
                    "error": str(e)
                })
        
        # Summary statistics
        valid_count = sum(1 for r in results if r.get("valid", False))
        invalid_count = len(results) - valid_count
        
        return JSONResponse(
            content={
                "total_proofs": len(proofs),
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "total_time_ms": total_time,
                "average_time_ms": total_time / len(proofs) if proofs else 0,
                "results": results
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in batch_verify_proofs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to batch verify proofs: {str(e)}"
        )


# Create database tables on module import
try:
    Base.metadata.create_all(bind=engine)
    logger.info("ZK proof database tables created successfully")
except Exception as e:
    logger.error(f"Failed to create ZK proof database tables: {e}")