"""Analysis models for GenomeVault API."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class AnalysisType(str, Enum):
    """Types of genomic analysis supported."""

    WHOLE_GENOME = "whole_genome"
    EXOME = "exome"
    TARGETED_PANEL = "targeted_panel"
    PHARMACOGENOMICS = "pharmacogenomics"
    ANCESTRY = "ancestry"
    RISK_ASSESSMENT = "risk_assessment"
    CARRIER_SCREENING = "carrier_screening"
    VARIANT_PATHOGENICITY = "variant_pathogenicity"


class FileFormat(str, Enum):
    """Supported input formats."""

    VCF = "vcf"
    VCF_GZ = "vcf.gz"
    FASTQ = "fastq"
    FASTQ_GZ = "fastq.gz"
    BAM = "bam"
    SAM = "sam"


class GenomeAnalysisRequest(BaseModel):
    """Request for genomic analysis."""

    analysis_type: AnalysisType
    file_format: FileFormat

    # File references (either file_path for uploaded files or file_content)
    file_path: Optional[Path] = None
    file_path_r2: Optional[Path] = None  # For paired-end FASTQ

    # Reference genome settings
    reference_genome: str = Field(default="GRCh38", description="Reference genome assembly")

    # Differential encoding settings
    k_anonymity: int = Field(default=3, ge=2, description="k-anonymity level")

    # HDC settings
    dimension: int = Field(default=10000, ge=1024, le=100000)
    binary_encoding: bool = Field(default=False)

    # Privacy settings
    enable_zk_proof: bool = Field(default=True, description="Generate ZK proof")
    enable_blockchain: bool = Field(default=False, description="Record on blockchain")
    zk_backend: str = Field(default="halo2", description="ZK backend: halo2, groth16, plonk")

    # PIR settings (optional)
    enable_pir: bool = Field(default=False)
    pir_database: Optional[str] = None
    pir_query_type: Optional[str] = None  # "cpir" or "it-pir"

    # Analysis-specific parameters
    analysis_params: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    patient_id_hash: Optional[str] = None  # SHA-256 hash for clinical use
    consent_hash: Optional[str] = None

    @field_validator('patient_id_hash', 'consent_hash')
    @classmethod
    def validate_hash(cls, v: Optional[str]) -> Optional[str]:
        """Validate SHA-256 hash format."""
        if v and not (len(v) == 64 and all(c in '0123456789abcdef' for c in v)):
            raise ValueError("Must be SHA-256 hash (64 hex chars)")
        return v


class AnalysisStageResult(BaseModel):
    """Result from a single pipeline stage."""

    stage_name: str
    success: bool
    duration_ms: float
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class GenomeAnalysisResponse(BaseModel):
    """Response from genomic analysis."""

    analysis_id: str  # Unique identifier
    status: str  # "success", "partial_success", "failed"

    # Pipeline results
    stages: List[AnalysisStageResult]

    # Key metrics
    total_duration_ms: float
    compression_ratio: Optional[float] = None
    variants_analyzed: Optional[int] = None

    # Encoded results
    hypervector_id: Optional[str] = None  # Reference to stored hypervector
    hypervector_dimension: Optional[int] = None

    # Privacy proofs
    zk_proof_id: Optional[str] = None
    zk_verification_status: Optional[bool] = None

    # PIR results
    pir_query_result: Optional[Dict[str, Any]] = None

    # Blockchain attestation
    blockchain_tx_hash: Optional[str] = None
    attestation_id: Optional[str] = None

    # Analysis-specific results
    analysis_results: Dict[str, Any] = Field(default_factory=dict)

    # Warnings and recommendations
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class AnalysisStatus(BaseModel):
    """Status of an ongoing analysis."""

    analysis_id: str
    status: str  # "queued", "processing", "completed", "failed"
    progress_percent: float
    current_stage: Optional[str] = None
    estimated_completion_seconds: Optional[float] = None
