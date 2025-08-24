"""
Pipeline management models for GenomeVault API Gateway.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, validator

from genomevault.api.gateway.models.base import BaseModel, ProcessingStatus


class PipelineType(str, Enum):
    """Types of processing pipelines."""

    GENOMIC_ANALYSIS = "genomic_analysis"
    CLINICAL_WORKFLOW = "clinical_workflow"
    RESEARCH_PIPELINE = "research_pipeline"
    ETL_PIPELINE = "etl_pipeline"
    TRAINING_PIPELINE = "training_pipeline"
    INFERENCE_PIPELINE = "inference_pipeline"


class StepType(str, Enum):
    """Types of pipeline steps."""

    DATA_INGESTION = "data_ingestion"
    PREPROCESSING = "preprocessing"
    HYPERVECTOR_ENCODING = "hypervector_encoding"
    PRIVACY_TRANSFORMATION = "privacy_transformation"
    ANALYSIS = "analysis"
    PROOF_GENERATION = "proof_generation"
    RESULT_AGGREGATION = "result_aggregation"
    OUTPUT_DELIVERY = "output_delivery"


class PipelineStep(BaseModel):
    """Individual pipeline step configuration."""

    step_id: str = Field(..., description="Unique step identifier")
    step_name: str = Field(..., description="Human-readable step name")
    step_type: StepType = Field(..., description="Type of processing step")

    # Dependencies
    depends_on: List[str] = Field(default_factory=list, description="Step IDs this step depends on")

    # Configuration
    config: Dict[str, Any] = Field(..., description="Step-specific configuration")
    resources: Optional[Dict[str, Any]] = Field(None, description="Resource requirements")

    # Execution parameters
    timeout_seconds: int = Field(300, ge=1, description="Step timeout in seconds")
    retry_count: int = Field(3, ge=0, description="Number of retries on failure")
    parallel: bool = Field(False, description="Whether step can run in parallel")

    # Privacy settings
    privacy_level: str = Field("standard", description="Privacy level for step")
    audit_required: bool = Field(True, description="Whether step requires audit logging")


class PipelineConfig(BaseModel):
    """Pipeline configuration model."""

    pipeline_id: Optional[str] = Field(None, description="Pipeline identifier (generated if not provided)")
    name: str = Field(..., description="Pipeline name")
    description: Optional[str] = Field(None, description="Pipeline description")
    pipeline_type: PipelineType = Field(..., description="Type of pipeline")

    # Pipeline steps
    steps: List[PipelineStep] = Field(..., description="Ordered list of pipeline steps")

    # Global configuration
    global_config: Optional[Dict[str, Any]] = Field(None, description="Global pipeline configuration")
    default_resources: Optional[Dict[str, Any]] = Field(None, description="Default resource allocation")

    # Execution options
    parallel_execution: bool = Field(False, description="Enable parallel step execution")
    fail_fast: bool = Field(True, description="Stop pipeline on first failure")
    cleanup_on_failure: bool = Field(True, description="Clean up resources on failure")

    # Privacy and compliance
    privacy_requirements: Optional[Dict[str, str]] = Field(None, description="Privacy requirements")
    compliance_tags: Optional[List[str]] = Field(None, description="Compliance requirement tags")

    @field_validator("steps")
    def validate_steps(cls, v):
        """Validate pipeline steps."""
        if not v:
            raise ValueError("Pipeline must have at least one step")

        # Check for duplicate step IDs
        step_ids = [step.step_id for step in v]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("Step IDs must be unique")

        # Validate dependencies
        for step in v:
            for dep_id in step.depends_on:
                if dep_id not in step_ids:
                    raise ValueError(f"Step {step.step_id} depends on non-existent step {dep_id}")

        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "Genomic Variant Analysis Pipeline",
                "description": "Privacy-preserving genomic variant analysis with ZK proofs",
                "pipeline_type": "genomic_analysis",
                "steps": [
                    {
                        "step_id": "ingest_vcf",
                        "step_name": "Ingest VCF Data",
                        "step_type": "data_ingestion",
                        "config": {
                            "input_format": "vcf",
                            "validation": True
                        },
                        "timeout_seconds": 600
                    },
                    {
                        "step_id": "encode_variants",
                        "step_name": "Encode to Hypervectors",
                        "step_type": "hypervector_encoding",
                        "depends_on": ["ingest_vcf"],
                        "config": {
                            "dimension": 8192,
                            "encoding_type": "unified"
                        }
                    }
                ],
                "parallel_execution": True,
                "privacy_requirements": {
                    "level": "clinical",
                    "epsilon": "0.1"
                }
            }
        }


class PipelineExecution(BaseModel):
    """Pipeline execution information."""

    execution_id: str = Field(..., description="Unique execution identifier")
    pipeline_id: str = Field(..., description="Pipeline identifier")
    pipeline_name: str = Field(..., description="Pipeline name")

    # Execution status
    status: ProcessingStatus = Field(..., description="Current execution status")
    progress_percent: int = Field(..., ge=0, le=100, description="Execution progress percentage")
    current_step: Optional[str] = Field(None, description="Currently executing step")

    # Timing information
    created_at: datetime = Field(..., description="Execution creation time")
    started_at: Optional[datetime] = Field(None, description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

    # Resource usage
    resources_allocated: Optional[Dict[str, Any]] = Field(None, description="Allocated resources")
    cpu_time_ms: Optional[int] = Field(None, description="CPU time used")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")

    # Input/Output
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data references")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data references")

    # Error information
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    failed_step: Optional[str] = Field(None, description="Step that caused failure")


class PipelineResult(BaseModel):
    """Pipeline execution result."""

    execution_id: str = Field(..., description="Execution identifier")
    pipeline_id: str = Field(..., description="Pipeline identifier")
    final_status: ProcessingStatus = Field(..., description="Final execution status")

    # Results
    results: Optional[Dict[str, Any]] = Field(None, description="Pipeline results")
    output_files: Optional[List[str]] = Field(None, description="Output file references")

    # Execution summary
    total_execution_time_ms: int = Field(..., description="Total execution time")
    steps_executed: int = Field(..., description="Number of steps executed")
    steps_successful: int = Field(..., description="Number of successful steps")
    steps_failed: int = Field(..., description="Number of failed steps")

    # Privacy and audit
    privacy_report: Optional[Dict[str, Any]] = Field(None, description="Privacy compliance report")
    audit_trail: Optional[str] = Field(None, description="Audit trail hash")

    # Quality metrics
    quality_scores: Optional[Dict[str, float]] = Field(None, description="Quality assessment scores")
    validation_results: Optional[Dict[str, bool]] = Field(None, description="Validation check results")


class PipelineCreateRequest(BaseModel):
    """Request model for creating a pipeline."""

    config: PipelineConfig = Field(..., description="Pipeline configuration")
    save_template: bool = Field(False, description="Save as reusable template")
    template_name: Optional[str] = Field(None, description="Template name if saving")

    model_config = {
        "json_schema_extra": {
            "example": {
                "config": {
                    "name": "Genomic Analysis Pipeline",
                    "pipeline_type": "genomic_analysis",
                    "steps": []
                },
                "save_template": True,
                "template_name": "standard_genomic_analysis"
            }
        }


class PipelineCreateResponse(BaseModel):
    """Response model for pipeline creation."""

    pipeline_id: str = Field(..., description="Created pipeline identifier")
    template_id: Optional[str] = Field(None, description="Template ID if saved as template")
    validation_results: Dict[str, bool] = Field(..., description="Configuration validation results")
    estimated_resources: Optional[Dict[str, Any]] = Field(None, description="Estimated resource requirements")

    model_config = {
        "json_schema_extra": {
            "example": {
                "pipeline_id": "pipeline_abc123456789",
                "template_id": "template_def456789012",
                "validation_results": {
                    "config_valid": True,
                    "steps_valid": True,
                    "dependencies_valid": True
                },
                "estimated_resources": {
                    "cpu_cores": 4,
                    "memory_gb": 8,
                    "estimated_runtime_minutes": 30
                }
            }
        }


class PipelineExecuteRequest(BaseModel):
    """Request model for pipeline execution."""

    pipeline_id: str = Field(..., description="Pipeline identifier")
    input_data: Dict[str, Any] = Field(..., description="Input data for pipeline execution")

    # Execution options
    priority: int = Field(5, ge=1, le=10, description="Execution priority")
    async_execution: bool = Field(False, description="Execute asynchronously")

    # Override options
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Configuration overrides")
    resource_overrides: Optional[Dict[str, Any]] = Field(None, description="Resource requirement overrides")

    # Notification settings
    notification_webhook: Optional[str] = Field(None, description="Webhook URL for status notifications")
    email_notifications: Optional[List[str]] = Field(None, description="Email addresses for notifications")

    model_config = {
        "json_schema_extra": {
            "example": {
                "pipeline_id": "pipeline_abc123456789",
                "input_data": {
                    "vcf_file_url": "https://storage.example.com/sample.vcf",
                    "reference_genome": "GRCh38",
                    "analysis_params": {
                        "population": "EUR",
                        "quality_threshold": 30
                    }
                },
                "priority": 7,
                "async_execution": True,
                "notification_webhook": "https://api.example.com/webhooks/pipeline"
            }
        }


class PipelineExecuteResponse(BaseModel):
    """Response model for pipeline execution."""

    execution_id: str = Field(..., description="Execution identifier")
    pipeline_id: str = Field(..., description="Pipeline identifier")
    status: ProcessingStatus = Field(..., description="Initial execution status")

    # If synchronous execution
    results: Optional[PipelineResult] = Field(None, description="Results (if synchronous)")

    # If asynchronous execution
    status_url: Optional[str] = Field(None, description="URL to check execution status")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

    model_config = {
        "json_schema_extra": {
            "example": {
                "execution_id": "exec_abc123456789def",
                "pipeline_id": "pipeline_abc123456789",
                "status": "running",
                "status_url": "/pipelines/exec_abc123456789def/status",
                "estimated_completion": "2024-01-15T11:30:00Z"
            }
        }


class PipelineListRequest(BaseModel):
    """Request model for listing pipelines."""

    pipeline_type: Optional[PipelineType] = Field(None, description="Filter by pipeline type")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    status: Optional[ProcessingStatus] = Field(None, description="Filter by status")
    template_only: bool = Field(False, description="Only return template pipelines")

    # Pagination
    page: int = Field(1, ge=1, description="Page number")
    per_page: int = Field(20, ge=1, le=100, description="Items per page")


class PipelineSummary(BaseModel):
    """Pipeline summary information."""

    pipeline_id: str = Field(..., description="Pipeline identifier")
    name: str = Field(..., description="Pipeline name")
    pipeline_type: PipelineType = Field(..., description="Pipeline type")
    created_at: datetime = Field(..., description="Creation timestamp")

    # Usage statistics
    execution_count: int = Field(..., description="Number of times executed")
    success_rate: float = Field(..., ge=0, le=1, description="Success rate (0-1)")
    average_runtime_ms: Optional[int] = Field(None, description="Average runtime in milliseconds")

    # Template information
    is_template: bool = Field(..., description="Whether pipeline is a template")
    template_usage_count: Optional[int] = Field(None, description="Template usage count")


class PipelineListResponse(BaseModel):
    """Response model for pipeline listing."""

    pipelines: List[PipelineSummary] = Field(..., description="Pipeline summaries")
    total: int = Field(..., description="Total number of pipelines")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
                    }
                }
            }
        }
    }
