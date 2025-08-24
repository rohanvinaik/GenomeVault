"""
Algorithm marketplace models for GenomeVault API Gateway.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from genomevault.api.gateway.models.base import BaseModel, ProcessingStatus


class AlgorithmCategory(str, Enum):
    """Algorithm categories in the marketplace."""
    
    GENOMIC_ANALYSIS = "genomic_analysis"
    CLINICAL_DECISION_SUPPORT = "clinical_decision_support"
    PRIVACY_PRESERVING = "privacy_preserving"
    MACHINE_LEARNING = "machine_learning"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    DATA_PREPROCESSING = "data_preprocessing"
    VISUALIZATION = "visualization"
    CUSTOM = "custom"


class AlgorithmType(str, Enum):
    """Types of algorithms."""
    
    EXECUTABLE = "executable"
    CONTAINER = "container"
    WASM_MODULE = "wasm_module"
    PYTHON_PACKAGE = "python_package"
    R_PACKAGE = "r_package"
    WEB_SERVICE = "web_service"


class PricingModel(str, Enum):
    """Pricing models for algorithms."""
    
    FREE = "free"
    PAY_PER_USE = "pay_per_use"
    SUBSCRIPTION = "subscription"
    FLAT_RATE = "flat_rate"
    CREDIT_BASED = "credit_based"


class ComputeRequirement(BaseModel):
    """Compute resource requirements."""
    
    cpu_cores: int = Field(..., ge=1, description="Required CPU cores")
    memory_mb: int = Field(..., ge=128, description="Required memory in MB")
    disk_space_mb: int = Field(..., ge=10, description="Required disk space in MB")
    gpu_required: bool = Field(False, description="Whether GPU is required")
    gpu_memory_mb: Optional[int] = Field(None, description="Required GPU memory in MB")
    max_runtime_seconds: Optional[int] = Field(None, description="Maximum runtime in seconds")


class AlgorithmMetadata(BaseModel):
    """Algorithm metadata information."""
    
    name: str = Field(..., description="Algorithm name")
    description: str = Field(..., description="Algorithm description")
    version: str = Field(..., description="Algorithm version")
    
    # Classification
    category: AlgorithmCategory = Field(..., description="Algorithm category")
    algorithm_type: AlgorithmType = Field(..., description="Algorithm type")
    tags: List[str] = Field(default_factory=list, description="Algorithm tags")
    
    # Technical details
    supported_inputs: List[str] = Field(..., description="Supported input formats")
    supported_outputs: List[str] = Field(..., description="Supported output formats")
    compute_requirements: ComputeRequirement = Field(..., description="Compute requirements")
    
    # Privacy and security
    privacy_preserving: bool = Field(False, description="Whether algorithm preserves privacy")
    security_level: str = Field("standard", description="Security level (standard, high, maximum)")
    audit_compliant: bool = Field(False, description="Whether algorithm is audit compliant")
    
    # Author and licensing
    author: str = Field(..., description="Algorithm author")
    author_organization: Optional[str] = Field(None, description="Author organization")
    license: str = Field(..., description="Algorithm license")
    source_code_url: Optional[str] = Field(None, description="Source code repository URL")
    
    # Documentation
    documentation_url: Optional[str] = Field(None, description="Documentation URL")
    example_usage: Optional[str] = Field(None, description="Example usage")
    changelog: Optional[str] = Field(None, description="Version changelog")


class AlgorithmPricing(BaseModel):
    """Algorithm pricing information."""
    
    pricing_model: PricingModel = Field(..., description="Pricing model")
    
    # Pricing details
    base_price: Optional[float] = Field(None, description="Base price (currency units)")
    price_per_execution: Optional[float] = Field(None, description="Price per execution")
    price_per_compute_unit: Optional[float] = Field(None, description="Price per compute unit")
    
    # Subscription pricing
    monthly_price: Optional[float] = Field(None, description="Monthly subscription price")
    annual_price: Optional[float] = Field(None, description="Annual subscription price")
    
    # Credit-based pricing
    credits_per_execution: Optional[int] = Field(None, description="Credits required per execution")
    
    # Discounts and limits
    bulk_discount_threshold: Optional[int] = Field(None, description="Executions needed for bulk discount")
    bulk_discount_percent: Optional[float] = Field(None, description="Bulk discount percentage")
    free_tier_executions: Optional[int] = Field(None, description="Free tier execution limit")
    
    # Currency and billing
    currency: str = Field("USD", description="Currency code")
    billing_cycle: Optional[str] = Field(None, description="Billing cycle (monthly, annual, etc.)")


class AlgorithmListRequest(BaseModel):
    """Request model for listing algorithms."""
    
    # Filtering
    category: Optional[AlgorithmCategory] = Field(None, description="Filter by category")
    algorithm_type: Optional[AlgorithmType] = Field(None, description="Filter by type")
    pricing_model: Optional[PricingModel] = Field(None, description="Filter by pricing model")
    privacy_preserving: Optional[bool] = Field(None, description="Filter by privacy preservation")
    
    # Search
    search_term: Optional[str] = Field(None, description="Search in names, descriptions, tags")
    tags: Optional[List[str]] = Field(None, description="Filter by tags (any match)")
    
    # Requirements matching
    max_cpu_cores: Optional[int] = Field(None, description="Maximum CPU cores available")
    max_memory_mb: Optional[int] = Field(None, description="Maximum memory available")
    gpu_available: Optional[bool] = Field(None, description="Whether GPU is available")
    
    # Sorting
    sort_by: str = Field(
        "popularity",
        pattern=r"^(name|created_at|updated_at|popularity|rating|price)$",
        description="Sort field"
    )
    sort_order: str = Field(
        "desc",
        pattern=r"^(asc|desc)$",
        description="Sort order"
    )
    
    # Pagination
    page: int = Field(1, ge=1, description="Page number")
    per_page: int = Field(20, ge=1, le=100, description="Items per page")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "category": "genomic_analysis",
                "privacy_preserving": True,
                "search_term": "variant analysis",
                "tags": ["GWAS", "population genetics"],
                "max_cpu_cores": 8,
                "max_memory_mb": 16384,
                "sort_by": "rating",
                "sort_order": "desc",
                "page": 1,
                "per_page": 20
            }
        }
    }


class AlgorithmSummary(BaseModel):
    """Algorithm summary for listing."""
    
    algorithm_id: str = Field(..., description="Algorithm identifier")
    name: str = Field(..., description="Algorithm name")
    description: str = Field(..., description="Short description")
    category: AlgorithmCategory = Field(..., description="Algorithm category")
    algorithm_type: AlgorithmType = Field(..., description="Algorithm type")
    version: str = Field(..., description="Current version")
    
    # Popularity and ratings
    rating: float = Field(..., ge=0, le=5, description="Average rating (0-5 stars)")
    rating_count: int = Field(..., description="Number of ratings")
    usage_count: int = Field(..., description="Number of times used")
    
    # Pricing summary
    pricing_model: PricingModel = Field(..., description="Pricing model")
    starting_price: Optional[float] = Field(None, description="Starting price")
    
    # Technical summary
    privacy_preserving: bool = Field(..., description="Privacy preserving")
    compute_requirements: ComputeRequirement = Field(..., description="Compute requirements")
    
    # Metadata
    author: str = Field(..., description="Algorithm author")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    # Featured status
    featured: bool = Field(False, description="Whether algorithm is featured")
    verified: bool = Field(False, description="Whether algorithm is verified")


class AlgorithmResponse(BaseModel):
    """Detailed algorithm information."""
    
    algorithm_id: str = Field(..., description="Algorithm identifier")
    metadata: AlgorithmMetadata = Field(..., description="Algorithm metadata")
    pricing: AlgorithmPricing = Field(..., description="Pricing information")
    
    # Popularity and ratings
    rating: float = Field(..., ge=0, le=5, description="Average rating")
    rating_count: int = Field(..., description="Number of ratings")
    usage_count: int = Field(..., description="Total usage count")
    
    # Reviews summary
    recent_reviews: Optional[List[Dict[str, Any]]] = Field(None, description="Recent reviews")
    
    # Technical details
    execution_examples: Optional[List[Dict[str, Any]]] = Field(None, description="Execution examples")
    performance_benchmarks: Optional[Dict[str, Any]] = Field(None, description="Performance benchmarks")
    
    # Status and availability
    status: str = Field(..., description="Algorithm status (active, deprecated, etc.)")
    availability_regions: List[str] = Field(..., description="Available regions")
    
    # Publisher information
    publisher_verified: bool = Field(..., description="Whether publisher is verified")
    last_audit_date: Optional[datetime] = Field(None, description="Last security audit date")
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "algorithm_id": "algo_abc123456789",
                "metadata": {
                    "name": "Genomic Variant Analyzer",
                    "description": "Privacy-preserving genomic variant analysis with ZK proofs",
                    "version": "2.1.0",
                    "category": "genomic_analysis",
                    "algorithm_type": "container",
                    "supported_inputs": ["vcf", "json"],
                    "supported_outputs": ["json", "csv"],
                    "privacy_preserving": True,
                    "author": "GenomeVault Research Team"
                },
                "pricing": {
                    "pricing_model": "credit_based",
                    "credits_per_execution": 10,
                    "free_tier_executions": 100
                },
                "rating": 4.7,
                "rating_count": 156,
                "usage_count": 2534,
                "status": "active"
            }
        }
    }


class AlgorithmExecutionRequest(BaseModel):
    """Request model for algorithm execution."""
    
    algorithm_id: str = Field(..., description="Algorithm identifier")
    
    # Input data
    input_data: Dict[str, Any] = Field(..., description="Input data for algorithm")
    input_format: Optional[str] = Field(None, description="Input data format")
    
    # Execution parameters
    parameters: Optional[Dict[str, Any]] = Field(None, description="Algorithm parameters")
    
    # Execution options
    priority: int = Field(5, ge=1, le=10, description="Execution priority")
    async_execution: bool = Field(False, description="Execute asynchronously")
    timeout_seconds: Optional[int] = Field(None, description="Execution timeout")
    
    # Resource allocation
    cpu_cores: Optional[int] = Field(None, description="CPU cores to allocate")
    memory_mb: Optional[int] = Field(None, description="Memory to allocate")
    use_gpu: bool = Field(False, description="Whether to use GPU")
    
    # Output options
    output_format: Optional[str] = Field(None, description="Desired output format")
    include_logs: bool = Field(False, description="Include execution logs in response")
    include_metrics: bool = Field(False, description="Include performance metrics")
    
    # Privacy options
    privacy_preserving: bool = Field(True, description="Use privacy-preserving execution")
    differential_privacy: bool = Field(False, description="Apply differential privacy")
    
    # Notification
    callback_url: Optional[str] = Field(None, description="Callback URL for completion notification")
    
    @field_validator("input_data")
    def validate_input_data(cls, v):
        """Validate input data is not empty."""
        if not v:
            raise ValueError("Input data cannot be empty")
        return v
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "algorithm_id": "algo_abc123456789",
                "input_data": {
                    "vcf_file_url": "https://storage.example.com/variants.vcf",
                    "reference_genome": "GRCh38",
                    "population": "EUR"
                },
                "parameters": {
                    "quality_threshold": 30,
                    "allele_frequency_cutoff": 0.01,
                    "include_synonymous": False
                },
                "priority": 7,
                "async_execution": True,
                "output_format": "json",
                "include_metrics": True,
                "privacy_preserving": True
            }
        }


class AlgorithmExecutionResponse(BaseModel):
    """Response model for algorithm execution."""
    
    execution_id: str = Field(..., description="Execution identifier")
    algorithm_id: str = Field(..., description="Algorithm identifier")
    status: ProcessingStatus = Field(..., description="Execution status")
    
    # Results (if completed synchronously)
    results: Optional[Dict[str, Any]] = Field(None, description="Execution results")
    output_files: Optional[List[str]] = Field(None, description="Output file URLs")
    
    # Execution details
    started_at: datetime = Field(..., description="Execution start time")
    completed_at: Optional[datetime] = Field(None, description="Execution completion time")
    execution_time_ms: Optional[int] = Field(None, description="Execution time in milliseconds")
    
    # Resource usage
    resources_used: Optional[Dict[str, Any]] = Field(None, description="Resources used")
    credits_consumed: Optional[int] = Field(None, description="Credits consumed")
    cost: Optional[float] = Field(None, description="Execution cost")
    
    # Monitoring (for async execution)
    status_url: Optional[str] = Field(None, description="URL to check execution status")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    
    # Logs and metrics (if requested)
    execution_logs: Optional[List[str]] = Field(None, description="Execution logs")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Performance metrics")
    
    # Privacy information
    privacy_guarantees: Optional[Dict[str, str]] = Field(None, description="Privacy guarantees provided")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "execution_id": "exec_def456789012",
                "algorithm_id": "algo_abc123456789",
                "status": "running",
                "started_at": "2024-01-15T10:30:00Z",
                "status_url": "/algorithms/exec_def456789012/status",
                "estimated_completion": "2024-01-15T10:45:00Z",
                "credits_consumed": 10
            }
        }


class ExecutionStatusResponse(BaseModel):
    """Response model for execution status."""
    
    execution_id: str = Field(..., description="Execution identifier")
    algorithm_id: str = Field(..., description="Algorithm identifier")
    status: ProcessingStatus = Field(..., description="Current execution status")
    progress_percent: int = Field(..., ge=0, le=100, description="Execution progress")
    
    # Timing
    started_at: datetime = Field(..., description="Start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    
    # Resource usage
    current_resource_usage: Optional[Dict[str, float]] = Field(None, description="Current resource usage")
    total_resources_used: Optional[Dict[str, float]] = Field(None, description="Total resources used")
    
    # Results (if completed)
    results: Optional[Dict[str, Any]] = Field(None, description="Execution results")
    
    # Error information (if failed)
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    error_details: Optional[Dict[str, str]] = Field(None, description="Detailed error information")