"""
Federated learning model management models for GenomeVault API Gateway.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, validator

from genomevault.api.gateway.models.base import BaseModel, ProcessingStatus


class ModelType(str, Enum):
    """Types of machine learning models."""

    NEURAL_NETWORK = "neural_network"
    LINEAR_MODEL = "linear_model"
    ENSEMBLE = "ensemble"
    KAN_MODEL = "kan_model"
    TRANSFORMER = "transformer"
    CUSTOM = "custom"


class ModelFramework(str, Enum):
    """Machine learning frameworks."""

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    SCIKIT_LEARN = "scikit_learn"
    CUSTOM_FRAMEWORK = "custom"


class TrainingStatus(str, Enum):
    """Training status values."""

    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AggregationMethod(str, Enum):
    """Federated learning aggregation methods."""

    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    FEDOPT = "fedopt"
    CUSTOM = "custom"


class ModelMetadata(BaseModel):
    """Model metadata information."""

    model_name: str = Field(..., description="Human-readable model name")
    description: Optional[str] = Field(None, description="Model description")
    version: str = Field(..., description="Model version")

    # Technical specifications
    model_type: ModelType = Field(..., description="Type of model")
    framework: ModelFramework = Field(..., description="ML framework used")
    input_shape: Optional[List[int]] = Field(None, description="Expected input shape")
    output_shape: Optional[List[int]] = Field(None, description="Model output shape")

    # Training information
    training_dataset_info: Optional[Dict[str, Any]] = Field(None, description="Training dataset information")
    hyperparameters: Optional[Dict[str, Any]] = Field(None, description="Model hyperparameters")

    # Privacy information
    privacy_guarantees: Optional[Dict[str, str]] = Field(None, description="Privacy guarantees provided")
    differential_privacy_epsilon: Optional[float] = Field(None, description="DP epsilon if applicable")

    # Performance metrics
    accuracy_metrics: Optional[Dict[str, float]] = Field(None, description="Model accuracy metrics")
    performance_benchmarks: Optional[Dict[str, Any]] = Field(None, description="Performance benchmarks")

    # Compliance and audit
    compliance_tags: Optional[List[str]] = Field(None, description="Compliance requirement tags")
    audit_trail: Optional[str] = Field(None, description="Audit trail hash")


class ModelCreateRequest(BaseModel):
    """Request model for creating a new model."""

    metadata: ModelMetadata = Field(..., description="Model metadata")

    # Model definition
    architecture: Dict[str, Any] = Field(..., description="Model architecture definition")
    initial_weights: Optional[str] = Field(None, description="Base64-encoded initial weights")

    # Federated learning configuration
    federated_config: Optional[Dict[str, Any]] = Field(None, description="Federated learning configuration")
    aggregation_method: AggregationMethod = Field(AggregationMethod.FEDAVG, description="Aggregation method")

    # Privacy configuration
    privacy_config: Optional[Dict[str, Any]] = Field(None, description="Privacy-preserving training config")

    # Storage and lifecycle
    storage_config: Optional[Dict[str, Any]] = Field(None, description="Model storage configuration")
    retention_days: Optional[int] = Field(None, ge=1, description="Model retention period in days")

    model_config = {
        "json_schema_extra": {
            "example": {
                "metadata": {
                    "model_name": "Genomic Risk Predictor",
                    "description": "Federated model for genomic risk assessment",
                    "version": "1.0.0",
                    "model_type": "neural_network",
                    "framework": "pytorch",
                    "input_shape": [8192],
                    "output_shape": [1],
                    "hyperparameters": {
                        "learning_rate": 0.001,
                        "batch_size": 32,
                        "hidden_layers": [512, 256, 128]
                    }
                },
                "architecture": {
                    "layers": [
                        {"type": "linear", "input_size": 8192, "output_size": 512},
                        {"type": "relu"},
                        {"type": "linear", "input_size": 512, "output_size": 256},
                        {"type": "relu"},
                        {"type": "linear", "input_size": 256, "output_size": 1},
                        {"type": "sigmoid"}
                    ]
                },
                "federated_config": {
                    "min_participants": 5,
                    "max_participants": 20,
                    "rounds": 100
                },
                "privacy_config": {
                    "differential_privacy": True,
                    "epsilon": 0.1,
                    "noise_multiplier": 1.0
                }
            }
        }


class ModelCreateResponse(BaseModel):
    """Response model for model creation."""

    model_id: str = Field(..., description="Created model identifier")
    model_name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")

    # Creation details
    created_at: datetime = Field(..., description="Model creation timestamp")
    storage_location: str = Field(..., description="Model storage location")

    # Validation results
    validation_results: Dict[str, bool] = Field(..., description="Architecture validation results")
    estimated_size_mb: Optional[float] = Field(None, description="Estimated model size in MB")

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "model_abc123456789",
                "model_name": "Genomic Risk Predictor",
                "version": "1.0.0",
                "created_at": "2024-01-15T10:30:00Z",
                "storage_location": "s3://models/model_abc123456789",
                "validation_results": {
                    "architecture_valid": True,
                    "hyperparameters_valid": True,
                    "privacy_config_valid": True
                },
                "estimated_size_mb": 25.7
            }
        }


class ModelUpdateRequest(BaseModel):
    """Request model for updating a model."""

    model_id: str = Field(..., description="Model identifier")

    # What to update
    metadata_updates: Optional[Dict[str, Any]] = Field(None, description="Metadata updates")
    architecture_updates: Optional[Dict[str, Any]] = Field(None, description="Architecture updates")
    weights_update: Optional[str] = Field(None, description="New weights (base64-encoded)")

    # Version management
    create_new_version: bool = Field(False, description="Create new version instead of updating")
    version_increment: str = Field("patch", pattern=r"^(major|minor|patch)$", description="Version increment type")

    # Update options
    validate_update: bool = Field(True, description="Validate updates before applying")
    backup_previous: bool = Field(True, description="Backup previous version")


class ModelUpdateResponse(BaseModel):
    """Response model for model updates."""

    model_id: str = Field(..., description="Model identifier")
    previous_version: str = Field(..., description="Previous model version")
    new_version: str = Field(..., description="New model version")

    # Update details
    updated_at: datetime = Field(..., description="Update timestamp")
    changes_applied: List[str] = Field(..., description="List of changes applied")

    # Validation results
    validation_results: Dict[str, bool] = Field(..., description="Update validation results")
    backup_location: Optional[str] = Field(None, description="Backup location if created")


class ModelResponse(BaseModel):
    """Response model for model information."""

    model_id: str = Field(..., description="Model identifier")
    metadata: ModelMetadata = Field(..., description="Model metadata")

    # Model state
    status: str = Field(..., description="Model status")
    current_version: str = Field(..., description="Current model version")
    size_mb: float = Field(..., description="Model size in MB")

    # Training information
    training_status: Optional[TrainingStatus] = Field(None, description="Training status if applicable")
    training_progress: Optional[int] = Field(None, ge=0, le=100, description="Training progress percentage")

    # Usage statistics
    usage_stats: Optional[Dict[str, Any]] = Field(None, description="Model usage statistics")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Current performance metrics")

    # Timestamps
    created_at: datetime = Field(..., description="Model creation timestamp")
    last_updated: datetime = Field(..., description="Last update timestamp")
    last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")


class ModelTrainingRequest(BaseModel):
    """Request model for federated model training."""

    model_id: str = Field(..., description="Model identifier to train")

    # Training configuration
    training_config: Dict[str, Any] = Field(..., description="Training configuration")
    federated_config: Dict[str, Any] = Field(..., description="Federated learning configuration")

    # Participants
    participant_requirements: Optional[Dict[str, Any]] = Field(None, description="Participant requirements")
    max_participants: int = Field(10, ge=1, le=100, description="Maximum number of participants")
    min_participants: int = Field(3, ge=1, description="Minimum number of participants")

    # Training schedule
    start_time: Optional[datetime] = Field(None, description="Training start time (immediate if not specified)")
    max_duration_hours: int = Field(24, ge=1, le=168, description="Maximum training duration in hours")

    # Privacy settings
    privacy_budget: Optional[float] = Field(None, ge=0.001, description="Differential privacy budget")
    secure_aggregation: bool = Field(True, description="Use secure aggregation")

    @field_validator("min_participants")
    def validate_min_participants(cls, v, values):
        """Validate minimum participants against maximum."""
        max_participants = values.get("max_participants", 10)
        if v > max_participants:
            raise ValueError("min_participants cannot exceed max_participants")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_id": "model_abc123456789",
                "training_config": {
                    "epochs": 10,
                    "local_epochs": 5,
                    "learning_rate": 0.001,
                    "batch_size": 32
                },
                "federated_config": {
                    "aggregation_method": "fedavg",
                    "communication_rounds": 20,
                    "client_fraction": 0.5
                },
                "max_participants": 15,
                "min_participants": 5,
                "max_duration_hours": 12,
                "privacy_budget": 1.0,
                "secure_aggregation": True
            }
        }


class ModelTrainingResponse(BaseModel):
    """Response model for training initiation."""

    training_id: str = Field(..., description="Training session identifier")
    model_id: str = Field(..., description="Model identifier")
    status: TrainingStatus = Field(..., description="Initial training status")

    # Training details
    scheduled_start: datetime = Field(..., description="Scheduled training start time")
    estimated_duration_hours: float = Field(..., description="Estimated training duration")

    # Participant information
    expected_participants: int = Field(..., description="Expected number of participants")
    registration_deadline: Optional[datetime] = Field(None, description="Participant registration deadline")

    # Monitoring
    status_url: str = Field(..., description="URL to monitor training progress")

    model_config = {
        "json_schema_extra": {
            "example": {
                "training_id": "training_def456789012",
                "model_id": "model_abc123456789",
                "status": "initializing",
                "scheduled_start": "2024-01-15T11:00:00Z",
                "estimated_duration_hours": 8.5,
                "expected_participants": 12,
                "registration_deadline": "2024-01-15T10:45:00Z",
                "status_url": "/models/training_def456789012/status"
            }
        }


class TrainingStatusResponse(BaseModel):
    """Response model for training status."""

    training_id: str = Field(..., description="Training session identifier")
    model_id: str = Field(..., description="Model identifier")
    status: TrainingStatus = Field(..., description="Current training status")

    # Progress information
    current_round: Optional[int] = Field(None, description="Current training round")
    total_rounds: Optional[int] = Field(None, description="Total training rounds")
    progress_percent: int = Field(..., ge=0, le=100, description="Training progress percentage")

    # Participant information
    registered_participants: int = Field(..., description="Number of registered participants")
    active_participants: int = Field(..., description="Number of active participants")
    participant_stats: Optional[Dict[str, Any]] = Field(None, description="Participant statistics")

    # Performance metrics
    current_metrics: Optional[Dict[str, float]] = Field(None, description="Current performance metrics")
    best_metrics: Optional[Dict[str, float]] = Field(None, description="Best metrics achieved")

    # Timing information
    started_at: Optional[datetime] = Field(None, description="Training start time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    last_update: datetime = Field(..., description="Last status update time")

    # Privacy information
    privacy_spent: Optional[float] = Field(None, description="Privacy budget spent")
    privacy_remaining: Optional[float] = Field(None, description="Remaining privacy budget")


class ModelListRequest(BaseModel):
    """Request model for listing models."""

    model_type: Optional[ModelType] = Field(None, description="Filter by model type")
    framework: Optional[ModelFramework] = Field(None, description="Filter by framework")
    created_after: Optional[datetime] = Field(None, description="Filter by creation date")
    status: Optional[str] = Field(None, description="Filter by status")

    # Search
    search_term: Optional[str] = Field(None, description="Search in model names and descriptions")

    # Pagination
    page: int = Field(1, ge=1, description="Page number")
    per_page: int = Field(20, ge=1, le=100, description="Items per page")


class ModelSummary(BaseModel):
    """Model summary information."""

    model_id: str = Field(..., description="Model identifier")
    model_name: str = Field(..., description="Model name")
    model_type: ModelType = Field(..., description="Model type")
    framework: ModelFramework = Field(..., description="Framework")
    version: str = Field(..., description="Current version")

    # Status
    status: str = Field(..., description="Model status")
    size_mb: float = Field(..., description="Model size in MB")

    # Usage statistics
    training_sessions: int = Field(..., description="Number of training sessions")
    inference_requests: int = Field(..., description="Number of inference requests")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")

    # Performance
    accuracy_score: Optional[float] = Field(None, description="Latest accuracy score")

    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class ModelListResponse(BaseModel):
    """Response model for model listing."""

    models: List[ModelSummary] = Field(..., description="Model summaries")
    total: int = Field(..., description="Total number of models")
    page: int = Field(..., description="Current page number")
    per_page: int = Field(..., description="Items per page")
    total_pages: int = Field(..., description="Total number of pages")
                }
            }
        }
    }
