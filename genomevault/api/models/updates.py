"""
Example models demonstrating PATCH/update patterns with Optional fields.

This module shows best practices for partial update models to avoid
"missing required field" type errors in PATCH-like handlers.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Any, Dict, List, TypedDict

from genomevault.api.utils import dict_for_update as _dict_for_update


class UpdateModelMixin(BaseModel):
    """Base mixin for models that support partial updates."""

    def dict_for_update(self) -> Dict[str, Any]:
        """Return only set fields for database update."""
        # Use shared implementation but keep method for backward compatibility
        return _dict_for_update(self)


# Example 1: Basic PATCH model with all Optional fields
class UserSettingsPatch(UpdateModelMixin):
    """
    PATCH model for user settings - all fields are optional.
    Only provided fields will be updated.
    """

    notification_enabled: Optional[bool] = None
    privacy_level: Optional[str] = None
    dimension_preference: Optional[int] = None
    compression_tier: Optional[str] = None

    class Config:
        """Configuration settings for ."""
        schema_extra = {
            "example": {
                "notification_enabled": True,
                "dimension_preference": 20000,
                # Other fields can be omitted in PATCH
            }
        }


# Example 2: Using model inheritance for create vs update
class HypervectorConfigBase(BaseModel):
    """Base configuration shared between create and update."""

    description: Optional[str] = Field(None, max_length=500)
    metadata: Optional[Dict[str, Any]] = None


class HypervectorConfigCreate(HypervectorConfigBase):
    """Create model - required fields."""

    name: str = Field(..., min_length=1, max_length=100)
    dimension: int = Field(..., ge=1000, le=100000)
    compression_tier: str = Field(...)


class HypervectorConfigUpdate(HypervectorConfigBase):
    """Update model - all fields optional for PATCH."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    dimension: Optional[int] = Field(None, ge=1000, le=100000)
    compression_tier: Optional[str] = None


# Example 3: Using TypedDict for partial updates


class AnalysisConfigUpdate(TypedDict, total=False):
    """
    TypedDict with NotRequired for partial updates.
    All fields are optional by setting total=False.
    """

    algorithm: str
    confidence_threshold: float
    max_iterations: int
    enable_caching: bool


# Example 4: Pydantic model with exclude_unset for PATCH
class ExperimentSettingsPatch(UpdateModelMixin):
    """
    Model for PATCH operations using exclude_unset.
    """

    name: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    parameters: Optional[Dict[str, Any]] = None


# Example 5: Advanced pattern with field validation but optional updates
class GenomicAnalysisPatch(BaseModel):
    """
    PATCH model with validation on optional fields.
    """

    analysis_type: Optional[str] = Field(
        None, pattern="^(SNV|CNV|SV|INDEL)$"
    )
    quality_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    reference_genome: Optional[str] = Field(
        None, pattern="^(hg19|hg38|GRCh37|GRCh38)$"
    )
    filters: Optional[Dict[str, Any]] = None

    @field_validator("quality_threshold")  # type: ignore
    def validate_quality(cls, v: Optional[float]) -> Optional[float]:
        """Validate quality threshold if provided."""
        if v is not None and v < 0.5:
            # Warning: low quality threshold
            # In production, log a warning here
            import logging

            logging.warning(f"Low quality threshold: {v}")
        return v

    class Config:
        """Configuration settings for ."""
        # Allow mutation for in-place updates
        allow_mutation = True
        schema_extra = {
            "example": {
                "quality_threshold": 0.95,
                "reference_genome": "GRCh38",
                # Other fields optional
            }
        }


# Example 6: Using Union types for flexible updates


class DataSourceUpdate(BaseModel):
    """
    Flexible update model supporting different data source types.
    """

    source_type: Optional[str] = None

    # Union of possible configurations
    s3_config: Optional[Dict[str, str]] = None
    gcs_config: Optional[Dict[str, str]] = None
    local_path: Optional[str] = None

    # Common fields
    compression: Optional[bool] = None
    encryption_key: Optional[str] = Field(None, exclude=True)  # Exclude from response

    @model_validator(mode="after")  # type: ignore
    def validate_source_config(cls, values: "DataSourceUpdate") -> "DataSourceUpdate":
        """Ensure only one source config is provided."""
        configs = [
            values.s3_config,
            values.gcs_config,
            values.local_path,
        ]
        if sum(c is not None for c in configs) > 1:
            raise ValueError("Only one source configuration can be updated at a time")
        return values


# Example 7: Nested partial updates
class PatientDataPatch(BaseModel):
    """
    Complex nested update model.
    """

    class ClinicalDataPatch(BaseModel):
        """Nested clinical data patch."""

        diagnosis: Optional[str] = None
        treatment_status: Optional[str] = None
        medications: Optional[List[str]] = None  # type: ignore

    class GenomicDataPatch(BaseModel):
        """Nested genomic data patch."""

        variants_count: Optional[int] = None
        quality_score: Optional[float] = None
        coverage_depth: Optional[float] = None

    patient_id: Optional[str] = Field(None, pattern="^P[0-9]{6}$")
    clinical: Optional[ClinicalDataPatch] = None
    genomic: Optional[GenomicDataPatch] = None
    last_updated: Optional[str] = None  # ISO timestamp

    def merge_with_existing(self, existing: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge patch data with existing data.
        """
        update_data = self.dict(exclude_unset=True, exclude_none=True)

        # Deep merge for nested objects
        for key, value in update_data.items():
            if key in existing and isinstance(existing[key], dict) and isinstance(value, dict):
                existing[key].update(value)
            else:
                existing[key] = value

        return existing


# Usage example in router:
"""
@router.patch("/config/{config_id}", response_model=ConfigResponse)
async def update_config(
    config_id: str,
    patch: HypervectorConfigUpdate  # All fields optional
) -> Any:
    # Update config
    # Only update provided fields
    update_data = patch.dict(exclude_unset=True)

    # Apply updates to database
    updated = await db.update_config(config_id, update_data)

    return updated
"""
