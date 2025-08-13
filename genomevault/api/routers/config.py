"""
Example configuration router demonstrating PATCH patterns.

This router shows best practices for handling partial updates
using Optional fields in Pydantic models.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(
    prefix="/api/v1/config",
    tags=["configuration"],
)


# Example: Configuration models with create vs patch patterns
class EncodingConfigBase(BaseModel):
    """Base configuration shared between create and update."""

    description: Optional[str] = Field(None, max_length=500)
    metadata: Optional[Dict[str, Any]] = None


class EncodingConfigCreate(EncodingConfigBase):
    """Create model - required fields."""

    name: str = Field(..., min_length=1, max_length=100)
    dimension: int = Field(..., ge=10000, le=100000)
    compression_tier: str = Field(..., regex="^(mini|clinical|full)$")
    enabled: bool = Field(True)


class EncodingConfigUpdate(EncodingConfigBase):
    """Update model - all fields optional for PATCH."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    dimension: Optional[int] = Field(None, ge=10000, le=100000)
    compression_tier: Optional[str] = Field(None, regex="^(mini|clinical|full)$")
    enabled: Optional[bool] = None

    def dict_for_update(self) -> Dict[str, Any]:
        """Return only set fields for database update."""
        return self.dict(exclude_unset=True, exclude_none=True)


class EncodingConfigResponse(BaseModel):
    """Response model for configuration."""

    id: str
    name: str
    dimension: int
    compression_tier: str
    enabled: bool
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


# Mock database storage
_configs_db: Dict[str, Dict[str, Any]] = {}


@router.post(
    "/encoding",
    response_model=EncodingConfigResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_encoding_config(config: EncodingConfigCreate) -> Any:
    """
    Create a new encoding configuration.

    All required fields must be provided.
    """
    import uuid
    from datetime import datetime

    config_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    # Store in mock database
    _configs_db[config_id] = {
        "id": config_id,
        **config.dict(),
        "created_at": now,
        "updated_at": now,
    }

    logger.info(f"Created encoding config: {config_id}")
    return _configs_db[config_id]


@router.patch("/encoding/{config_id}", response_model=EncodingConfigResponse)
async def update_encoding_config(config_id: str, patch: EncodingConfigUpdate) -> Any:
    """
    Partially update an encoding configuration.

    Only provided fields will be updated. All fields are optional.
    This is a true PATCH operation - unset fields are not modified.
    """
    from datetime import datetime

    if config_id not in _configs_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration {config_id} not found",
        )

    # Get only the fields that were actually set in the request
    update_data = patch.dict_for_update()

    if not update_data:
        # No fields to update
        return _configs_db[config_id]

    # Update only provided fields
    _configs_db[config_id].update(update_data)
    _configs_db[config_id]["updated_at"] = datetime.utcnow().isoformat()

    logger.info(f"Updated config {config_id} with fields: {list(update_data.keys())}")
    return _configs_db[config_id]


@router.put("/encoding/{config_id}", response_model=EncodingConfigResponse)
async def replace_encoding_config(config_id: str, config: EncodingConfigCreate) -> Any:
    """
    Replace an entire encoding configuration.

    This is a PUT operation - all fields must be provided.
    The entire resource is replaced.
    """
    from datetime import datetime

    if config_id not in _configs_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration {config_id} not found",
        )

    # Preserve creation timestamp
    created_at = _configs_db[config_id]["created_at"]

    # Replace entire configuration
    _configs_db[config_id] = {
        "id": config_id,
        **config.dict(),
        "created_at": created_at,
        "updated_at": datetime.utcnow().isoformat(),
    }

    logger.info(f"Replaced entire config: {config_id}")
    return _configs_db[config_id]


@router.get("/encoding/{config_id}", response_model=EncodingConfigResponse)
async def get_encoding_config(config_id: str) -> Any:
    """Get an encoding configuration by ID."""

    if config_id not in _configs_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Configuration {config_id} not found",
        )

    return _configs_db[config_id]


# Example: Nested partial updates
class UserPreferencesBase(BaseModel):
    """Base preferences model."""

    class NotificationSettings(BaseModel):
        """Nested notification settings."""

        email_enabled: bool = True
        push_enabled: bool = False
        frequency: str = "daily"  # daily, weekly, monthly

    class PrivacySettings(BaseModel):
        """Nested privacy settings."""

        share_data: bool = False
        allow_research: bool = False
        retention_days: int = 365


class UserPreferencesPatch(BaseModel):
    """
    PATCH model for user preferences with nested objects.
    All fields optional, including nested fields.
    """

    class NotificationSettingsPatch(BaseModel):
        """Partial update for notifications."""

        email_enabled: Optional[bool] = None
        push_enabled: Optional[bool] = None
        frequency: Optional[str] = None

    class PrivacySettingsPatch(BaseModel):
        """Partial update for privacy."""

        share_data: Optional[bool] = None
        allow_research: Optional[bool] = None
        retention_days: Optional[int] = Field(None, ge=1, le=3650)

    theme: Optional[str] = Field(None, regex="^(light|dark|auto)$")
    language: Optional[str] = Field(None, regex="^[a-z]{2}(-[A-Z]{2})?$")
    notifications: Optional[NotificationSettingsPatch] = None
    privacy: Optional[PrivacySettingsPatch] = None

    def merge_with_existing(self, existing: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge patch data with existing data.
        Handles nested objects properly.
        """
        update_data = self.dict(exclude_unset=True, exclude_none=True)

        for key, value in update_data.items():
            if key in existing and isinstance(existing[key], dict) and isinstance(value, dict):
                # Deep merge nested objects
                existing[key].update(value)
            else:
                existing[key] = value

        return existing


@router.patch("/preferences/{user_id}")
async def update_user_preferences(user_id: str, patch: UserPreferencesPatch) -> Any:
    """
    Update user preferences with deep nested partial updates.

    Example request:
    ```json
    {
        "theme": "dark",
        "notifications": {
            "email_enabled": false
        }
    }
    ```

    This will only update the theme and email notification setting,
    leaving all other preferences unchanged.
    """
    # In a real app, fetch from database
    existing_preferences = {
        "user_id": user_id,
        "theme": "light",
        "language": "en-US",
        "notifications": {
            "email_enabled": True,
            "push_enabled": False,
            "frequency": "daily",
        },
        "privacy": {
            "share_data": False,
            "allow_research": False,
            "retention_days": 365,
        },
    }

    # Apply partial updates
    updated = patch.merge_with_existing(existing_preferences)

    logger.info(f"Updated preferences for user {user_id}")
    return updated
