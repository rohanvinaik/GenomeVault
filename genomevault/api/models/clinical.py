"""Clinical module."""
from __future__ import annotations

from pydantic import BaseModel, Field
class ClinicalEvalRequest(BaseModel):
    """ClinicalEvalRequest implementation."""

    y_true: list[int] = Field(..., description="Binary labels 0/1")
    y_score: list[float] = Field(..., description="Uncalibrated scores in [0,1]")
    calibrator: str = Field("none", description="none | platt | isotonic")
    bins: int = Field(10, ge=2, le=50)


class ClinicalEvalResponse(BaseModel):
    """ClinicalEvalResponse implementation."""

    metrics: dict[str, float]
    threshold: float
    confusion: dict[str, float]
    calibration_bins: list[tuple[float, float, float]]  # (center, mean_pred, frac_pos)
