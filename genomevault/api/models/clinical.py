from __future__ import annotations

from typing import List, Tuple, Dict
from pydantic import BaseModel, Field


class ClinicalEvalRequest(BaseModel):
    y_true: List[int] = Field(..., description="Binary labels 0/1")
    y_score: List[float] = Field(..., description="Uncalibrated scores in [0,1]")
    calibrator: str = Field("none", description="none | platt | isotonic")
    bins: int = Field(10, ge=2, le=50)


class ClinicalEvalResponse(BaseModel):
    metrics: Dict[str, float]
    threshold: float
    confusion: Dict[str, float]
    calibration_bins: List[Tuple[float, float, float]]  # (center, mean_pred, frac_pos)
