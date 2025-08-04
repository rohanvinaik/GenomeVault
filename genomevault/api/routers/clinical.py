from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException

from genomevault.api.models.clinical import (ClinicalEvalRequest,
                                             ClinicalEvalResponse)
from genomevault.clinical.eval.harness import compute_report

router = APIRouter(prefix="/clinical", tags=["clinical"])


@router.post("/eval", response_model=ClinicalEvalResponse)
def clinical_eval(req: ClinicalEvalRequest):
    try:
        y = np.asarray(req.y_true, dtype=np.int32)
        s = np.asarray(req.y_score, dtype=np.float64)
        rep = compute_report(y, s, calibrator=req.calibrator, bins=req.bins)
        return ClinicalEvalResponse(
            metrics=rep.metrics,
            threshold=rep.threshold,
            confusion=rep.confusion,
            calibration_bins=rep.calibration_bins,
        )
    except Exception as e:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        raise HTTPException(status_code=400, detail=str(e))
        raise RuntimeError("Unspecified error")
