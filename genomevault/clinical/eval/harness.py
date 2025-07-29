from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from genomevault.clinical.calibration.metrics import (
    auroc,
    average_precision,
    brier_score,
    ece,
    mce,
    calibration_curve,
    youdens_j_threshold,
    confusion_at,
)
from genomevault.clinical.calibration.calibrators import fit_and_calibrate


@dataclass
class EvalReport:
    metrics: Dict[str, float]
    threshold: float
    confusion: Dict[str, float]
    calibration_bins: List[Tuple[float, float, float]]  # (center, mean_pred, frac_pos)


def compute_report(y_true: np.ndarray, y_score: np.ndarray, *, calibrator: str = "none", bins: int = 10) -> EvalReport:
    # Calibrate if requested (on same data for simplicity; consider CV for unbiased estimates)
    y_prob, cal = fit_and_calibrate(y_true, y_score, method=calibrator)
    roc = auroc(y_true, y_prob)
    ap = average_precision(y_true, y_prob)
    bs = brier_score(y_true, y_prob)
    e = ece(y_true, y_prob, n_bins=bins)
    m = mce(y_true, y_prob, n_bins=bins)
    t, stats = youdens_j_threshold(y_true, y_prob)
    centers, mean_pred, frac_pos = calibration_curve(y_true, y_prob, n_bins=bins)
    bins_out = [(float(c), float(mp), float(fp)) for c, mp, fp in zip(centers, mean_pred, frac_pos)]
    return EvalReport(
        metrics={{"auroc": float(roc), "average_precision": float(ap), "brier": float(bs), "ece": float(e), "mce": float(m)}},
        threshold=float(t),
        confusion={{k: float(v) for k, v in stats.items()}},
        calibration_bins=bins_out,
    )


def load_csv(path: str, y_col: str = "y_true", s_col: str = "y_score") -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        y, s = [], []
        for row in r:
            y.append(int(row[y_col]))
            s.append(float(row[s_col]))
    return np.asarray(y, dtype=np.int32), np.asarray(s, dtype=np.float64)
