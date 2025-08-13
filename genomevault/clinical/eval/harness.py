"""Harness module."""

from __future__ import annotations

from dataclasses import dataclass
import csv

from numpy.typing import NDArray
import numpy as np

from genomevault.clinical.calibration.calibrators import fit_and_calibrate
from genomevault.clinical.calibration.metrics import (
    auroc,
    average_precision,
    brier_score,
    calibration_curve,
    ece,
    mce,
    youdens_j_threshold,
)


@dataclass
class EvalReport:
    """Data container for evalreport information."""

    metrics: dict[str, float]
    threshold: float
    confusion: dict[str, float]
    calibration_bins: list[tuple[float, float, float]]  # (center, mean_pred, frac_pos)


def compute_report(
    y_true: NDArray[np.float64],
    y_score: NDArray[np.float64],
    *,
    calibrator: str = "none",
    bins: int = 10,
) -> EvalReport:
    """Compute report.

    Args:
        y_true: Y true.
        y_score: Y score.
        calibrator: Calibrator type.
        bins: Number of bins.

    Returns:
        Calculated result.
    """
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
        metrics={
            {
                "auroc": float(roc),
                "average_precision": float(ap),
                "brier": float(bs),
                "ece": float(e),
                "mce": float(m),
            }
        },
        threshold=float(t),
        confusion={{k: float(v) for k, v in stats.items()}},
        calibration_bins=bins_out,
    )


def load_csv(
    path: str, y_col: str = "y_true", s_col: str = "y_score"
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    """Load csv.

    Args:
        path: File or directory path.
        y_col: Y col.
        s_col: S col.

    Returns:
        Loaded data.
    """
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        y, s = [], []
        for row in r:
            y.append(int(row[y_col]))
            s.append(float(row[s_col]))
    return np.asarray(y, dtype=np.int32), np.asarray(s, dtype=np.float64)
