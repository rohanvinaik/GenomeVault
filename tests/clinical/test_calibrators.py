import numpy as np
from genomevault.clinical.calibration.calibrators import PlattCalibrator, IsotonicCalibrator, fit_and_calibrate
from genomevault.clinical.calibration.metrics import brier_score


def test_platt_reduces_brier_on_synthetic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    # create poorly calibrated scores: square of true probs (shrinks extremes)
    p_true = 0.2 + 0.6 * rng.random(200)
    s = np.clip(p_true**0.5 + rng.normal(0, 0.05, size=200), 0, 1)
    b0 = brier_score(y, s)
    cal = PlattCalibrator().fit(y, s)
    p_cal = cal.predict_proba(s)
    b1 = brier_score(y, p_cal)
    assert b1 <= b0 or abs(b1 - b0) < 1e-6


def test_isotonic_monotone_and_reasonable():
    x = np.array([0.1, 0.4, 0.3, 0.8, 0.7])
    y = np.array([0, 0, 1, 1, 1])
    iso = IsotonicCalibrator().fit(y, x)
    p = iso.predict_proba(x)
    # monotone non-decreasing over sorted x
    order = np.argsort(x)
    assert np.all(np.diff(p[order]) >= -1e-9)


def test_fit_and_calibrate_dispatch():
    x = np.array([0.2, 0.9])
    y = np.array([0, 1])
    p, cal = fit_and_calibrate(y, x, method="platt")
    assert p.shape == x.shape
    p2, cal2 = fit_and_calibrate(y, x, method="none")
    assert np.allclose(p2, x)
