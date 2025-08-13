import numpy as np

from genomevault.clinical.calibration.metrics import (

    auroc,
    average_precision,
    brier_score,
    calibration_curve,
    confusion_at,
    ece,
    mce,
    youdens_j_threshold,
)


def test_auroc_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    s_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    s_random = np.array([0.1, 0.9, 0.2, 0.8])
    assert abs(auroc(y, s_perfect) - 1.0) < 1e-9
    # random order here yields auc 0.5
    assert abs(auroc(y, s_random) - 0.5) < 1e-9


def test_ap_and_brier_and_calibration_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    p = rng.random(200)
    ap = average_precision(y, p)
    bs = brier_score(y, p)
    assert 0.0 <= ap <= 1.0
    assert 0.0 <= bs <= 1.0
    e = ece(y, p, n_bins=10)
    m = mce(y, p, n_bins=10)
    assert 0.0 <= e <= 1.0 and 0.0 <= m <= 1.0
    bins, mp, fp = calibration_curve(y, p, n_bins=10)
    assert bins.shape == mp.shape == fp.shape


def test_confusion_and_j():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    stats = confusion_at(y, p, threshold=0.5)
    assert stats["tp"] == 2 and stats["tn"] == 2
    t, s = youdens_j_threshold(y, p)
    assert abs(t - 0.8) < 1e-9 or abs(t - 0.9) < 1e-9
