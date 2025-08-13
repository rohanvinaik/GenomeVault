import numpy as np

from genomevault.clinical.eval.harness import compute_report

def test_compute_report_smoke():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=200)
    s = rng.random(200)
    rep = compute_report(y, s, calibrator="none", bins=8)
    assert "auroc" in rep.metrics and 0.0 <= rep.metrics["auroc"] <= 1.0
    assert len(rep.calibration_bins) == 8
