import numpy as np

from genomevault.kan.hybrid import CompressionMetrics, HybridKANHD


def synthetic_genomes(n=64, d=1024, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 4, size=(n, d)) - 1).astype(np.int8)


def test_hybrid_kan_hd_smoke():
    X = synthetic_genomes(16, 256)
    model = HybridKANHD(dimension=128, spline_mode="bspline", privacy_tier="standard")
    enc, metrics = model.encode_genomic_data(X)
    assert isinstance(metrics, CompressionMetrics)
    rep = model.generate_interpretability_report(top_k=5)
    assert isinstance(rep, dict) and "patterns" in rep
