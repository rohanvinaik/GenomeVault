import numpy as np
import pytest

from genomevault.core.exceptions import ProjectionError
from genomevault.hypervector.encoding.orthogonal_projection import OrthogonalProjection


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_dimension_and_angle_preservation():
    rng = np.random.default_rng(123)
    X = rng.standard_normal((4, 256))  # small for speed
    proj = OrthogonalProjection(n_components=64, seed=42).fit(n_features=256)
    Y = proj.transform(X)
    assert Y.shape == (4, 64)

    # Compare pairwise cosines
    cos_before = [cosine(X[0], X[1]), cosine(X[0], X[2]), cosine(X[1], X[2])]
    cos_after = [cosine(Y[0], Y[1]), cosine(Y[0], Y[2]), cosine(Y[1], Y[2])]
    for cb, ca in zip(cos_before, cos_after):
        assert np.isfinite(ca)
        assert np.allclose(cb, ca, rtol=1e-5, atol=1e-6)


def test_invalid_params_and_usage():
    with pytest.raises(ProjectionError):
        _ = OrthogonalProjection(n_components=0)

    proj = OrthogonalProjection(n_components=16)
    with pytest.raises(ProjectionError):
        _ = proj.transform(np.zeros((2, 16)))  # not fitted

    with pytest.raises(ProjectionError):
        _ = proj.fit(n_features=8)  # n_features < n_components
