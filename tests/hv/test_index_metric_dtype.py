import numpy as np
import pytest
from genomevault.hypervector.index import build, search, load_index_metadata


def test_build_defaults_metric_based_on_dtype(tmp_path):
    """Building infers metric from vector dtype."""
    float_vecs = [
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        np.array([0.2, 0.1, 0.4], dtype=np.float32),
    ]
    build(float_vecs, ["a", "b"], tmp_path)
    manifest = load_index_metadata(tmp_path)
    assert manifest["metric"] == "cosine"

    binary_vecs = [
        np.array([0, 1, 0], dtype=np.uint8),
        np.array([1, 0, 1], dtype=np.uint8),
    ]
    build(binary_vecs, ["c", "d"], tmp_path / "bin")
    manifest_bin = load_index_metadata(tmp_path / "bin")
    assert manifest_bin["metric"] == "hamming"


def test_build_rejects_mismatched_metric(tmp_path):
    """Invalid metric-dtype pairs raise errors during build."""
    float_vecs = [
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([0.2, 0.3], dtype=np.float32),
    ]
    with pytest.raises(ValueError):
        build(float_vecs, ["a", "b"], tmp_path, metric="hamming")

    binary_vecs = [
        np.array([0, 1], dtype=np.uint8),
        np.array([1, 0], dtype=np.uint8),
    ]
    with pytest.raises(ValueError):
        build(binary_vecs, ["a", "b"], tmp_path / "b", metric="cosine")


def test_search_rejects_mismatched_query_dtype(tmp_path):
    """Search validates query dtype against index metric."""
    binary_vecs = [
        np.array([0, 1, 0], dtype=np.uint8),
        np.array([1, 0, 1], dtype=np.uint8),
    ]
    build(binary_vecs, ["a", "b"], tmp_path, metric="hamming")
    with pytest.raises(ValueError):
        search(np.array([0.1, 0.2, 0.3], dtype=np.float32), tmp_path)

    float_vecs = [
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        np.array([0.2, 0.1, 0.4], dtype=np.float32),
    ]
    build(float_vecs, ["c", "d"], tmp_path / "f", metric="cosine")
    with pytest.raises(ValueError):
        search(np.array([0, 1, 0], dtype=np.uint8), tmp_path / "f")


def test_search_allows_valid_query(tmp_path):
    """Valid metric and dtype combinations search successfully."""
    binary_vecs = [
        np.array([0, 1, 0], dtype=np.uint8),
        np.array([1, 0, 1], dtype=np.uint8),
    ]
    build(binary_vecs, ["a", "b"], tmp_path / "h", metric="hamming")
    res = search(np.array([0, 1, 0], dtype=np.uint8), tmp_path / "h")
    assert res and {r["id"] for r in res} <= {"a", "b"}

    float_vecs = [
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
        np.array([0.2, 0.1, 0.4], dtype=np.float32),
    ]
    build(float_vecs, ["c", "d"], tmp_path / "c", metric="cosine")
    res = search(np.array([0.1, 0.2, 0.3], dtype=np.float32), tmp_path / "c")
    assert res and {r["id"] for r in res} <= {"c", "d"}
