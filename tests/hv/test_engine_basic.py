from genomevault.core.constants import HYPERVECTOR_DIMENSIONS
from genomevault.core.exceptions import ProjectionError, ValidationError
from genomevault.hypervector.engine import HypervectorEngine


def test_encode_and_similarity_roundtrip():
    engine = HypervectorEngine()
    dim = HYPERVECTOR_DIMENSIONS["base"]
    r1 = engine.encode(
        data={"genomic": [1.0, 0.0, 2.0, 3.0]}, dimension=dim, compression_tier="mini"
    )
    assert r1["dimension"] == dim and "vector_id" in r1
    sim = engine.calculate_similarity(r1["vector_id"], r1["vector_id"])
    assert 0.99 <= sim <= 1.0


def test_operate_and_bundle():
    engine = HypervectorEngine()
    dim = HYPERVECTOR_DIMENSIONS["base"]
    a = engine.encode(data={"genomic": [1, 2, 3]}, dimension=dim)
    b = engine.encode(data={"genomic": [3, 2, 1]}, dimension=dim)
    out = engine.operate(operation="bundle", vector_ids=[a["vector_id"], b["vector_id"]])
    assert "result_vector_id" in out


def test_perm_and_bind():
    engine = HypervectorEngine()
    dim = HYPERVECTOR_DIMENSIONS["base"]
    a = engine.encode(data={"genomic": [1, 0, 0]}, dimension=dim)
    r = engine.operate(operation="permute", vector_ids=[a["vector_id"]], parameters={"shift": 3})
    assert "result_vector_id" in r
    # bind/multiply
    b = engine.encode(data={"genomic": [0, 1, 0]}, dimension=dim)
    r2 = engine.operate(operation="bind", vector_ids=[a["vector_id"], b["vector_id"]])
    assert "result_vector_id" in r2


def test_invalid_dimension_raises():
    engine = HypervectorEngine()
    try:
        engine.encode(data={"genomic": [1, 2]}, dimension=123)
        assert False, "should have raised"
    except ProjectionError:
        logger.exception("Unhandled exception")
        raise


def test_missing_vector_id_raises():
    engine = HypervectorEngine()
    try:
        engine.calculate_similarity("nope", "nope2")
        assert False, "should have raised"
    except ValidationError:
        logger.exception("Unhandled exception")
        raise
