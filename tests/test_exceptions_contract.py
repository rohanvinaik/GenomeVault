import pytest

from genomevault.exceptions import (
    GVError,
    GVInputError,
    GVNotFound,
    GVTimeout,
    error_response,
    CODEC,
)


@pytest.mark.parametrize(
    "exc_cls,kwargs",
    [
        (GVError, {"message": "base boom"}),
        (GVInputError, {"message": "bad input", "details": {"field": "vcf"}}),
        (GVNotFound, {"message": "panel missing"}),
        (GVTimeout, {"message": "zk prover slow", "details": {"stage": "prove"}}),
    ],
)
def test_exception_to_dict(exc_cls, kwargs):
    """Test exception to dict.
    Args:        exc_cls: List of items.        kwargs: List of items.
    Returns:
        Result of the operation."""
    exc = exc_cls(**kwargs)
    d = error_response(exc)
    assert d["type"] == exc_cls.__name__
    assert "message" in d and d["message"]
    assert "code" in d and isinstance(d["code"], str)
    assert isinstance(d.get("details", {}), dict)


def test_codec_roundtrip_known_codes():
    """Test codec roundtrip known codes.
    Returns:
        Result of the operation."""
    for code, cls in CODEC.items():
        assert issubclass(cls, GVError)
        inst = cls("msg")
        assert error_response(inst)["code"] == code
