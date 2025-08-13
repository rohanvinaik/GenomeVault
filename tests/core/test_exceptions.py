import pytest

from genomevault.core.exceptions import (
    APISchemaError,
    ConfigurationError,
    EncodingError,
    FederatedError,
    GenomeVaultError,
    LedgerError,
    PIRProtocolError,
    ProjectionError,
    ValidationError,
    ZKProofError,
)


def test_instantiation_and_str_contains_class():
    """Test instantiation and str contains class.
    Returns:
        Result of the operation."""
    e = ProjectionError("bad dims", context={"dim": 100})
    s = str(e)
    assert "ProjectionError" in s
    assert "bad dims" in s
    assert "dim" in s


@pytest.mark.parametrize(
    "exc_cls",
    [
        GenomeVaultError,
        ConfigurationError,
        ValidationError,
        ProjectionError,
        EncodingError,
        ZKProofError,
        PIRProtocolError,
        LedgerError,
        FederatedError,
        APISchemaError,
    ],
)
def test_all_exceptions_constructible(exc_cls):
    """Test all exceptions constructible.
    Args:        exc_cls: List of items.
    Returns:
        Result of the operation."""
    e = exc_cls("message", context={"k": 1})
    assert isinstance(e, Exception)
    assert e.context == {"k": 1}
