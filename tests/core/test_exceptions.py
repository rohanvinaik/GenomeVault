"""
Tests for GenomeVault core exceptions
"""

import pytest

from genomevault.core.exceptions import (
    APISchemaError,
    BindingError,
    ConfigurationError,
    EncodingError,
    FederatedError,
    GenomeVaultError,
    HypervectorError,
    LedgerError,
    PIRProtocolError,
    ProjectionError,
    ValidationError,
    ZKProofError,
)


class TestGenomeVaultError:
    """Test the base GenomeVaultError exception"""

    def test_basic_instantiation(self):
        """Test that basic instantiation works"""
        error = GenomeVaultError("Test error")
        assert str(error) == "GenomeVaultError: Test error"
        assert error.message == "Test error"
        assert error.context == {}

    def test_with_context(self):
        """Test exception with context"""
        context = {"user_id": 123, "operation": "encode"}
        error = GenomeVaultError("Test error with context", context)

        assert error.message == "Test error with context"
        assert error.context == context
        assert "user_id=123" in str(error)
        assert "operation=encode" in str(error)
        assert str(error).startswith("GenomeVaultError: Test error with context")

    def test_empty_context(self):
        """Test exception with empty context dict"""
        error = GenomeVaultError("Test error", {})
        assert str(error) == "GenomeVaultError: Test error"
        assert error.context == {}

    def test_none_context(self):
        """Test exception with None context"""
        error = GenomeVaultError("Test error", None)
        assert str(error) == "GenomeVaultError: Test error"
        assert error.context == {}


class TestDomainSpecificExceptions:
    """Test domain-specific exception subclasses"""

    def test_configuration_error(self):
        """Test ConfigurationError"""
        error = ConfigurationError("Invalid config", {"key": "hypervector_dim"})
        assert isinstance(error, GenomeVaultError)
        assert "ConfigurationError" in str(error)
        assert "key=hypervector_dim" in str(error)

    def test_validation_error(self):
        """Test ValidationError"""
        error = ValidationError("Invalid data format")
        assert isinstance(error, GenomeVaultError)
        assert str(error) == "ValidationError: Invalid data format"

    def test_projection_error(self):
        """Test ProjectionError"""
        context = {"dimensions": 10000, "actual": 5000}
        error = ProjectionError("Dimension mismatch", context)
        assert isinstance(error, GenomeVaultError)
        assert "dimensions=10000" in str(error)
        assert "actual=5000" in str(error)

    def test_encoding_error(self):
        """Test EncodingError"""
        error = EncodingError("Failed to encode genomic data", {"gene": "BRCA1"})
        assert isinstance(error, GenomeVaultError)
        assert "EncodingError" in str(error)
        assert "gene=BRCA1" in str(error)

    def test_zkproof_error(self):
        """Test ZKProofError"""
        error = ZKProofError("Proof verification failed", {"circuit": "diabetes_risk"})
        assert isinstance(error, GenomeVaultError)
        assert "ZKProofError" in str(error)
        assert "circuit=diabetes_risk" in str(error)

    def test_pir_protocol_error(self):
        """Test PIRProtocolError"""
        error = PIRProtocolError("Server timeout", {"server_id": 3, "timeout": 30})
        assert isinstance(error, GenomeVaultError)
        assert "PIRProtocolError" in str(error)
        assert "server_id=3" in str(error)
        assert "timeout=30" in str(error)

    def test_ledger_error(self):
        """Test LedgerError"""
        error = LedgerError("Block validation failed", {"block_height": 1000})
        assert isinstance(error, GenomeVaultError)
        assert "LedgerError" in str(error)
        assert "block_height=1000" in str(error)

    def test_federated_error(self):
        """Test FederatedError"""
        error = FederatedError("Aggregation failed", {"round": 5, "participants": 10})
        assert isinstance(error, GenomeVaultError)
        assert "FederatedError" in str(error)
        assert "round=5" in str(error)
        assert "participants=10" in str(error)

    def test_api_schema_error(self):
        """Test APISchemaError"""
        error = APISchemaError("Invalid request schema", {"endpoint": "/vectors/encode"})
        assert isinstance(error, GenomeVaultError)
        assert "APISchemaError" in str(error)
        assert "endpoint=/vectors/encode" in str(error)


class TestInheritanceChain:
    """Test inheritance relationships"""

    def test_hypervector_inheritance(self):
        """Test HypervectorError inheritance chain"""
        error = HypervectorError("Vector operation failed")
        assert isinstance(error, GenomeVaultError)

        binding_error = BindingError("Binding failed", {"operation": "circular_convolution"})
        assert isinstance(binding_error, HypervectorError)
        assert isinstance(binding_error, GenomeVaultError)
        assert "BindingError" in str(binding_error)
        assert "operation=circular_convolution" in str(binding_error)


class TestContextPreservation:
    """Test that context is properly preserved and accessible"""

    def test_context_preservation(self):
        """Test that context is preserved as-is"""
        complex_context = {
            "user_id": 123,
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "none_value": None,
        }
        error = GenomeVaultError("Complex error", complex_context)

        assert error.context["user_id"] == 123
        assert error.context["nested"] == {"key": "value"}
        assert error.context["list"] == [1, 2, 3]
        assert error.context["none_value"] is None

    def test_context_immutability(self):
        """Test that modifying original context doesn't affect exception"""
        original_context = {"key": "value"}
        error = GenomeVaultError("Test", original_context)

        # Modify original context
        original_context["key"] = "modified"
        original_context["new_key"] = "new_value"

        # Exception context should remain unchanged
        assert error.context == {"key": "value"}


class TestExceptionRaising:
    """Test raising and catching exceptions"""

    def test_raise_and_catch_base(self):
        """Test raising and catching base exception"""
        with pytest.raises(GenomeVaultError) as exc_info:
            raise GenomeVaultError("Test error", {"code": 500})

        assert exc_info.value.message == "Test error"
        assert exc_info.value.context == {"code": 500}

    def test_raise_and_catch_specific(self):
        """Test raising and catching specific exception"""
        with pytest.raises(ValidationError) as exc_info:
            raise ValidationError("Invalid input", {"field": "genome_data"})

        assert isinstance(exc_info.value, GenomeVaultError)
        assert exc_info.value.message == "Invalid input"
        assert exc_info.value.context["field"] == "genome_data"

    def test_catch_hierarchy(self):
        """Test catching exceptions through hierarchy"""
        # BindingError should be caught by HypervectorError handler
        try:
            raise BindingError("Binding failed")
        except HypervectorError as e:
            assert isinstance(e, BindingError)
            assert e.message == "Binding failed"

        # BindingError should also be caught by GenomeVaultError handler
        try:
            raise BindingError("Binding failed")
        except GenomeVaultError as e:
            assert isinstance(e, BindingError)
            assert e.message == "Binding failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
