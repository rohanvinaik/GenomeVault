"""Basic smoke test to verify test infrastructure is working"""
import pytest

def test_pytest_is_working():
    """Verify pytest is installed and working"""
    assert True

def test_imports_work():
    """Verify basic imports work"""
    import numpy as np
    import cryptography
    import fastapi
    assert True

@pytest.mark.unit
def test_unit_marker():
    """Test that unit marker works"""
    assert 1 + 1 == 2

@pytest.mark.integration
def test_integration_marker():
    """Test that integration marker works"""
    # This would normally connect to a service
    assert True

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_parametrize_works(input, expected):
    """Test that parametrize works"""
    assert input * 2 == expected
