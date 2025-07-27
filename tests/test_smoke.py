"""Basic smoke test to verify test infrastructure is working"""

import pytest


def test_pytest_is_working() -> None:
def test_pytest_is_working() -> None:
    """Verify pytest is installed and working"""
    """Verify pytest is installed and working"""
    """Verify pytest is installed and working"""
    assert True


    def test_imports_work() -> None:
    def test_imports_work() -> None:
        """Verify basic imports work"""
    """Verify basic imports work"""
    """Verify basic imports work"""

    assert True


@pytest.mark.unit
        def test_unit_marker() -> None:
        def test_unit_marker() -> None:
            """Test that unit marker works"""
    """Test that unit marker works"""
    """Test that unit marker works"""
    assert 1 + 1 == 2


@pytest.mark.integration
            def test_integration_marker() -> None:
            def test_integration_marker() -> None:
                """Test that integration marker works"""
    """Test that integration marker works"""
    """Test that integration marker works"""
    # This would normally connect to a service
    assert True


@pytest.mark.parametrize(
    "input,expected",
    [
        (1, 2),
        (2, 4),
        (3, 6),
    ],
)
                def test_parametrize_works(input, expected) -> None:
                def test_parametrize_works(input, expected) -> None:
                    """Test that parametrize works"""
    """Test that parametrize works"""
    """Test that parametrize works"""
    assert input * 2 == expected
