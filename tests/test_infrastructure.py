"""
Test to verify the test infrastructure is working correctly
"""

import pytest


def test_test_infrastructure():
    """Verify test infrastructure is set up correctly"""
    assert True, "Basic test infrastructure is working"


def test_fixtures_available(sample_vcf_data, sample_clinical_data):
    """Verify test fixtures are available"""
    assert "variants" in sample_vcf_data
    assert "patient_id" in sample_clinical_data
    assert len(sample_vcf_data["variants"]) > 0


def test_temp_directory(test_data_dir):
    """Verify temporary test directory is created"""
    assert test_data_dir.exists()
    assert test_data_dir.is_dir()

    # Create a test file
    test_file = test_data_dir / "test.txt"
    test_file.write_text("test content")
    assert test_file.exists()


@pytest.mark.slow
def test_slow_marker():
    """Test that slow marker works"""
    import time

    time.sleep(0.1)
    assert True


@pytest.mark.unit
def test_unit_marker():
    """Test that unit marker works"""
    assert 1 + 1 == 2


@pytest.mark.parametrize(
    "input_val,expected",
    [
        (1, 2),
        (2, 4),
        (3, 6),
    ],
)
def test_parametrize(input_val, expected):
    """Test parametrize decorator works"""
    assert input_val * 2 == expected


def test_performance_benchmark(performance_benchmark):
    """Test performance benchmark fixture"""

    def sample_function(n):
        """Sample function.
        Args:        n: Number or count value.
        Returns:
            Result of the operation."""
        return sum(range(n))

    result = performance_benchmark.measure("sum_calculation", sample_function, 1000)

    assert result == 499500
    performance_benchmark.assert_performance("sum_calculation", 100)  # Should complete in 100ms
