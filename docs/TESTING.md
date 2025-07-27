# GenomeVault Testing Guide

## Overview

GenomeVault 3.0 includes a comprehensive testing framework to ensure code quality, security, and performance. This guide covers how to run tests, write new tests, and maintain test coverage.

## Quick Start

### Initial Setup

```bash
# Run the development setup script
./setup-dev.sh

# Or manually install dependencies
pip install -r requirements-dev.txt
pre-commit install
```

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
make coverage

# Run specific test suites
./run-tests.sh unit        # Unit tests only
./run-tests.sh integration # Integration tests only
./run-tests.sh security    # Security tests and scans
./run-tests.sh performance # Performance benchmarks

# Watch mode (auto-rerun on changes)
./run-tests.sh watch
```

## Test Organization

Tests are organized into the following categories:

```
tests/
├── unit/           # Fast, isolated unit tests
├── integration/    # Tests that require external services
├── performance/    # Performance benchmarks
├── security/       # Security-specific tests
├── fixtures/       # Test data and fixtures
└── conftest.py     # Shared pytest configuration
```

### Test Markers

Tests are marked with the following pytest markers:

- `@pytest.mark.unit` - Unit tests (default)
- `@pytest.mark.integration` - Integration tests requiring services
- `@pytest.mark.slow` - Tests that take > 1 second
- `@pytest.mark.security` - Security-related tests
- `@pytest.mark.performance` - Performance benchmarks

Run specific markers:
```bash
pytest -m "not slow"      # Skip slow tests
pytest -m integration     # Only integration tests
pytest -m "unit and not slow"  # Fast unit tests only
```

## Writing Tests

### Unit Test Example

```python
# tests/unit/test_example.py
import pytest
from genomevault.module import MyClass

class TestMyClass:
    """Test suite for MyClass"""

    @pytest.fixture
    def instance(self):
        """Create an instance for testing"""
        return MyClass()

    def test_method_behavior(self, instance):
        """Test specific method behavior"""
        result = instance.method(input_data)
        assert result == expected_output

    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_multiple_cases(self, instance, input, expected):
        """Test with multiple input cases"""
        assert instance.double(input) == expected
```

### Integration Test Example

```python
# tests/integration/test_database.py
import pytest
from genomevault.db import Database

@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations"""

    @pytest.fixture
    def db(self):
        """Create database connection"""
        db = Database(test_mode=True)
        yield db
        db.cleanup()

    def test_data_persistence(self, db):
        """Test that data persists correctly"""
        db.save("key", "value")
        assert db.get("key") == "value"
```

### Performance Test Example

```python
# tests/performance/test_encoding_perf.py
import pytest
from genomevault.hypervector import HypervectorEncoder

@pytest.mark.performance
def test_encoding_speed(benchmark):
    """Benchmark hypervector encoding"""
    encoder = HypervectorEncoder(dimensions=10000)
    data = np.random.randn(1000)

    # Benchmark the encoding operation
    result = benchmark(encoder.encode, data)

    # Assertions on the result
    assert result.shape == (10000,)

    # Performance assertions
    assert benchmark.stats['mean'] < 0.05  # < 50ms average
```

## Test Coverage

We maintain a minimum of 80% test coverage. Check coverage with:

```bash
# Generate coverage report
make coverage

# View HTML report
open htmlcov/index.html
```

### Coverage Configuration

Coverage is configured in `pyproject.toml`:
- Minimum coverage: 80%
- Branch coverage: enabled
- Excluded patterns: test files, migrations, type checking blocks

## Continuous Integration

Tests run automatically on every push and pull request via GitHub Actions.

### CI Pipeline Stages

1. **Linting & Code Quality**
   - Black formatting check
   - isort import sorting
   - Flake8 style violations
   - Pylint code quality

2. **Type Checking**
   - mypy static type analysis

3. **Unit Tests**
   - Run on Python 3.9, 3.10, 3.11
   - Coverage report uploaded to Codecov

4. **Security Scanning**
   - Bandit security scan
   - Safety dependency check
   - Semgrep analysis

5. **Integration Tests**
   - Tests with PostgreSQL and Redis

6. **Build & Package**
   - Build Python package
   - Build Docker images

7. **Smoke Tests**
   - Basic API health checks

## Testing Best Practices

### 1. Test Isolation
- Each test should be independent
- Use fixtures for setup/teardown
- Mock external dependencies

### 2. Clear Test Names
- Use descriptive test function names
- Follow pattern: `test_<what>_<condition>_<expected>`
- Example: `test_encoding_with_invalid_input_raises_error`

### 3. Arrange-Act-Assert
```python
def test_example():
    # Arrange
    data = prepare_test_data()
    instance = MyClass()

    # Act
    result = instance.process(data)

    # Assert
    assert result.status == 'success'
    assert len(result.items) == 3
```

### 4. Use Fixtures Effectively
```python
@pytest.fixture(scope='module')
def expensive_resource():
    """Create resource once per module"""
    resource = create_expensive_resource()
    yield resource
    resource.cleanup()
```

### 5. Parametrize Tests
```python
@pytest.mark.parametrize("tier,expected_size", [
    (CompressionTier.MINI, 25),
    (CompressionTier.CLINICAL, 300),
    (CompressionTier.FULL_HDC, 150)
])
def test_compression_sizes(tier, expected_size):
    compressor = Compressor(tier=tier)
    assert abs(compressor.estimate_size() - expected_size) < 10
```

## Security Testing

Security tests ensure our cryptographic operations and data handling are secure:

```bash
# Run security test suite
./run-tests.sh security

# Run bandit scan
bandit -r genomevault/

# Check dependencies
safety check
```

### Security Test Categories

1. **Cryptographic Tests**
   - Key generation randomness
   - Encryption/decryption correctness
   - Side-channel resistance

2. **Privacy Tests**
   - Data leakage prevention
   - Reconstruction attack resistance
   - Differential privacy verification

3. **Input Validation**
   - SQL injection prevention
   - XSS protection
   - Path traversal prevention

## Performance Testing

Performance tests ensure operations meet timing requirements:

```bash
# Run performance benchmarks
./run-tests.sh performance

# Run with detailed output
pytest tests/performance/ -v --benchmark-verbose
```

### Key Performance Requirements

- Hypervector encoding: < 50ms
- ZK proof generation: < 25ms for standard proofs
- PIR query: < 500ms round trip
- Compression: > 10:1 ratio

## Debugging Tests

### Verbose Output
```bash
pytest -vv tests/unit/test_failing.py
```

### Drop into debugger on failure
```bash
pytest --pdb tests/unit/test_failing.py
```

### Run specific test
```bash
pytest tests/unit/test_module.py::TestClass::test_method
```

### Show local variables on failure
```bash
pytest -l tests/
```

## Test Data Management

Test fixtures are stored in `tests/fixtures/`:
- `sample_variants.json` - Example genomic variants
- `test_keys/` - Test cryptographic keys (DO NOT use in production)
- `mock_data/` - Mock API responses

## Troubleshooting

### Common Issues

1. **Import errors**
   ```bash
   # Ensure genomevault is in PYTHONPATH
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Database connection errors**
   ```bash
   # Start test database
   docker-compose -f docker-compose.test.yml up -d
   ```

3. **Coverage not meeting threshold**
   ```bash
   # Find uncovered lines
   coverage report -m
   ```

4. **Slow tests**
   ```bash
   # Profile test execution
   pytest --durations=10
   ```

## Contributing Tests

When contributing to GenomeVault:

1. Write tests for new features
2. Ensure existing tests pass
3. Maintain or improve coverage
4. Follow the existing test patterns
5. Add appropriate markers
6. Document complex test scenarios

## Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing best practices](https://docs.python-guide.org/writing/tests/)
