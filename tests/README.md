# GenomeVault Test Suite

This directory contains the comprehensive test suite for GenomeVault 3.0.

## Test Structure

```
tests/
├── conftest.py          # Shared pytest fixtures and configuration
├── fixtures/            # Test data files
│   └── sample_variants.json
├── unit/               # Unit tests for individual components
│   ├── test_zk_*.py    # ZK-related unit tests
│   ├── test_pir_*.py   # PIR-related unit tests
│   ├── test_hdc_*.py   # HDC-related unit tests
│   └── ...
├── integration/        # Integration tests
│   └── test_hipaa_governance.py
├── e2e/               # End-to-end tests
│   ├── test_pir_e2e.py
│   └── test_zk_e2e.py
├── property/          # Property-based tests (Hypothesis)
│   ├── test_hdc_properties.py
│   └── test_zk_properties.py
├── adversarial/       # Adversarial/security tests
│   ├── test_hdc_adversarial.py
│   ├── test_pir_adversarial.py
│   └── test_zk_adversarial.py
├── pir/               # PIR-specific tests
│   └── test_pir_protocol.py
└── zk/                # ZK-specific tests
    └── test_zk_property_circuits.py
```

## Running Tests

### Quick Start

```bash
# Run all tests
make test

# Run with coverage
make coverage

# Run lane-specific tests
make test-zk      # Run all ZK tests
make test-pir     # Run all PIR tests
make test-hdc     # Run all HDC tests

# Run test categories
make test-unit
make test-integration
make test-e2e
make test-property
make test-adversarial

# Run specific test categories (alternative)
./run-tests.sh unit
./run-tests.sh integration
./run-tests.sh security
./run-tests.sh performance

# Watch mode (re-runs on file changes)
./run-tests.sh watch
```

### Detailed Commands

```bash
# Run only unit tests
pytest tests/unit/ -v

# Run tests with coverage report
pytest tests/ --cov=genomevault --cov-report=html

# Run tests in parallel
pytest tests/ -n auto

# Run tests with specific markers
pytest tests/ -m "not slow"
pytest tests/ -m integration

# Run with maximum verbosity
pytest tests/ -vvv

# Stop on first failure
pytest tests/ -x
```

## Test Categories

### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Fast execution (<1s per test)
- Located in `tests/unit/`

### Integration Tests
- Test component interactions
- May use real databases/services
- Slower execution (1-10s per test)
- Located in `tests/integration/`
- Marked with `@pytest.mark.integration`

### Security Tests
- Test security controls and crypto operations
- Verify privacy guarantees
- Located in `tests/security/`
- Marked with `@pytest.mark.security`

### Performance Tests
- Benchmark critical operations
- Verify performance requirements
- Located in `tests/performance/`
- Marked with `@pytest.mark.performance`

## Writing Tests

### Test Naming Convention

#### Lane-Specific Naming
All test files should be prefixed with their lane identifier:
- **ZK tests**: `test_zk_*.py`
- **PIR tests**: `test_pir_*.py`
- **HDC tests**: `test_hdc_*.py`

#### Function Naming
```python
def test_<component>_<action>_<expected_result>():
    """Test description"""
    pass
```

Example:
```python
# In test_pir_protocol.py
def test_pir_query_generation_constant_time():
    """Test that PIR query generation runs in constant time"""
    pass

# In test_hdc_encoding.py
def test_hdc_encode_variant_preserves_similarity():
    """Test that HDC encoding preserves variant similarity"""
    pass
```

### Using Fixtures

```python
def test_compression_mini_tier(sample_snp_data):
    """Test mini tier compression achieves target size"""
    compressor = SNPCompressor(tier=CompressionTier.MINI)
    compressed = compressor.compress(sample_snp_data['mini'])
    assert 22.5 <= len(compressed)/1024 <= 27.5
```

### Performance Assertions

```python
def test_encoding_performance(performance_benchmark, sample_data):
    """Test encoding completes within 50ms"""
    result = performance_benchmark.measure(
        'encoding',
        encoder.encode,
        sample_data
    )
    performance_benchmark.assert_performance('encoding', 50)
```

## Coverage Requirements

- Minimum overall coverage: 80%
- Critical components (crypto, privacy): 95%
- New code must include tests
- Coverage reports in `htmlcov/`

## CI Integration

Tests run automatically on:
- Push to main/develop branches
- Pull requests
- Nightly builds

See `.github/workflows/ci.yml` for CI configuration.

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're in the virtual environment
   ```bash
   source .venv/bin/activate
   ```

2. **Database connection errors**: Start test services
   ```bash
   make docker-up
   ```

3. **Slow tests**: Run without slow tests
   ```bash
   pytest tests/ -m "not slow"
   ```

4. **Coverage not generating**: Install coverage dependencies
   ```bash
   pip install pytest-cov
   ```

### Debug Mode

Run tests with detailed output:
```bash
pytest tests/ -vvv --tb=long --capture=no
```

## Contributing

1. Write tests for new features
2. Ensure all tests pass
3. Maintain or improve coverage
4. Follow naming conventions
5. Add appropriate markers
