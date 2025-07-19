# GenomeVault Test Suite

This directory contains the comprehensive test suite for GenomeVault 3.0.

## Test Structure

```
tests/
├── conftest.py          # Shared pytest fixtures and configuration
├── fixtures/            # Test data files
│   └── sample_variants.json
├── unit/               # Unit tests for individual components
│   ├── test_compression.py
│   ├── test_hypervector.py
│   └── ...
├── integration/        # Integration tests
│   └── test_hipaa_governance.py
├── performance/        # Performance benchmarks
└── security/          # Security-focused tests
```

## Running Tests

### Quick Start

```bash
# Run all tests
make test

# Run with coverage
make coverage

# Run specific test categories
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

```python
def test_<component>_<action>_<expected_result>():
    """Test description"""
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
