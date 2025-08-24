# GenomeVault Integration Tests

Comprehensive integration tests for the GenomeVault API endpoints.

## Overview

This test suite validates the complete functionality of GenomeVault's privacy-preserving genomic computing platform, including:

- **HDC Encoding**: Hyperdimensional computing operations
- **ZK Proofs**: Zero-knowledge proof generation and verification
- **PIR Queries**: Private information retrieval
- **Error Handling**: Robust error responses
- **Rate Limiting**: API rate limiting with Redis
- **Database Persistence**: Data storage and retrieval

## Prerequisites

### Required Services

1. **PostgreSQL** (or SQLite for testing)
2. **Redis** (for rate limiting)
3. **Python 3.8+**

### Python Dependencies

```bash
pip install pytest pytest-asyncio pytest-cov pytest-timeout
pip install fastapi httpx sqlalchemy redis
pip install -e "../../[dev]"  # Install GenomeVault with dev dependencies
```

## Running Tests

### Run All Integration Tests

```bash
# From the tests/integration directory
pytest

# With coverage report
pytest --cov=genomevault.api --cov-report=html

# Run specific test class
pytest test_api_integration.py::TestHDCEncoding

# Run specific test
pytest test_api_integration.py::TestHDCEncoding::test_encode_variants_success
```

### Run by Category

```bash
# Fast tests only
pytest -m "not slow"

# Tests that don't require external services
pytest -m "not requires_redis and not requires_postgres"

# Async tests only
pytest -m "async"
```

### Run with Different Verbosity

```bash
# Quiet mode
pytest -q

# Verbose mode
pytest -v

# Very verbose (show all output)
pytest -vv

# Show print statements
pytest -s
```

## Test Structure

```
tests/integration/
├── __init__.py               # Package initialization
├── conftest.py               # Shared fixtures and configuration
├── pytest.ini                # Pytest configuration
├── test_api_integration.py  # Main integration tests
└── README.md                 # This file
```

## Test Classes

### TestHDCEncoding
Tests hyperdimensional computing operations:
- Variant encoding
- VCF file processing
- Encoding comparison
- Batch operations
- Round-trip verification

### TestZKProofs
Tests zero-knowledge proof system:
- Proof generation
- Proof verification
- Circuit listing
- Batch verification
- Error handling

### TestPIRQueries
Tests private information retrieval:
- Database setup
- Query execution
- Byzantine fault tolerance
- Status monitoring

### TestErrorHandling
Tests API error responses:
- Malformed requests
- Validation errors
- Server errors
- Method not allowed
- Not found errors

### TestRateLimiting
Tests rate limiting with Redis:
- Per-minute limits
- Rate limit headers
- Burst handling
- IP-based limiting
- Exempted endpoints

### TestDatabasePersistence
Tests data persistence:
- Encoding storage
- Proof storage
- Transaction rollback
- Concurrent access
- Index performance

### TestIntegrationFlows
Tests complete workflows:
- HDC encoding workflow
- ZK proof workflow
- PIR query workflow

## Fixtures

Key fixtures provided by `conftest.py`:

- `test_db`: In-memory SQLite database for testing
- `test_client`: FastAPI test client
- `async_client`: Async HTTP client
- `mock_redis`: Mocked Redis client
- `sample_genomic_data`: Sample variant data
- `vcf_content`: Sample VCF file content
- `mock_zk_engine`: Mocked ZK engine
- `mock_pir_client`: Mocked PIR client

## Environment Variables

Set these for testing:

```bash
export TESTING=true
export DATABASE_URL=sqlite:///:memory:
export REDIS_URL=redis://localhost:6379/15
```

## Coverage Reports

After running tests with coverage:

1. **Terminal Report**: Shown automatically
2. **HTML Report**: Open `htmlcov/index.html` in browser
3. **XML Report**: `coverage.xml` for CI/CD integration

## Continuous Integration

Example GitHub Actions workflow:

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    services:
      postgres:
        image: postgres:14
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
        pip install pytest-cov

    - name: Run integration tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost/test
        REDIS_URL: redis://localhost:6379
      run: |
        cd tests/integration
        pytest --cov=genomevault.api --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./tests/integration/coverage.xml
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure GenomeVault is installed
   pip install -e ../..
   ```

2. **Database Connection Errors**
   ```bash
   # Use in-memory SQLite for testing
   export DATABASE_URL=sqlite:///:memory:
   ```

3. **Redis Connection Errors**
   ```bash
   # Start Redis or use mock
   docker run -d -p 6379:6379 redis:7
   ```

4. **Async Test Failures**
   ```bash
   # Ensure pytest-asyncio is installed
   pip install pytest-asyncio
   ```

## Best Practices

1. **Isolation**: Each test should be independent
2. **Fixtures**: Use fixtures for common setup
3. **Mocking**: Mock external services when appropriate
4. **Assertions**: Use specific assertions with meaningful messages
5. **Cleanup**: Ensure proper cleanup in fixtures
6. **Documentation**: Document complex test logic

## Contributing

When adding new tests:

1. Follow existing patterns
2. Use appropriate markers
3. Add docstrings to test methods
4. Update this README if needed
5. Ensure tests are deterministic
6. Mock external dependencies

## License

Same as GenomeVault project.
