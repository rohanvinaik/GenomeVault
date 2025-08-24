"""
Shared fixtures and configuration for integration tests.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Generator, AsyncGenerator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Add project root to path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from genomevault.api.app import app


# Configure test environment
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["REDIS_URL"] = "redis://localhost:6379/15"  # Use test DB 15


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def test_db_engine():
    """Create test database engine."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Import all models to ensure tables are created
    from genomevault.api.routers.hdc import Base as HDCBase
    from genomevault.api.routers.zk import Base as ZKBase

    # Create all tables
    HDCBase.metadata.create_all(bind=engine)
    ZKBase.metadata.create_all(bind=engine)

    yield engine

    # Clean up
    HDCBase.metadata.drop_all(bind=engine)
    ZKBase.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def test_db(test_db_engine) -> Generator[Session, None, None]:
    """Create test database session."""
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_db_engine)

    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="function")
def test_client() -> TestClient:
    """Create FastAPI test client."""
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create async HTTP client for testing."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="function")
def mock_redis() -> MagicMock:
    """Mock Redis client for testing."""
    mock = MagicMock()

    # Mock common Redis operations
    mock.get = MagicMock(return_value=None)
    mock.set = MagicMock(return_value=True)
    mock.setex = MagicMock(return_value=True)
    mock.incr = MagicMock(return_value=1)
    mock.decr = MagicMock(return_value=0)
    mock.expire = MagicMock(return_value=True)
    mock.ttl = MagicMock(return_value=60)
    mock.delete = MagicMock(return_value=1)
    mock.exists = MagicMock(return_value=0)
    mock.pipeline = MagicMock(return_value=mock)
    mock.execute = MagicMock(return_value=[True, True])

    return mock


@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_genomic_data():
    """Generate sample genomic data for testing."""
    return {
        "variants": [
            {
                "chromosome": "1",
                "position": 12345,
                "ref": "A",
                "alt": "G",
                "quality": 30.0,
                "gene": "GENE1",
                "impact": "MODERATE",
            },
            {
                "chromosome": "2",
                "position": 67890,
                "ref": "T",
                "alt": "C",
                "quality": 40.0,
                "gene": "GENE2",
                "impact": "HIGH",
            },
            {
                "chromosome": "X",
                "position": 11111,
                "ref": "G",
                "alt": "A",
                "quality": 35.0,
                "gene": "GENE3",
                "impact": "LOW",
            },
            {
                "chromosome": "7",
                "position": 117559590,
                "ref": "CTT",
                "alt": "C",
                "quality": 99.0,
                "gene": "CFTR",
                "impact": "HIGH",
            },
        ],
        "samples": [
            {
                "sample_id": "SAMPLE_001",
                "patient_id": "PATIENT_001",
                "tissue_type": "Blood",
                "sequencing_platform": "Illumina",
            },
            {
                "sample_id": "SAMPLE_002",
                "patient_id": "PATIENT_002",
                "tissue_type": "Tumor",
                "sequencing_platform": "Nanopore",
            },
        ],
    }


@pytest.fixture
def vcf_content():
    """Generate sample VCF file content."""
    return """##fileformat=VCFv4.3
##fileDate=20240101
##source=GenomeVault_Test
##reference=GRCh38
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##INFO=<ID=GENE,Number=1,Type=String,Description="Gene Name">
##INFO=<ID=IMPACT,Number=1,Type=String,Description="Variant Impact">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">
##contig=<ID=1>
##contig=<ID=2>
##contig=<ID=X>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE001
1	12345	rs123	A	G	30	PASS	AF=0.5;DP=50;GENE=GENE1;IMPACT=MODERATE	GT:GQ:DP	0/1:30:50
2	67890	.	T	C	40	PASS	AF=0.3;DP=60;GENE=GENE2;IMPACT=HIGH	GT:GQ:DP	0/1:40:60
X	11111	rs456	G	A	35	PASS	AF=0.4;DP=45;GENE=GENE3;IMPACT=LOW	GT:GQ:DP	1/1:35:45
"""


@pytest.fixture
def mock_zk_engine():
    """Mock ZK engine for testing."""
    mock_engine = MagicMock()

    # Mock proof generation
    mock_proof = {
        "proof": {
            "pi_a": ["0x123", "0x456"],
            "pi_b": [["0x789", "0xabc"], ["0xdef", "0x012"]],
            "pi_c": ["0x345", "0x678"],
            "protocol": "groth16",
            "curve": "bn128",
        },
        "public": {"c": 42},
        "circuit_type": "sum64",
    }

    mock_engine.generate_proof = MagicMock(return_value=mock_proof)
    mock_engine.verify_proof = MagicMock(return_value=True)
    mock_engine.toolchain_available = True

    return mock_engine


@pytest.fixture
def mock_pir_client():
    """Mock PIR client for testing."""
    mock_client = MagicMock()

    mock_client.database_size = 100
    mock_client.generate_it_pir_query = MagicMock(
        return_value={"queries": [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1], [1, 1, 1, 1, 1]], "index": 42}
    )

    mock_client.retrieve = MagicMock(
        return_value={"result": b"test_data", "success": True, "server_responses": 3}
    )

    return mock_client


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # Reset ZK engine singleton
    import genomevault.api.routers.zk as zk_module

    zk_module._zk_engine = None

    yield

    # Clean up after test
    zk_module._zk_engine = None


@pytest.fixture
def api_headers():
    """Common API headers for testing."""
    return {
        "Content-Type": "application/json",
        "User-Agent": "GenomeVault-Test/1.0",
        "X-Request-ID": "test-request-123",
    }


@pytest.fixture
def auth_headers():
    """Authentication headers for protected endpoints."""
    # In a real app, this would include JWT tokens or API keys
    return {"Authorization": "Bearer test-token-123", "X-API-Key": "test-api-key"}


# Markers for different test types
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "requires_redis: marks tests that require Redis")
    config.addinivalue_line("markers", "requires_postgres: marks tests that require PostgreSQL")
