# tests/conftest.py
"""
Shared pytest fixtures and configuration for GenomeVault tests
"""

import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

# Set test environment
os.environ["GENOMEVAULT_ENV"] = "test"
os.environ["LOG_LEVEL"] = "DEBUG"


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="genomevault_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_vcf_data():
    """Generate sample VCF-like genomic data"""
    return {
        "header": {
            "fileformat": "VCFv4.3",
            "reference": "GRCh38",
            "date": datetime.now().isoformat(),
            "source": "GenomeVault-Test",
        },
        "variants": [
            {
                "chrom": "1",
                "pos": 14370,
                "id": "rs6054257",
                "re": "G",
                "alt": "A",
                "qual": 29,
                "filter": "PASS",
                "info": "NS=3;DP=14;AF=0.5",
                "format": "GT:GQ:DP",
                "sample": "0/1:48:14",
            },
            {
                "chrom": "2",
                "pos": 17330,
                "id": "rs1234567",
                "re": "T",
                "alt": "C",
                "qual": 99,
                "filter": "PASS",
                "info": "NS=3;DP=20;AF=0.75",
                "format": "GT:GQ:DP",
                "sample": "1/1:99:20",
            },
        ],
    }


@pytest.fixture
def sample_clinical_data():
    """Generate sample clinical/phenotypic data"""
    return {
        "patient_id": "TEST001",
        "demographics": {"age": 45, "sex": "F", "ethnicity": "European"},
        "clinical_features": {
            "height": 165,
            "weight": 70,
            "bmi": 25.7,
            "blood_pressure": {"systolic": 120, "diastolic": 80},
            "glucose": 95,
            "hba1c": 5.4,
        },
        "medications": [
            {"name": "metformin", "dose": "500mg", "frequency": "BID"},
            {"name": "lisinopril", "dose": "10mg", "frequency": "daily"},
        ],
        "conditions": ["T2DM", "HTN"],
    }


@pytest.fixture
def mock_pir_servers():
    """Mock PIR server configuration"""
    return [
        {
            "id": "pir-server-1",
            "url": "https://pir1.genomevault.test",
            "type": "LN",  # Light Node
            "location": "us-east-1",
            "honesty_probability": 0.95,
        },
        {
            "id": "pir-server-2",
            "url": "https://pir2.genomevault.test",
            "type": "LN",
            "location": "eu-west-1",
            "honesty_probability": 0.95,
        },
        {
            "id": "pir-server-3",
            "url": "https://pir3.genomevault.test",
            "type": "TS",  # Trusted Signatory
            "location": "us-west-2",
            "honesty_probability": 0.98,
            "hipaa_certified": True,
        },
        {
            "id": "pir-server-4",
            "url": "https://pir4.genomevault.test",
            "type": "TS",
            "location": "ap-southeast-1",
            "honesty_probability": 0.98,
            "hipaa_certified": True,
        },
    ]


@pytest.fixture
def mock_blockchain_node():
    """Mock blockchain node for testing"""
    node = Mock()
    node.get_block_height.return_value = 12345
    node.get_block_by_height.return_value = {
        "height": 12345,
        "hash": "0x" + "a" * 64,
        "timestamp": datetime.now().timestamp(),
        "transactions": [],
    }
    node.submit_transaction.return_value = {
        "tx_hash": "0x" + "b" * 64,
        "status": "pending",
    }
    return node


@pytest.fixture
def zk_test_vectors():
    """Test vectors for zero-knowledge proofs"""
    return {
        "variant_verification": {
            "public_inputs": {
                "variant_hash": "0x" + "c" * 64,
                "reference_hash": "0x" + "d" * 64,
                "commitment_root": "0x" + "e" * 64,
            },
            "private_inputs": {
                "variant_data": {"chr": "1", "pos": 14370, "re": "G", "alt": "A"},
                "merkle_proo": ["0x" + "f" * 64] * 20,
                "witness_randomness": np.random.bytes(32).hex(),
            },
            "expected_proof_size": 384,
            "expected_verification_time_ms": 25,
        },
        "diabetes_risk": {
            "public_inputs": {
                "g_threshold": 100,
                "r_threshold": 0.15,
                "result_commitment": "0x" + "a" * 64,
            },
            "private_inputs": {
                "glucose_reading": 105,
                "risk_score": 0.18,
                "witness_randomness": np.random.bytes(32).hex(),
            },
            "expected_proof_size": 384,
            "expected_verification_time_ms": 25,
        },
    }


@pytest.fixture
def performance_benchmark():
    """Performance benchmarking utility"""

    class PerformanceBenchmark:
        def __init__(self):
            self.results = {}

        def measure(self, name: str, func, *args, **kwargs):
            import time

            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            self.results[name] = {
                "elapsed_seconds": elapsed,
                "elapsed_ms": elapsed * 1000,
            }
            return result

        def assert_performance(self, name: str, max_ms: float):
            if name not in self.results:
                pytest.fail("No benchmark results for {name}")
            actual_ms = self.results[name]["elapsed_ms"]
            assert actual_ms <= max_ms, "{name} took {actual_ms:.1f}ms, expected <= {max_ms}ms"

    return PerformanceBenchmark()


@pytest.fixture
def cleanup_test_files():
    """Cleanup any test files created during tests"""
    files_to_cleanup = []

    def _register(filepath):
        files_to_cleanup.append(filepath)

    yield _register

    # Cleanup after test
    for filepath in files_to_cleanup:
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
        except Exception:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            pass
            raise


# Pytest hooks
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "slow: marks test as slow")
    config.addinivalue_line("markers", "integration: marks test as integration test")
    config.addinivalue_line("markers", "security: marks test as security-related")
    config.addinivalue_line("markers", "performance: marks test as performance benchmark")


def pytest_collection_modifyitems(config, items):
    """Add markers to tests based on their location"""
    for item in items:
        # Add markers based on test file location
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        else:
            item.add_marker(pytest.mark.unit)
