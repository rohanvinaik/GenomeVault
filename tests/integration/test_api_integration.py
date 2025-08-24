#!/usr/bin/env python3
"""
Integration tests for GenomeVault API endpoints.

Tests the complete flow of HDC encoding, ZK proofs, PIR queries,
error handling, rate limiting, and database persistence.
"""

import asyncio
import base64
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from fastapi import FastAPI
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
from genomevault.api.routers.hdc import HDCEncoding, Base as HDCBase
from genomevault.api.routers.zk import ZKProofRecord, Base as ZKBase
from genomevault.core.constants import OmicsType
from genomevault.hypervector.featurizers.variants import variant_to_numeric
from genomevault.zk.real_engine import RealZKEngine


# Test database URL (in-memory SQLite for tests)
TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
def test_db():
    """Create a test database session."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    # Create all tables
    HDCBase.metadata.create_all(bind=engine)
    ZKBase.metadata.create_all(bind=engine)

    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        # Clean up tables after test
        HDCBase.metadata.drop_all(bind=engine)
        ZKBase.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client():
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="function")
def mock_redis():
    """Mock Redis client for rate limiting tests."""
    mock = MagicMock()
    mock.get = MagicMock(return_value=None)
    mock.setex = MagicMock(return_value=True)
    mock.incr = MagicMock(return_value=1)
    mock.expire = MagicMock(return_value=True)
    mock.ttl = MagicMock(return_value=60)
    return mock


@pytest.fixture
def sample_variants():
    """Generate sample variant data for testing."""
    return [
        {"chromosome": "1", "position": 12345, "ref": "A", "alt": "G", "quality": 30.0},
        {"chromosome": "2", "position": 67890, "ref": "T", "alt": "C", "quality": 40.0},
        {"chromosome": "X", "position": 11111, "ref": "G", "alt": "A", "quality": 35.0},
    ]


@pytest.fixture
def sample_vcf():
    """Generate sample VCF file content."""
    vcf_content = """##fileformat=VCFv4.3
##fileDate=20240101
##source=GenomeVault_Test
##reference=GRCh38
##contig=<ID=1>
##contig=<ID=2>
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
1\t12345\trs123\tA\tG\t30\tPASS\tAF=0.5
2\t67890\t.\tT\tC\t40\tPASS\tAF=0.3
"""
    return vcf_content.encode()


class TestHDCEncoding:
    """Test HDC encoding functionality."""

    def test_encode_variants_success(self, test_client, sample_variants):
        """Test successful variant encoding."""
        response = test_client.post(
            "/api/hdc/encode",
            json={"variants": sample_variants, "dimension": 10000, "normalize": True},
        )

        assert response.status_code == 200
        data = response.json()

        assert "encoding_id" in data
        assert data["dimension"] == 10000
        assert data["variant_count"] == len(sample_variants)
        assert "encoding" in data
        assert "checksum" in data
        assert "created_at" in data

    def test_encode_vcf_file(self, test_client, sample_vcf):
        """Test encoding from VCF file upload."""
        files = {"file": ("test.vcf", sample_vcf, "text/plain")}
        data = {"dimension": 5000, "normalize": True}

        response = test_client.post("/api/hdc/encode-vcf", files=files, data=data)

        assert response.status_code == 200
        result = response.json()

        assert result["dimension"] == 5000
        assert result["variant_count"] == 2  # Based on sample VCF
        assert "encoding_id" in result

    def test_encoding_round_trip(self, test_client, sample_variants):
        """Test encoding and retrieval round-trip."""
        # Create encoding
        create_response = test_client.post(
            "/api/hdc/encode",
            json={"variants": sample_variants, "dimension": 10000, "normalize": True},
        )

        assert create_response.status_code == 200
        encoding_id = create_response.json()["encoding_id"]

        # Retrieve encoding
        get_response = test_client.get(f"/api/hdc/{encoding_id}")

        assert get_response.status_code == 200
        data = get_response.json()

        assert data["encoding_id"] == encoding_id
        assert data["dimension"] == 10000
        assert data["variant_count"] == len(sample_variants)

    def test_compare_encodings(self, test_client, sample_variants):
        """Test comparing two encodings."""
        # Create first encoding
        response1 = test_client.post(
            "/api/hdc/encode", json={"variants": sample_variants[:2], "dimension": 10000}
        )
        encoding_id_1 = response1.json()["encoding_id"]

        # Create second encoding
        response2 = test_client.post(
            "/api/hdc/encode", json={"variants": sample_variants[1:], "dimension": 10000}
        )
        encoding_id_2 = response2.json()["encoding_id"]

        # Compare encodings
        compare_response = test_client.post(
            "/api/hdc/compare",
            json={
                "encoding_id_1": encoding_id_1,
                "encoding_id_2": encoding_id_2,
                "metric": "hamming",
            },
        )

        assert compare_response.status_code == 200
        data = compare_response.json()

        assert data["encoding_id_1"] == encoding_id_1
        assert data["encoding_id_2"] == encoding_id_2
        assert 0 <= data["similarity"] <= 1
        assert data["distance"] >= 0
        assert data["metric"] == "hamming"

    def test_batch_compare(self, test_client, sample_variants):
        """Test batch comparison of encodings."""
        # Create multiple encodings
        encoding_ids = []
        for i in range(3):
            response = test_client.post(
                "/api/hdc/encode", json={"variants": [sample_variants[i]], "dimension": 5000}
            )
            encoding_ids.append(response.json()["encoding_id"])

        # Batch compare
        response = test_client.post(
            "/api/hdc/batch-compare",
            params={"encoding_id": encoding_ids[0], "metric": "cosine"},
            json=encoding_ids[1:],
        )

        assert response.status_code == 200
        data = response.json()

        assert data["base_encoding_id"] == encoding_ids[0]
        assert data["metric"] == "cosine"
        assert len(data["comparisons"]) == 2

    def test_invalid_chromosome(self, test_client):
        """Test encoding with invalid chromosome."""
        response = test_client.post(
            "/api/hdc/encode",
            json={
                "variants": [
                    {
                        "chromosome": "99",  # Invalid
                        "position": 12345,
                        "ref": "A",
                        "alt": "G",
                    }
                ]
            },
        )

        assert response.status_code == 422  # Validation error

    def test_encoding_not_found(self, test_client):
        """Test retrieving non-existent encoding."""
        response = test_client.get("/api/hdc/nonexistent_id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


class TestZKProofs:
    """Test Zero-Knowledge proof functionality."""

    def test_generate_proof_sum64(self, test_client):
        """Test generating a sum64 proof."""
        response = test_client.post(
            "/api/zk/prove",
            json={
                "circuit_name": "sum64",
                "inputs": [
                    {"name": "a", "value": 15, "is_public": False},
                    {"name": "b", "value": 27, "is_public": False},
                    {"name": "c", "value": 42, "is_public": True},
                ],
                "store_proof": True,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "proof_id" in data
        assert data["circuit_name"] == "sum64"
        assert "proof" in data
        assert data["public_inputs"]["c"] == 42
        assert data["stored"] is True

    def test_verify_proof(self, test_client):
        """Test verifying a proof."""
        # Generate proof
        generate_response = test_client.post(
            "/api/zk/prove",
            json={
                "circuit_name": "sum64",
                "inputs": [
                    {"name": "a", "value": 10, "is_public": False},
                    {"name": "b", "value": 20, "is_public": False},
                    {"name": "c", "value": 30, "is_public": True},
                ],
            },
        )

        assert generate_response.status_code == 200
        proof_data = generate_response.json()

        # Verify proof
        verify_response = test_client.post(
            "/api/zk/verify",
            json={
                "proof": proof_data["proof"],
                "public_inputs": proof_data["public_inputs"],
                "circuit_name": "sum64",
            },
        )

        assert verify_response.status_code == 200
        verify_data = verify_response.json()

        assert verify_data["valid"] is True
        assert verify_data["circuit_name"] == "sum64"
        assert "verification_time_ms" in verify_data

    def test_list_circuits(self, test_client):
        """Test listing available circuits."""
        response = test_client.get("/api/zk/circuits")

        assert response.status_code == 200
        circuits = response.json()

        assert isinstance(circuits, list)
        assert len(circuits) > 0

        # Check sum64 circuit exists
        sum64 = next((c for c in circuits if c["name"] == "sum64"), None)
        assert sum64 is not None
        assert sum64["type"] == "arithmetic"
        assert sum64["supported"] is True

    def test_get_circuit_info(self, test_client):
        """Test getting specific circuit information."""
        response = test_client.get("/api/zk/circuits/sum64")

        assert response.status_code == 200
        data = response.json()

        assert data["name"] == "sum64"
        assert data["type"] == "arithmetic"
        assert len(data["required_inputs"]) == 3
        assert data["supported"] is True

    def test_batch_verify_proofs(self, test_client):
        """Test batch verification of proofs."""
        # Generate multiple proofs
        proofs = []
        for i in range(3):
            response = test_client.post(
                "/api/zk/prove",
                json={
                    "circuit_name": "sum64",
                    "inputs": [
                        {"name": "a", "value": i, "is_public": False},
                        {"name": "b", "value": i * 2, "is_public": False},
                        {"name": "c", "value": i * 3, "is_public": True},
                    ],
                },
            )
            proof_data = response.json()
            proofs.append(
                {
                    "proof": proof_data["proof"],
                    "public_inputs": proof_data["public_inputs"],
                    "circuit_name": "sum64",
                }
            )

        # Batch verify
        response = test_client.post("/api/zk/batch-verify", json=proofs)

        assert response.status_code == 200
        data = response.json()

        assert data["total_proofs"] == 3
        assert data["valid_count"] == 3
        assert data["invalid_count"] == 0
        assert len(data["results"]) == 3

    def test_proof_not_found(self, test_client):
        """Test retrieving non-existent proof."""
        response = test_client.get("/api/zk/proofs/nonexistent_proof")

        assert response.status_code == 404

    def test_invalid_circuit(self, test_client):
        """Test using invalid circuit name."""
        response = test_client.post(
            "/api/zk/prove",
            json={
                "circuit_name": "invalid_circuit",
                "inputs": [{"name": "x", "value": 1, "is_public": True}],
            },
        )

        assert response.status_code == 400
        assert "Unknown circuit" in response.json()["detail"]


class TestPIRQueries:
    """Test Private Information Retrieval functionality."""

    @pytest.mark.asyncio
    async def test_pir_setup(self, async_client):
        """Test PIR database setup."""
        response = await async_client.post(
            "/api/pir/setup",
            json={"dataset_size": 100, "element_size": 1024, "metadata": {"test": True}},
        )

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "ready"
        assert data["dataset_size"] == 100
        assert data["element_size"] == 1024
        assert "setup_id" in data

    @pytest.mark.asyncio
    async def test_pir_query(self, async_client):
        """Test executing a PIR query."""
        # Setup database first
        setup_response = await async_client.post(
            "/api/pir/setup", json={"dataset_size": 50, "element_size": 512}
        )

        setup_data = setup_response.json()

        # Execute query
        query_response = await async_client.post(
            "/api/pir/query",
            json={"index": 10, "dataset_size": 50, "setup_id": setup_data.get("setup_id")},
        )

        assert query_response.status_code == 200
        query_data = query_response.json()

        assert "result" in query_data
        assert query_data["index"] == 10
        assert query_data["success"] is True
        assert "retrieval_time_ms" in query_data

    @pytest.mark.asyncio
    async def test_pir_status(self, async_client):
        """Test getting PIR system status."""
        response = await async_client.get("/api/pir/status")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "servers_available" in data
        assert "byzantine_threshold" in data
        assert isinstance(data["statistics"], dict)

    def test_pir_byzantine_detection(self, test_client):
        """Test Byzantine fault detection in PIR."""
        # This would require mocking multiple servers
        # For now, test the endpoint exists
        response = test_client.get("/api/pir/status")

        assert response.status_code == 200
        data = response.json()
        assert "byzantine_threshold" in data
        assert data["byzantine_threshold"] == 1  # Default threshold

    def test_pir_invalid_index(self, test_client):
        """Test PIR query with invalid index."""
        response = test_client.post(
            "/api/pir/query",
            json={
                "index": -1,  # Invalid
                "dataset_size": 100,
            },
        )

        assert response.status_code == 422  # Validation error


class TestErrorHandling:
    """Test API error handling."""

    def test_malformed_json(self, test_client):
        """Test handling of malformed JSON."""
        response = test_client.post(
            "/api/hdc/encode", data="not json", headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_missing_required_fields(self, test_client):
        """Test handling of missing required fields."""
        response = test_client.post(
            "/api/hdc/encode",
            json={},  # Missing required 'variants' field
        )

        assert response.status_code == 422
        error = response.json()
        assert "variants" in str(error["detail"])

    def test_invalid_field_types(self, test_client):
        """Test handling of invalid field types."""
        response = test_client.post(
            "/api/hdc/encode",
            json={
                "variants": "not_a_list",  # Should be a list
                "dimension": "not_a_number",  # Should be an integer
            },
        )

        assert response.status_code == 422

    def test_out_of_range_values(self, test_client):
        """Test handling of out-of-range values."""
        response = test_client.post(
            "/api/hdc/encode",
            json={
                "variants": [
                    {
                        "chromosome": "1",
                        "position": -1,  # Invalid negative position
                        "ref": "A",
                        "alt": "G",
                    }
                ],
                "dimension": 100,  # Too small (min is 1000)
            },
        )

        assert response.status_code == 422

    def test_server_error_handling(self, test_client):
        """Test handling of internal server errors."""
        # Mock a database error
        with patch("genomevault.api.routers.hdc.get_db") as mock_db:
            mock_db.side_effect = Exception("Database connection failed")

            response = test_client.post(
                "/api/hdc/encode",
                json={"variants": [{"chromosome": "1", "position": 12345, "ref": "A", "alt": "G"}]},
            )

            assert response.status_code == 500

    def test_method_not_allowed(self, test_client):
        """Test handling of invalid HTTP methods."""
        response = test_client.get("/api/hdc/encode")  # Should be POST

        assert response.status_code == 405

    def test_endpoint_not_found(self, test_client):
        """Test handling of non-existent endpoints."""
        response = test_client.get("/api/nonexistent/endpoint")

        assert response.status_code == 404


class TestRateLimiting:
    """Test rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_rate_limit_per_minute(self, async_client, mock_redis):
        """Test per-minute rate limiting."""
        with patch("genomevault.api.middleware.rate_limiter.redis_client", mock_redis):
            # Simulate hitting rate limit
            mock_redis.incr.return_value = 61  # Over the 60/min limit

            response = await async_client.get("/api/hdc/test_id")

            # Should get rate limit error
            assert response.status_code == 429
            assert "rate limit" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_rate_limit_headers(self, async_client, mock_redis):
        """Test rate limit headers in response."""
        with patch("genomevault.api.middleware.rate_limiter.redis_client", mock_redis):
            mock_redis.incr.return_value = 10
            mock_redis.ttl.return_value = 45

            response = await async_client.get("/api/zk/circuits")

            # Check rate limit headers
            assert "X-RateLimit-Limit" in response.headers
            assert "X-RateLimit-Remaining" in response.headers
            assert "X-RateLimit-Reset" in response.headers

            assert response.headers["X-RateLimit-Limit"] == "60"
            assert int(response.headers["X-RateLimit-Remaining"]) == 50

    @pytest.mark.asyncio
    async def test_rate_limit_burst(self, async_client, mock_redis):
        """Test burst rate limiting."""
        with patch("genomevault.api.middleware.rate_limiter.redis_client", mock_redis):
            # Simulate rapid requests
            for i in range(10):
                mock_redis.incr.return_value = i + 1
                response = await async_client.get("/api/zk/circuits")

                if i < 10:  # Within burst limit
                    assert response.status_code == 200
                else:  # Exceeds burst
                    assert response.status_code == 429

    def test_rate_limit_by_ip(self, test_client, mock_redis):
        """Test rate limiting by IP address."""
        with patch("genomevault.api.middleware.rate_limiter.redis_client", mock_redis):
            # Different IPs should have separate limits
            headers1 = {"X-Forwarded-For": "192.168.1.1"}
            headers2 = {"X-Forwarded-For": "192.168.1.2"}

            mock_redis.incr.return_value = 1
            response1 = test_client.get("/api/pir/status", headers=headers1)
            assert response1.status_code == 200

            mock_redis.incr.return_value = 1  # Reset for different IP
            response2 = test_client.get("/api/pir/status", headers=headers2)
            assert response2.status_code == 200

    @pytest.mark.asyncio
    async def test_rate_limit_exemption(self, async_client, mock_redis):
        """Test rate limit exemption for certain endpoints."""
        with patch("genomevault.api.middleware.rate_limiter.redis_client", mock_redis):
            mock_redis.incr.return_value = 100  # Way over limit

            # Health check should be exempt
            response = await async_client.get("/health")
            assert response.status_code == 200

            # Metrics should be exempt
            response = await async_client.get("/metrics")
            assert response.status_code == 200


class TestDatabasePersistence:
    """Test database persistence functionality."""

    def test_hdc_encoding_persistence(self, test_client, test_db):
        """Test HDC encoding is persisted to database."""
        with patch("genomevault.api.routers.hdc.get_db", return_value=test_db):
            # Create encoding
            response = test_client.post(
                "/api/hdc/encode",
                json={
                    "variants": [{"chromosome": "1", "position": 12345, "ref": "A", "alt": "G"}],
                    "dimension": 5000,
                },
            )

            assert response.status_code == 200
            encoding_id = response.json()["encoding_id"]

            # Check database
            encoding = (
                test_db.query(HDCEncoding).filter(HDCEncoding.encoding_id == encoding_id).first()
            )

            assert encoding is not None
            assert encoding.dimension == 5000
            assert encoding.variant_count == 1

    def test_zk_proof_persistence(self, test_client, test_db):
        """Test ZK proof is persisted to database."""
        with patch("genomevault.api.routers.zk.get_db", return_value=test_db):
            # Generate and store proof
            response = test_client.post(
                "/api/zk/prove",
                json={
                    "circuit_name": "sum64",
                    "inputs": [
                        {"name": "a", "value": 5, "is_public": False},
                        {"name": "b", "value": 10, "is_public": False},
                        {"name": "c", "value": 15, "is_public": True},
                    ],
                    "store_proof": True,
                },
            )

            assert response.status_code == 200
            proof_id = response.json()["proof_id"]

            # Check database
            proof_record = (
                test_db.query(ZKProofRecord).filter(ZKProofRecord.proof_id == proof_id).first()
            )

            assert proof_record is not None
            assert proof_record.circuit_name == "sum64"
            assert json.loads(proof_record.public_inputs)["c"] == 15

    def test_database_transaction_rollback(self, test_client, test_db):
        """Test database transaction rollback on error."""
        with patch("genomevault.api.routers.hdc.get_db", return_value=test_db):
            # Mock an error during commit
            original_commit = test_db.commit
            test_db.commit = MagicMock(side_effect=Exception("Commit failed"))

            response = test_client.post(
                "/api/hdc/encode",
                json={"variants": [{"chromosome": "1", "position": 12345, "ref": "A", "alt": "G"}]},
            )

            # Should get error response
            assert response.status_code == 500

            # Restore original commit
            test_db.commit = original_commit

            # Check nothing was persisted
            count = test_db.query(HDCEncoding).count()
            assert count == 0

    def test_concurrent_database_access(self, test_client, test_db):
        """Test handling of concurrent database access."""
        import threading

        results = []

        def create_encoding():
            with patch("genomevault.api.routers.hdc.get_db", return_value=test_db):
                response = test_client.post(
                    "/api/hdc/encode",
                    json={
                        "variants": [{"chromosome": "1", "position": 12345, "ref": "A", "alt": "G"}]
                    },
                )
                results.append(response.status_code)

        # Create multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=create_encoding)
            threads.append(t)
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # All should succeed
        assert all(status == 200 for status in results)

        # Check all were persisted
        count = test_db.query(HDCEncoding).count()
        assert count == 5

    def test_database_cleanup_on_delete(self, test_client, test_db):
        """Test database cleanup when deleting records."""
        with patch("genomevault.api.routers.hdc.get_db", return_value=test_db):
            # Create encoding
            response = test_client.post(
                "/api/hdc/encode",
                json={"variants": [{"chromosome": "1", "position": 12345, "ref": "A", "alt": "G"}]},
            )

            encoding_id = response.json()["encoding_id"]

            # Verify it exists
            encoding = (
                test_db.query(HDCEncoding).filter(HDCEncoding.encoding_id == encoding_id).first()
            )
            assert encoding is not None

            # Delete it
            test_db.delete(encoding)
            test_db.commit()

            # Verify it's gone
            encoding = (
                test_db.query(HDCEncoding).filter(HDCEncoding.encoding_id == encoding_id).first()
            )
            assert encoding is None

    def test_database_index_performance(self, test_client, test_db):
        """Test database index performance with many records."""
        with patch("genomevault.api.routers.hdc.get_db", return_value=test_db):
            # Create many encodings
            encoding_ids = []
            for i in range(100):
                response = test_client.post(
                    "/api/hdc/encode",
                    json={
                        "variants": [
                            {
                                "chromosome": str((i % 22) + 1),
                                "position": 10000 + i,
                                "ref": "A",
                                "alt": "G",
                            }
                        ]
                    },
                )
                encoding_ids.append(response.json()["encoding_id"])

            # Test indexed lookup performance
            import time

            start_time = time.time()
            for encoding_id in encoding_ids[:10]:
                encoding = (
                    test_db.query(HDCEncoding)
                    .filter(HDCEncoding.encoding_id == encoding_id)
                    .first()
                )
                assert encoding is not None

            lookup_time = time.time() - start_time

            # Should be fast with index (< 100ms for 10 lookups)
            assert lookup_time < 0.1


class TestIntegrationFlows:
    """Test complete integration flows."""

    def test_complete_hdc_workflow(self, test_client, sample_variants):
        """Test complete HDC encoding workflow."""
        # 1. Create first encoding
        response1 = test_client.post(
            "/api/hdc/encode",
            json={
                "variants": sample_variants[:2],
                "dimension": 10000,
                "metadata": {"sample": "patient1"},
            },
        )
        assert response1.status_code == 200
        encoding1 = response1.json()

        # 2. Create second encoding
        response2 = test_client.post(
            "/api/hdc/encode",
            json={
                "variants": sample_variants[1:],
                "dimension": 10000,
                "metadata": {"sample": "patient2"},
            },
        )
        assert response2.status_code == 200
        encoding2 = response2.json()

        # 3. Compare encodings
        compare_response = test_client.post(
            "/api/hdc/compare",
            json={
                "encoding_id_1": encoding1["encoding_id"],
                "encoding_id_2": encoding2["encoding_id"],
                "metric": "cosine",
            },
        )
        assert compare_response.status_code == 200
        comparison = compare_response.json()

        # 4. Download encoding
        download_response = test_client.get(f"/api/hdc/{encoding1['encoding_id']}/download")
        assert download_response.status_code == 200
        download_data = download_response.json()

        # Verify data integrity
        assert download_data["checksum"] == encoding1["checksum"]
        assert download_data["encoding"] == encoding1["encoding"]

    def test_complete_zk_workflow(self, test_client):
        """Test complete ZK proof workflow."""
        # 1. List available circuits
        circuits_response = test_client.get("/api/zk/circuits")
        assert circuits_response.status_code == 200
        circuits = circuits_response.json()

        # 2. Get specific circuit info
        circuit_response = test_client.get("/api/zk/circuits/sum64")
        assert circuit_response.status_code == 200

        # 3. Generate proof
        prove_response = test_client.post(
            "/api/zk/prove",
            json={
                "circuit_name": "sum64",
                "inputs": [
                    {"name": "a", "value": 25, "is_public": False},
                    {"name": "b", "value": 17, "is_public": False},
                    {"name": "c", "value": 42, "is_public": True},
                ],
                "store_proof": True,
                "metadata": {"purpose": "test"},
            },
        )
        assert prove_response.status_code == 200
        proof_data = prove_response.json()

        # 4. Verify proof
        verify_response = test_client.post(
            "/api/zk/verify",
            json={
                "proof": proof_data["proof"],
                "public_inputs": proof_data["public_inputs"],
                "circuit_name": "sum64",
                "proof_id": proof_data["proof_id"],
            },
        )
        assert verify_response.status_code == 200
        verify_data = verify_response.json()
        assert verify_data["valid"] is True

        # 5. Get proof info
        info_response = test_client.get(f"/api/zk/proofs/{proof_data['proof_id']}")
        assert info_response.status_code == 200
        info_data = info_response.json()
        assert info_data["verified"] is True

    @pytest.mark.asyncio
    async def test_complete_pir_workflow(self, async_client):
        """Test complete PIR workflow."""
        # 1. Setup PIR database
        setup_response = await async_client.post(
            "/api/pir/setup",
            json={
                "dataset_size": 100,
                "element_size": 1024,
                "metadata": {"dataset": "test_genomic"},
            },
        )
        assert setup_response.status_code == 200
        setup_data = setup_response.json()

        # 2. Check status
        status_response = await async_client.get("/api/pir/status")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["status"] == "operational"

        # 3. Execute query
        query_response = await async_client.post(
            "/api/pir/query",
            json={"index": 42, "dataset_size": 100, "setup_id": setup_data.get("setup_id")},
        )
        assert query_response.status_code == 200
        query_data = query_response.json()
        assert query_data["success"] is True
        assert query_data["index"] == 42

        # 4. Check updated statistics
        status_response2 = await async_client.get("/api/pir/status")
        assert status_response2.status_code == 200
        status_data2 = status_response2.json()
        assert status_data2["statistics"]["total_queries"] > 0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
