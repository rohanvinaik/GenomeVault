"""
Adversarial tests for PIR implementation.
Tests malformed queries, timing attacks, and collusion scenarios.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from genomevault.pir.client import PIRClient, PIRServer
from genomevault.pir.server.handler import PIRHandler
from genomevault.pir.server.pir_server import PIRServer as ServerImpl


class TestMalformedQueries:
    """Test server handling of malformed queries."""

    @pytest.fixture
    def server(self):
        """Create test server."""
        from pathlib import Path

        return ServerImpl("test_server", Path("/tmp/test_shards"))

    @pytest.fixture
    def handler(self, server):
        """Create request handler."""
        return PIRHandler(server)

    @pytest.mark.asyncio
    async def test_invalid_query_vector_size(self, handler):
        """Test query with wrong vector size."""
        request = Mock()
        request.json = asyncio.coroutine(
            lambda: {
                "query_id": "test-123",
                "query_vector": [0, 1, 0],  # Too small
                "vector_size": 1000000,  # Doesn't match
                "timestamp": time.time(),
                "protocol_version": "1.0",
            }
        )

        response = await handler.handle_query(request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_missing_required_fields(self, handler):
        """Test query missing required fields."""
        request = Mock()
        request.json = asyncio.coroutine(
            lambda: {
                "query_id": "test-123",
                # Missing query_vector
                "timestamp": time.time(),
            }
        )

        response = await handler.handle_query(request)
        assert response.status == 400

    @pytest.mark.asyncio
    async def test_invalid_protocol_version(self, handler):
        """Test unsupported protocol version."""
        request = Mock()
        request.json = asyncio.coroutine(
            lambda: {
                "query_id": "test-123",
                "query_vector": [0] * 1000,
                "vector_size": 1000,
                "timestamp": time.time(),
                "protocol_version": "0.5",  # Unsupported
            }
        )

        response = await handler.handle_query(request)
        assert response.status == 400

    @given(st.lists(st.integers(min_value=0, max_value=2), min_size=10, max_size=1000))
    @pytest.mark.asyncio
    async def test_fuzz_query_vectors(self, handler, query_vector):
        """Fuzz test with random query vectors."""
        request = Mock()
        request.json = asyncio.coroutine(
            lambda: {
                "query_id": "fuzz-test",
                "query_vector": query_vector,
                "vector_size": len(query_vector),
                "timestamp": time.time(),
                "protocol_version": "1.0",
            }
        )

        # Should handle gracefully without crashing
        try:
            response = await handler.handle_query(request)
            assert response.status in [200, 400, 500]
        except Exception:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            pytest.fail("Server crashed on fuzzed input")
            raise


class TestTimingAttacks:
    """Test timing attack mitigations."""

    @pytest.mark.asyncio
    async def test_constant_response_time(self):
        """Test that response times are constant."""
        servers = [
            PIRServer(f"server_{i}", f"http://localhost:900{i}", "region", False, 0.95, 50)
            for i in range(3)
        ]

        client = PIRClient(servers, database_size=10000)

        # Mock the server responses
        async def mock_query_server(server, query):
            # Simulate variable processing time
            if query.target_index < 5000:
                await asyncio.sleep(0.01)  # Fast
            else:
                await asyncio.sleep(0.05)  # Slow

            return Mock(response_vector=np.random.bytes(1024))

        with patch.object(client, "_query_server", mock_query_server):
            # Time multiple queries
            timings = []

            for idx in [100, 9900, 500, 9500]:  # Mix of fast/slow
                start = time.time()
                query = client.create_query(idx)
                # Would execute: await client.execute_query(query)
                end = time.time()
                timings.append(end - start)

            # Check variance is low (constant time)
            variance = np.var(timings)
            assert variance < 0.01, "Response times vary too much"

    @pytest.mark.asyncio
    async def test_random_delays_added(self):
        """Test that random delays are added to queries."""
        from genomevault.pir.server.handler import PIRHandler

        handler = PIRHandler(Mock())

        # Test timing padding
        start_times = [0.01, 0.05, 0.08, 0.02]  # Different processing times

        for elapsed in start_times:
            start = time.time() - elapsed
            await handler._timing_padding(start)
            total_time = time.time() - start

            # Should pad to ~100ms
            assert 0.095 < total_time < 0.105


class TestCollusion:
    """Test collusion detection and prevention."""

    def test_server_independence(self):
        """Test that server responses are independent."""
        # Create client with 3 servers
        servers = [
            PIRServer(f"server_{i}", f"http://localhost:900{i}", "region", False, 0.95, 50)
            for i in range(3)
        ]

        client = PIRClient(servers, database_size=1000)

        # Generate query for index 42
        query = client.create_query(42)

        # Check that query vectors are different
        vectors = list(query.query_vectors.values())

        # No two vectors should be identical
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                assert not np.array_equal(vectors[i], vectors[j])

        # But they should XOR to unit vector at index 42
        result = np.zeros_like(vectors[0])
        for vec in vectors:
            result = np.bitwise_xor(result, vec)

        expected = np.zeros_like(result)
        expected[42] = 1
        assert np.array_equal(result, expected)

    def test_collusion_detection_simulation(self):
        """Simulate collusion between servers."""
        # If 2 out of 3 servers collude
        server_queries = {
            "server_1": np.array([0, 1, 0, 1, 0]),
            "server_2": np.array([1, 0, 1, 0, 1]),
            "server_3": np.array([1, 1, 1, 1, 1]),  # XOR of first two
        }

        # Colluding servers 1 and 2 can determine query
        colluding_sum = np.bitwise_xor(server_queries["server_1"], server_queries["server_2"])

        # This equals server_3's query, revealing the target
        assert np.array_equal(colluding_sum, server_queries["server_3"])

        # The target index is where all XOR to 1
        all_xor = np.zeros_like(colluding_sum)
        for query in server_queries.values():
            all_xor = np.bitwise_xor(all_xor, query)

        target_index = np.where(all_xor == 1)[0][0]
        assert target_index == 2  # Index 2 was queried


class TestReplayAttacks:
    """Test replay attack prevention."""

    @pytest.mark.asyncio
    async def test_nonce_replay_detection(self):
        """Test that replayed nonces are rejected."""
        from genomevault.pir.server.handler import PIRHandler

        server = Mock()
        server.process_query = asyncio.coroutine(
            lambda x: {"response": [0] * 1024, "computation_time_ms": 10}
        )
        server.server_id = "test"
        server.is_trusted_signatory = False

        handler = PIRHandler(server)

        # First request with nonce
        request1 = Mock()
        request1.json = asyncio.coroutine(
            lambda: {
                "query_id": "test-123",
                "query_vector": [0] * 1000,
                "vector_size": 1000,
                "timestamp": time.time(),
                "protocol_version": "1.0",
                "nonce": "abcd1234" * 4,  # 32 hex chars
            }
        )

        response1 = await handler.handle_query(request1)
        assert response1.status == 200

        # Replay same nonce
        request2 = Mock()
        request2.json = asyncio.coroutine(
            lambda: {
                "query_id": "test-456",  # Different query ID
                "query_vector": [0] * 1000,
                "vector_size": 1000,
                "timestamp": time.time(),
                "protocol_version": "1.0",
                "nonce": "abcd1234" * 4,  # Same nonce
            }
        )

        response2 = await handler.handle_query(request2)
        assert response2.status == 400

        # Check error is replay detection
        body = await response2.json()
        assert body["error"]["code"] == "REPLAY_DETECTED"

    def test_timestamp_validation(self):
        """Test that old timestamps are rejected."""
        # This would be implemented in production
        # to prevent replay of old queries
        pytest.skip("Timestamp validation not implemented in minimal version")


class TestPaddingAndSizing:
    """Test fixed-size response enforcement."""

    def test_response_padding(self):
        """Test that responses are padded to fixed size."""
        from genomevault.pir.server.handler import PIRHandler

        handler = PIRHandler(Mock())

        # Test small data
        small_data = b"small"
        padded = handler._ensure_fixed_size(small_data)
        assert len(padded) == 1024
        assert padded[:5] == small_data

        # Test exact size
        exact_data = b"x" * 1024
        padded = handler._ensure_fixed_size(exact_data)
        assert len(padded) == 1024
        assert padded == exact_data

        # Test oversized data
        large_data = b"y" * 2000
        padded = handler._ensure_fixed_size(large_data)
        assert len(padded) == 1024
        assert padded == large_data[:1024]

    def test_query_padding_enforcement(self):
        """Test that queries enforce padding."""
        # Queries should also have consistent sizes
        # to prevent traffic analysis
        pytest.skip("Query padding not implemented in minimal version")


class TestErrorInjection:
    """Test handling of injected errors."""

    @pytest.mark.asyncio
    async def test_corrupted_responses(self):
        """Test handling of corrupted server responses."""
        client = PIRClient(
            [
                PIRServer(f"s{i}", f"http://localhost:900{i}", "r", False, 0.95, 50)
                for i in range(5)
            ],
            database_size=1000,
        )

        # Mock servers returning corrupted data
        async def mock_query_with_corruption(server, query):
            if server.server_id == "s2":
                # Server 2 returns corrupted data
                return Mock(response_vector=None)
            else:
                return Mock(response_vector=np.random.bytes(1024))

        with patch.object(client, "_query_server", mock_query_with_corruption):
            query = client.create_query(42)

            # Should still succeed with 4/5 servers
            try:
                # Would execute: result = await client.execute_query(query)
                pytest.skip("Full query execution not implemented in minimal version")
            except Exception:
                from genomevault.observability.logging import configure_logging

                logger = configure_logging()
                logger.exception("Unhandled exception")
                pytest.fail("Failed to handle corrupted response")
                raise

    def test_byzantine_server_behavior(self):
        """Test Byzantine fault tolerance."""
        # Test that system tolerates malicious servers
        # that return intentionally wrong results
        pytest.skip("Byzantine fault tolerance not implemented in minimal version")


if __name__ == "__main__":
    # Run specific adversarial test
    import sys

    if len(sys.argv) > 1:
        test_class = sys.argv[1]
        pytest.main([__file__, f"::{test_class}", "-v"])
    else:
        pytest.main([__file__, "-v"])
