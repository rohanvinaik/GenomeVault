"""
PIR client: build and decode queries with information-theoretic privacy.
Implements multi-server PIR with optimal communication complexity.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import aiohttp
import numpy as np

from genomevault.utils.config import get_config

config = get_config()
from genomevault.utils.logging import get_logger, logger, performance_logger

logger = get_logger(__name__)


@dataclass
class PIRServer:
    """PIR server information."""

    server_id: str
    endpoint: str
    region: str
    is_trusted_signatory: bool
    honesty_probability: float
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "server_id": self.server_id,
            "endpoint": self.endpoint,
            "region": self.region,
            "is_trusted_signatory": self.is_trusted_signatory,
            "honesty_probability": self.honesty_probability,
            "latency_ms": self.latency_ms,
        }


@dataclass
class PIRQuery:
    """PIR query structure."""

    query_id: str
    target_index: int
    query_vectors: dict[str, np.ndarray]
    timestamp: float
    metadata: dict | None = None


@dataclass
class PIRResponse:
    """PIR response from server."""

    server_id: str
    query_id: str
    response_vector: np.ndarray
    computation_time_ms: float
    timestamp: float


class PIRClient:
    """
    Information-theoretic PIR client.
    Implements Chor et al. PIR scheme with enhanced security.
    """

    def __init__(self, servers: list[PIRServer], database_size: int):
        """
        Initialize PIR client.

        Args:
            servers: List of available PIR servers
            database_size: Size of the database being queried
        """
        self.servers = servers
        self.database_size = database_size
        self.min_servers = config.pir.min_honest_servers

        # Validate server configuration
        self._validate_servers()

        # Initialize connection pool
        self.session = None

        logger.info(
            "PIRClient initialized with {len(servers)} servers",
            extra={"privacy_safe": True},
        )

    def _validate_servers(self):
        """Validate server configuration meets security requirements."""
        if len(self.servers) < self.min_servers:
            raise ValueError("Insufficient servers: {len(self.servers)} < {self.min_servers}")

        # Calculate privacy guarantee
        ts_servers = [s for s in self.servers if s.is_trusted_signatory]
        if len(ts_servers) >= self.min_servers:
            # Use HIPAA TS probability
            failure_prob = self.calculate_privacy_failure_probability(
                len(ts_servers), config.pir.server_honesty_hipaa
            )
        else:
            # Use generic probability
            failure_prob = self.calculate_privacy_failure_probability(
                len(self.servers), config.pir.server_honesty_generic
            )

        if failure_prob > config.pir.target_failure_probability:
            logger.warning(
                "Privacy failure probability {failure_prob:.2e} exceeds target {config.pir.target_failure_probability:.2e}",
                extra={"privacy_safe": True},
            )

    def calculate_privacy_failure_probability(self, k: int, q: float) -> float:
        """
        Calculate privacy breach probability P_fail(k,q) = (1-q)^k.

        Args:
            k: Number of honest servers
            q: Server honesty probability

        Returns:
            Privacy failure probability
        """
        return (1 - q) ** k

    def calculate_min_servers_needed(self, target_failure: float, honesty_prob: float) -> int:
        """
        Calculate minimum servers needed for target failure probability.

        Args:
            target_failure: Target failure probability
            honesty_prob: Server honesty probability

        Returns:
            Minimum number of servers needed
        """
        import math

        return math.ceil(math.log(target_failure) / math.log(1 - honesty_prob))

    @performance_logger.log_operation("create_query")
    def create_query(self, target_index: int) -> PIRQuery:
        """
        Create PIR query for retrieving item at target_index.

        Args:
            target_index: Index of desired database item

        Returns:
            PIR query object
        """
        if target_index >= self.database_size:
            raise ValueError("Index {target_index} out of bounds")

        # Generate query ID
        query_id = hashlib.sha256(b"{target_index}:{time.time()}").hexdigest()[:16]

        # Create unit vector for target
        e_j = np.zeros(self.database_size)
        e_j[target_index] = 1

        # Generate random query vectors that sum to e_j
        query_vectors = self._generate_query_vectors(e_j)

        query = PIRQuery(
            query_id=query_id,
            target_index=target_index,
            query_vectors=query_vectors,
            timestamp=time.time(),
            metadata={
                "database_size": self.database_size,
                "num_servers": len(self.servers),
            },
        )

        logger.info(f"PIR query created for index {target_index}", extra={"privacy_safe": True})

        return query

    def _generate_query_vectors(self, target_vector: np.ndarray) -> dict[str, np.ndarray]:
        """
        Generate random query vectors that sum to target vector.

        Args:
            target_vector: Target unit vector e_j

        Returns:
            Dictionary mapping server IDs to query vectors
        """
        n = len(self.servers)
        query_vectors = {}

        # Generate n-1 random vectors
        random_vectors = []
        for i in range(n - 1):
            # Use binary vectors for efficiency
            random_vec = np.random.randint(0, 2, size=self.database_size)
            random_vectors.append(random_vec)
            query_vectors[self.servers[i].server_id] = random_vec

        # Calculate last vector to ensure sum equals target
        last_vector = target_vector.copy()
        for vec in random_vectors:
            last_vector = (last_vector - vec) % 2

        query_vectors[self.servers[-1].server_id] = last_vector

        # Verify sum equals target (in GF(2))
        total = np.zeros_like(target_vector)
        for vec in query_vectors.values():
            total = (total + vec) % 2

        assert np.array_equal(total, target_vector), "Query vectors don't sum to target"

        return query_vectors

    async def execute_query(self, query: PIRQuery) -> Any:
        """
        Execute PIR query across servers.

        Args:
            query: PIR query to execute

        Returns:
            Retrieved data item
        """
        if not self.session:
            self.session = aiohttp.ClientSession()

        # Send queries to all servers in parallel
        tasks = []
        for server in self.servers:
            task = self._query_server(server, query)
            tasks.append(task)

        # Wait for all responses
        responses = await asyncio.gather(*tasks)

        # Filter out failed responses
        valid_responses = [r for r in responses if r is not None]

        if len(valid_responses) < self.min_servers:
            raise RuntimeError(
                "Insufficient responses: {len(valid_responses)} < {self.min_servers}"
            )

        # Reconstruct data
        result = self._reconstruct_data(valid_responses)

        logger.info(f"PIR query {query.query_id} completed", extra={"privacy_safe": True})

        return result

    async def _query_server(self, server: PIRServer, query: PIRQuery) -> PIRResponse | None:
        """
        Query individual PIR server.

        Args:
            server: Server to query
            query: PIR query

        Returns:
            Server response or None if failed
        """
        try:
            # Get query vector for this server
            query_vector = query.query_vectors[server.server_id]

            # Prepare request
            request_data = {
                "query_id": query.query_id,
                "query_vector": query_vector.tolist(),
                "database_size": self.database_size,
            }

            # Send request
            start_time = time.time()

            async with self.session.post(
                "{server.endpoint}/pir/query",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=config.pir.query_timeout_seconds),
            ) as response:
                if response.status != 200:
                    logger.error(f"PIR query failed on {server.server_id}: {response.status}")
                    return None

                result = await response.json()

            computation_time = (time.time() - start_time) * 1000

            # Parse response
            pir_response = PIRResponse(
                server_id=server.server_id,
                query_id=query.query_id,
                response_vector=np.array(result["response"]),
                computation_time_ms=computation_time,
                timestamp=time.time(),
            )

            return pir_response

        except Exception:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            logger.error(f"Error querying server {server.server_id}: {e}")
            return None
            raise

    def _reconstruct_data(self, responses: list[PIRResponse]) -> Any:
        """
        Reconstruct data from server responses.

        Args:
            responses: List of server responses

        Returns:
            Reconstructed data item
        """
        # Sum all response vectors
        result = np.zeros_like(responses[0].response_vector)

        for response in responses:
            result = result + response.response_vector

        # The result is the desired database item
        return result

    def decode_response(self, response_data: np.ndarray, data_type: str = "genomic") -> Any:
        """
        Decode PIR response based on data type.

        Args:
            response_data: Raw response data
            data_type: Type of data being retrieved

        Returns:
            Decoded data
        """
        if data_type == "genomic":
            return self._decode_genomic_data(response_data)
        elif data_type == "reference":
            return self._decode_reference_data(response_data)
        elif data_type == "annotation":
            return self._decode_annotation_data(response_data)
        else:
            # Generic decoding
            return response_data

    def _decode_genomic_data(self, data: np.ndarray) -> dict:
        """
        Decode genomic reference data.

        Args:
            data: Raw genomic data

        Returns:
            Decoded genomic information
        """
        # Convert from numpy array to genomic format
        # This is a simplified example
        decoded = {"sequence": "", "annotations": [], "quality_scores": []}

        # Decode sequence (2-bit encoding)
        bases = ["A", "C", "G", "T"]
        for i in range(0, len(data), 2):
            if i + 1 < len(data):
                base_idx = int(data[i]) * 2 + int(data[i + 1])
                decoded["sequence"] += bases[base_idx % 4]

        return decoded

    def _decode_reference_data(self, data: np.ndarray) -> dict:
        """Decode reference genome data."""
        # Decode pangenome graph structure
        return {
            "nodes": self._extract_nodes(data),
            "edges": self._extract_edges(data),
            "paths": self._extract_paths(data),
        }

    def _decode_annotation_data(self, data: np.ndarray) -> dict:
        """Decode functional annotations."""
        # Decode variant annotations
        return {
            "gene_impact": self._extract_gene_impact(data),
            "conservation_scores": self._extract_conservation(data),
            "population_frequencies": self._extract_frequencies(data),
        }

    def _extract_nodes(self, data: np.ndarray) -> list[dict]:
        """Extract graph nodes from data."""
        # Placeholder implementation
        return []

    def _extract_edges(self, data: np.ndarray) -> list[dict]:
        """Extract graph edges from data."""
        # Placeholder implementation
        return []

    def _extract_paths(self, data: np.ndarray) -> list[dict]:
        """Extract reference paths from data."""
        # Placeholder implementation
        return []

    def _extract_gene_impact(self, data: np.ndarray) -> dict:
        """Extract gene impact predictions."""
        # Placeholder implementation
        return {}

    def _extract_conservation(self, data: np.ndarray) -> np.ndarray:
        """Extract conservation scores."""
        # Placeholder implementation
        return np.array([])

    def _extract_frequencies(self, data: np.ndarray) -> dict:
        """Extract population frequencies."""
        # Placeholder implementation
        return {}

    async def batch_query(self, indices: list[int]) -> list[Any]:
        """
        Execute multiple PIR queries in batch.

        Args:
            indices: List of database indices to retrieve

        Returns:
            List of retrieved items
        """
        # Create queries
        queries = [self.create_query(idx) for idx in indices]

        # Execute in parallel with rate limiting
        results = []
        batch_size = 10  # Process 10 queries at a time

        for i in range(0, len(queries), batch_size):
            batch = queries[i : i + batch_size]
            batch_tasks = [self.execute_query(q) for q in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)

            # Rate limiting
            if i + batch_size < len(queries):
                await asyncio.sleep(0.1)  # 100ms between batches

        return results

    def estimate_communication_cost(self, num_queries: int = 1) -> dict[str, float]:
        """
        Estimate communication cost for PIR queries.

        Args:
            num_queries: Number of queries to estimate for

        Returns:
            Communication cost estimates
        """
        n = len(self.servers)
        N = self.database_size

        # Communication complexity: O(N^(1/n))
        download_per_server = N ** (1 / n)
        upload_per_server = N ** (1 / n)

        total_download = download_per_server * n * num_queries
        total_upload = upload_per_server * n * num_queries

        # Estimate latency
        avg_latency = np.mean([s.latency_ms for s in self.servers])
        total_latency = avg_latency * n  # Sequential server queries

        return {
            "download_bytes": total_download,
            "upload_bytes": total_upload,
            "total_bytes": total_download + total_upload,
            "estimated_latency_ms": total_latency,
            "servers_used": n,
            "queries": num_queries,
        }

    def get_optimal_server_configuration(self) -> dict[str, Any]:
        """
        Calculate optimal server configuration for current setup.

        Returns:
            Optimal configuration details
        """
        # Separate TS and non-TS servers
        ts_servers = [s for s in self.servers if s.is_trusted_signatory]
        ln_servers = [s for s in self.servers if not s.is_trusted_signatory]

        # Calculate different configurations
        configs = []

        # Config 1: 3 LN + 2 TS
        if len(ln_servers) >= 3 and len(ts_servers) >= 2:
            config1_servers = ln_servers[:3] + ts_servers[:2]
            config1_latency = sum(s.latency_ms for s in config1_servers)
            config1_failure = self.calculate_privacy_failure_probability(
                2, config.pir.server_honesty_hipaa
            )
            configs.append(
                {
                    "name": "3 LN + 2 TS",
                    "servers": 5,
                    "latency_ms": config1_latency,
                    "failure_probability": config1_failure,
                }
            )

        # Config 2: 1 LN + 2 TS
        if len(ln_servers) >= 1 and len(ts_servers) >= 2:
            config2_servers = ln_servers[:1] + ts_servers[:2]
            config2_latency = sum(s.latency_ms for s in config2_servers)
            config2_failure = self.calculate_privacy_failure_probability(
                2, config.pir.server_honesty_hipaa
            )
            configs.append(
                {
                    "name": "1 LN + 2 TS",
                    "servers": 3,
                    "latency_ms": config2_latency,
                    "failure_probability": config2_failure,
                }
            )

        # Find optimal based on latency while meeting security requirement
        valid_configs = [
            c for c in configs if c["failure_probability"] <= config.pir.target_failure_probability
        ]

        if valid_configs:
            optimal = min(valid_configs, key=lambda c: c["latency_ms"])
        else:
            optimal = None

        return {
            "configurations": configs,
            "optimal": optimal,
            "current_servers": len(self.servers),
            "ts_servers": len(ts_servers),
            "ln_servers": len(ln_servers),
        }

    async def close(self):
        """Close client connections."""
        if self.session:
            await self.session.close()


# Example usage
if __name__ == "__main__":
    import asyncio

    # Example server configuration
    servers = [
        # Light nodes (non-TS)
        PIRServer("ln1", "http://ln1.genomevault.com", "us-east", False, 0.95, 70),
        PIRServer("ln2", "http://ln2.genomevault.com", "eu-west", False, 0.95, 80),
        PIRServer("ln3", "http://ln3.genomevault.com", "asia-pac", False, 0.95, 90),
        # Trusted signatories (HIPAA-compliant)
        PIRServer("ts1", "http://ts1.genomevault.com", "us-west", True, 0.98, 60),
        PIRServer("ts2", "http://ts2.genomevault.com", "us-central", True, 0.98, 50),
    ]

    # Initialize client
    database_size = 1000000  # 1M items
    client = PIRClient(servers, database_size)

    # Show optimal configuration
    optimal_config = client.get_optimal_server_configuration()
    print("Optimal PIR Configuration:")
    print(json.dumps(optimal_config, indent=2))

    # Calculate privacy guarantees
    print("\nPrivacy Guarantees:")
    print("2 TS servers: P_fail = {client.calculate_privacy_failure_probability(2, 0.98):.2e}")
    print("3 TS servers: P_fail = {client.calculate_privacy_failure_probability(3, 0.98):.2e}")

    # Estimate communication cost
    cost = client.estimate_communication_cost(num_queries=10)
    print("\nCommunication Cost (10 queries):")
    print("Total: {cost['total_bytes'] / 1024:.2f} KB")
    print("Latency: {cost['estimated_latency_ms']:.0f} ms")

    # Example query (would be async in practice)
    async def example_query():
        query = client.create_query(target_index=42)
        # In practice, would execute: result = await client.execute_query(query)
        await client.close()

    # Run example
    # asyncio.run(example_query())
