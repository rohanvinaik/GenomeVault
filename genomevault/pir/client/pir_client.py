"""
PIR Client implementation for private genomic queries
"""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

from genomevault.core.constants import (
    MIN_PIR_SERVERS,
    PIR_QUERY_TIMEOUT_MS,
    PIR_THRESHOLD,
)
from genomevault.core.exceptions import PIRError


@dataclass
class PIRQuery:
    """Represents a PIR query"""

    indices: List[int]  # Database indices to query
    seed: Optional[int] = None  # Seed for deterministic masking
    metadata: Dict[str, Any] = None  # Additional query metadata
    query_vector: Optional[np.ndarray] = None  # Encoded query vector
    nonce: Optional[bytes] = None  # Query nonce

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.nonce is None:
            self.nonce = np.random.bytes(32)


class PIRClient:
    """
    Client for Private Information Retrieval
    Queries multiple servers without revealing the queried position
    """

    def __init__(self, server_urls: List[str], database_size: int, threshold: int = 2):
        self.server_urls = server_urls
        self.database_size = database_size
        self.threshold = threshold
        self.session: Optional[aiohttp.ClientSession] = None

        if len(server_urls) < MIN_PIR_SERVERS:
            raise PIRError(f"Need at least {MIN_PIR_SERVERS} servers, got {len(server_urls)}")

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def query_position(self, chromosome: str, position: int, length: int = 1000) -> bytes:
        """
        Query a genomic position privately

        Args:
            chromosome: Chromosome identifier
            position: Genomic position
            length: Number of bases to retrieve

        Returns:
            Reference sequence at the position
        """
        # Create oblivious query vector
        query = self._create_query(chromosome, position, length)

        # Query all servers in parallel
        tasks = []
        for server_url in self.server_urls:
            task = self._query_server(server_url, query)
            tasks.append(task)

        # Collect responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed responses
        valid_responses = [r for r in responses if not isinstance(r, Exception)]

        if len(valid_responses) < self.threshold:
            raise PIRError(
                "Insufficient responses: got {len(valid_responses)}, need {self.threshold}"
            )

        # Reconstruct data from responses
        result = self._reconstruct_data(valid_responses, query)

        return result

    def _create_query(self, chromosome: str, position: int, length: int) -> PIRQuery:
        """
        Create an oblivious query vector
        """
        # Create selection vector
        # In real implementation, this would be cryptographically secure
        total_positions = 3_000_000_000  # Approximate genome size

        # Create sparse binary vector
        query_vector = np.zeros(min(total_positions // 1000, 1000000), dtype=np.float32)

        # Encode position obliviously
        # This is simplified - real implementation would use homomorphic encryption
        idx = hash("{chromosome}:{position}") % len(query_vector)
        query_vector[idx] = 1.0

        # Add noise for privacy
        noise_positions = np.random.choice(len(query_vector), size=10, replace=False)
        query_vector[noise_positions] = np.random.random(10) * 0.1

        nonce = np.random.bytes(32)

        return PIRQuery(position=position, length=length, query_vector=query_vector, nonce=nonce)

    async def _query_server(self, server_url: str, query: PIRQuery) -> Dict[str, Any]:
        """Query a single PIR server"""
        if not self.session:
            raise PIRError("Session not initialized")

        query_data = {
            "vector": query.query_vector.tolist(),
            "nonce": query.nonce.hex(),
            "length": query.length,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=PIR_QUERY_TIMEOUT_MS / 1000)
            async with self.session.post(
                "{server_url}/query", json=query_data, timeout=timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise PIRError("Server returned status {response.status}")
        except asyncio.TimeoutError:
            raise PIRError("Query timeout for server {server_url}")
        except Exception as e:
            raise PIRError("Query failed for server {server_url}: {str(e)}")

    def _reconstruct_data(self, responses: List[Dict[str, Any]], query: PIRQuery) -> bytes:
        """
        Reconstruct data from PIR responses
        Uses threshold reconstruction
        """
        # Extract response vectors
        response_vectors = []
        trusted_responses = []

        for response in responses:
            vector = np.array(response["response"])
            response_vectors.append(vector)

            if response.get("is_trusted", False):
                trusted_responses.append(response)

        # Prefer trusted responses if available
        if len(trusted_responses) >= self.threshold:
            response_vectors = [np.array(r["response"]) for r in trusted_responses]

        # Simple reconstruction (XOR for demonstration)
        # Real implementation would use proper secret sharing
        result_vector = response_vectors[0]
        for vector in response_vectors[1 : self.threshold]:
            result_vector = result_vector ^ vector

        # Extract the queried data
        # This is simplified - real implementation would properly decode
        result_bytes = result_vector.astype(np.uint8).tobytes()

        # Extract the specific position requested
        start_byte = query.position % len(result_bytes)
        end_byte = min(start_byte + query.length, len(result_bytes))

        return result_bytes[start_byte:end_byte]

    async def get_server_status(self) -> List[Dict[str, Any]]:
        """Get status of all PIR servers"""
        if not self.session:
            raise PIRError("Session not initialized")

        statuses = []
        for server_url in self.server_urls:
            try:
                async with self.session.get("{server_url}/status") as response:
                    if response.status == 200:
                        status = await response.json()
                        status["url"] = server_url
                        status["online"] = True
                        statuses.append(status)
                    else:
                        statuses.append(
                            {
                                "url": server_url,
                                "online": False,
                                "error": "Status code {response.status}",
                            }
                        )
            except Exception as e:
                statuses.append({"url": server_url, "online": False, "error": str(e)})

        return statuses

    def calculate_privacy_guarantee(self, num_honest_servers: int) -> float:
        """
        Calculate privacy failure probability
        P_fail = (1 - q)^k where q is server honesty probability
        """
        honesty_prob = PIR_THRESHOLD
        privacy_failure_prob = (1 - honesty_prob) ** num_honest_servers
        return privacy_failure_prob

    def create_query(self, db_index: int, seed: Optional[int] = None) -> PIRQuery:
        """
        Create a PIR query for a database index with optional seed

        Args:
            db_index: Database index to query
            seed: Optional seed for deterministic query generation

        Returns:
            PIRQuery object
        """
        # Use seed for deterministic randomness if provided
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random

        # Create query vector
        query_vector = np.zeros(self.database_size, dtype=np.float32)

        # Set the target index
        query_vector[db_index] = 1.0

        # Add obfuscation noise
        noise_indices = rng.choice(
            self.database_size, size=min(10, self.database_size // 100), replace=False
        )
        query_vector[noise_indices] = rng.random(len(noise_indices)) * 0.1

        # Generate nonce
        nonce = rng.bytes(32) if seed is not None else np.random.bytes(32)

        return PIRQuery(
            indices=[db_index],
            seed=seed,
            query_vector=query_vector,
            nonce=nonce,
            metadata={"target_index": db_index},
        )

    async def execute_query(self, query: PIRQuery) -> Any:
        """
        Execute a PIR query across servers

        Args:
            query: PIRQuery to execute

        Returns:
            Reconstructed data from servers
        """
        if not query.query_vector:
            raise PIRError("Query vector not initialized")

        # Query all servers in parallel
        tasks = []
        for server_url in self.server_urls:
            task = self._query_server_v2(server_url, query)
            tasks.append(task)

        # Collect responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out failed responses
        valid_responses = [r for r in responses if not isinstance(r, Exception)]

        if len(valid_responses) < self.threshold:
            raise PIRError(
                f"Insufficient responses: got {len(valid_responses)}, need {self.threshold}"
            )

        # Reconstruct data from responses
        result = self._reconstruct_data_v2(valid_responses, query)

        return result

    async def _query_server_v2(self, server_url: str, query: PIRQuery) -> Dict[str, Any]:
        """Query a single PIR server with enhanced query format"""
        if not self.session:
            raise PIRError("Session not initialized")

        query_data = {
            "indices": query.indices,
            "vector": query.query_vector.tolist(),
            "nonce": query.nonce.hex(),
            "metadata": query.metadata,
        }

        try:
            timeout = aiohttp.ClientTimeout(total=PIR_QUERY_TIMEOUT_MS / 1000)
            async with self.session.post(
                f"{server_url}/query", json=query_data, timeout=timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise PIRError(f"Server returned status {response.status}")
        except asyncio.TimeoutError:
            raise PIRError(f"Query timeout for server {server_url}")
        except Exception as e:
            raise PIRError(f"Query failed for server {server_url}: {str(e)}")

    def _reconstruct_data_v2(self, responses: List[Dict[str, Any]], query: PIRQuery) -> Any:
        """
        Reconstruct data from PIR responses (enhanced version)
        """
        # Extract response data
        response_data = []
        for response in responses:
            if "data" in response:
                response_data.append(response["data"])
            elif "response" in response:
                response_data.append(response["response"])

        if not response_data:
            raise PIRError("No valid data in responses")

        # For single index queries, return the first valid response
        # In production, would use proper secret sharing reconstruction
        return response_data[0]

    async def batch_query(self, indices: List[int]) -> List[Any]:
        """
        Execute batch queries for multiple indices

        Args:
            indices: List of database indices to query

        Returns:
            List of results for each index
        """
        # Create queries for all indices
        queries = [self.create_query(idx) for idx in indices]

        # Execute all queries
        tasks = [self.execute_query(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                raise PIRError(f"Query for index {indices[i]} failed: {str(result)}")
            final_results.append(result)

        return final_results

    def decode_response(self, response_data: Any, response_type: str = "genomic") -> Any:
        """
        Decode response data based on type

        Args:
            response_data: Raw response data
            response_type: Type of response (genomic, variant, etc.)

        Returns:
            Decoded data
        """
        if response_type == "genomic":
            # Decode genomic data
            if isinstance(response_data, dict):
                return response_data
            elif isinstance(response_data, (bytes, bytearray)):
                return json.loads(response_data.decode())
            else:
                return response_data
        else:
            # Default decoding
            return response_data
