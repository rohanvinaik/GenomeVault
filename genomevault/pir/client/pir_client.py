from typing import Any, Dict, List, Optional, Tuple

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

    position: int
    length: int
    query_vector: np.ndarray
    nonce: bytes


class PIRClient:
    """
    Client for Private Information Retrieval
    Queries multiple servers without revealing the queried position
    """

    def __init__(self, server_urls: List[str], threshold: int = 2):
        self.server_urls = server_urls
        self.threshold = threshold
        self.session: Optional[aiohttp.ClientSession] = None

        if len(server_urls) < MIN_PIR_SERVERS:
            raise PIRError("Need at least {MIN_PIR_SERVERS} servers, got {len(server_urls)}")

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
            raise PIRError("Session not initialized") from e
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
                    raise PIRError("Server returned status {response.status}") from e
        except asyncio.TimeoutError:
            raise PIRError("Query timeout for server {server_url}") from e
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
            raise PIRError("Session not initialized") from e
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
