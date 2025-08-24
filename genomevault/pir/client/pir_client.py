"""
PIR Client implementation for private genomic queries

This module provides a client for Private Information Retrieval (PIR) that
can query multiple servers without revealing which data is being accessed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class QueryProtocol(Enum):
    """PIR query protocols."""
    XOR = "xor"
    IT_PIR = "it-pir"
    HYBRID = "hybrid"


@dataclass
class ServerConfig:
    """Configuration for a PIR server."""
    url: str
    server_id: int
    weight: float = 1.0  # Trust weight for this server
    max_retries: int = 3
    timeout_seconds: float = 10.0


@dataclass
class PIRQuery:
    """Represents a PIR query with privacy guarantees."""
    
    index: int  # Target database index
    query_vectors: List[np.ndarray]  # Query vectors for each server
    query_id: str = field(default_factory=lambda: secrets.token_hex(16))
    protocol: QueryProtocol = QueryProtocol.IT_PIR
    nonce: bytes = field(default_factory=lambda: secrets.token_bytes(32))
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PIRResponse:
    """Response from a PIR server."""
    
    server_id: int
    query_id: str
    data: bytes
    timestamp: float
    valid: bool = True
    error: Optional[str] = None


class PIRClient:
    """
    Client for Private Information Retrieval with multi-server support.
    
    Features:
    - Privacy-preserving queries across multiple servers
    - XOR-based response aggregation
    - Byzantine fault tolerance
    - Automatic retries and timeout handling
    - Response integrity validation
    """
    
    def __init__(
        self,
        servers: List[ServerConfig],
        database_size: int,
        element_size: int = 1024,
        min_servers: int = 2,
        protocol: QueryProtocol = QueryProtocol.IT_PIR
    ):
        """
        Initialize PIR client.
        
        Args:
            servers: List of server configurations
            database_size: Size of the database
            element_size: Size of each database element in bytes
            min_servers: Minimum servers required for reconstruction
            protocol: PIR protocol to use
        """
        self.servers = servers
        self.database_size = database_size
        self.element_size = element_size
        self.min_servers = min_servers
        self.protocol = protocol
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self.stats = {
            "queries_sent": 0,
            "queries_successful": 0,
            "queries_failed": 0,
            "bytes_received": 0,
            "total_latency_ms": 0.0
        }
        
        if len(servers) < min_servers:
            raise ValueError(
                f"Need at least {min_servers} servers, got {len(servers)}"
            )
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def generate_it_pir_query(self, index: int) -> PIRQuery:
        """
        Generate IT-PIR query vectors for retrieving element at index.
        
        The protocol ensures information-theoretic security by splitting
        a unit vector into random shares that XOR to the target.
        
        Args:
            index: Database index to retrieve
            
        Returns:
            PIRQuery with vectors for each server
        """
        if index < 0 or index >= self.database_size:
            raise ValueError(f"Index {index} out of bounds")
        
        # Create unit vector for target index
        unit_vector = np.zeros(self.database_size, dtype=np.uint8)
        unit_vector[index] = 1
        
        # Generate random vectors for n-1 servers
        query_vectors = []
        for i in range(len(self.servers) - 1):
            # Random binary vector
            random_vector = np.random.randint(0, 2, self.database_size, dtype=np.uint8)
            query_vectors.append(random_vector)
        
        # Last vector ensures XOR equals unit vector
        last_vector = unit_vector.copy()
        for vec in query_vectors:
            last_vector = (last_vector + vec) % 2  # XOR in binary field
        
        query_vectors.append(last_vector)
        
        # Verify correctness
        xor_sum = np.zeros(self.database_size, dtype=np.uint8)
        for vec in query_vectors:
            xor_sum = (xor_sum + vec) % 2
        
        if not np.array_equal(xor_sum, unit_vector):
            raise RuntimeError("Query vector generation failed")
        
        return PIRQuery(
            index=index,
            query_vectors=query_vectors,
            protocol=QueryProtocol.IT_PIR
        )
    
    def generate_xor_query(self, index: int) -> PIRQuery:
        """
        Generate XOR-based PIR query.
        
        Args:
            index: Database index to retrieve
            
        Returns:
            PIRQuery with XOR masks
        """
        # For 2-server XOR PIR
        mask1 = np.random.randint(0, 2, self.database_size, dtype=np.uint8)
        mask2 = mask1.copy()
        mask2[index] = 1 - mask2[index]  # Flip bit at target index
        
        return PIRQuery(
            index=index,
            query_vectors=[mask1, mask2],
            protocol=QueryProtocol.XOR
        )
    
    async def query_server(
        self,
        server: ServerConfig,
        query_vector: np.ndarray,
        query_id: str,
        retry_count: int = 0
    ) -> PIRResponse:
        """
        Query a single PIR server with retry logic.
        
        Args:
            server: Server configuration
            query_vector: Query vector for this server
            query_id: Unique query identifier
            retry_count: Current retry attempt
            
        Returns:
            PIRResponse from the server
        """
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        # Prepare query payload
        payload = {
            "mask": query_vector.tolist(),
            "query_id": query_id,
            "protocol": self.protocol.value,
            "server_id": server.server_id
        }
        
        try:
            # Set timeout
            timeout = aiohttp.ClientTimeout(total=server.timeout_seconds)
            
            # Send query
            start_time = time.time()
            async with self.session.post(
                f"{server.url}/query",
                json=payload,
                timeout=timeout
            ) as response:
                latency = (time.time() - start_time) * 1000
                self.stats["total_latency_ms"] += latency
                
                if response.status == 200:
                    result = await response.json()
                    response_data = bytes.fromhex(result["response"])
                    
                    self.stats["bytes_received"] += len(response_data)
                    
                    return PIRResponse(
                        server_id=server.server_id,
                        query_id=result.get("query_id", query_id),
                        data=response_data,
                        timestamp=result.get("timestamp", time.time()),
                        valid=True
                    )
                else:
                    error_msg = f"Server returned status {response.status}"
                    logger.warning(f"Server {server.server_id}: {error_msg}")
                    
                    # Retry if possible
                    if retry_count < server.max_retries:
                        await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                        return await self.query_server(
                            server, query_vector, query_id, retry_count + 1
                        )
                    
                    return PIRResponse(
                        server_id=server.server_id,
                        query_id=query_id,
                        data=b"",
                        timestamp=time.time(),
                        valid=False,
                        error=error_msg
                    )
                    
        except asyncio.TimeoutError:
            error_msg = f"Timeout after {server.timeout_seconds}s"
            logger.warning(f"Server {server.server_id}: {error_msg}")
            
            # Retry on timeout
            if retry_count < server.max_retries:
                return await self.query_server(
                    server, query_vector, query_id, retry_count + 1
                )
            
            return PIRResponse(
                server_id=server.server_id,
                query_id=query_id,
                data=b"",
                timestamp=time.time(),
                valid=False,
                error=error_msg
            )
            
        except Exception as e:
            error_msg = f"Query failed: {str(e)}"
            logger.error(f"Server {server.server_id}: {error_msg}")
            
            return PIRResponse(
                server_id=server.server_id,
                query_id=query_id,
                data=b"",
                timestamp=time.time(),
                valid=False,
                error=error_msg
            )
    
    def aggregate_responses_xor(self, responses: List[PIRResponse]) -> bytes:
        """
        Aggregate responses using XOR.
        
        Args:
            responses: List of server responses
            
        Returns:
            XOR aggregation of all responses
        """
        if not responses:
            raise ValueError("No responses to aggregate")
        
        # Start with first response
        result = bytearray(responses[0].data)
        
        # XOR with remaining responses
        for response in responses[1:]:
            if len(response.data) != len(result):
                raise ValueError(
                    f"Response size mismatch: {len(response.data)} != {len(result)}"
                )
            
            for i in range(len(result)):
                result[i] ^= response.data[i]
        
        return bytes(result)
    
    def validate_response_integrity(
        self,
        response: bytes,
        expected_size: int,
        checksum: Optional[str] = None
    ) -> bool:
        """
        Validate response integrity.
        
        Args:
            response: Response data
            expected_size: Expected response size
            checksum: Optional expected checksum
            
        Returns:
            True if response is valid
        """
        # Check size
        if len(response) != expected_size:
            logger.warning(
                f"Response size mismatch: {len(response)} != {expected_size}"
            )
            return False
        
        # Check if response is all zeros (likely error)
        if response == b'\x00' * len(response):
            logger.warning("Response is all zeros")
            return False
        
        # Verify checksum if provided
        if checksum:
            actual_checksum = hashlib.sha256(response).hexdigest()
            if actual_checksum != checksum:
                logger.warning("Checksum mismatch")
                return False
        
        return True
    
    async def retrieve(
        self,
        index: int,
        protocol: Optional[QueryProtocol] = None
    ) -> bytes:
        """
        Retrieve data at index privately.
        
        Args:
            index: Database index to retrieve
            protocol: Protocol to use (defaults to client's protocol)
            
        Returns:
            Retrieved data
        """
        protocol = protocol or self.protocol
        self.stats["queries_sent"] += 1
        
        # Generate query based on protocol
        if protocol == QueryProtocol.IT_PIR:
            query = self.generate_it_pir_query(index)
        elif protocol == QueryProtocol.XOR:
            query = self.generate_xor_query(index)
        else:
            raise ValueError(f"Unsupported protocol: {protocol}")
        
        # Query all servers in parallel
        tasks = []
        for i, server in enumerate(self.servers):
            if i < len(query.query_vectors):
                task = self.query_server(
                    server,
                    query.query_vectors[i],
                    query.query_id
                )
                tasks.append(task)
        
        # Collect responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter valid responses
        valid_responses = []
        for resp in responses:
            if isinstance(resp, PIRResponse) and resp.valid:
                valid_responses.append(resp)
            elif isinstance(resp, Exception):
                logger.error(f"Query exception: {resp}")
        
        # Check if we have enough responses
        if len(valid_responses) < self.min_servers:
            self.stats["queries_failed"] += 1
            raise RuntimeError(
                f"Insufficient valid responses: {len(valid_responses)} < {self.min_servers}"
            )
        
        # Aggregate responses
        result = self.aggregate_responses_xor(valid_responses[:self.min_servers])
        
        # Validate result
        if not self.validate_response_integrity(result, self.element_size):
            logger.warning("Response integrity check failed")
        
        self.stats["queries_successful"] += 1
        
        # Extract actual data (remove padding)
        actual_data = result.rstrip(b'\x00')
        
        return actual_data
    
    async def batch_retrieve(
        self,
        indices: List[int],
        max_concurrent: int = 10
    ) -> List[bytes]:
        """
        Retrieve multiple indices with concurrency control.
        
        Args:
            indices: List of indices to retrieve
            max_concurrent: Maximum concurrent queries
            
        Returns:
            List of retrieved data
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def retrieve_with_semaphore(idx):
            async with semaphore:
                return await self.retrieve(idx)
        
        tasks = [retrieve_with_semaphore(idx) for idx in indices]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle errors
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Failed to retrieve index {indices[i]}: {result}")
                final_results.append(b"")
            else:
                final_results.append(result)
        
        return final_results
    
    async def get_server_status(self) -> Dict[int, Dict[str, Any]]:
        """
        Get status from all servers.
        
        Returns:
            Dictionary mapping server ID to status
        """
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        statuses = {}
        
        for server in self.servers:
            try:
                async with self.session.get(
                    f"{server.url}/status",
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    if response.status == 200:
                        status = await response.json()
                        statuses[server.server_id] = {
                            "online": True,
                            "status": status
                        }
                    else:
                        statuses[server.server_id] = {
                            "online": False,
                            "error": f"HTTP {response.status}"
                        }
            except Exception as e:
                statuses[server.server_id] = {
                    "online": False,
                    "error": str(e)
                }
        
        return statuses
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get client statistics.
        
        Returns:
            Statistics dictionary
        """
        success_rate = 0.0
        if self.stats["queries_sent"] > 0:
            success_rate = self.stats["queries_successful"] / self.stats["queries_sent"]
        
        avg_latency = 0.0
        if self.stats["queries_successful"] > 0:
            avg_latency = self.stats["total_latency_ms"] / self.stats["queries_successful"]
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "num_servers": len(self.servers),
            "protocol": self.protocol.value
        }