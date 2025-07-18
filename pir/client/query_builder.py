"""
PIR Client Implementation
"""

import asyncio
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import httpx

from core.constants import MIN_PIR_SERVERS, PIR_THRESHOLD
from core.exceptions import PIRError


@dataclass
class PIRQuery:
    """Represents a PIR query"""
    query_id: str
    index: int
    shard_queries: Dict[int, bytes]
    num_shards: int
    threshold: int


@dataclass
class PIRResponse:
    """Response from PIR server"""
    shard_id: int
    data: bytes
    proof: Optional[bytes] = None


class PIRClient:
    """
    Client for Private Information Retrieval
    """
    
    def __init__(self, servers: List[str], threshold: int = 2):
        """
        Initialize PIR client
        
        Args:
            servers: List of PIR server URLs
            threshold: Minimum number of honest servers required
        """
        if len(servers) < MIN_PIR_SERVERS:
            raise PIRError(f"At least {MIN_PIR_SERVERS} servers required")
        
        self.servers = servers
        self.threshold = threshold
        self.client = httpx.AsyncClient(timeout=30.0)
        
    async def query(self, index: int, database_size: int) -> bytes:
        """
        Perform a private information retrieval query
        
        Args:
            index: Index of the item to retrieve
            database_size: Total size of the database
            
        Returns:
            Retrieved data
        """
        # Generate query shards
        query = self._generate_query(index, database_size)
        
        # Send queries to all servers in parallel
        tasks = []
        for shard_id, server in enumerate(self.servers):
            task = self._query_server(server, shard_id, query)
            tasks.append(task)
        
        # Collect responses
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed responses
        valid_responses = []
        for resp in responses:
            if isinstance(resp, PIRResponse):
                valid_responses.append(resp)
            else:
                print(f"Server failed: {resp}")
        
        # Check if we have enough responses
        if len(valid_responses) < self.threshold:
            raise PIRError(f"Insufficient responses: {len(valid_responses)} < {self.threshold}")
        
        # Reconstruct the data
        return self._reconstruct_data(valid_responses, query)
    
    def _generate_query(self, index: int, database_size: int) -> PIRQuery:
        """
        Generate PIR query using information-theoretic secure sharing
        """
        num_shards = len(self.servers)
        query_id = hashlib.sha256(f"{index}:{database_size}".encode()).hexdigest()[:16]
        
        # Create random masks for all servers except the last
        shard_queries = {}
        accumulated_query = np.zeros(database_size, dtype=np.uint8)
        
        # Generate random queries for n-1 servers
        for i in range(num_shards - 1):
            # Random binary vector
            shard_query = np.random.randint(0, 2, size=database_size, dtype=np.uint8)
            shard_queries[i] = shard_query.tobytes()
            accumulated_query ^= shard_query
        
        # Last shard ensures the queries XOR to a one-hot vector at the desired index
        target_query = np.zeros(database_size, dtype=np.uint8)
        target_query[index] = 1
        final_query = accumulated_query ^ target_query
        shard_queries[num_shards - 1] = final_query.tobytes()
        
        return PIRQuery(
            query_id=query_id,
            index=index,
            shard_queries=shard_queries,
            num_shards=num_shards,
            threshold=self.threshold
        )
    
    async def _query_server(self, server_url: str, shard_id: int, query: PIRQuery) -> PIRResponse:
        """
        Query a single PIR server
        """
        try:
            # Prepare request
            request_data = {
                "query_id": query.query_id,
                "shard_id": shard_id,
                "query": query.shard_queries[shard_id].hex(),
                "threshold": query.threshold
            }
            
            # Send request
            response = await self.client.post(
                f"{server_url}/pir/query",
                json=request_data
            )
            
            if response.status_code != 200:
                raise PIRError(f"Server returned {response.status_code}")
            
            # Parse response
            data = response.json()
            return PIRResponse(
                shard_id=shard_id,
                data=bytes.fromhex(data["response"]),
                proof=bytes.fromhex(data.get("proof", ""))
            )
            
        except Exception as e:
            raise PIRError(f"Failed to query server {server_url}: {str(e)}")
    
    def _reconstruct_data(self, responses: List[PIRResponse], query: PIRQuery) -> bytes:
        """
        Reconstruct the original data from PIR responses
        """
        # Sort responses by shard ID
        responses.sort(key=lambda r: r.shard_id)
        
        # For threshold PIR, we need at least 'threshold' responses
        if len(responses) < self.threshold:
            raise PIRError("Insufficient responses for reconstruction")
        
        # XOR all responses together
        result = None
        for response in responses[:self.threshold]:
            if result is None:
                result = np.frombuffer(response.data, dtype=np.uint8)
            else:
                result ^= np.frombuffer(response.data, dtype=np.uint8)
        
        return result.tobytes()
    
    async def batch_query(self, indices: List[int], database_size: int) -> Dict[int, bytes]:
        """
        Perform multiple PIR queries in parallel
        """
        tasks = []
        for index in indices:
            task = self.query(index, database_size)
            tasks.append((index, task))
        
        results = {}
        for index, task in tasks:
            try:
                data = await task
                results[index] = data
            except Exception as e:
                print(f"Failed to retrieve index {index}: {e}")
        
        return results
    
    def calculate_privacy_guarantee(self, num_servers: int, honesty_rate: float) -> float:
        """
        Calculate the probability of privacy failure
        
        Args:
            num_servers: Total number of servers
            honesty_rate: Probability that a server is honest (e.g., 0.98)
            
        Returns:
            Probability of privacy failure
        """
        # Privacy fails if all servers collude (none are honest)
        p_fail = (1 - honesty_rate) ** num_servers
        return p_fail
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class CachedPIRClient(PIRClient):
    """
    PIR Client with local caching for improved performance
    """
    
    def __init__(self, servers: List[str], threshold: int = 2, cache_size: int = 1000):
        super().__init__(servers, threshold)
        self.cache = {}
        self.cache_size = cache_size
        self.access_count = {}
    
    async def query(self, index: int, database_size: int) -> bytes:
        """
        Query with caching
        """
        cache_key = f"{index}:{database_size}"
        
        # Check cache
        if cache_key in self.cache:
            self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
            return self.cache[cache_key]
        
        # Perform PIR query
        result = await super().query(index, database_size)
        
        # Update cache with LRU eviction
        self._update_cache(cache_key, result)
        
        return result
    
    def _update_cache(self, key: str, value: bytes):
        """Update cache with LRU eviction"""
        if len(self.cache) >= self.cache_size:
            # Find least recently used item
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]
        
        self.cache[key] = value
        self.access_count[key] = 1
