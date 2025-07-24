"""
Information-Theoretic PIR implementation.
Provides unconditional privacy guarantees without computational assumptions.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PIRQuery:
    """Information-theoretic PIR query."""
    
    query_id: str
    server_queries: List[np.ndarray]  # One query per server
    index: int  # Target index (private)
    num_servers: int
    metadata: Dict[str, Any]
    
    def get_server_query(self, server_id: int) -> np.ndarray:
        """Get query for specific server."""
        if server_id >= self.num_servers:
            raise ValueError(f"Invalid server ID: {server_id}")
        return self.server_queries[server_id]


@dataclass
class PIRResponse:
    """Response from PIR servers."""
    
    response_id: str
    server_responses: List[np.ndarray]
    reconstructed_data: Optional[bytes] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if all server responses received."""
        return all(r is not None for r in self.server_responses)


class InformationTheoreticPIR:
    """
    Information-Theoretic Private Information Retrieval.
    
    Implements a k-out-of-n threshold PIR scheme where privacy
    is guaranteed as long as fewer than k servers collude.
    """
    
    def __init__(
        self,
        num_servers: int = 3,
        threshold: int = 2,
        field_size: int = 2**32 - 5  # Prime for finite field arithmetic
    ):
        """
        Initialize IT-PIR system.
        
        Args:
            num_servers: Total number of PIR servers
            threshold: Maximum number of colluding servers tolerated
            field_size: Size of finite field for computations
        """
        if threshold >= num_servers:
            raise ValueError("Threshold must be less than number of servers")
        
        self.num_servers = num_servers
        self.threshold = threshold
        self.field_size = field_size
        
        logger.info(
            f"IT-PIR initialized: {num_servers} servers, "
            f"{threshold}-private, field size {field_size}"
        )
    
    def generate_query(
        self,
        index: int,
        database_size: int,
        block_size: int = 1024
    ) -> PIRQuery:
        """
        Generate PIR queries for all servers.
        
        Args:
            index: Index of desired element (private)
            database_size: Total number of elements
            block_size: Size of each database element in bytes
            
        Returns:
            PIR query containing server-specific queries
        """
        logger.info(f"Generating PIR query for index {index} (database size: {database_size})")
        
        # Create unit vector for desired index
        unit_vector = np.zeros(database_size, dtype=np.uint64)
        unit_vector[index] = 1
        
        # Generate random shares that sum to unit vector
        queries = self._split_vector_randomly(unit_vector)
        
        # Create query object
        query = PIRQuery(
            query_id=self._generate_query_id(index),
            server_queries=queries,
            index=index,
            num_servers=self.num_servers,
            metadata={
                "database_size": database_size,
                "block_size": block_size,
                "threshold": self.threshold,
                "timestamp": time.time()
            }
        )
        
        return query
    
    def _split_vector_randomly(self, vector: np.ndarray) -> List[np.ndarray]:
        """
        Split vector into random shares that sum to original.
        
        This implements additive secret sharing in finite field.
        """
        n = self.num_servers
        shares = []
        
        # Generate n-1 random shares
        for i in range(n - 1):
            share = np.random.randint(0, self.field_size, size=len(vector), dtype=np.uint64)
            shares.append(share)
        
        # Compute last share to ensure sum equals original vector
        last_share = vector.copy()
        for share in shares:
            last_share = (last_share - share) % self.field_size
        
        shares.append(last_share)
        
        # Verify reconstruction
        reconstructed = np.zeros_like(vector)
        for share in shares:
            reconstructed = (reconstructed + share) % self.field_size
        
        assert np.array_equal(reconstructed % self.field_size, vector % self.field_size), \
            "Failed to correctly split vector"
        
        return shares
    
    def process_server_query(
        self,
        server_id: int,
        query: PIRQuery,
        database: List[bytes]
    ) -> np.ndarray:
        """
        Process PIR query on server side.
        
        Args:
            server_id: ID of this server
            query: PIR query from client
            database: Server's database
            
        Returns:
            Response vector
        """
        server_query = query.get_server_query(server_id)
        
        # Compute inner product of query with database
        response = np.zeros(len(database[0]) if database else 0, dtype=np.uint64)
        
        for i, db_entry in enumerate(database):
            if i < len(server_query):
                # Convert database entry to numeric vector
                entry_vector = np.frombuffer(db_entry, dtype=np.uint8)
                
                # Scale by query coefficient and add to response
                coefficient = server_query[i]
                for j in range(min(len(entry_vector), len(response))):
                    response[j] = (response[j] + coefficient * int(entry_vector[j])) % self.field_size
        
        return response
    
    def reconstruct_response(
        self,
        query: PIRQuery,
        server_responses: List[np.ndarray]
    ) -> bytes:
        """
        Reconstruct data from server responses.
        
        Args:
            query: Original PIR query
            server_responses: Responses from all servers
            
        Returns:
            Reconstructed data
        """
        if len(server_responses) != self.num_servers:
            raise ValueError(
                f"Expected {self.num_servers} responses, got {len(server_responses)}"
            )
        
        # Sum all responses in finite field
        reconstructed = np.zeros_like(server_responses[0])
        
        for response in server_responses:
            reconstructed = (reconstructed + response) % self.field_size
        
        # Convert back to bytes
        # Assuming values fit in uint8 after modular reduction
        reconstructed_bytes = (reconstructed % 256).astype(np.uint8).tobytes()
        
        return reconstructed_bytes
    
    def _generate_query_id(self, index: int) -> str:
        """Generate unique query ID."""
        data = {
            "index": index,
            "timestamp": time.time(),
            "nonce": np.random.bytes(8).hex()
        }
        
        return hashlib.sha256(
            json.dumps(data, sort_keys=True).encode()
        ).hexdigest()[:16]
    
    def create_multi_query(
        self,
        indices: List[int],
        database_size: int,
        block_size: int = 1024
    ) -> List[PIRQuery]:
        """
        Create multiple PIR queries for batch retrieval.
        
        Uses query combination techniques to improve efficiency.
        """
        queries = []
        
        # For small batches, use individual queries
        if len(indices) <= 10:
            for idx in indices:
                query = self.generate_query(idx, database_size, block_size)
                queries.append(query)
        else:
            # For larger batches, use batching optimizations
            # Group indices to reduce total communication
            batch_size = min(100, len(indices))
            
            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]
                
                # Create combined query for batch
                combined_vector = np.zeros(database_size, dtype=np.uint64)
                for idx in batch_indices:
                    combined_vector[idx] = 1
                
                # Split combined vector
                batch_queries = self._split_vector_randomly(combined_vector)
                
                query = PIRQuery(
                    query_id=self._generate_query_id(hash(tuple(batch_indices))),
                    server_queries=batch_queries,
                    index=-1,  # Batch query
                    num_servers=self.num_servers,
                    metadata={
                        "database_size": database_size,
                        "block_size": block_size,
                        "batch_indices": batch_indices,
                        "batch_size": len(batch_indices),
                        "timestamp": time.time()
                    }
                )
                
                queries.append(query)
        
        return queries


class RobustITPIR(InformationTheoreticPIR):
    """
    Robust IT-PIR with Byzantine fault tolerance.
    
    Handles malicious servers that may return incorrect responses.
    """
    
    def __init__(
        self,
        num_servers: int = 5,
        threshold: int = 2,
        byzantine_threshold: int = 1
    ):
        """
        Initialize robust IT-PIR.
        
        Args:
            num_servers: Total servers
            threshold: Privacy threshold
            byzantine_threshold: Maximum Byzantine servers
        """
        super().__init__(num_servers, threshold)
        self.byzantine_threshold = byzantine_threshold
        
        # Need enough servers for error correction
        min_servers = 2 * byzantine_threshold + threshold + 1
        if num_servers < min_servers:
            raise ValueError(
                f"Need at least {min_servers} servers for "
                f"{byzantine_threshold}-Byzantine, {threshold}-private PIR"
            )
    
    def process_responses_with_verification(
        self,
        query: PIRQuery,
        server_responses: List[Tuple[np.ndarray, bytes]]
    ) -> bytes:
        """
        Process responses with Byzantine fault detection.
        
        Args:
            query: Original query
            server_responses: List of (response, proof) tuples
            
        Returns:
            Verified reconstructed data
        """
        # Use error-correcting codes to detect/correct errors
        valid_responses = []
        
        for i, (response, proof) in enumerate(server_responses):
            if self._verify_response_proof(query, i, response, proof):
                valid_responses.append(response)
            else:
                logger.warning(f"Invalid response from server {i}")
        
        if len(valid_responses) < self.num_servers - self.byzantine_threshold:
            raise ValueError("Too many invalid responses")
        
        # Use Reed-Solomon decoding for error correction
        return self._reed_solomon_decode(valid_responses)
    
    def _verify_response_proof(
        self,
        query: PIRQuery,
        server_id: int,
        response: np.ndarray,
        proof: bytes
    ) -> bool:
        """Verify response authenticity (simplified)."""
        # In practice, would use Merkle proofs or other authentication
        expected_hash = hashlib.sha256(
            response.tobytes() + query.query_id.encode()
        ).digest()
        
        return proof == expected_hash
    
    def _reed_solomon_decode(self, responses: List[np.ndarray]) -> bytes:
        """Decode using Reed-Solomon error correction."""
        # Simplified - in practice use proper RS decoding
        # For now, use majority voting
        
        result = np.zeros_like(responses[0])
        
        for i in range(len(result)):
            values = [r[i] for r in responses]
            # Take most common value
            result[i] = max(set(values), key=values.count)
        
        return (result % 256).astype(np.uint8).tobytes()


# Example usage
if __name__ == "__main__":
    # Initialize IT-PIR system
    pir = InformationTheoreticPIR(num_servers=3, threshold=2)
    
    # Simulate database
    database_size = 1000
    block_size = 1024
    
    # Create mock database on each server
    databases = []
    for server_id in range(pir.num_servers):
        db = []
        for i in range(database_size):
            # Each entry contains genomic data
            data = f"Genomic_Data_Block_{i}_Server_{server_id}".encode()
            data = data.ljust(block_size, b'\0')
            db.append(data)
        databases.append(db)
    
    # Client wants to retrieve index 42 privately
    target_index = 42
    
    print(f"IT-PIR Example: Retrieving index {target_index}")
    print(f"Database size: {database_size} blocks")
    print(f"Servers: {pir.num_servers}, Threshold: {pir.threshold}")
    print("="*60)
    
    # Generate PIR query
    start_time = time.time()
    query = pir.generate_query(target_index, database_size, block_size)
    query_time = time.time() - start_time
    
    print(f"\nQuery generation time: {query_time*1000:.2f} ms")
    print(f"Query size per server: {query.server_queries[0].nbytes / 1024:.2f} KB")
    
    # Each server processes its query
    server_responses = []
    
    for server_id in range(pir.num_servers):
        start_time = time.time()
        response = pir.process_server_query(
            server_id,
            query,
            databases[server_id]
        )
        response_time = time.time() - start_time
        
        server_responses.append(response)
        
        print(f"\nServer {server_id} response time: {response_time*1000:.2f} ms")
        print(f"Response size: {response.nbytes / 1024:.2f} KB")
    
    # Client reconstructs the data
    start_time = time.time()
    reconstructed = pir.reconstruct_response(query, server_responses)
    reconstruction_time = time.time() - start_time
    
    print(f"\nReconstruction time: {reconstruction_time*1000:.2f} ms")
    
    # Verify correctness
    expected = databases[0][target_index]  # All servers have same data
    actual = reconstructed[:len(expected)]
    
    print(f"\nRetrieved data matches: {actual == expected}")
    print(f"Retrieved: {actual[:50]}...")  # First 50 bytes
    
    # Test batch queries
    print("\n\nBatch Query Example:")
    print("="*60)
    
    batch_indices = [10, 42, 100, 200, 500]
    batch_queries = pir.create_multi_query(
        batch_indices,
        database_size,
        block_size
    )
    
    print(f"Created {len(batch_queries)} queries for {len(batch_indices)} indices")
    
    # Test robust PIR
    print("\n\nRobust IT-PIR Example (Byzantine tolerance):")
    print("="*60)
    
    robust_pir = RobustITPIR(
        num_servers=5,
        threshold=2,
        byzantine_threshold=1
    )
    
    print(f"Servers: {robust_pir.num_servers}")
    print(f"Privacy threshold: {robust_pir.threshold}")
    print(f"Byzantine threshold: {robust_pir.byzantine_threshold}")
    print("Can tolerate 1 malicious server while maintaining privacy")
