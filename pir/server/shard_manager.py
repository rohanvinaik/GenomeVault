"""
PIR Server Implementation
"""

import asyncio
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import logging
from pathlib import Path

from core.constants import PIR_THRESHOLD
from core.exceptions import PIRError

logger = logging.getLogger(__name__)


@dataclass
class ShardConfig:
    """Configuration for a PIR shard server"""
    server_id: int
    total_shards: int
    data_path: Path
    is_trusted_signatory: bool = False
    signatory_weight: int = 0


class PIRServer:
    """
    Server for Private Information Retrieval
    """
    
    def __init__(self, config: ShardConfig):
        self.config = config
        self.database = {}
        self.query_cache = {}
        self._load_database()
    
    def _load_database(self):
        """Load the reference genome database shard"""
        logger.info(f"Loading database shard {self.config.server_id}")
        
        # In production, this would load actual reference genome data
        # For now, we'll simulate with deterministic random data
        np.random.seed(self.config.server_id)
        
        # Simulate reference genome shards (e.g., 1000 chunks of 1KB each)
        self.database_size = 1000
        self.chunk_size = 1024  # 1KB per chunk
        
        for i in range(self.database_size):
            # Generate deterministic data based on server ID and chunk index
            chunk_seed = f"{self.config.server_id}:{i}".encode()
            chunk_hash = hashlib.sha256(chunk_seed).digest()
            self.database[i] = chunk_hash * (self.chunk_size // 32)
    
    async def handle_query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a PIR query from a client
        
        Args:
            request: Query request containing:
                - query_id: Unique query identifier
                - shard_id: Which shard this query is for
                - query: The actual query vector (hex encoded)
                - threshold: Required threshold for reconstruction
                
        Returns:
            Response containing the computed result
        """
        try:
            # Extract request parameters
            query_id = request["query_id"]
            shard_id = request["shard_id"]
            query_hex = request["query"]
            threshold = request.get("threshold", 2)
            
            # Validate shard_id matches our server
            if shard_id != self.config.server_id:
                raise PIRError(f"Wrong shard: expected {self.config.server_id}, got {shard_id}")
            
            # Decode query vector
            query_bytes = bytes.fromhex(query_hex)
            query_vector = np.frombuffer(query_bytes, dtype=np.uint8)
            
            # Validate query size
            if len(query_vector) != self.database_size:
                raise PIRError(f"Query size mismatch: {len(query_vector)} != {self.database_size}")
            
            # Compute PIR response
            response_data = await self._compute_pir_response(query_vector)
            
            # Generate proof if we're a trusted signatory
            proof = None
            if self.config.is_trusted_signatory:
                proof = self._generate_proof(query_id, response_data)
            
            # Cache the query for potential auditing
            self.query_cache[query_id] = {
                "timestamp": asyncio.get_event_loop().time(),
                "query_hash": hashlib.sha256(query_bytes).hexdigest(),
                "response_hash": hashlib.sha256(response_data).hexdigest()
            }
            
            return {
                "query_id": query_id,
                "shard_id": self.config.server_id,
                "response": response_data.hex(),
                "proof": proof.hex() if proof else "",
                "is_trusted_signatory": self.config.is_trusted_signatory
            }
            
        except Exception as e:
            logger.error(f"Query handling failed: {str(e)}")
            raise PIRError(f"Query processing failed: {str(e)}")
    
    async def _compute_pir_response(self, query_vector: np.ndarray) -> bytes:
        """
        Compute the PIR response using the query vector
        
        In standard PIR, this computes the dot product of the query
        vector with the database
        """
        # Initialize result
        result = np.zeros(self.chunk_size, dtype=np.uint8)
        
        # For each database entry where query bit is 1
        for i in range(self.database_size):
            if query_vector[i] == 1:
                # XOR the data chunk into the result
                chunk_data = np.frombuffer(self.database[i], dtype=np.uint8)
                result ^= chunk_data
        
        return result.tobytes()
    
    def _generate_proof(self, query_id: str, response_data: bytes) -> bytes:
        """
        Generate a proof of correct computation for trusted signatories
        """
        # Create proof data
        proof_content = {
            "server_id": self.config.server_id,
            "query_id": query_id,
            "response_hash": hashlib.sha256(response_data).hexdigest(),
            "signatory_weight": self.config.signatory_weight,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        # In production, this would use actual cryptographic signatures
        proof_json = json.dumps(proof_content, sort_keys=True)
        proof_hash = hashlib.sha256(proof_json.encode()).digest()
        
        return proof_hash
    
    async def handle_batch_query(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle multiple queries in parallel"""
        tasks = []
        for request in requests:
            task = self.handle_query(request)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed queries
        valid_responses = []
        for resp in responses:
            if isinstance(resp, dict):
                valid_responses.append(resp)
            else:
                logger.error(f"Batch query failed: {resp}")
        
        return valid_responses
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get server statistics"""
        return {
            "server_id": self.config.server_id,
            "total_shards": self.config.total_shards,
            "database_size": self.database_size,
            "chunk_size": self.chunk_size,
            "total_queries": len(self.query_cache),
            "is_trusted_signatory": self.config.is_trusted_signatory,
            "signatory_weight": self.config.signatory_weight
        }
    
    async def audit_query(self, query_id: str) -> Optional[Dict[str, Any]]:
        """
        Audit a previous query for verification
        """
        if query_id in self.query_cache:
            return self.query_cache[query_id]
        return None


class ThresholdPIRServer(PIRServer):
    """
    Enhanced PIR server with threshold encryption support
    """
    
    def __init__(self, config: ShardConfig, threshold_key_share: Optional[bytes] = None):
        super().__init__(config)
        self.threshold_key_share = threshold_key_share
        
    async def handle_query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle threshold PIR query with additional encryption
        """
        # Get base response
        response = await super().handle_query(request)
        
        # If we have a threshold key share, add encrypted layer
        if self.threshold_key_share:
            response_data = bytes.fromhex(response["response"])
            encrypted_response = self._threshold_encrypt(response_data)
            response["response"] = encrypted_response.hex()
            response["threshold_encrypted"] = True
        
        return response
    
    def _threshold_encrypt(self, data: bytes) -> bytes:
        """
        Apply threshold encryption to the response
        """
        # In production, this would use actual threshold cryptography
        # For now, we'll XOR with the key share
        key_stream = hashlib.sha256(self.threshold_key_share).digest()
        key_stream = key_stream * (len(data) // 32 + 1)
        key_stream = key_stream[:len(data)]
        
        encrypted = bytes(a ^ b for a, b in zip(data, key_stream))
        return encrypted
