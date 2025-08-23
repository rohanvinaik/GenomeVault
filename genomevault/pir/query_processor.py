"""
Constant-time query processor for PIR operations.

Implements timing attack mitigation through constant-time operations
and query processing with replay protection.
"""

import hashlib
import hmac
import secrets
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Set
from enum import Enum
import threading
from collections import deque

import numpy as np

from genomevault.utils.logging import get_logger
from genomevault.pir.xor_scheme import PIRQuery, PIRResponse, XORSchemeParams
from genomevault.pir.byzantine_handler import ByzantineHandler, ByzantineConfig

logger = get_logger(__name__)


class ProcessingMode(Enum):
    """Query processing modes."""
    CONSTANT_TIME = "constant_time"
    FAST_PATH = "fast_path"
    SECURE_MODE = "secure_mode"


@dataclass
class QueryMetadata:
    """Metadata for query tracking and replay protection."""
    query_id: str
    client_id: str
    timestamp: float
    nonce: bytes
    query_hash: str
    response_hash: Optional[str] = None
    processing_time: Optional[float] = None
    server_responses: Dict[int, bytes] = field(default_factory=dict)
    status: str = "pending"


@dataclass
class ProcessorConfig:
    """Configuration for query processor."""
    constant_time_ns: int = 10_000_000  # 10ms constant processing time
    max_query_age: float = 60.0  # Maximum query age in seconds
    nonce_cache_size: int = 10000  # Size of nonce cache
    query_cache_size: int = 1000  # Size of query result cache
    enable_caching: bool = True  # Enable query result caching
    processing_mode: ProcessingMode = ProcessingMode.CONSTANT_TIME
    batch_size: int = 100  # Maximum batch size
    num_worker_threads: int = 4  # Worker threads for processing


class ConstantTimeQueryProcessor:
    """
    Query processor with constant-time guarantees.
    
    Implements:
    1. Constant-time query processing
    2. Replay attack protection
    3. Query caching and batching
    4. Timing attack mitigation
    """
    
    def __init__(self, 
                 config: ProcessorConfig,
                 xor_params: XORSchemeParams,
                 byzantine_config: ByzantineConfig):
        """
        Initialize query processor.
        
        Args:
            config: Processor configuration
            xor_params: XOR scheme parameters
            byzantine_config: Byzantine handler configuration
        """
        self.config = config
        self.xor_params = xor_params
        self.byzantine_handler = ByzantineHandler(byzantine_config)
        
        # Query tracking
        self.active_queries: Dict[str, QueryMetadata] = {}
        self.query_history: deque = deque(maxlen=config.query_cache_size)
        
        # Nonce tracking for replay protection
        self.used_nonces: Set[bytes] = set()
        self.nonce_timestamps: Dict[bytes, float] = {}
        
        # Query result cache
        self.result_cache: Dict[str, bytes] = {}
        
        # Threading for constant-time processing
        self.processing_lock = threading.Lock()
        self.worker_threads = []
        
        # Statistics
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'replay_attempts': 0,
            'timing_violations': 0,
            'byzantine_detections': 0
        }
        
        logger.info(f"Initialized query processor in {config.processing_mode.value} mode")
    
    def process_query(self, 
                     query: PIRQuery,
                     database: np.ndarray,
                     server_id: int,
                     client_id: str) -> PIRResponse:
        """
        Process PIR query with constant-time guarantee.
        
        Args:
            query: PIR query to process
            database: Database array
            server_id: Processing server ID
            client_id: Client identifier
            
        Returns:
            PIR response
        """
        start_time = time.perf_counter_ns()
        
        # Create query metadata
        metadata = QueryMetadata(
            query_id=query.query_id,
            client_id=client_id,
            timestamp=query.timestamp,
            nonce=query.nonce,
            query_hash=self._hash_query(query)
        )
        
        # Check replay protection
        if not self._check_replay_protection(metadata):
            # Still process to maintain constant time
            response = self._generate_dummy_response(query, server_id)
            self.stats['replay_attempts'] += 1
        else:
            # Check cache if enabled
            if self.config.enable_caching and metadata.query_hash in self.result_cache:
                cached_result = self.result_cache[metadata.query_hash]
                response = self._create_response(cached_result, query, server_id)
                self.stats['cache_hits'] += 1
            else:
                # Process query
                response = self._process_query_internal(query, database, server_id)
                
                # Cache result
                if self.config.enable_caching:
                    self.result_cache[metadata.query_hash] = response.response_data
        
        # Update metadata
        metadata.processing_time = (time.perf_counter_ns() - start_time) / 1e9
        metadata.server_responses[server_id] = response.response_data
        metadata.response_hash = hashlib.sha256(response.response_data).hexdigest()
        metadata.status = "completed"
        
        # Store in history
        with self.processing_lock:
            self.active_queries[query.query_id] = metadata
            self.query_history.append(metadata)
            self.stats['total_queries'] += 1
        
        # Ensure constant execution time
        if self.config.processing_mode == ProcessingMode.CONSTANT_TIME:
            self._enforce_constant_time(start_time)
        
        return response
    
    def _process_query_internal(self, 
                               query: PIRQuery,
                               database: np.ndarray,
                               server_id: int) -> PIRResponse:
        """
        Internal query processing with constant-time operations.
        
        Args:
            query: Query to process
            database: Database array
            server_id: Server ID
            
        Returns:
            PIR response
        """
        # Validate query
        if not self._validate_query_integrity(query):
            return self._generate_dummy_response(query, server_id)
        
        # Perform constant-time computation
        result = self._constant_time_computation(database, query.query_vector)
        
        # Encode result
        result_bytes = self._encode_result_constant_time(result)
        
        # Pad to fixed size (1KB blocks)
        padded_result = self._pad_to_blocks(result_bytes)
        
        # Create response
        response = PIRResponse(
            response_data=padded_result,
            server_id=server_id,
            query_id=query.query_id,
            timestamp=time.time(),
            padding=b'',
            checksum=self._compute_checksum_constant_time(padded_result)
        )
        
        return response
    
    def _constant_time_computation(self, 
                                  database: np.ndarray,
                                  query_vector: np.ndarray) -> np.ndarray:
        """
        Perform computation in constant time.
        
        Uses bit-slicing and masking to ensure timing independence
        from query pattern.
        
        Args:
            database: Database matrix
            query_vector: Query vector
            
        Returns:
            Computation result
        """
        # Ensure binary inputs
        database = database.astype(np.uint8)
        query_vector = query_vector.astype(np.uint8)
        
        # Initialize result
        num_cols = database.shape[1] if len(database.shape) > 1 else len(database)
        result = np.zeros(num_cols, dtype=np.uint8)
        
        # Process each row with constant-time selection
        for i in range(len(query_vector)):
            # Create mask (all 0s or all 1s) without branching
            mask = np.full(num_cols, query_vector[i], dtype=np.uint8)
            
            # Get database row (handle both 1D and 2D arrays)
            if len(database.shape) > 1:
                row = database[i] if i < len(database) else np.zeros(num_cols, dtype=np.uint8)
            else:
                row = database if i == 0 else np.zeros(num_cols, dtype=np.uint8)
            
            # Apply mask and XOR (constant time)
            masked_row = row & mask
            result ^= masked_row
        
        return result
    
    def _encode_result_constant_time(self, result: np.ndarray) -> bytes:
        """
        Encode result with constant-time operations.
        
        Args:
            result: Result array
            
        Returns:
            Encoded bytes
        """
        # Pad to fixed size to avoid timing leaks
        fixed_size = 1024 * 8  # 1KB in bits
        
        if len(result) < fixed_size:
            padded = np.pad(result, (0, fixed_size - len(result)), mode='constant')
        else:
            padded = result[:fixed_size]
        
        # Pack bits to bytes
        packed = np.packbits(padded)
        
        return packed.tobytes()
    
    def _pad_to_blocks(self, data: bytes) -> bytes:
        """
        Pad data to fixed-size blocks (1KB).
        
        Args:
            data: Data to pad
            
        Returns:
            Padded data
        """
        block_size = 1024  # 1KB blocks
        
        if len(data) >= block_size:
            return data[:block_size]
        
        # Pad with random data
        padding_needed = block_size - len(data) - 4  # 4 bytes for length
        padding = secrets.token_bytes(padding_needed)
        
        # Encode original length
        length_bytes = len(data).to_bytes(4, 'big')
        
        return data + padding + length_bytes
    
    def _compute_checksum_constant_time(self, data: bytes) -> bytes:
        """
        Compute checksum in constant time.
        
        Args:
            data: Data to checksum
            
        Returns:
            Checksum bytes
        """
        # Use HMAC for constant-time comparison
        key = secrets.token_bytes(32)
        h = hmac.new(key, data, hashlib.sha256)
        
        # Process in fixed-size chunks to maintain constant time
        chunk_size = 64
        for i in range(0, 1024, chunk_size):  # Process full 1KB
            chunk = data[i:i+chunk_size] if i < len(data) else b'\x00' * chunk_size
            h.update(chunk)
        
        return h.digest()
    
    def _check_replay_protection(self, metadata: QueryMetadata) -> bool:
        """
        Check for replay attacks.
        
        Args:
            metadata: Query metadata
            
        Returns:
            True if query is valid (not a replay)
        """
        current_time = time.time()
        
        # Check query age
        if current_time - metadata.timestamp > self.config.max_query_age:
            logger.warning(f"Query {metadata.query_id} is too old")
            return False
        
        # Check nonce
        with self.processing_lock:
            if metadata.nonce in self.used_nonces:
                logger.warning(f"Nonce replay detected for query {metadata.query_id}")
                return False
            
            # Add nonce to used set
            self.used_nonces.add(metadata.nonce)
            self.nonce_timestamps[metadata.nonce] = current_time
            
            # Clean old nonces
            self._cleanup_old_nonces(current_time)
        
        return True
    
    def _cleanup_old_nonces(self, current_time: float) -> None:
        """
        Clean up old nonces to prevent memory growth.
        
        Args:
            current_time: Current timestamp
        """
        if len(self.used_nonces) > self.config.nonce_cache_size:
            # Remove nonces older than max_query_age
            cutoff_time = current_time - self.config.max_query_age
            
            old_nonces = [
                nonce for nonce, timestamp in self.nonce_timestamps.items()
                if timestamp < cutoff_time
            ]
            
            for nonce in old_nonces:
                self.used_nonces.discard(nonce)
                del self.nonce_timestamps[nonce]
    
    def _validate_query_integrity(self, query: PIRQuery) -> bool:
        """
        Validate query integrity.
        
        Args:
            query: Query to validate
            
        Returns:
            True if query is valid
        """
        # Check signature exists
        if not query.signature:
            return False
        
        # Check query vector size
        if len(query.query_vector) == 0:
            return False
        
        # Verify timestamp is reasonable
        current_time = time.time()
        if abs(current_time - query.timestamp) > self.config.max_query_age:
            return False
        
        return True
    
    def _generate_dummy_response(self, query: PIRQuery, server_id: int) -> PIRResponse:
        """
        Generate dummy response for invalid queries.
        
        Maintains constant time by returning random data.
        
        Args:
            query: Original query
            server_id: Server ID
            
        Returns:
            Dummy response
        """
        # Generate random data of fixed size
        dummy_data = secrets.token_bytes(1024)  # 1KB
        
        return PIRResponse(
            response_data=dummy_data,
            server_id=server_id,
            query_id=query.query_id,
            timestamp=time.time(),
            padding=b'',
            checksum=hashlib.sha256(dummy_data).digest()
        )
    
    def _create_response(self, data: bytes, query: PIRQuery, server_id: int) -> PIRResponse:
        """
        Create response from cached data.
        
        Args:
            data: Cached response data
            query: Original query
            server_id: Server ID
            
        Returns:
            PIR response
        """
        return PIRResponse(
            response_data=data,
            server_id=server_id,
            query_id=query.query_id,
            timestamp=time.time(),
            padding=b'',
            checksum=hashlib.sha256(data).digest()
        )
    
    def _hash_query(self, query: PIRQuery) -> str:
        """
        Create hash of query for caching.
        
        Args:
            query: Query to hash
            
        Returns:
            Query hash
        """
        h = hashlib.sha256()
        h.update(query.query_vector.tobytes())
        h.update(query.query_type.value.encode())
        
        return h.hexdigest()
    
    def _enforce_constant_time(self, start_time: int) -> None:
        """
        Enforce constant execution time.
        
        Args:
            start_time: Start time in nanoseconds
        """
        elapsed = time.perf_counter_ns() - start_time
        target = self.config.constant_time_ns
        
        if elapsed < target:
            # Busy wait for remaining time
            remaining = target - elapsed
            end_time = time.perf_counter_ns() + remaining
            
            while time.perf_counter_ns() < end_time:
                # Perform dummy operations to avoid optimization
                _ = secrets.token_bytes(1)
        elif elapsed > target * 1.1:  # 10% tolerance
            self.stats['timing_violations'] += 1
            logger.warning(f"Timing violation: {elapsed}ns > {target}ns")
    
    def batch_process_queries(self, 
                            queries: List[Tuple[PIRQuery, str]],
                            database: np.ndarray,
                            server_id: int) -> List[PIRResponse]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of (query, client_id) tuples
            database: Database array
            server_id: Server ID
            
        Returns:
            List of responses
        """
        responses = []
        
        # Process in batches for efficiency
        for i in range(0, len(queries), self.config.batch_size):
            batch = queries[i:i + self.config.batch_size]
            
            # Process each query in batch
            batch_responses = []
            for query, client_id in batch:
                response = self.process_query(query, database, server_id, client_id)
                batch_responses.append(response)
            
            responses.extend(batch_responses)
        
        return responses
    
    def combine_server_responses(self, 
                                responses: Dict[int, PIRResponse]) -> Optional[bytes]:
        """
        Combine responses from multiple servers.
        
        Args:
            responses: Dictionary of server_id -> response
            
        Returns:
            Combined result or None if failed
        """
        # Check Byzantine behavior
        response_list = [(sid, r.response_data) for sid, r in responses.items()]
        suspicious = self.byzantine_handler.detect_byzantine_behavior(response_list)
        
        if suspicious:
            self.stats['byzantine_detections'] += 1
            logger.warning(f"Byzantine behavior detected from servers: {suspicious}")
        
        # Verify consistency
        if not self.byzantine_handler.verify_response_consistency(response_list):
            logger.error("Response consistency check failed")
            return None
        
        # Decode with Byzantine tolerance
        expected_size = 1024  # 1KB blocks
        result = self.byzantine_handler.decode_responses(response_list, expected_size)
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processor statistics.
        
        Returns:
            Statistics dictionary
        """
        with self.processing_lock:
            stats = self.stats.copy()
            stats['active_queries'] = len(self.active_queries)
            stats['cached_results'] = len(self.result_cache)
            stats['used_nonces'] = len(self.used_nonces)
            stats['cache_hit_rate'] = (
                stats['cache_hits'] / stats['total_queries'] 
                if stats['total_queries'] > 0 else 0
            )
            
        return stats


class SecureQueryProcessor(ConstantTimeQueryProcessor):
    """
    Enhanced query processor with additional security features.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize secure processor."""
        super().__init__(*args, **kwargs)
        
        # Additional security features
        self.query_rate_limits: Dict[str, List[float]] = {}
        self.max_queries_per_minute = 100
        
    def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client exceeds rate limit.
        
        Args:
            client_id: Client identifier
            
        Returns:
            True if within rate limit
        """
        current_time = time.time()
        
        if client_id not in self.query_rate_limits:
            self.query_rate_limits[client_id] = []
        
        # Remove old timestamps
        cutoff = current_time - 60  # 1 minute window
        self.query_rate_limits[client_id] = [
            t for t in self.query_rate_limits[client_id] if t > cutoff
        ]
        
        # Check limit
        if len(self.query_rate_limits[client_id]) >= self.max_queries_per_minute:
            logger.warning(f"Rate limit exceeded for client {client_id}")
            return False
        
        # Add current timestamp
        self.query_rate_limits[client_id].append(current_time)
        
        return True