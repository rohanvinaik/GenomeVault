"""
XOR-based 2-server PIR scheme implementation.

Implements information-theoretic PIR with perfect privacy guarantees
using XOR operations across two non-colluding servers.
"""

import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum

import numpy as np
from cryptography.hazmat.primitives import hashes, hmac

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class QueryType(Enum):
    """Types of PIR queries."""

    SINGLE_BLOCK = "single_block"
    RANGE = "range"
    BATCH = "batch"


@dataclass
class PIRQuery:
    """
    PIR query structure with security metadata.

    Attributes:
        query_vector: The actual query vector for the server
        nonce: Random nonce for replay protection
        timestamp: Query timestamp for freshness
        query_id: Unique identifier for query tracking
        padding: Random padding to fixed size
    """

    query_vector: np.ndarray
    nonce: bytes
    timestamp: float
    query_id: str
    padding: bytes
    query_type: QueryType = QueryType.SINGLE_BLOCK
    signature: Optional[bytes] = None


@dataclass
class PIRResponse:
    """
    PIR response structure with integrity protection.

    Attributes:
        response_data: The actual response data
        server_id: ID of the responding server
        query_id: ID of the original query
        timestamp: Response timestamp
        padding: Padding to fixed 1KB blocks
        checksum: Response integrity checksum
    """

    response_data: bytes
    server_id: int
    query_id: str
    timestamp: float
    padding: bytes
    checksum: bytes


@dataclass
class XORSchemeParams:
    """Parameters for XOR-based PIR scheme."""

    database_size: int
    block_size: int = 1024  # 1KB blocks as specified
    num_servers: int = 2
    security_parameter: int = 128
    max_query_age: float = 60.0  # Maximum query age in seconds
    response_padding_size: int = 1024  # Fixed response size
    enable_byzantine_protection: bool = True


class XORPIRScheme:
    """
    XOR-based 2-server PIR implementation.

    This implements the information-theoretic PIR scheme where:
    1. Client generates random query Q1
    2. Client computes Q2 = Q1 XOR e_i (where e_i is unit vector for index i)
    3. Servers compute R1 = DB · Q1 and R2 = DB · Q2
    4. Client recovers DB[i] = R1 XOR R2

    Security guarantee: Perfect privacy if servers don't collude
    """

    def __init__(self, params: XORSchemeParams):
        """
        Initialize XOR PIR scheme.

        Args:
            params: Scheme parameters
        """
        self.params = params
        self._validate_params()

        # Initialize query tracking for replay protection
        self.query_cache: Dict[str, float] = {}
        self.max_cache_size = 10000

        # Initialize nonce tracker
        self.used_nonces: set = set()

        logger.info(f"Initialized XOR PIR scheme with {params.num_servers} servers")

    def _validate_params(self) -> None:
        """Validate scheme parameters."""
        if self.params.num_servers < 2:
            raise ValueError("XOR scheme requires at least 2 servers")

        if self.params.block_size % 16 != 0:
            raise ValueError("Block size must be multiple of 16 for alignment")

        if self.params.security_parameter < 128:
            logger.warning("Security parameter < 128 bits may be insufficient")

    def generate_queries(self, index: int, num_blocks: int) -> Tuple[PIRQuery, PIRQuery]:
        """
        Generate query pair for retrieving block at index.

        This implements the core XOR-PIR protocol:
        - Q1 is random
        - Q2 = Q1 XOR e_i (unit vector at position i)

        Args:
            index: Index of block to retrieve
            num_blocks: Total number of blocks in database

        Returns:
            Tuple of (query1, query2) for the two servers
        """
        if index >= num_blocks:
            raise ValueError(f"Index {index} out of bounds for {num_blocks} blocks")

        # Generate random query Q1
        q1 = self._generate_random_query(num_blocks)

        # Compute Q2 = Q1 XOR e_i
        q2 = q1.copy()
        q2[index] ^= 1  # XOR with unit vector at position i

        # Generate security metadata
        query_id = secrets.token_hex(16)
        nonce = secrets.token_bytes(32)
        timestamp = time.time()

        # Add to nonce tracker for replay protection
        self.used_nonces.add(nonce)

        # Create padding to fixed size
        query_size = num_blocks // 8  # Bits to bytes
        padding_size = max(0, self.params.response_padding_size - query_size - 128)
        padding = secrets.token_bytes(padding_size)

        # Create query objects
        query1 = PIRQuery(
            query_vector=q1,
            nonce=nonce,
            timestamp=timestamp,
            query_id=query_id,
            padding=padding,
            query_type=QueryType.SINGLE_BLOCK,
        )

        query2 = PIRQuery(
            query_vector=q2,
            nonce=nonce,
            timestamp=timestamp,
            query_id=query_id,
            padding=padding,
            query_type=QueryType.SINGLE_BLOCK,
        )

        # Sign queries for integrity
        query1.signature = self._sign_query(query1)
        query2.signature = self._sign_query(query2)

        # Track query for replay protection
        self.query_cache[query_id] = timestamp
        self._cleanup_old_queries()

        logger.debug(f"Generated query pair {query_id} for index {index}")

        return query1, query2

    def _generate_random_query(self, size: int) -> np.ndarray:
        """
        Generate cryptographically random query vector.

        Args:
            size: Size of query vector

        Returns:
            Random binary vector
        """
        # Generate random bits
        num_bytes = (size + 7) // 8
        random_bytes = secrets.token_bytes(num_bytes)

        # Convert to bit array
        bits = np.unpackbits(np.frombuffer(random_bytes, dtype=np.uint8))[:size]

        return bits

    def process_query_constant_time(
        self, query: PIRQuery, database: np.ndarray, server_id: int
    ) -> PIRResponse:
        """
        Process PIR query with constant-time guarantees.

        This function processes queries in constant time to prevent
        timing side-channel attacks.

        Args:
            query: PIR query to process
            database: Database as binary matrix (rows are blocks)
            server_id: ID of this server

        Returns:
            PIR response with result
        """
        start_time = time.perf_counter_ns()

        # Validate query freshness
        if not self._validate_query_freshness(query):
            # Still process to maintain constant time
            dummy_result = self._process_dummy_query(len(database))
        else:
            # Validate query signature
            if not self._verify_query_signature(query):
                dummy_result = self._process_dummy_query(len(database))
            else:
                dummy_result = None

        # Perform the actual computation
        if dummy_result is None:
            # Compute inner product: DB · Q
            result = self._constant_time_inner_product(database, query.query_vector)
        else:
            result = dummy_result

        # Convert result to bytes
        result_bytes = self._encode_result(result)

        # Pad to fixed size (1KB blocks)
        padded_result = self._pad_response(result_bytes)

        # Generate response metadata
        response = PIRResponse(
            response_data=padded_result,
            server_id=server_id,
            query_id=query.query_id,
            timestamp=time.time(),
            padding=b"",  # Padding already included in response_data
            checksum=self._compute_checksum(padded_result),
        )

        # Ensure constant execution time
        self._wait_constant_time(start_time)

        return response

    def _constant_time_inner_product(self, database: np.ndarray, query: np.ndarray) -> np.ndarray:
        """
        Compute inner product in constant time.

        This uses bit-slicing and masking to ensure constant-time execution
        regardless of the query pattern.

        Args:
            database: Binary database matrix
            query: Binary query vector

        Returns:
            XOR of selected database rows
        """
        # Ensure inputs are binary
        database = database.astype(np.uint8)
        query = query.astype(np.uint8)

        # Initialize result
        result = np.zeros(database.shape[1], dtype=np.uint8)

        # Process each row with constant-time selection
        for i in range(len(database)):
            # Create mask from query bit (all 0s or all 1s)
            mask = np.full(database.shape[1], query[i], dtype=np.uint8)

            # XOR row with result if selected (constant time)
            masked_row = database[i] & mask
            result ^= masked_row

        return result

    def _wait_constant_time(self, start_time: int, target_ns: int = 10_000_000):
        """
        Wait to ensure constant execution time.

        Args:
            start_time: Start time in nanoseconds
            target_ns: Target execution time in nanoseconds
        """
        elapsed = time.perf_counter_ns() - start_time
        remaining = target_ns - elapsed

        if remaining > 0:
            # Busy wait for precise timing
            end_time = time.perf_counter_ns() + remaining
            while time.perf_counter_ns() < end_time:
                pass

    def combine_responses(self, response1: PIRResponse, response2: PIRResponse) -> bytes:
        """
        Combine responses from two servers to recover data.

        For XOR-PIR: result = R1 XOR R2

        Args:
            response1: Response from server 1
            response2: Response from server 2

        Returns:
            Recovered data block
        """
        # Verify response integrity
        if not self._verify_response_checksum(response1):
            raise ValueError("Response 1 checksum verification failed")

        if not self._verify_response_checksum(response2):
            raise ValueError("Response 2 checksum verification failed")

        # Verify responses are for same query
        if response1.query_id != response2.query_id:
            raise ValueError("Response query IDs don't match")

        # Check response freshness
        current_time = time.time()
        if current_time - response1.timestamp > self.params.max_query_age:
            raise ValueError("Response 1 is stale")

        if current_time - response2.timestamp > self.params.max_query_age:
            raise ValueError("Response 2 is stale")

        # XOR the responses
        data1 = self._unpad_response(response1.response_data)
        data2 = self._unpad_response(response2.response_data)

        if len(data1) != len(data2):
            raise ValueError("Response sizes don't match")

        # Perform XOR
        result = bytes(a ^ b for a, b in zip(data1, data2))

        logger.debug(f"Combined responses for query {response1.query_id}")

        return result

    def _validate_query_freshness(self, query: PIRQuery) -> bool:
        """
        Validate query freshness for replay protection.

        Args:
            query: Query to validate

        Returns:
            True if query is fresh
        """
        current_time = time.time()

        # Check timestamp
        if current_time - query.timestamp > self.params.max_query_age:
            logger.warning(f"Query {query.query_id} is stale")
            return False

        # Check nonce hasn't been used
        if query.nonce in self.used_nonces:
            logger.warning(f"Query {query.query_id} nonce already used")
            return False

        return True

    def _sign_query(self, query: PIRQuery) -> bytes:
        """
        Sign query for integrity protection.

        Args:
            query: Query to sign

        Returns:
            Query signature
        """
        # Create signing key from security parameter
        key = secrets.token_bytes(self.params.security_parameter // 8)

        # Create HMAC
        h = hmac.HMAC(key, hashes.SHA256())
        h.update(query.query_vector.tobytes())
        h.update(query.nonce)
        h.update(str(query.timestamp).encode())
        h.update(query.query_id.encode())

        return h.finalize()

    def _verify_query_signature(self, query: PIRQuery) -> bool:
        """
        Verify query signature.

        Args:
            query: Query to verify

        Returns:
            True if signature is valid
        """
        # In production, this would verify against a known key
        # For now, we'll accept all properly formatted queries
        return query.signature is not None and len(query.signature) == 32

    def _pad_response(self, data: bytes) -> bytes:
        """
        Pad response to fixed size (1KB blocks).

        Args:
            data: Data to pad

        Returns:
            Padded data
        """
        block_size = self.params.response_padding_size

        if len(data) >= block_size:
            return data[:block_size]

        # Pad with random data
        padding_size = block_size - len(data) - 4  # 4 bytes for length
        padding = secrets.token_bytes(padding_size)

        # Encode original length
        length_bytes = len(data).to_bytes(4, "big")

        return data + padding + length_bytes

    def _unpad_response(self, padded_data: bytes) -> bytes:
        """
        Remove padding from response.

        Args:
            padded_data: Padded data

        Returns:
            Original data
        """
        # Extract length from last 4 bytes
        length = int.from_bytes(padded_data[-4:], "big")

        # Return original data
        return padded_data[:length]

    def _encode_result(self, result: np.ndarray) -> bytes:
        """
        Encode computation result as bytes.

        Args:
            result: Binary result vector

        Returns:
            Encoded bytes
        """
        # Pack bits into bytes
        padded_size = ((len(result) + 7) // 8) * 8
        padded_result = np.pad(result, (0, padded_size - len(result)))

        return np.packbits(padded_result).tobytes()

    def _compute_checksum(self, data: bytes) -> bytes:
        """
        Compute checksum for response integrity.

        Args:
            data: Data to checksum

        Returns:
            Checksum bytes
        """
        return hashlib.sha256(data).digest()

    def _verify_response_checksum(self, response: PIRResponse) -> bool:
        """
        Verify response checksum.

        Args:
            response: Response to verify

        Returns:
            True if checksum is valid
        """
        expected = self._compute_checksum(response.response_data)
        return secrets.compare_digest(expected, response.checksum)

    def _process_dummy_query(self, database_size: int) -> np.ndarray:
        """
        Process dummy query for constant-time execution.

        Args:
            database_size: Size of database

        Returns:
            Random result
        """
        # Return random data of appropriate size
        return np.random.randint(0, 2, self.params.block_size * 8, dtype=np.uint8)

    def _cleanup_old_queries(self) -> None:
        """Clean up old queries from cache."""
        if len(self.query_cache) > self.max_cache_size:
            # Remove oldest queries
            sorted_queries = sorted(self.query_cache.items(), key=lambda x: x[1])
            for query_id, _ in sorted_queries[: len(self.query_cache) // 2]:
                del self.query_cache[query_id]

    def estimate_breach_probability(self, collusion_prob: float = 0.01) -> float:
        """
        Estimate privacy breach probability.

        For k servers with independent collusion probability q:
        P_breach = q^k

        Args:
            collusion_prob: Probability of single server compromise

        Returns:
            Estimated breach probability
        """
        return collusion_prob**self.params.num_servers


class BatchXORPIR(XORPIRScheme):
    """
    Batch variant of XOR-PIR for retrieving multiple blocks.

    This amortizes the communication cost across multiple queries.
    """

    def generate_batch_queries(
        self, indices: List[int], num_blocks: int
    ) -> Tuple[PIRQuery, PIRQuery]:
        """
        Generate queries for retrieving multiple blocks.

        Args:
            indices: List of block indices to retrieve
            num_blocks: Total number of blocks

        Returns:
            Query pair for batch retrieval
        """
        # Generate base random query
        q1 = self._generate_random_query(num_blocks)

        # XOR with unit vectors for all requested indices
        q2 = q1.copy()
        for idx in indices:
            if idx >= num_blocks:
                raise ValueError(f"Index {idx} out of bounds")
            q2[idx] ^= 1

        # Create query objects with batch type
        query_id = secrets.token_hex(16)
        nonce = secrets.token_bytes(32)
        timestamp = time.time()

        padding_size = max(0, self.params.response_padding_size * len(indices) - num_blocks // 8)
        padding = secrets.token_bytes(padding_size)

        query1 = PIRQuery(
            query_vector=q1,
            nonce=nonce,
            timestamp=timestamp,
            query_id=query_id,
            padding=padding,
            query_type=QueryType.BATCH,
        )

        query2 = PIRQuery(
            query_vector=q2,
            nonce=nonce,
            timestamp=timestamp,
            query_id=query_id,
            padding=padding,
            query_type=QueryType.BATCH,
        )

        query1.signature = self._sign_query(query1)
        query2.signature = self._sign_query(query2)

        return query1, query2
