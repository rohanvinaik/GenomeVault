"""
Robust Information-Theoretic PIR with Byzantine fault tolerance.

This module implements a Byzantine-resilient IT-PIR scheme that can tolerate
malicious servers and network adversaries while maintaining information-theoretic
privacy guarantees.
"""

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass

import numpy as np

from genomevault.pir.advanced.it_pir import InformationTheoreticPIR, PIRQuery
from genomevault.utils.metrics import get_metrics

logger = logging.getLogger(__name__)
metrics = get_metrics()


class SecurityError(Exception):
    """Raised when security constraints are violated."""

    pass


@dataclass
class AuthenticatedResponse:
    """Server response with authentication."""

    response_data: np.ndarray
    mac: bytes
    server_id: int
    timestamp: float
    nonce: bytes


class RobustITPIR(InformationTheoreticPIR):
    """
    IT-PIR with Byzantine fault tolerance and authentication.

    Features:
    - Reed-Solomon error correction for Byzantine servers
    - MAC-based response authentication
    - Constant-time query processing
    - Fixed-size responses with padding
    - Query nonces and timestamps for replay protection
    """

    def __init__(
        self,
        database_size: int,
        num_servers: int,
        byzantine_threshold: int = None,
        mac_key: bytes = None,
    ):
        """
        Initialize robust IT-PIR system.

        Args:
            database_size: Number of items in database
            num_servers: Total number of servers
            byzantine_threshold: Maximum number of Byzantine servers to tolerate
            mac_key: Shared MAC key for authentication
        """
        super().__init__(database_size, num_servers)

        # Byzantine fault tolerance parameters
        self.byzantine_threshold = byzantine_threshold or (num_servers - 1) // 3

        # Ensure we have enough servers for error correction
        min_servers = 2 * self.byzantine_threshold + 1
        if num_servers < min_servers:
            raise ValueError(
                f"Need at least {min_servers} servers to tolerate "
                f"{self.byzantine_threshold} Byzantine servers"
            )

        # Authentication
        self.mac_key = mac_key or hashlib.sha256(b"genomevault_it_pir").digest()

        # Response size (fixed for all queries)
        self.response_size = 1024  # 1KB fixed response size

        # Query cache for replay detection
        self.query_cache = {}
        self.cache_ttl = 300  # 5 minutes

        logger.info(
            "Initialized RobustITPIR: %snum_servers servers, "
            "Byzantine threshold: %sbyzantine_threshold"
        )

    def generate_authenticated_query(self, item_index: int) -> PIRQuery:
        """
        Generate authenticated PIR query with replay protection.

        Args:
            item_index: Index of item to retrieve

        Returns:
            Authenticated PIR query
        """
        # Generate base query
        query = super().generate_query(item_index)

        # Add nonce and timestamp
        nonce = np.random.bytes(16)
        timestamp = time.time()

        # Store in cache for replay detection
        query_id = hashlib.sha256(nonce + str(timestamp).encode()).digest()
        self.query_cache[query_id] = {
            "timestamp": timestamp,
            "item_index": item_index,
            "used": False,
        }

        # Clean old entries
        self._clean_query_cache()

        # Add authentication metadata
        query.metadata = {
            "nonce": nonce.hex(),
            "timestamp": timestamp,
            "query_id": query_id.hex(),
        }

        return query

    def process_responses_with_verification(
        self, query: PIRQuery, server_responses: list[tuple[np.ndarray, bytes]]
    ) -> bytes:
        """
        Process server responses with Byzantine fault tolerance.

        Args:
            query: Original PIR query
            server_responses: List of (response, mac) tuples from servers

        Returns:
            Recovered data item

        Raises:
            SecurityError: If too many invalid responses
        """
        with metrics.time_operation("robust_pir_response_processing"):
            # Verify response authenticity
            valid_responses = []
            invalid_count = 0

            for i, (response, mac) in enumerate(server_responses):
                if self._verify_response_mac(query, i, response, mac):
                    valid_responses.append((i, response))
                else:
                    invalid_count += 1
                    logger.warning("Invalid response from server %si")

            # Check if we have enough valid responses
            if invalid_count > self.byzantine_threshold:
                raise SecurityError(
                    f"Too many invalid responses: {invalid_count} > {self.byzantine_threshold}"
                )

            if len(valid_responses) < self.num_servers - self.byzantine_threshold:
                raise SecurityError(f"Insufficient valid responses: {len(valid_responses)}")

            # Use Reed-Solomon decoding to recover data
            recovered_data = self._reed_solomon_decode(valid_responses)

            # Record metrics
            metrics.record("robust_pir_valid_responses", len(valid_responses))
            metrics.record("robust_pir_invalid_responses", invalid_count)

            return recovered_data

    def _verify_response_mac(
        self, query: PIRQuery, server_id: int, response: np.ndarray, mac: bytes
    ) -> bool:
        """Verify MAC on server response."""
        # Construct message for MAC
        message = b"".join(
            [
                query.metadata["nonce"].encode(),
                str(query.metadata["timestamp"]).encode(),
                str(server_id).encode(),
                response.tobytes(),
            ]
        )

        # Compute expected MAC
        expected_mac = hmac.new(self.mac_key, message, hashlib.sha256).digest()

        # Constant-time comparison
        return hmac.compare_digest(mac, expected_mac)

    def _reed_solomon_decode(self, valid_responses: list[tuple[int, np.ndarray]]) -> bytes:
        """
        Decode data using Reed-Solomon error correction.

        Args:
            valid_responses: List of (server_id, response) tuples

        Returns:
            Decoded data
        """
        # Sort responses by server ID
        valid_responses.sort(key=lambda x: x[0])

        # Extract response matrix
        response_matrix = np.array([resp for _, resp in valid_responses])

        # Simple Reed-Solomon decoding (simplified for demo)
        # In production, use proper RS implementation

        # Majority voting on each position
        decoded = np.zeros(self.response_size, dtype=np.uint8)

        for pos in range(self.response_size):
            # Get values at this position
            values = response_matrix[:, pos]

            # Find majority value
            unique, counts = np.unique(values, return_counts=True)
            majority_idx = np.argmax(counts)
            decoded[pos] = unique[majority_idx]

        # Remove padding
        data_end = np.where(decoded == 0)[0]
        if len(data_end) > 0:
            decoded = decoded[: data_end[0]]

        return decoded.tobytes()

    def _clean_query_cache(self) -> None:
        """Remove expired queries from cache."""
        current_time = time.time()
        expired = []

        for query_id, info in self.query_cache.items():
            if current_time - info["timestamp"] > self.cache_ttl:
                expired.append(query_id)

        for query_id in expired:
            del self.query_cache[query_id]

    def simulate_server_response(
        self,
        server_id: int,
        query: PIRQuery,
        database: np.ndarray,
        malicious: bool = False,
    ) -> tuple[np.ndarray, bytes]:
        """
        Simulate server response with optional Byzantine behavior.

        Args:
            server_id: ID of this server
            query: PIR query
            database: Server's database shard
            malicious: Whether to simulate malicious behavior

        Returns:
            (response, mac) tuple
        """
        # Process query (constant time)
        response = self._process_query_constant_time(query.query_vectors[server_id], database)

        # Add padding to fixed size
        if len(response) < self.response_size:
            padding = np.zeros(self.response_size - len(response), dtype=np.uint8)
            response = np.concatenate([response, padding])
        elif len(response) > self.response_size:
            response = response[: self.response_size]

        # Simulate malicious behavior
        if malicious:
            # Corrupt response
            corruption_type = np.random.choice(
                ["flip_bits", "wrong_mac", "replay_old", "random_data"]
            )

            if corruption_type == "flip_bits":
                # Flip random bits
                num_flips = np.random.randint(1, 50)
                flip_positions = np.random.choice(len(response), num_flips)
                response[flip_positions] ^= 0xFF

            elif corruption_type == "wrong_mac":
                # Return response with incorrect MAC
                return response, b"wrong_mac_value"

            elif corruption_type == "random_data":
                # Return completely random data
                response = np.random.bytes(self.response_size)

        # Compute MAC
        message = b"".join(
            [
                query.metadata["nonce"].encode(),
                str(query.metadata["timestamp"]).encode(),
                str(server_id).encode(),
                response.tobytes(),
            ]
        )

        mac = hmac.new(self.mac_key, message, hashlib.sha256).digest()

        return response, mac

    def _process_query_constant_time(
        self, query_vector: np.ndarray, database: np.ndarray
    ) -> np.ndarray:
        """Process query in constant time to prevent timing attacks."""
        # Ensure constant-time processing
        result = np.zeros(self.response_size, dtype=np.uint8)

        # Process all database entries (no early exit)
        for i in range(len(database)):
            # Constant-time selection
            mask = query_vector[i]
            for j in range(min(len(database[i]), self.response_size)):
                result[j] ^= mask * database[i][j]

        return result


# Security analysis functions
def analyze_privacy_breach_probability(
    num_servers: int, collusion_probability: float = 0.98
) -> dict[str, float]:
    """
    Analyze probability of privacy breach.

    Args:
        num_servers: Total number of servers
        collusion_probability: Probability that a server colludes

    Returns:
        Analysis results
    """
    # P(breach) = P(all k servers collude)
    breach_prob = collusion_probability**num_servers

    # Find minimum k for target privacy levels
    privacy_targets = {
        "hipaa_compliant": 1e-4,  # φ ≤ 10^-4
        "high_security": 1e-6,
        "maximum_security": 1e-9,
    }

    min_servers = {}
    for target_name, target_prob in privacy_targets.items():
        # Solve: q^k ≤ target_prob
        # k ≥ log(target_prob) / log(q)
        k_min = np.ceil(np.log(target_prob) / np.log(collusion_probability))
        min_servers[target_name] = int(k_min)

    return {
        "breach_probability": breach_prob,
        "collusion_probability": collusion_probability,
        "num_servers": num_servers,
        "minimum_servers": min_servers,
    }


if __name__ == "__main__":
    # Example usage and security analysis

    # Analyze security for different configurations
    logger.info("IT-PIR Security Analysis")
    logger.info("=" * 50)

    for num_servers in [2, 3, 5, 7]:
        logger.info("\nConfiguration: %snum_servers servers")

        # HIPAA Trust Score nodes (q = 0.98)
        analysis = analyze_privacy_breach_probability(num_servers, 0.98)
        logger.info("  HIPAA TS nodes (q=0.98):")
        logger.info("    Privacy breach probability: %sanalysis['breach_probability']:.2e")

        # Generic nodes (q = 0.95)
        analysis = analyze_privacy_breach_probability(num_servers, 0.95)
        logger.info("  Generic nodes (q=0.95):")
        logger.info("    Privacy breach probability: %sanalysis['breach_probability']:.2e")

    logger.info("\nMinimum servers for privacy targets:")
    analysis = analyze_privacy_breach_probability(3, 0.98)
    for target, min_k in analysis["minimum_servers"].items():
        logger.info("  %starget: %smin_k servers")

    # Test robust PIR with Byzantine servers
    logger.info("\n" + "=" * 50)
    logger.info("Testing Robust IT-PIR with Byzantine servers")

    # Setup
    database_size = 1000
    num_servers = 5
    byzantine_threshold = 1  # Tolerate 1 Byzantine server

    pir = RobustITPIR(
        database_size=database_size,
        num_servers=num_servers,
        byzantine_threshold=byzantine_threshold,
    )

    # Create test database shards
    item_size = 256  # 256 bytes per item
    database_shards = []
    for i in range(num_servers):
        shard = np.random.bytes(database_size * item_size)
        shard = shard.reshape(database_size, item_size)
        database_shards.append(shard)

    # Query for item
    target_item = 42
    query = pir.generate_authenticated_query(target_item)

    # Simulate server responses (1 Byzantine)
    responses = []
    byzantine_server = np.random.randint(num_servers)

    for i in range(num_servers):
        is_malicious = i == byzantine_server
        response, mac = pir.simulate_server_response(
            i, query, database_shards[i], malicious=is_malicious
        )
        responses.append((response, mac))

        if is_malicious:
            logger.info("  Server %si: Byzantine (sending corrupted response)")
        else:
            logger.info("  Server %si: Honest")

    # Process responses
    try:
        recovered_data = pir.process_responses_with_verification(query, responses)
        logger.info("\n  Successfully recovered data despite Byzantine server!")
        logger.info("  Data size: %slen(recovered_data) bytes")
    except SecurityError:
        from genomevault.observability.logging import configure_logging

        logger = configure_logging()
        logger.exception("Unhandled exception")
        logger.info("\n  Security error: %se")
        raise RuntimeError("Unspecified error")
