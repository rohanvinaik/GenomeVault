"""
Byzantine fault tolerance handler for PIR servers.

Implements Reed-Solomon error correction to handle malicious or faulty servers
while maintaining information-theoretic privacy guarantees.
"""

import hashlib
import secrets
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

import numpy as np
from reedsolo import RSCodec, ReedSolomonError

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class ServerStatus(Enum):
    """PIR server status."""

    HEALTHY = "healthy"
    SUSPICIOUS = "suspicious"
    FAULTY = "faulty"
    UNKNOWN = "unknown"


@dataclass
class ServerReputation:
    """
    Track server reputation for Byzantine detection.

    Attributes:
        server_id: Server identifier
        total_queries: Total queries sent to server
        successful_responses: Number of valid responses
        failed_verifications: Number of failed integrity checks
        suspicious_behaviors: Count of suspicious patterns
        status: Current server status
    """

    server_id: int
    total_queries: int = 0
    successful_responses: int = 0
    failed_verifications: int = 0
    suspicious_behaviors: int = 0
    status: ServerStatus = ServerStatus.UNKNOWN
    last_update: float = 0.0

    @property
    def reliability_score(self) -> float:
        """Calculate server reliability score."""
        if self.total_queries == 0:
            return 0.5  # Unknown reliability

        success_rate = self.successful_responses / self.total_queries
        failure_penalty = self.failed_verifications * 0.1
        suspicious_penalty = self.suspicious_behaviors * 0.05

        return max(0.0, min(1.0, success_rate - failure_penalty - suspicious_penalty))

    def update_status(self) -> None:
        """Update server status based on reputation."""
        score = self.reliability_score

        if score >= 0.9:
            self.status = ServerStatus.HEALTHY
        elif score >= 0.7:
            self.status = ServerStatus.SUSPICIOUS
        else:
            self.status = ServerStatus.FAULTY


@dataclass
class ByzantineConfig:
    """Configuration for Byzantine fault tolerance."""

    num_servers: int = 3  # Total number of servers
    min_servers: int = 2  # Minimum servers for reconstruction
    redundancy_factor: int = 2  # Reed-Solomon redundancy
    error_correction_symbols: int = 10  # RS error correction symbols
    reputation_threshold: float = 0.7  # Minimum reputation for trust
    verification_samples: int = 5  # Samples for response verification
    max_retries: int = 3  # Maximum query retries


class ByzantineHandler:
    """
    Byzantine fault tolerance handler for PIR.

    Implements:
    1. Reed-Solomon error correction for response recovery
    2. Server reputation tracking
    3. Malicious server detection
    4. Response verification and validation
    """

    def __init__(self, config: ByzantineConfig):
        """
        Initialize Byzantine handler.

        Args:
            config: Byzantine configuration
        """
        self.config = config
        self._validate_config()

        # Initialize Reed-Solomon codec
        self.rs_codec = RSCodec(config.error_correction_symbols)

        # Server reputation tracking
        self.server_reputations: Dict[int, ServerReputation] = {}
        for i in range(config.num_servers):
            self.server_reputations[i] = ServerReputation(server_id=i)

        # Response verification cache
        self.verification_cache: Dict[str, bytes] = {}

        logger.info(f"Initialized Byzantine handler for {config.num_servers} servers")

    def _validate_config(self) -> None:
        """Validate Byzantine configuration."""
        # Check Byzantine fault tolerance requirement: n >= 3t + 1
        max_faulty = (self.config.num_servers - 1) // 3

        if self.config.min_servers <= max_faulty:
            raise ValueError(
                f"Insufficient servers for Byzantine tolerance. "
                f"Need min_servers > {max_faulty} for {self.config.num_servers} total servers"
            )

        if self.config.error_correction_symbols < 2:
            raise ValueError("Need at least 2 error correction symbols")

    def encode_for_servers(self, data: bytes) -> List[bytes]:
        """
        Encode data with Reed-Solomon for distribution to servers.

        This creates redundant shares that can tolerate Byzantine failures.

        Args:
            data: Original data to encode

        Returns:
            List of encoded shares for each server
        """
        # Add Reed-Solomon redundancy
        try:
            encoded = self.rs_codec.encode(data)
        except Exception as e:
            logger.error(f"Reed-Solomon encoding failed: {e}")
            raise

        # Split into shares for each server
        share_size = len(encoded) // self.config.num_servers
        shares = []

        for i in range(self.config.num_servers):
            start = i * share_size
            end = start + share_size if i < self.config.num_servers - 1 else len(encoded)
            share = encoded[start:end]

            # Add integrity checksum
            checksum = hashlib.sha256(share).digest()[:8]
            shares.append(share + checksum)

        logger.debug(f"Encoded {len(data)} bytes into {len(shares)} shares")

        return shares

    def decode_responses(
        self, responses: List[Tuple[int, bytes]], expected_size: int
    ) -> Optional[bytes]:
        """
        Decode responses from multiple servers with Byzantine tolerance.

        Args:
            responses: List of (server_id, response_data) tuples
            expected_size: Expected size of decoded data

        Returns:
            Decoded data if successful, None otherwise
        """
        if len(responses) < self.config.min_servers:
            logger.error(f"Insufficient responses: {len(responses)} < {self.config.min_servers}")
            return None

        # Verify response integrity
        valid_responses = []
        for server_id, response_data in responses:
            if self._verify_response_integrity(server_id, response_data):
                # Remove checksum
                valid_responses.append((server_id, response_data[:-8]))
                self._update_reputation(server_id, success=True)
            else:
                logger.warning(f"Response from server {server_id} failed integrity check")
                self._update_reputation(server_id, success=False)

        if len(valid_responses) < self.config.min_servers:
            logger.error("Too many corrupted responses")
            return None

        # Reconstruct data from shares
        try:
            # Combine shares in order
            combined = b""
            for i in range(self.config.num_servers):
                # Find response from server i
                for sid, data in valid_responses:
                    if sid == i:
                        combined += data
                        break
                else:
                    # Missing response, use erasure
                    combined += b"\x00" * (len(valid_responses[0][1]))

            # Decode with Reed-Solomon
            decoded = self.rs_codec.decode(combined)[0]

            # Verify decoded size
            if len(decoded) != expected_size:
                logger.warning(f"Decoded size mismatch: {len(decoded)} != {expected_size}")
                return None

            return decoded

        except ReedSolomonError as e:
            logger.error(f"Reed-Solomon decoding failed: {e}")
            return None

    def detect_byzantine_behavior(self, responses: List[Tuple[int, bytes]]) -> List[int]:
        """
        Detect potentially Byzantine servers.

        Args:
            responses: Server responses to analyze

        Returns:
            List of suspicious server IDs
        """
        suspicious_servers = []

        # Group responses by content hash
        response_groups: Dict[str, List[int]] = {}
        for server_id, response in responses:
            content_hash = hashlib.sha256(response).hexdigest()
            if content_hash not in response_groups:
                response_groups[content_hash] = []
            response_groups[content_hash].append(server_id)

        # Find minority responses (potential Byzantine)
        if len(response_groups) > 1:
            # Sort groups by size
            sorted_groups = sorted(response_groups.values(), key=len, reverse=True)

            # Servers in minority groups are suspicious
            for group in sorted_groups[1:]:
                suspicious_servers.extend(group)
                for server_id in group:
                    self._mark_suspicious(server_id)

        # Check timing anomalies
        # (In production, would analyze response times)

        return suspicious_servers

    def verify_response_consistency(self, responses: List[Tuple[int, bytes]]) -> bool:
        """
        Verify consistency across multiple responses.

        Args:
            responses: Server responses to verify

        Returns:
            True if responses are consistent
        """
        if len(responses) < self.config.min_servers:
            return False

        # For XOR-PIR, responses should XOR to the same value
        # We'll check a sample of combinations
        sample_size = min(self.config.verification_samples, len(responses))

        for i in range(sample_size - 1):
            for j in range(i + 1, sample_size):
                # XOR two responses
                xor1 = self._xor_bytes(responses[i][1], responses[j][1])

                # Compare with another pair
                if i + 2 < len(responses) and j + 1 < len(responses):
                    xor2 = self._xor_bytes(responses[i + 1][1], responses[j + 1][1])

                    # Check if XORs are related (they should follow a pattern)
                    if not self._check_xor_relation(xor1, xor2):
                        logger.warning("Inconsistent XOR relationship detected")
                        return False

        return True

    def select_reliable_servers(self, num_needed: int) -> List[int]:
        """
        Select most reliable servers for query.

        Args:
            num_needed: Number of servers needed

        Returns:
            List of selected server IDs
        """
        # Sort servers by reliability score
        sorted_servers = sorted(
            self.server_reputations.values(), key=lambda x: x.reliability_score, reverse=True
        )

        # Filter by minimum reputation threshold
        reliable_servers = [
            s.server_id
            for s in sorted_servers
            if s.reliability_score >= self.config.reputation_threshold
        ]

        if len(reliable_servers) < num_needed:
            logger.warning(
                f"Only {len(reliable_servers)} reliable servers available, " f"need {num_needed}"
            )
            # Add less reliable servers if necessary
            for server in sorted_servers:
                if server.server_id not in reliable_servers:
                    reliable_servers.append(server.server_id)
                    if len(reliable_servers) >= num_needed:
                        break

        return reliable_servers[:num_needed]

    def handle_server_failure(self, server_id: int, error: Exception) -> None:
        """
        Handle server failure.

        Args:
            server_id: Failed server ID
            error: Failure exception
        """
        logger.error(f"Server {server_id} failed: {error}")

        # Update reputation
        rep = self.server_reputations[server_id]
        rep.failed_verifications += 1
        rep.update_status()

        # Mark as faulty if too many failures
        if rep.failed_verifications > 5:
            rep.status = ServerStatus.FAULTY
            logger.warning(f"Server {server_id} marked as FAULTY")

    def _verify_response_integrity(self, server_id: int, response: bytes) -> bool:
        """
        Verify response integrity using checksum.

        Args:
            server_id: Server ID
            response: Response data with checksum

        Returns:
            True if integrity check passes
        """
        if len(response) < 8:
            return False

        data = response[:-8]
        checksum = response[-8:]
        expected = hashlib.sha256(data).digest()[:8]

        return secrets.compare_digest(checksum, expected)

    def _update_reputation(self, server_id: int, success: bool) -> None:
        """
        Update server reputation.

        Args:
            server_id: Server ID
            success: Whether interaction was successful
        """
        rep = self.server_reputations[server_id]
        rep.total_queries += 1

        if success:
            rep.successful_responses += 1
        else:
            rep.failed_verifications += 1

        rep.update_status()

    def _mark_suspicious(self, server_id: int) -> None:
        """
        Mark server as suspicious.

        Args:
            server_id: Server ID
        """
        rep = self.server_reputations[server_id]
        rep.suspicious_behaviors += 1
        rep.update_status()

        logger.warning(f"Server {server_id} marked as suspicious")

    def _xor_bytes(self, a: bytes, b: bytes) -> bytes:
        """
        XOR two byte strings.

        Args:
            a: First byte string
            b: Second byte string

        Returns:
            XOR result
        """
        min_len = min(len(a), len(b))
        return bytes(a[i] ^ b[i] for i in range(min_len))

    def _check_xor_relation(self, xor1: bytes, xor2: bytes) -> bool:
        """
        Check if two XOR results have expected relationship.

        Args:
            xor1: First XOR result
            xor2: Second XOR result

        Returns:
            True if relationship is valid
        """
        # For valid PIR responses, XORs should have predictable patterns
        # This is a simplified check - production would be more sophisticated

        # Check if XORs are similar (low Hamming distance)
        if len(xor1) != len(xor2):
            return False

        differences = sum(a != b for a, b in zip(xor1, xor2))
        threshold = len(xor1) * 0.1  # Allow 10% difference

        return differences <= threshold

    def get_server_statistics(self) -> Dict[int, Dict[str, Any]]:
        """
        Get statistics for all servers.

        Returns:
            Dictionary of server statistics
        """
        stats = {}
        for server_id, rep in self.server_reputations.items():
            stats[server_id] = {
                "status": rep.status.value,
                "reliability_score": rep.reliability_score,
                "total_queries": rep.total_queries,
                "successful_responses": rep.successful_responses,
                "failed_verifications": rep.failed_verifications,
                "suspicious_behaviors": rep.suspicious_behaviors,
            }

        return stats


class AdaptiveByzantineHandler(ByzantineHandler):
    """
    Adaptive Byzantine handler that adjusts strategies based on observed behavior.
    """

    def __init__(self, config: ByzantineConfig):
        """Initialize adaptive handler."""
        super().__init__(config)

        # Adaptive parameters
        self.adaptive_threshold = 0.8  # Threshold for switching strategies
        self.fast_path_enabled = True  # Use optimistic fast path
        self.verification_level = 1  # Current verification level (1-3)

    def adapt_strategy(self) -> None:
        """Adapt strategy based on observed server behavior."""
        # Calculate overall system reliability
        avg_reliability = np.mean(
            [rep.reliability_score for rep in self.server_reputations.values()]
        )

        if avg_reliability > 0.9:
            # High reliability - use fast path
            self.fast_path_enabled = True
            self.verification_level = 1
            logger.info("Switching to fast path mode")

        elif avg_reliability > 0.7:
            # Medium reliability - normal verification
            self.fast_path_enabled = False
            self.verification_level = 2
            logger.info("Using normal verification mode")

        else:
            # Low reliability - maximum verification
            self.fast_path_enabled = False
            self.verification_level = 3
            logger.warning("Switching to maximum verification mode")

    def fast_path_decode(self, responses: List[Tuple[int, bytes]]) -> Optional[bytes]:
        """
        Fast path decoding for high-reliability scenarios.

        Args:
            responses: Server responses

        Returns:
            Decoded data if successful
        """
        if not self.fast_path_enabled:
            return None

        # Quick check with minimal verification
        if len(responses) >= self.config.min_servers:
            # Take first min_servers responses from reliable servers
            reliable_responses = [
                r for r in responses if self.server_reputations[r[0]].reliability_score > 0.9
            ]

            if len(reliable_responses) >= self.config.min_servers:
                # Fast decode without full verification
                try:
                    combined = b"".join([r[1] for r in reliable_responses])
                    return self.rs_codec.decode(combined)[0]
                except:
                    # Fall back to normal path
                    self.fast_path_enabled = False
                    return None

        return None
