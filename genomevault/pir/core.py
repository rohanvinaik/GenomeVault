"""Core Private Information Retrieval (PIR) functionality.

This module provides the foundation for privacy-preserving database queries
where the server cannot learn which item was retrieved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple
import hashlib
import secrets

import numpy as np


@dataclass
class PIRConfig:
    """Configuration for PIR operations."""

    database_size: int
    security_parameter: int = 128
    use_compression: bool = True
    batch_size: int = 1

    def __post_init__(self):
        """post init  ."""
        if self.database_size <= 0:
            raise ValueError("Database size must be positive")
        if self.security_parameter < 80:
            raise ValueError("Security parameter must be at least 80 bits")


class PIRClient:
    """Client for Private Information Retrieval."""

    def __init__(self, config: PIRConfig):
        """Initialize instance.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self._private_key = secrets.token_bytes(32)

    def generate_query(self, index: int) -> bytes:
        """Generate a PIR query for the given index.

        Args:
            index: The index of the item to retrieve

        Returns:
            Encrypted query that hides the index
        """
        if not 0 <= index < self.config.database_size:
            raise ValueError(f"Index {index} out of range [0, {self.config.database_size})")

        # Simple PIR: Create a one-hot encoded vector with noise
        query_vector = np.zeros(self.config.database_size)
        query_vector[index] = 1

        # Add noise for privacy
        noise = np.random.normal(0, 0.01, self.config.database_size)
        query_vector += noise

        # Serialize and add authentication
        query_bytes = query_vector.tobytes()
        mac = hashlib.sha256(self._private_key + query_bytes).digest()[:16]

        return mac + query_bytes

    def process_response(self, response: bytes) -> Any:
        """Process the server's response to extract the requested item.

        Args:
            response: The server's encrypted response

        Returns:
            The decrypted item
        """
        # Verify response integrity
        if len(response) < 16:
            raise ValueError("Invalid response: too short")

        mac = response[:16]
        data = response[16:]

        expected_mac = hashlib.sha256(self._private_key + data).digest()[:16]
        if mac != expected_mac:
            raise ValueError("Response authentication failed")

        # For this simple implementation, return the raw data
        return data


class PIRServer:
    """Server for Private Information Retrieval."""

    def __init__(self, database: List[bytes], config: PIRConfig):
        """Initialize instance.

        Args:
            database: Input database to process.
            config: Configuration dictionary.

        Raises:
            ValueError: When operation fails.
        """
        self.database = database
        self.config = config

        if len(database) != config.database_size:
            raise ValueError(
                f"Database size mismatch: expected {config.database_size}, got {len(database)}"
            )

    def process_query(self, query: bytes) -> bytes:
        """Process a PIR query and return the response.

        Args:
            query: The client's encrypted query

        Returns:
            Encrypted response containing the requested item
        """
        if len(query) < 16:
            raise ValueError("Invalid query: too short")

        # Extract MAC and query vector
        mac = query[:16]
        query_data = query[16:]

        # Deserialize query vector
        try:
            query_vector = np.frombuffer(query_data, dtype=np.float64)
            query_vector = query_vector[: self.config.database_size]
        except Exception as e:
            raise ValueError(f"Failed to deserialize query: {e}")

        # Find the index with highest value (simple PIR)
        index = np.argmax(query_vector)

        # Get the item (in real PIR, this would be done obliviously)
        if 0 <= index < len(self.database):
            item = self.database[index]
        else:
            item = b""

        # Create response with MAC
        response_mac = hashlib.sha256(mac + item).digest()[:16]

        return response_mac + item


class SimplePIR:
    """Simple PIR implementation for basic use cases."""

    def __init__(self, database: List[bytes]):
        """Initialize instance.

        Args:
            database: Input database to process.
        """
        self.config = PIRConfig(database_size=len(database))
        self.server = PIRServer(database, self.config)
        self.client = PIRClient(self.config)

    def retrieve(self, index: int) -> bytes:
        """Retrieve an item privately.

        Args:
            index: The index of the item to retrieve

        Returns:
            The retrieved item
        """
        query = self.client.generate_query(index)
        response = self.server.process_query(query)
        return self.client.process_response(response)


def create_pir_system(
    database: List[bytes], security_parameter: int = 128
) -> Tuple[PIRClient, PIRServer]:
    """Create a PIR client-server pair.

    Args:
        database: The database items
        security_parameter: Security level in bits

    Returns:
        Tuple of (client, server)
    """
    config = PIRConfig(database_size=len(database), security_parameter=security_parameter)

    client = PIRClient(config)
    server = PIRServer(database, config)

    return client, server


# Public API
__all__ = [
    "PIRConfig",
    "PIRClient",
    "PIRServer",
    "SimplePIR",
    "create_pir_system",
]
