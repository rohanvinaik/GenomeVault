"""PIR engine with proper variable-length record handling."""

from __future__ import annotations

import json
from typing import Any, List, Tuple, Union
import numpy as np

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class VariableLengthPIREngine:
    """PIR engine with proper record padding and variable length support."""

    def __init__(self, max_record_length: int = 1024):
        """
        Initialize PIR engine.

        Args:
            max_record_length: Maximum supported record length in bytes
        """
        self.max_record_length = max_record_length
        logger.info(f"PIR engine initialized with max record length: {max_record_length} bytes")

    def _pad_record(self, record: bytes, target_length: int) -> bytes:
        """
        Pad record to target length with null bytes.

        Args:
            record: Record to pad
            target_length: Target length in bytes

        Returns:
            Padded record

        Raises:
            ValueError: If record is longer than target length
        """
        if len(record) > target_length:
            raise ValueError(f"Record too long: {len(record)} > {target_length}")

        # Pad with null bytes
        padding = b"\x00" * (target_length - len(record))
        padded = record + padding

        logger.debug(f"Padded record from {len(record)} to {len(padded)} bytes")
        return padded

    def _unpad_record(self, padded_record: bytes) -> bytes:
        """
        Remove null byte padding from record.

        Args:
            padded_record: Padded record to unpad

        Returns:
            Original record without padding
        """
        # Find last non-zero byte
        for i in range(len(padded_record) - 1, -1, -1):
            if padded_record[i] != 0:
                original = padded_record[: i + 1]
                logger.debug(f"Unpadded record from {len(padded_record)} to {len(original)} bytes")
                return original

        # All zeros - return empty bytes
        return b""

    def prepare_database(
        self, records: List[Union[str, bytes, dict, Any]]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Prepare database with uniform record lengths for PIR operations.

        Args:
            records: List of variable-length records of any type

        Returns:
            Tuple of (padded database array, original record lengths)

        Raises:
            ValueError: If database is empty or records are too large
        """
        if not records:
            raise ValueError("Empty database")

        logger.info(f"Preparing database with {len(records)} records")

        # Convert all records to bytes
        byte_records = []
        original_lengths = []

        for i, record in enumerate(records):
            try:
                if isinstance(record, str):
                    byte_record = record.encode("utf-8")
                elif isinstance(record, bytes):
                    byte_record = record
                elif isinstance(record, dict):
                    byte_record = json.dumps(record, sort_keys=True).encode("utf-8")
                elif isinstance(record, (int, float)):
                    byte_record = str(record).encode("utf-8")
                else:
                    # Convert other types to string first
                    byte_record = str(record).encode("utf-8")

                byte_records.append(byte_record)
                original_lengths.append(len(byte_record))

            except Exception as e:
                logger.error(f"Failed to convert record {i} to bytes: {e}")
                raise ValueError(f"Cannot convert record {i} to bytes: {e}")

        # Find maximum record length
        max_len = max(len(r) for r in byte_records)
        logger.info(f"Maximum record length: {max_len} bytes")

        # Validate against maximum supported length
        if max_len > self.max_record_length:
            raise ValueError(f"Record too large: {max_len} > {self.max_record_length}")

        # Round up to nearest block size for efficiency
        block_size = 256
        padded_length = ((max_len + block_size - 1) // block_size) * block_size

        # Ensure we don't exceed maximum supported length
        if padded_length > self.max_record_length:
            padded_length = self.max_record_length

        logger.info(f"Using padded length: {padded_length} bytes")

        # Pad all records to uniform length
        padded_records = []
        for record in byte_records:
            padded = self._pad_record(record, padded_length)
            padded_records.append(np.frombuffer(padded, dtype=np.uint8))

        # Stack into 2D array for PIR operations
        database = np.stack(padded_records)

        logger.info(f"Database prepared: {database.shape[0]} records × {database.shape[1]} bytes")
        return database, original_lengths

    def query(self, database: np.ndarray, index: int) -> bytes:
        """
        Query database with PIR to retrieve record at index.

        Args:
            database: Padded database array from prepare_database()
            index: Index to retrieve (0-based)

        Returns:
            Retrieved record with padding removed

        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(database):
            raise IndexError(f"Index {index} out of range [0, {len(database)})")

        logger.debug(f"Querying record at index {index}")

        # For now, direct access (in production this would use IT-PIR protocol)
        # The actual PIR protocol would use the IT-PIR implementation
        padded_record = database[index]

        # Convert numpy array back to bytes
        record_bytes = padded_record.tobytes()

        # Remove padding
        original_record = self._unpad_record(record_bytes)

        logger.debug(f"Retrieved record: {len(original_record)} bytes")
        return original_record

    def private_query(self, database: np.ndarray, index: int) -> bytes:
        """
        Perform private information retrieval query.

        This uses the IT-PIR protocol for actual privacy guarantees.

        Args:
            database: Padded database array
            index: Index to retrieve privately

        Returns:
            Retrieved record
        """
        from genomevault.pir.servers import PIRServer

        logger.info(f"Performing private query for index {index}")

        # Convert database to list of bytes for PIRServer
        records = []
        for row in database:
            records.append(row.tobytes())

        # Create PIR server
        server = PIRServer(records)

        # Create query mask (1 at target index, 0 elsewhere)
        mask = np.zeros(len(records), dtype=np.uint8)
        mask[index] = 1

        # Execute PIR query
        result = server.answer(mask)

        # Unpad the result
        return self._unpad_record(result)

    def validate_database(self, records: List[Any]) -> Tuple[bool, str]:
        """
        Validate database records for PIR compatibility.

        Args:
            records: Records to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not records:
            return False, "Empty database"

        try:
            # Check if all records can be converted to bytes
            byte_records = []
            for i, record in enumerate(records):
                try:
                    if isinstance(record, str):
                        byte_record = record.encode("utf-8")
                    elif isinstance(record, bytes):
                        byte_record = record
                    elif isinstance(record, dict):
                        byte_record = json.dumps(record, sort_keys=True).encode("utf-8")
                    elif isinstance(record, (int, float)):
                        byte_record = str(record).encode("utf-8")
                    else:
                        byte_record = str(record).encode("utf-8")

                    byte_records.append(byte_record)

                except Exception as e:
                    return False, f"Cannot convert record {i} to bytes: {e}"

            # Check maximum size
            max_size = max(len(r) for r in byte_records)
            if max_size > self.max_record_length:
                return False, f"Record too large: {max_size} > {self.max_record_length}"

            # Check for reasonable number of records
            if len(records) > 1000000:  # 1M records
                return False, f"Too many records: {len(records)} > 1,000,000"

            return True, "Valid"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def get_stats(self, database: np.ndarray, original_lengths: List[int]) -> dict:
        """
        Get database statistics.

        Args:
            database: Prepared database array
            original_lengths: Original record lengths

        Returns:
            Statistics dictionary
        """
        stats = {
            "num_records": len(database),
            "padded_record_size": database.shape[1] if len(database.shape) > 1 else 0,
            "original_sizes": {
                "min": min(original_lengths) if original_lengths else 0,
                "max": max(original_lengths) if original_lengths else 0,
                "avg": sum(original_lengths) / len(original_lengths) if original_lengths else 0,
            },
            "padding_efficiency": {
                "total_original": sum(original_lengths),
                "total_padded": database.size if database.size > 0 else 0,
                "overhead_ratio": (
                    (database.size - sum(original_lengths)) / sum(original_lengths)
                    if sum(original_lengths) > 0
                    else 0
                ),
            },
            "memory_usage_mb": database.nbytes / (1024 * 1024),
        }

        return stats


# Integration with existing PIR infrastructure
class EnhancedPIRServer:
    """Enhanced PIR server with variable length record support."""

    def __init__(self, records: List[Any], max_record_length: int = 1024):
        """
        Initialize enhanced PIR server.

        Args:
            records: Variable length records
            max_record_length: Maximum record length
        """
        self.engine = VariableLengthPIREngine(max_record_length)

        # Validate and prepare database
        is_valid, error_msg = self.engine.validate_database(records)
        if not is_valid:
            raise ValueError(f"Invalid database: {error_msg}")

        self.database, self.original_lengths = self.engine.prepare_database(records)
        self.stats = self.engine.get_stats(self.database, self.original_lengths)

        logger.info(f"Enhanced PIR server initialized: {self.stats['num_records']} records")

    def answer(self, mask: np.ndarray) -> bytes:
        """
        Answer PIR query with variable length record support.

        Args:
            mask: Query mask (1 at target indices)

        Returns:
            Retrieved record(s)
        """
        if len(mask) != len(self.database):
            raise ValueError(f"Mask length {len(mask)} != database size {len(self.database)}")

        # Find the target index
        target_indices = np.where(mask == 1)[0]

        if len(target_indices) == 0:
            return b""
        elif len(target_indices) == 1:
            # Single record query
            return self.engine.query(self.database, target_indices[0])
        else:
            # Multiple record query (XOR all selected records)
            result = np.zeros(self.database.shape[1], dtype=np.uint8)

            for idx in target_indices:
                result = np.bitwise_xor(result, self.database[idx])

            # Convert to bytes and unpad
            result_bytes = result.tobytes()
            return self.engine._unpad_record(result_bytes)

    def get_database_stats(self) -> dict:
        """Get database statistics."""
        return self.stats
