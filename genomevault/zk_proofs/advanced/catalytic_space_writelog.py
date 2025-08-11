"""
Write-log based Catalytic Space implementation.

This implementation demonstrates the catalytic computing concept more explicitly
by maintaining a write log of all modifications, allowing perfect restoration
while being more space-efficient for sparse modifications.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WriteEntry:
    """Single write operation entry in the log."""

    offset: int
    original_data: bytes
    new_data: bytes
    sequence_num: int


class CatalyticSpaceWriteLog:
    """
    Catalytic memory space with write-log based restoration.

    This implementation is more space-efficient when modifications are sparse,
    as it only stores the changes rather than a full copy of the initial state.
    It also better demonstrates the "catalytic" concept where the auxiliary
    space is borrowed temporarily and must be restored exactly.
    """

    def __init__(self, size: int):
        """
        Initialize catalytic space with write logging.

        Args:
            size: Size of catalytic space in bytes
        """
        self.size = size

        # Initialize with secure random data
        self.data = bytearray(os.urandom(size))

        # Write log for restoration
        self.write_log: list[WriteEntry] = []
        self.sequence_counter = 0

        # Compute initial fingerprint
        self.initial_fingerprint = self._compute_fingerprint()

        # Statistics
        self.access_count = 0
        self.modification_count = 0
        self.max_log_size = 0

        logger.info(f"Initialized write-log catalytic space: {size} bytes")

    def _compute_fingerprint(self) -> str:
        """Compute cryptographic fingerprint of current state."""
        return hashlib.sha256(self.data).hexdigest()

    def read(self, offset: int, length: int) -> bytes:
        """
        Read from catalytic space.

        Args:
            offset: Starting offset
            length: Number of bytes to read

        Returns:
            Bytes read from the space
        """
        if offset < 0 or offset + length > self.size:
            raise ValueError(f"Read out of bounds: offset={offset}, length={length}")

        self.access_count += 1
        return bytes(self.data[offset : offset + length])

    def write(self, offset: int, data: bytes) -> None:
        """
        Write to catalytic space with logging for restoration.

        Args:
            offset: Starting offset
            data: Data to write
        """
        if offset < 0 or offset + len(data) > self.size:
            raise ValueError(f"Write out of bounds: offset={offset}, length={len(data)}")

        # Record the original data before overwriting
        original_data = bytes(self.data[offset : offset + len(data)])

        # Create write log entry
        entry = WriteEntry(
            offset=offset,
            original_data=original_data,
            new_data=data,
            sequence_num=self.sequence_counter,
        )
        self.write_log.append(entry)
        self.sequence_counter += 1

        # Perform the write
        self.data[offset : offset + len(data)] = data

        # Update statistics
        self.modification_count += 1
        self.max_log_size = max(self.max_log_size, len(self.write_log))

        logger.debug(
            f"Write logged: offset={offset}, size={len(data)}, "
            f"log_entries={len(self.write_log)}"
        )

    def reset(self) -> bool:
        """
        Reset catalytic space to initial state using write log.

        Returns:
            True if successfully reset
        """
        current_fingerprint = self._compute_fingerprint()

        if current_fingerprint == self.initial_fingerprint:
            # Already in initial state
            self.write_log.clear()
            self.access_count = 0
            self.modification_count = 0
            return True

        logger.info(
            f"Restoring catalytic space: {len(self.write_log)} writes to undo, "
            f"max log size was {self.max_log_size} entries"
        )

        # Restore in reverse order (LIFO)
        for entry in reversed(self.write_log):
            # Restore original data
            self.data[entry.offset : entry.offset + len(entry.original_data)] = entry.original_data

        # Clear the write log
        self.write_log.clear()
        self.sequence_counter = 0

        # Reset statistics
        self.access_count = 0
        self.modification_count = 0

        # Verify restoration
        restored_fingerprint = self._compute_fingerprint()
        if restored_fingerprint != self.initial_fingerprint:
            logger.error(
                f"Failed to restore catalytic space! "
                f"Expected: {self.initial_fingerprint}, Got: {restored_fingerprint}"
            )
            return False

        logger.info("Catalytic space successfully restored to initial state")
        return True

    def get_usage_stats(self) -> dict[str, Any]:
        """
        Get usage statistics for the catalytic space.

        Returns:
            Dictionary with usage statistics
        """
        current_log_bytes = sum(
            len(entry.original_data) + len(entry.new_data) + 16  # overhead
            for entry in self.write_log
        )

        return {
            "size": self.size,
            "access_count": self.access_count,
            "modification_count": self.modification_count,
            "current_log_entries": len(self.write_log),
            "max_log_entries": self.max_log_size,
            "current_log_bytes": current_log_bytes,
            "log_efficiency": current_log_bytes / self.size if self.size > 0 else 0,
            "fingerprint": self._compute_fingerprint(),
            "is_pristine": self._compute_fingerprint() == self.initial_fingerprint,
        }

    def compact_log(self) -> int:
        """
        Compact the write log by merging overlapping writes.

        This optimization reduces memory usage when the same regions
        are written multiple times.

        Returns:
            Number of entries removed
        """
        if len(self.write_log) < 2:
            return 0

        # Build a map of regions to their earliest original data
        region_map: dict[tuple[int, int], bytes] = {}
        compacted_log: list[WriteEntry] = []

        for entry in self.write_log:
            region = (entry.offset, entry.offset + len(entry.original_data))

            # Check for overlaps with existing regions
            overlaps = False
            for (start, end), original in list(region_map.items()):
                if not (region[1] <= start or region[0] >= end):
                    # Regions overlap - this is a complex case
                    # For simplicity, we'll keep both entries
                    overlaps = True
                    break

            if not overlaps and region not in region_map:
                # First write to this region - record original data
                region_map[region] = entry.original_data
                compacted_log.append(entry)
            elif region in region_map:
                # Subsequent write to same region - update the new_data
                # but keep the original_data from first write
                for i, existing in enumerate(compacted_log):
                    if existing.offset == entry.offset and len(existing.original_data) == len(
                        entry.original_data
                    ):
                        compacted_log[i] = WriteEntry(
                            offset=entry.offset,
                            original_data=existing.original_data,  # Keep first original
                            new_data=entry.new_data,  # Update to latest write
                            sequence_num=entry.sequence_num,
                        )
                        break
            else:
                # Overlapping writes - keep as is for correctness
                compacted_log.append(entry)

        removed = len(self.write_log) - len(compacted_log)
        self.write_log = compacted_log

        if removed > 0:
            logger.info(f"Compacted write log: removed {removed} redundant entries")

        return removed


# Example usage demonstrating the catalytic concept
if __name__ == "__main__":
    # Create a catalytic space
    space = CatalyticSpaceWriteLog(1024 * 1024)  # 1MB

    logger.info("Demonstrating catalytic space with write-log restoration")
    logger.info("=" * 60)

    # Simulate some computations that use the space temporarily
    logger.info("\n1. Writing temporary computation data...")

    # Write some data
    for i in range(10):
        offset = i * 1024
        data = os.urandom(512)
        space.write(offset, data)

    stats = space.get_usage_stats()
    logger.info(
        f"After writes: {stats['current_log_entries']} log entries, "
        f"{stats['current_log_bytes']} bytes in log"
    )

    # Overwrite some regions (demonstrates log compaction potential)
    logger.info("\n2. Overwriting some regions...")
    for i in range(5):
        offset = i * 1024
        data = os.urandom(512)
        space.write(offset, data)

    stats = space.get_usage_stats()
    logger.info(f"After overwrites: {stats['current_log_entries']} log entries")

    # Compact the log
    removed = space.compact_log()
    stats = space.get_usage_stats()
    logger.info(
        f"After compaction: removed {removed} entries, "
        f"now {stats['current_log_entries']} entries"
    )

    # Verify space is modified
    logger.info(f"\n3. Space is modified: {not stats['is_pristine']}")

    # Reset to initial state
    logger.info("\n4. Resetting catalytic space...")
    success = space.reset()

    # Verify restoration
    stats = space.get_usage_stats()
    logger.info(f"Reset successful: {success}")
    logger.info(f"Space is pristine: {stats['is_pristine']}")
    logger.info(f"Log entries after reset: {stats['current_log_entries']}")

    # Demonstrate space efficiency
    logger.info("\n5. Space efficiency comparison:")
    logger.info(f"  - Full snapshot approach: {space.size} bytes")
    logger.info(f"  - Write-log approach (max): {stats['max_log_entries'] * 1024} bytes (approx)")
    logger.info(f"  - Efficiency gain: {space.size / (stats['max_log_entries'] * 1024 + 1):.1f}x")
