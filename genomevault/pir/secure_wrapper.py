from __future__ import annotations

"""Secure Wrapper module."""
from dataclasses import dataclass
from typing import Sequence


@dataclass
class PIRServer:
    """Toy 2-server info-theoretic PIR server holding a shard."""

    data: Sequence[int]

    def get(self, mask: Sequence[int]) -> int:
        """Get.

        Args:
            mask: Mask.

        Returns:
            Integer result.
        """
        # Return XOR of selected entries
        acc = 0
        for bit, val in zip(mask, self.data):
            if bit:
                acc ^= int(val)
        return acc


class SecurePIRWrapper:
    """
    Minimal 2-server XOR PIR wrapper:
    - query(index) builds two random masks whose XOR selects exactly index
    - correctness: server1(mask1) ^ server2(mask2) == data[index]
    """

    def __init__(self, server1: PIRServer, server2: PIRServer):
        """Initialize instance.

        Args:
            server1: Server1.
            server2: Server2.
        """
        assert len(server1.data) == len(server2.data), "server sizes must match"
        self.n = len(server1.data)
        self.s1 = server1
        self.s2 = server2

    def query(self, index: int) -> int:
        """Query.

        Args:
            index: Index position.

        Returns:
            Integer result.

        Raises:
            ValueError: When operation fails.
        """
        if not (0 <= index < self.n):
            raise ValueError("index out of range")
        # mask1 random; mask2 = mask1 with index bit flipped
        import random

        mask1 = [random.randint(0, 1) for _ in range(self.n)]
        mask2 = mask1[:]
        mask2[index] ^= 1
        r1 = self.s1.get(mask1)
        r2 = self.s2.get(mask2)
        return r1 ^ r2

    @staticmethod
    def from_single_array(arr: Sequence[int]) -> "SecurePIRWrapper":
        """From single array.

        Args:
            arr: Arr.

        Returns:
            Operation result.
        """
        # Duplicate data across both servers (fine for tests)
        return SecurePIRWrapper(PIRServer(arr), PIRServer(arr))
