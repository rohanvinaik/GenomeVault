"""Packed module."""
from __future__ import annotations

from typing import cast

from numpy.typing import NDArray
import numpy as np
import torch
def pack_bits(x: torch.Tensor) -> NDArray[np.uint8]:
    """Pack {-1,+1} or {0,1} signs into bytes (numpy)."""
    x = x.detach().cpu().view(-1)
    bits: NDArray[np.uint8] = cast(NDArray[np.uint8], (x > 0).to(torch.uint8).numpy())
    # pad to multiple of 8
    pad = (-len(bits)) % 8
    bits = np.pad(bits, (0, pad))
    return np.packbits(bits, bitorder="little")


def unpack_bits(packed: NDArray[np.uint8], length: int) -> torch.Tensor:
    """Unpack bits.

    Args:
        packed: Packed.
        length: Length.

    Returns:
        Operation result.
    """
    bits: NDArray[np.uint8] = np.unpackbits(packed, bitorder="little")[:length].astype(np.uint8)
    unpacked_array: NDArray[np.int8] = (bits * 2 - 1).astype(np.int8)
    return torch.from_numpy(unpacked_array.astype(np.int8, copy=False))
