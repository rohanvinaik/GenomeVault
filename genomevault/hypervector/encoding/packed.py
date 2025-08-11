from __future__ import annotations

"""Packed module."""
import numpy as np
import torch


def pack_bits(x: torch.Tensor) -> np.ndarray:
    """Pack {-1,+1} or {0,1} signs into bytes (numpy)."""
    x = x.detach().cpu().view(-1)
    bits = (x > 0).to(torch.uint8).numpy()
    # pad to multiple of 8
    pad = (-len(bits)) % 8
    bits = np.pad(bits, (0, pad))
    return np.packbits(bits, bitorder="little")


def unpack_bits(packed: np.ndarray, length: int) -> torch.Tensor:
    """Unpack bits.

    Args:
        packed: Packed.
        length: Length.

    Returns:
        Operation result.
    """
    bits = np.unpackbits(packed, bitorder="little")[:length].astype(np.uint8)
    return torch.from_numpy((bits * 2 - 1).astype(np.int8))
