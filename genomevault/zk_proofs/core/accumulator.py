from __future__ import annotations

from typing import Dict, List

from genomevault.crypto.commit import TAGS, H

INITIAL_ACC = H(TAGS["ACC"], b"INITIAL_ACC")


def step(acc: bytes, proof_commit: bytes, vk_hash: bytes) -> bytes:
    return H(TAGS["ACC"], acc, proof_commit, vk_hash)


def verify_chain(chain: List[Dict[str, str]], final_hex: str | None) -> bool:
    """
    chain: [{"proof_commit": <hex>, "vk_hash": <hex>, "proof_id": "..."}...]
    """
    if final_hex is None:
        return False
    acc = INITIAL_ACC
    for s in chain:
        pc = bytes.fromhex(s["proof_commit"])
        vk = bytes.fromhex(s["vk_hash"])
        acc = step(acc, pc, vk)
    return acc.hex() == final_hex
