from __future__ import annotations
from typing import Dict, Iterable, List
import hashlib
import numpy as np

CHROM_MAP = {f"chr{i}": i for i in range(1, 23)} | {
    "chrX": 23,
    "chrY": 24,
    "X": 23,
    "Y": 24,
}

IMPACT_MAP = {
    "HIGH": 3.0,
    "MODERATE": 2.0,
    "LOW": 1.0,
    "MODIFIER": 0.5,
    "high": 3.0,
    "moderate": 2.0,
    "low": 1.0,
    "modifier": 0.5,
}


def _hash01(s: str) -> float:
    # Stable hash → [0,1]
    h = hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, "little") / (2**64 - 1)


def variant_to_numeric(v: Dict) -> List[float]:
    """
    Map a single VCF-like variant dict to a numeric feature vector.
    Expect keys: chrom, pos, ref, alt, impact (where available).
    """
    chrom = float(CHROM_MAP.get(str(v.get("chrom", "")).strip(), 0))
    pos = float(v.get("pos", 0)) % 1_000_000  # localize to manageable scale
    ref = _hash01(str(v.get("ref", "")))
    alt = _hash01(str(v.get("alt", "")))
    imp = IMPACT_MAP.get(str(v.get("impact", "")).strip(), 0.0)
    # You can add genotype, quality, depth, etc., similarly
    return [chrom, pos, ref, alt, imp]


def featurize_variants(variants: Iterable[Dict]) -> np.ndarray:
    feats = [variant_to_numeric(v) for v in variants]
    if not feats:
        return np.zeros((0, 5), dtype="float32")
    return np.asarray(feats, dtype="float32").mean(axis=0)  # simple pooling
