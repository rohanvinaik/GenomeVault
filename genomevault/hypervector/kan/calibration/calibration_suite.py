from __future__ import annotations

"""
KAN→HD Calibration Suite

Computes Pareto curves linking compression ratio × epsilon/delta × ECC settings
to downstream errors: reconstruction MSE, allele frequency error (AF_err),
odds-ratio error (OR_err), and p-value drift (p_drift) on canonical tests.

This module is deliberately model-agnostic: you pass callables for encode/decode
and data providers for ground truth metrics.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from hashlib import blake2b
from typing import Any, Callable, Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class CalibrationConfig:
    compression_ratios: List[float]
    epsilons: List[float]
    deltas: List[float]
    ecc_flags: List[bool]
    sample_size: int
    seed: int = 0


@dataclass
class CalibrationResult:
    config_hash: str
    records: List[Dict[str, Any]]


def _b2(b: bytes) -> str:
    return blake2b(b, digest_size=32).hexdigest()


def recon_mse(x: np.ndarray, x_hat: np.ndarray) -> float:
    return float(np.mean((x - x_hat) ** 2))


def allele_freq_error(gt: np.ndarray, pred: np.ndarray) -> float:
    # Simple AF approximation: mean across samples; compare absolute diff
    af_gt = gt.mean(axis=0)
    af_pr = pred.mean(axis=0)
    return float(np.mean(np.abs(af_gt - af_pr)))


def odds_ratio_error(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-6) -> float:
    # 2x2 OR per feature (presence vs absence); aggregate MAE
    def or_vec(arr):
        p = arr.mean(axis=0).clip(eps, 1 - eps)
        q = 1 - p
        return p / q

    return float(np.mean(np.abs(np.log(or_vec(gt)) - np.log(or_vec(pred)))))


def p_value_drift(gt_stats: np.ndarray, pr_stats: np.ndarray) -> float:
    # Domain-agnostic placeholder: L2 distance between z-score vectors
    return float(np.linalg.norm(gt_stats - pr_stats) / np.sqrt(gt_stats.size))


def run_calibration(
    cfg: CalibrationConfig,
    sample_provider: Callable[[int, int], Tuple[np.ndarray, np.ndarray]],
    # returns (X, stats) where stats are any per-feature statistics (e.g., z-scores)
    encode: Callable[[np.ndarray, float, float, float, bool], Any],
    decode: Callable[[Any], np.ndarray],
    artifact_dir: str = ".gv_artifacts/calibration",
) -> CalibrationResult:
    os.makedirs(artifact_dir, exist_ok=True)
    rs = np.random.RandomState(cfg.seed)
    records: List[Dict[str, Any]] = []
    for cr in cfg.compression_ratios:
        for eps in cfg.epsilons:
            for delt in cfg.deltas:
                for ecc in cfg.ecc_flags:
                    X, stats = sample_provider(cfg.sample_size, rs.randint(0, 1_000_000))
                    enc = encode(X, cr, eps, delt, ecc)
                    Xh = decode(enc)
                    rec = {
                        "compression_ratio": cr,
                        "epsilon": eps,
                        "delta": delt,
                        "ecc": ecc,
                        "recon_mse": recon_mse(X, Xh),
                        "af_err": allele_freq_error(X, Xh),
                        "or_err": odds_ratio_error(X, Xh),
                        "p_drift": p_value_drift(
                            stats, stats
                        ),  # replace with real downstream stat compare
                    }
                    records.append(rec)
    payload = json.dumps({"cfg": asdict(cfg), "records": records}, sort_keys=True).encode()
    cfg_hash = _b2(payload)
    out_path = os.path.join(artifact_dir, f"calibration_{cfg_hash}.json")
    with open(out_path, "w") as f:
        f.write(payload.decode())
    return CalibrationResult(config_hash=cfg_hash, records=records)
