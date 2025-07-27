from __future__ import annotations

"""
SRS Manager with versioned IDs, VK registry, and domain-separated transcript utility.

- Stores SRS and VK artifacts under a versioned directory (default: .gv_artifacts/zk/).
- Computes stable IDs using BLAKE2b over content + metadata JSON.
- Provides transcript() helper with domain separation tags for PLONK-like protocols.
- Pure-Python and file-based so it works without external services.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from hashlib import blake2b
from typing import Any, Dict, Optional, Tuple

DEFAULT_BASE_DIR = os.environ.get("GV_ZK_ARTIFACT_DIR", ".gv_artifacts/zk")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _b2(content: bytes) -> str:
    return blake2b(content, digest_size=32).hexdigest()


@dataclass(frozen=True)
class SRSMetadata:
    curve: str
    size: int
    toolchain: str
    toolchain_version: str


@dataclass(frozen=True)
class SRSRecord:
    srs_id: str
    path: str
    meta: SRSMetadata
    created_at: float


@dataclass(frozen=True)
class VKRecord:
    circuit_id: str
    vk_hash: str
    srs_id: str
    path: str
    created_at: float


class SRSManager:
    def __init__(self, base_dir: str = DEFAULT_BASE_DIR) -> None:
        self.base_dir = base_dir
        self.srs_dir = os.path.join(base_dir, "srs")
        self.vk_dir = os.path.join(base_dir, "vk")
        self.index_path = os.path.join(base_dir, "index.json")
        _ensure_dir(self.srs_dir)
        _ensure_dir(self.vk_dir)
        self._load_index()

    def _load_index(self) -> None:
        if os.path.exists(self.index_path):
            with open(self.index_path, "r") as f:
                idx = json.load(f)
        else:
            idx = {"srs": {}, "vk": {}}
        self._srs: Dict[str, SRSRecord] = {
            k: SRSRecord(
                srs_id=v["srs_id"],
                path=v["path"],
                meta=SRSMetadata(**v["meta"]),
                created_at=v["created_at"],
            )
            for k, v in idx.get("srs", {}).items()
        }
        self._vk: Dict[str, VKRecord] = {k: VKRecord(**v) for k, v in idx.get("vk", {}).items()}

    def _save_index(self) -> None:
        idx = {
            "srs": {
                k: {
                    "srs_id": rec.srs_id,
                    "path": rec.path,
                    "meta": asdict(rec.meta),
                    "created_at": rec.created_at,
                }
                for k, rec in self._srs.items()
            },
            "vk": {k: asdict(rec) for k, rec in self._vk.items()},
        }
        _ensure_dir(self.base_dir)
        with open(self.index_path, "w") as f:
            json.dump(idx, f, indent=2, sort_keys=True)

    # -- SRS --
    def register_srs(self, srs_bytes: bytes, meta: SRSMetadata) -> str:
        payload = json.dumps(asdict(meta), sort_keys=True).encode() + b"|" + srs_bytes
        srs_id = _b2(payload)
        path = os.path.join(self.srs_dir, f"{srs_id}.srs")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(srs_bytes)
        self._srs[srs_id] = SRSRecord(srs_id=srs_id, path=path, meta=meta, created_at=time.time())
        self._save_index()
        return srs_id

    def get_srs(self, srs_id: str) -> bytes:
        rec = self._srs.get(srs_id)
        if not rec:
            raise KeyError(f"SRS not found: {srs_id}")
        with open(rec.path, "rb") as f:
            return f.read()

    # -- VK --
    def register_vk(self, circuit_id: str, vk_bytes: bytes, srs_id: str) -> str:
        if srs_id not in self._srs:
            raise KeyError(f"SRS {srs_id} must exist before registering VK")
        vk_hash = _b2(vk_bytes)
        path = os.path.join(self.vk_dir, f"{circuit_id}.{vk_hash}.vk")
        with open(path, "wb") as f:
            f.write(vk_bytes)
        rec = VKRecord(
            circuit_id=circuit_id, vk_hash=vk_hash, srs_id=srs_id, path=path, created_at=time.time()
        )
        self._vk[circuit_id] = rec
        self._save_index()
        return vk_hash

    def get_vk(self, circuit_id: str) -> Tuple[str, str, bytes]:
        rec = self._vk.get(circuit_id)
        if not rec:
            raise KeyError(f"VK not found for circuit: {circuit_id}")
        with open(rec.path, "rb") as f:
            vk_bytes = f.read()
        return rec.vk_hash, rec.srs_id, vk_bytes

    # -- Transcript helper --
    @staticmethod
    def transcript(domain: str, *parts: bytes) -> bytes:
        """
        Domain-separated transcript hash using BLAKE2b(32).
        Example usage: transcript("median_circuit", public_inputs_bytes, witness_commitment)
        """
        h = blake2b(digest_size=32)
        h.update(b"GV::DOMAIN::" + domain.encode())
        for p in parts:
            h.update(b"|")
            h.update(p)
        return h.digest()
