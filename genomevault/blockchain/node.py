from __future__ import annotations

"""
Minimal governance node registry and quadratic voting proposal lifecycle.

This is a skeleton that is *functional* and testable; integrate with on-chain contracts as needed.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from hashlib import blake2b
from math import sqrt
from typing import Dict, List, Optional

ARTIFACT_DIR = os.environ.get("GV_GOVERNANCE_DIR", ".gv_artifacts/governance")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _b2(b: bytes) -> str:
    return blake2b(b, digest_size=32).hexdigest()


@dataclass
class Node:
    node_id: str
    role: str  # 'validator', 'observer', etc.
    hipaa_verified: bool
    pubkey: str
    joined_at: float


@dataclass
class Proposal:
    proposal_id: str
    title: str
    description: str
    created_by: str
    created_at: float
    votes: Dict[str, float]  # voter_id -> credits spent
    executed: bool = False


class Governance:
    def __init__(self, dir_path: str = ARTIFACT_DIR) -> None:
        self.dir = dir_path
        _ensure_dir(self.dir)
        self.nodes_path = os.path.join(self.dir, "nodes.json")
        self.props_path = os.path.join(self.dir, "proposals.json")
        self._nodes: Dict[str, Node] = {}
        self._props: Dict[str, Proposal] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.nodes_path):
            with open(self.nodes_path, "r") as f:
                nodes = json.load(f)
            self._nodes = {k: Node(**v) for k, v in nodes.items()}
        if os.path.exists(self.props_path):
            with open(self.props_path, "r") as f:
                props = json.load(f)
            self._props = {k: Proposal(**v) for k, v in props.items()}

    def _save(self) -> None:
        with open(self.nodes_path, "w") as f:
            json.dump({k: asdict(v) for k, v in self._nodes.items()}, f, indent=2, sort_keys=True)
        with open(self.props_path, "w") as f:
            json.dump({k: asdict(v) for k, v in self._props.items()}, f, indent=2, sort_keys=True)

    def add_node(self, node_id: str, role: str, hipaa_verified: bool, pubkey: str) -> None:
        self._nodes[node_id] = Node(
            node_id=node_id,
            role=role,
            hipaa_verified=hipaa_verified,
            pubkey=pubkey,
            joined_at=time.time(),
        )
        self._save()

    def propose(self, title: str, description: str, created_by: str) -> str:
        pid = _b2((title + description + created_by + str(time.time())).encode())
        self._props[pid] = Proposal(
            proposal_id=pid,
            title=title,
            description=description,
            created_by=created_by,
            created_at=time.time(),
            votes={},
        )
        self._save()
        return pid

    @staticmethod
    def _qv_weight(credits: float) -> float:
        # Quadratic voting: weight = sqrt(credits)
        return sqrt(max(0.0, credits))

    def vote(self, proposal_id: str, voter_id: str, credits: float) -> None:
        prop = self._props[proposal_id]
        prop.votes[voter_id] = prop.votes.get(voter_id, 0.0) + max(0.0, credits)
        self._save()

    def tally(self, proposal_id: str) -> float:
        prop = self._props[proposal_id]
        return sum(self._qv_weight(c) for c in prop.votes.values())

    def execute(self, proposal_id: str, threshold: float) -> bool:
        prop = self._props[proposal_id]
        if prop.executed:
            return True
        if self.tally(proposal_id) >= threshold:
            prop.executed = True
            self._save()
            return True
        return False
