from typing import Any, Dict, List, Optional

"""
Base node implementation for GenomeVault blockchain
"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from genomevault.core.config import get_config
from genomevault.core.constants import (
    BLOCK_TIME_SECONDS,
    NodeClassWeight,
    NodeType,
    SignatoryWeight,
)


@dataclass
class Block:
    """Blockchain block structure"""

    index: int
    timestamp: float
    transactions: List[Dict[str, Any]]
    proof_hashes: List[str]
    previous_hash: str
    nonce: int
    hash: str


class BaseNode(ABC):
    """
    Abstract base class for all node types
    Implements core blockchain functionality
    """

    def __init__(self, node_type: NodeType, is_signatory: bool = False):
        self.node_type = node_type
        self.is_signatory = is_signatory
        self.config = get_config()

        # Calculate voting power: w = c + s
        self.hardware_weight = self._get_hardware_weight()
        self.signatory_weight = (
            SignatoryWeight.TRUSTED_SIGNATORY if is_signatory else SignatoryWeight.NON_SIGNER
        )
        self.voting_power = self.hardware_weight + self.signatory_weight

        # Blockchain state
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict[str, Any]] = []
        self.peers: List[str] = []

        # Node-specific storage
        self.storage_limit = self._get_storage_limit()
        self.stored_proofs: Dict[str, bytes] = {}

    def _get_hardware_weight(self) -> int:
        """Get hardware class weight based on node type"""
        weight_map = {
            NodeType.LIGHT: NodeClassWeight.LIGHT,
            NodeType.FULL: NodeClassWeight.FULL,
            NodeType.ARCHIVE: NodeClassWeight.ARCHIVE,
        }
        return weight_map[self.node_type]

    @abstractmethod
    def _get_storage_limit(self) -> int:
        """Get storage limit in bytes based on node type"""
        pass

    @abstractmethod
    async def sync_chain(self):
        """Sync blockchain with network"""
        pass

    @abstractmethod
    async def validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Validate a transaction based on node capabilities"""
        pass

    def calculate_credits_per_block(self) -> int:
        """
        Calculate credits earned per block
        credits = c + 2 * [s > 0]
        """
        base_credits = self.hardware_weight
        signatory_bonus = 2 if self.is_signatory else 0
        return base_credits + signatory_bonus

    async def create_block(self) -> Optional[Block]:
        """Create a new block if this node is selected"""
        if not self.pending_transactions:
            return None

        # Simple leader selection based on voting power
        # In production, use proper consensus mechanism
        if not self._is_block_producer():
            return None

        block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=self.pending_transactions[:100],  # Limit transactions per block
            proof_hashes=[tx.get("proof_hash", "") for tx in self.pending_transactions[:100]],
            previous_hash=self.chain[-1].hash if self.chain else "0",
            nonce=0,
            hash="",
        )

        # Simple proof of work (for demonstration)
        block.nonce = self._proof_of_work(block)
        block.hash = self._calculate_hash(block)

        return block

    def _is_block_producer(self) -> bool:
        """Determine if this node should produce the next block"""
        # Simplified selection based on voting power
        # In production, use proper BFT consensus
        current_slot = int(time.time() / BLOCK_TIME_SECONDS)
        selection_hash = hashlib.sha256("{current_slot}:{self.voting_power}".encode()).hexdigest()
        selection_value = int(selection_hash[:8], 16)

        # Probability proportional to voting power
        threshold = (self.voting_power / 100) * (2**32)
        return selection_value < threshold

    def _proof_of_work(self, block: Block) -> int:
        """Simple proof of work for demonstration"""
        nonce = 0
        while True:
            if self._valid_proof(block, nonce):
                return nonce
            nonce += 1

    def _valid_proof(self, block: Block, nonce: int) -> bool:
        """Check if proof is valid"""
        guess = "{block.index}{block.timestamp}{block.transactions}{block.previous_hash}{nonce}"
        guess_hash = hashlib.sha256(guess.encode()).hexdigest()
        return guess_hash[:4] == "0000"  # Require 4 leading zeros

    def _calculate_hash(self, block: Block) -> str:
        """Calculate block hash"""
        block_string = json.dumps(
            {
                "index": block.index,
                "timestamp": block.timestamp,
                "transactions": block.transactions,
                "proof_hashes": block.proof_hashes,
                "previous_hash": block.previous_hash,
                "nonce": block.nonce,
            },
            sort_keys=True,
        )
        return hashlib.sha256(block_string.encode()).hexdigest()

    async def add_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Add a transaction to pending pool"""
        if await self.validate_transaction(transaction):
            self.pending_transactions.append(transaction)
            return True
        return False

    def get_node_info(self) -> Dict[str, Any]:
        """Get node information"""
        return {
            "node_type": self.node_type.value,
            "is_signatory": self.is_signatory,
            "voting_power": self.voting_power,
            "hardware_weight": self.hardware_weight,
            "signatory_weight": self.signatory_weight,
            "credits_per_block": self.calculate_credits_per_block(),
            "chain_length": len(self.chain),
            "pending_transactions": len(self.pending_transactions),
            "storage_used": sum(len(proof) for proof in self.stored_proofs.values()),
            "storage_limit": self.storage_limit,
        }
