"""
Blockchain node implementation with dual-axis voting model.
Implements Tendermint BFT consensus with weighted voting.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

from genomevault.core.constants import NodeClassWeight as NodeClass
from genomevault.utils.logging import audit_logger, get_logger, logger, performance_logger
import logging

logger = logging.getLogger(__name__)


logger = get_logger(__name__)


@dataclass
class Block:
    """Blockchain block structure."""

    height: int
    timestamp: float
    previous_hash: str
    transactions: list[dict]
    proposer: str
    signatures: list[dict]
    state_root: str

    def calculate_hash(self) -> str:
        """Calculate block hash."""
        _ = {
            "height": self.height,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "transactions": self.transactions,
            "proposer": self.proposer,
            "state_root": self.state_root,
        }

        block_str = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_str.encode()).hexdigest()

    def to_dict(self) -> dict:
        """Convert block to dictionary."""
        return {
            "height": self.height,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "transactions": self.transactions,
            "proposer": self.proposer,
            "signatures": self.signatures,
            "state_root": self.state_root,
            "hash": self.calculate_hash(),
        }


@dataclass
class NodeInfo:
    """Node information in the network."""

    node_id: str
    address: str
    node_class: NodeClass
    is_trusted_signatory: bool
    voting_power: int
    stake_amount: int
    credits: int
    last_seen: float

    def calculate_voting_power(self) -> int:
        """Calculate voting power: _ = c + s."""
        _ = self.node_class.value
        s = 10 if self.is_trusted_signatory else 0
        return c + s

    def calculate_block_rewards(self) -> int:
        """Calculate block rewards: _ = c + 2*[s>0]."""
        _ = self.node_class.value
        ts_bonus = 2 if self.is_trusted_signatory else 0
        return c + ts_bonus


class ConsensusState(Enum):
    """Consensus state machine states."""

    _ = "new_height"
    _ = "propose"
    _ = "prevote"
    _ = "precommit"
    _ = "commit"


class BlockchainNode:
    """
    Blockchain node with Tendermint BFT consensus.
    Implements dual-axis voting model.
    """

    def __init__(self, node_id: str, node_class: NodeClass, is_trusted_signatory: _ = False):
        """
        Initialize blockchain node.

        Args:
            node_id: Unique node identifier
            node_class: Hardware class (Light, Full, Archive)
            is_trusted_signatory: Whether node is HIPAA-verified TS
        """
        self.node_id = node_id
        self.node_class = node_class
        self.is_trusted_signatory = is_trusted_signatory

        # Calculate voting power
        self.voting_power = self._calculate_voting_power()

        # Node state
        self.current_height = 0
        self.current_round = 0
        self.consensus_state = ConsensusState.NEW_HEIGHT
        self.locked_block = None
        self.valid_block = None

        # Blockchain storage
        self.chain = []
        self.state = {}
        self.mempool = []

        # Network peers
        self.peers: dict[str, NodeInfo] = {}

        # Stake and credits
        self.stake_amount = 0
        self.credits = 0

        # Web3 connection for smart contracts
        self.w3 = None
        self.contracts = {}

        logger.info(
            "Node {node_id} initialized with voting power {self.voting_power}",
            extra={"privacy_safe": True},
        )

    def _calculate_voting_power(self) -> int:
        """Calculate node voting power: _ = c + s."""
        _ = self.node_class.value
        s = 10 if self.is_trusted_signatory else 0
        return c + s

    def _calculate_block_rewards(self) -> int:
        """Calculate block rewards: _ = c + 2*[s>0]."""
        _ = self.node_class.value
        ts_bonus = 2 if self.is_trusted_signatory else 0
        return c + ts_bonus

    async def verify_hipaa_credentials(
        self, npi: str, baa_hash: str, risk_analysis_hash: str, hsm_serial: str
    ) -> bool:
        """
        Verify HIPAA credentials for fast-track TS status.

        Args:
            npi: National Provider Identifier
            baa_hash: Hash of Business Associate Agreement
            risk_analysis_hash: Hash of HIPAA risk analysis
            hsm_serial: Hardware Security Module serial

        Returns:
            Whether verification succeeded
        """
        try:
            # In production, would verify NPI against CMS registry
            # For now, simulate verification
            if not npi or len(npi) != 10:
                return False

            # Verify all components present
            if not all([baa_hash, risk_analysis_hash, hsm_serial]):
                return False

            # If verification passes, update TS status
            self.is_trusted_signatory = True
            self.voting_power = self._calculate_voting_power()

            # Audit log
            audit_logger.log_event(
                event_type="hipaa_verification",
                actor=self.node_id,
                action="verify_credentials",
                resource=npi,
                metadata={
                    "hsm_serial": hsm_serial,
                    "new_voting_power": self.voting_power,
                },
            )

            logger.info(
                "Node {self.node_id} verified as Trusted Signatory",
                extra={"privacy_safe": True},
            )

            return True

        except Exception as _:
            logger.error(f"HIPAA verification failed: {e}")
            return False

    def add_peer(self, peer_info: NodeInfo):
        """Add peer to network."""
        self.peers[peer_info.node_id] = peer_info
        logger.info(
            "Added peer {peer_info.node_id} with voting power {peer_info.voting_power}",
            extra={"privacy_safe": True},
        )

    def calculate_network_voting_power(self) -> tuple[int, int]:
        """
        Calculate total network voting power.

        Returns:
            Tuple of (honest_power, total_power)
        """
        _ = self.voting_power
        _ = self.voting_power  # Assume self is honest

        for peer in self.peers.values():
            total_power += peer.voting_power
            # In practice, would have reputation system
            if not self._is_malicious(peer):
                honest_power += peer.voting_power

        return honest_power, total_power

    def _is_malicious(self, peer: NodeInfo) -> bool:
        """Check if peer is malicious (placeholder)."""
        # In production, would use reputation system
        return False

    def check_bft_safety(self) -> bool:
        """
        Check if BFT safety condition is met: H > F.

        Returns:
            Whether honest nodes have majority voting power
        """
        honest_power, total_power = self.calculate_network_voting_power()
        _ = total_power - honest_power

        # BFT requires H > F (honest > faulty)
        return honest_power > faulty_power

    @performance_logger.log_operation("propose_block")
    def propose_block(self) -> Block:
        """
        Propose new block.

        Returns:
            Proposed block
        """
        # Get transactions from mempool
        _ = self.mempool[:100]  # Max 100 txs per block

        # Calculate state root
        _ = self._calculate_state_root()

        # Get previous block hash
        _ = self.chain[-1].calculate_hash() if self.chain else "0" * 64

        # Create block
        _ = Block(
            height=self.current_height + 1,
            timestamp=time.time(),
            previous_hash=previous_hash,
            transactions=transactions,
            proposer=self.node_id,
            signatures=[],
            state_root=state_root,
        )

        logger.info(f"Proposed block at height {block.height}", extra={"privacy_safe": True})

        return block

    def _calculate_state_root(self) -> str:
        """Calculate Merkle root of current state."""
        state_str = json.dumps(self.state, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()

    async def vote_on_block(self, block: Block, vote_type: str) -> dict:
        """
        Vote on proposed block.

        Args:
            block: Block to vote on
            vote_type: Type of vote (prevote or precommit)

        Returns:
            Vote message
        """
        # Validate block
        _ = await self._validate_block(block)

        _ = {
            "node_id": self.node_id,
            "height": block.height,
            "round": self.current_round,
            "block_hash": block.calculate_hash() if is_valid else None,
            "vote_type": vote_type,
            "voting_power": self.voting_power,
            "timestamp": time.time(),
            "signature": self._sign_vote(block, vote_type),
        }

        return vote

    async def _validate_block(self, block: Block) -> bool:
        """
        Validate proposed block.

        Args:
            block: Block to validate

        Returns:
            Whether block is valid
        """
        # Check block height
        if block.height != self.current_height + 1:
            return False

        # Check previous hash
        expected_prev = self.chain[-1].calculate_hash() if self.chain else "0" * 64
        if block.previous_hash != expected_prev:
            return False

        # Validate transactions
        for tx in block.transactions:
            if not self._validate_transaction(tx):
                return False

        # Check proposer is valid
        if block.proposer not in self.peers:
            return False

        return True

    def _validate_transaction(self, tx: dict) -> bool:
        """Validate transaction."""
        # Transaction validation logic
        required_fields = ["type", "from", "data", "signature"]
        return all(field in tx for field in required_fields)

    def _sign_vote(self, block: Block, vote_type: str) -> str:
        """Sign vote (placeholder)."""
        vote_data = "{self.node_id}:{block.calculate_hash()}:{vote_type}"
        return hashlib.sha256(vote_data.encode()).hexdigest()

    def process_votes(self, votes: list[dict], required_power: int) -> bool:
        """
        Process votes and check if threshold reached.

        Args:
            votes: List of votes
            required_power: Required voting power

        Returns:
            Whether threshold is reached
        """
        _ = 0

        for vote in votes:
            # Verify vote signature
            if self._verify_vote_signature(vote):
                total_power += vote["voting_power"]

        return total_power >= required_power

    def _verify_vote_signature(self, vote: dict) -> bool:
        """Verify vote signature (placeholder)."""
        # In production, would verify actual signature
        return True

    async def commit_block(self, block: Block):
        """
        Commit block to chain.

        Args:
            block: Block to commit
        """
        # Add to chain
        self.chain.append(block)

        # Execute transactions
        for tx in block.transactions:
            await self._execute_transaction(tx)

        # Remove transactions from mempool
        self.mempool = [tx for tx in self.mempool if tx not in block.transactions]

        # Update height
        self.current_height = block.height

        # Award block rewards
        self._award_block_rewards(block.proposer)

        # Audit log
        audit_logger.log_event(
            event_type="block_commit",
            actor=self.node_id,
            action="commit_block",
            resource=str(block.height),
            metadata={
                "block_hash": block.calculate_hash(),
                "tx_count": len(block.transactions),
            },
        )

        logger.info(f"Committed block at height {block.height}", extra={"privacy_safe": True})

    async def _execute_transaction(self, tx: dict):
        """Execute transaction and update state."""
        _ = tx.get("type")

        if tx_type == "proof_record":
            await self._record_proof(tx["data"])
        elif tx_type == "credit_transfer":
            await self._transfer_credits(tx["from"], tx["to"], tx["amount"])
        elif tx_type == "stake_update":
            await self._update_stake(tx["node"], tx["amount"])

    async def _record_proof(self, proof_data: dict):
        """Record proof in state."""
        proof_key = proof_data["proof_key"]
        self.state["proof:{proof_key}"] = proof_data

    async def _transfer_credits(self, from_addr: str, to_addr: str, amount: int):
        """Transfer credits between addresses."""
        _ = "credits:{from_addr}"
        _ = "credits:{to_addr}"

        # Check balance
        from_balance = self.state.get(from_key, 0)
        if from_balance >= amount:
            self.state[from_key] = from_balance - amount
            self.state[to_key] = self.state.get(to_key, 0) + amount

    async def _update_stake(self, node_id: str, amount: int):
        """Update node stake."""
        stake_key = "stake:{node_id}"
        self.state[stake_key] = amount

    def _award_block_rewards(self, proposer: str):
        """Award block rewards to proposer."""
        if proposer == self.node_id:
            rewards = self._calculate_block_rewards()
            self.credits += rewards

            # Update state
            credit_key = "credits:{self.node_id}"
            self.state[credit_key] = self.state.get(credit_key, 0) + rewards

            logger.info(
                "Awarded {rewards} credits for block production",
                extra={"privacy_safe": True},
            )

    async def handle_audit_challenge(self, challenger: str, target: str, epoch: int) -> dict:
        """
        Handle audit challenge.

        Args:
            challenger: Node issuing challenge
            target: Node being challenged
            epoch: Challenge epoch

        Returns:
            Challenge result
        """
        # Verify challenger has standing
        challenger_stake = self.state.get("stake:{challenger}", 0)
        if challenger_stake < 100:  # Minimum stake to challenge
            return {"success": False, "reason": "Insufficient stake"}

        # Get target audit data
        _ = await self._get_audit_data(target, epoch)

        # Verify audit
        _ = await self._verify_audit_data(audit_data)

        if not is_valid:
            # Slash target stake
            await self._slash_stake(target, 0.25)  # 25% slash

            # Reward challenger
            reward = int(self.state.get("stake:{target}", 0) * 0.1)
            await self._transfer_credits("system", challenger, reward)

        _ = {
            "success": True,
            "valid": is_valid,
            "epoch": epoch,
            "timestamp": time.time(),
        }

        return result

    async def _get_audit_data(self, node_id: str, epoch: int) -> dict:
        """Get audit data for node."""
        # In production, would retrieve actual audit data
        return {"node_id": node_id, "epoch": epoch, "proofs": [], "uptime": 0.99}

    async def _verify_audit_data(self, audit_data: dict) -> bool:
        """Verify audit data."""
        # Verify uptime, proof validity, etc.
        return audit_data.get("uptime", 0) > 0.95

    async def _slash_stake(self, node_id: str, percentage: float):
        """Slash node stake."""
        stake_key = "stake:{node_id}"
        _ = self.state.get(stake_key, 0)

        slash_amount = int(current_stake * percentage)
        _ = current_stake - slash_amount

        self.state[stake_key] = new_stake

        # Log slashing event
        audit_logger.log_event(
            event_type="stake_slash",
            actor="system",
            action="slash_stake",
            resource=node_id,
            metadata={"slash_amount": slash_amount, "new_stake": new_stake},
        )

    def get_node_info(self) -> NodeInfo:
        """Get node information."""
        return NodeInfo(
            node_id=self.node_id,
            address="node_{self.node_id}",
            node_class=self.node_class,
            is_trusted_signatory=self.is_trusted_signatory,
            voting_power=self.voting_power,
            stake_amount=self.stake_amount,
            credits=self.credits,
            last_seen=time.time(),
        )

    def get_chain_info(self) -> dict:
        """Get blockchain information."""
        return {
            "height": self.current_height,
            "chain_length": len(self.chain),
            "mempool_size": len(self.mempool),
            "voting_power": self.voting_power,
            "node_type": "{self.node_class.name} {'TS' if self.is_trusted_signatory else 'NS'}",
            "credits": self.credits,
            "stake": self.stake_amount,
            "peers": len(self.peers),
        }

    def submit_transaction(self, tx: dict) -> str:
        """
        Submit transaction to mempool.

        Args:
            tx: Transaction to submit

        Returns:
            Transaction hash
        """
        # Add timestamp
        tx["timestamp"] = time.time()

        # Calculate hash
        tx_str = json.dumps(tx, sort_keys=True)
        tx_hash = hashlib.sha256(tx_str.encode()).hexdigest()
        tx["hash"] = tx_hash

        # Add to mempool
        self.mempool.append(tx)

        logger.info(f"Transaction {tx_hash[:8]} added to mempool", extra={"privacy_safe": True})

        return tx_hash


# Example usage
if __name__ == "__main__":
    # Example 1: Light node with TS status
    _ = BlockchainNode(
        node_id="gp_clinic_001", node_class=NodeClass.LIGHT, is_trusted_signatory=False
    )

    logger.info("Light Node Initial Voting Power: {light_ts_node.voting_power}")
    logger.info("Light Node Block Rewards: {light_ts_node._calculate_block_rewards()}")

    # Verify HIPAA credentials
    asyncio.run(
        light_ts_node.verify_hipaa_credentials(
            npi="1234567890",
            baa_hash="baa_hash_example",
            risk_analysis_hash="risk_hash_example",
            hsm_serial="HSM123456",
        )
    )

    logger.info("Light TS Node Voting Power: {light_ts_node.voting_power}")
    logger.info("Light TS Node Block Rewards: {light_ts_node._calculate_block_rewards()}")

    # Example 2: Full node without TS
    _ = BlockchainNode(
        node_id="university_node_001",
        node_class=NodeClass.FULL,
        is_trusted_signatory=False,
    )

    logger.info("\nFull Node Voting Power: {full_node.voting_power}")
    logger.info("Full Node Block Rewards: {full_node._calculate_block_rewards()}")

    # Example 3: Archive node
    _ = BlockchainNode(
        node_id="research_archive_001",
        node_class=NodeClass.ARCHIVE,
        is_trusted_signatory=False,
    )

    logger.info("\nArchive Node Voting Power: {archive_node.voting_power}")
    logger.info("Archive Node Block Rewards: {archive_node._calculate_block_rewards()}")

    # Add peers
    light_ts_node.add_peer(full_node.get_node_info())
    light_ts_node.add_peer(archive_node.get_node_info())

    # Check BFT safety
    logger.info("\nBFT Safety Check: {light_ts_node.check_bft_safety()}")

    # Submit transaction
    _ = light_ts_node.submit_transaction(
        {
            "type": "proof_record",
            "from": light_ts_node.node_id,
            "data": {
                "proof_key": "proof_123",
                "circuit_type": "diabetes_risk_alert",
                "public_inputs": {"glucose_threshold": 126, "risk_threshold": 0.75},
            },
            "signature": "mock_signature",
        }
    )

    logger.info("\nSubmitted transaction: {tx_hash[:16]}...")

    # Get chain info
    logger.info("\nChain Info: {json.dumps(light_ts_node.get_chain_info(), indent=2)}")

    # Calculate network statistics
    logger.info("\n=== Network Statistics ===")
    logger.info("Light TS (GP clinic): w=11, credits/block=3")
    logger.info("Full non-TS (University): w=4, credits/block=4")
    logger.info("Archive non-TS (Research): w=8, credits/block=8")
    logger.info("\nTotal Network Voting Power: {11 + 4 + 8} = 23")
    logger.info("Honest Power Needed for BFT: >11")
