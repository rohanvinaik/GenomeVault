"""
Weighted voting consensus implementation with dual-axis node model.

Based on Section 2.3.1 and 4.1.3, implements a Byzantine Fault Tolerant consensus
with node weights determined by resource contribution and trusted signatory status.

Key Features:
- Dual-axis weighting: resource class (c) + signatory status (s)
- BFT safety with H > 2F/3 requirement
- Credit rewards and slashing mechanisms
- HIPAA fast-track verification for healthcare entities
"""

from __future__ import annotations

from enum import IntEnum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
import hashlib
import random
from collections import defaultdict

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class ResourceClass(IntEnum):
    """
    Resource contribution classes with associated weights.
    Higher resource contribution = higher voting weight.
    """

    LIGHT = 1  # Minimal resources, basic validation
    FULL = 4  # Full node, complete blockchain state
    ARCHIVE = 8  # Archive node, historical data + analytics


class SignatoryStatus(IntEnum):
    """
    Trusted signatory status with associated weights.
    HIPAA-compliant entities get trusted status.
    """

    NON_SIGNER = 0  # Regular node, no special trust
    TRUSTED_SIGNATORY = 10  # HIPAA-verified entity


@dataclass
class DualAxisNode:
    """
    Node in the dual-axis voting system.

    Total voting weight: w = c + s
    where c = resource class weight, s = signatory status weight
    """

    node_id: str
    resource_class: ResourceClass = ResourceClass.LIGHT
    signatory_status: SignatoryStatus = SignatoryStatus.NON_SIGNER
    stake: float = 100.0  # Initial stake
    credits: float = 0.0  # Earned credits

    # HIPAA verification
    npi_number: Optional[str] = None  # National Provider Identifier
    hipaa_verified: bool = False
    verification_date: Optional[datetime] = None

    # Performance metrics
    blocks_validated: int = 0
    failed_audits: int = 0
    honesty_probability: float = 0.95  # Default q=0.95, HIPAA q=0.98

    # Network participation
    is_active: bool = True
    last_seen: datetime = field(default_factory=datetime.now)

    @property
    def voting_weight(self) -> int:
        """Calculate total voting weight: w = c + s"""
        if not self.is_active:
            return 0
        return self.resource_class.value + self.signatory_status.value

    @property
    def weight_components(self) -> Tuple[int, int]:
        """Return (resource_weight, signatory_weight) components."""
        return (self.resource_class.value, self.signatory_status.value)

    def slash_stake(self, percentage: float = 0.25):
        """
        Slash stake on failed audit or misbehavior.
        Default: 25% slashing as specified.
        """
        slash_amount = self.stake * percentage
        self.stake = max(0, self.stake - slash_amount)
        self.failed_audits += 1

        logger.warning(
            f"Node {self.node_id} slashed {slash_amount:.2f} ({percentage*100:.0f}%), "
            f"remaining stake: {self.stake:.2f}"
        )

        # Deactivate if stake too low
        if self.stake <= 10.0:  # Changed from < to <= to match test expectation
            self.is_active = False
            logger.warning(f"Node {self.node_id} deactivated due to low stake")

    def award_credits(self, base_reward: float = 1.0):
        """
        Award credits for block validation.
        Reward = c + (s>0)×2 credits
        """
        resource_reward = self.resource_class.value * base_reward
        signatory_bonus = 2.0 * base_reward if self.signatory_status > 0 else 0

        total_reward = resource_reward + signatory_bonus
        self.credits += total_reward
        self.blocks_validated += 1

        logger.debug(
            f"Node {self.node_id} awarded {total_reward:.2f} credits "
            f"(resource: {resource_reward:.2f}, signatory: {signatory_bonus:.2f})"
        )

        return total_reward

    def __str__(self) -> str:
        """String representation showing key attributes."""
        status = "TS" if self.signatory_status > 0 else "NS"
        return (
            f"Node({self.node_id}, {self.resource_class.name}_{status}, "
            f"w={self.voting_weight}, stake={self.stake:.0f})"
        )


class HIPAAVerifier:
    """
    HIPAA fast-track verification for healthcare entities.
    Verifies NPI numbers against CMS registry (simulated).
    """

    def __init__(self):
        """Initialize HIPAA verifier with simulated CMS registry."""
        # Simulated NPI registry (in production, would query real CMS database)
        self.cms_registry = {
            "1234567890": {"name": "Memorial Hospital", "type": "Hospital"},
            "2345678901": {"name": "City Medical Center", "type": "Hospital"},
            "3456789012": {"name": "Regional Health System", "type": "Health System"},
            "4567890123": {"name": "University Medical", "type": "Academic Medical Center"},
            "5678901234": {"name": "Community Clinic", "type": "FQHC"},
        }

        logger.info(f"HIPAA Verifier initialized with {len(self.cms_registry)} NPIs")

    def verify_npi(self, npi: str) -> bool:
        """
        Verify NPI in CMS registry.

        Args:
            npi: National Provider Identifier (10 digits)

        Returns:
            True if NPI is valid and active
        """
        # Basic format validation
        if not npi or len(npi) != 10 or not npi.isdigit():
            return False

        # Check registry (simulated)
        return npi in self.cms_registry

    def get_entity_info(self, npi: str) -> Optional[Dict]:
        """Get entity information for verified NPI."""
        return self.cms_registry.get(npi)

    def fast_track_verification(self, node: DualAxisNode) -> bool:
        """
        Fast-track HIPAA verification for a node.
        Sets s=10 and q=0.98 for verified entities.

        Args:
            node: Node requesting verification

        Returns:
            True if verification successful
        """
        if not node.npi_number:
            logger.debug(f"Node {node.node_id} has no NPI for verification")
            return False

        if self.verify_npi(node.npi_number):
            # Set trusted signatory status
            node.signatory_status = SignatoryStatus.TRUSTED_SIGNATORY
            node.hipaa_verified = True
            node.verification_date = datetime.now()

            # Higher honesty probability for HIPAA entities
            node.honesty_probability = 0.98  # vs 0.95 for regular nodes

            entity_info = self.get_entity_info(node.npi_number)
            logger.info(
                f"✅ HIPAA fast-track verified: {node.node_id} "
                f"({entity_info['name']}, NPI: {node.npi_number})"
            )

            return True
        else:
            logger.warning(f"❌ NPI verification failed for {node.node_id}: {node.npi_number}")
            return False


@dataclass
class VoteMessage:
    """Message representing a vote in consensus."""

    node_id: str
    block_hash: str
    round: int
    timestamp: datetime
    signature: str  # Simplified signature


class BFTConsensus:
    """
    Byzantine Fault Tolerant consensus with weighted voting.

    Safety requirement: H > 2F/3 where:
    - H = total honest weight
    - F = total faulty weight
    - Total weight W = H + F
    """

    def __init__(self, nodes: List[DualAxisNode], byzantine_ratio: float = 0.33):
        """
        Initialize BFT consensus.

        Args:
            nodes: List of participating nodes
            byzantine_ratio: Maximum fraction of Byzantine weight (default 1/3)
        """
        self.nodes = {node.node_id: node for node in nodes}
        self.byzantine_ratio = byzantine_ratio

        # Voting state
        self.current_round = 0
        self.votes: Dict[int, Dict[str, List[VoteMessage]]] = defaultdict(lambda: defaultdict(list))

        # Calculate weight distribution
        self._calculate_weights()

        logger.info(
            f"BFT Consensus initialized with {len(nodes)} nodes, "
            f"total weight: {self.total_weight}"
        )

    def _calculate_weights(self):
        """Calculate total weights and safety thresholds."""
        self.total_weight = sum(node.voting_weight for node in self.nodes.values())
        self.byzantine_weight = int(self.total_weight * self.byzantine_ratio)
        self.honest_weight = self.total_weight - self.byzantine_weight

        # Safety threshold: need > 2/3 of total weight
        self.threshold = (2 * self.total_weight) // 3 + 1

        # Verify safety condition: Byzantine must be < 1/3 of total
        max_byzantine = self.total_weight // 3
        self.is_safe = self.byzantine_weight < max_byzantine

        logger.info(
            f"Weight distribution: Total={self.total_weight}, "
            f"Honest={self.honest_weight}, Byzantine={self.byzantine_weight}, "
            f"Threshold={self.threshold}, Safe={self.is_safe}"
        )

    def verify_safety(self) -> Tuple[bool, str]:
        """
        Verify H > 2F/3 safety condition.

        Returns:
            (is_safe, explanation)
        """
        # The BFT safety condition is: H > 2F/3
        # But we need to check if H > 2W/3 for consensus (where W is total weight)
        # Safety fails when F >= W/3 (Byzantine has 1/3 or more of total weight)

        # Check if Byzantine weight is less than 1/3 of total
        max_byzantine = self.total_weight // 3

        if self.byzantine_weight < max_byzantine:
            return True, f"Safety verified: F={self.byzantine_weight} < W/3={max_byzantine}"
        else:
            return False, f"Safety violated: F={self.byzantine_weight} >= W/3={max_byzantine}"

    def calculate_minimum_honest_weight(self) -> int:
        """
        Calculate minimum honest weight required for safety.

        Returns:
            Minimum honest weight needed
        """
        # For BFT consensus, we need more than 2/3 of total weight to be honest
        # This means Byzantine must be less than 1/3 of total weight
        # So minimum honest weight = total - (total/3) = 2*total/3 + 1

        min_honest = (2 * self.total_weight) // 3 + 1
        return min_honest

    def submit_vote(self, node_id: str, block_hash: str, signature: str = "mock_sig") -> bool:
        """
        Submit a weighted vote from a node.

        Args:
            node_id: ID of voting node
            block_hash: Hash of block being voted on
            signature: Vote signature (simplified)

        Returns:
            True if vote accepted
        """
        if node_id not in self.nodes:
            logger.warning(f"Unknown node {node_id} attempted to vote")
            return False

        node = self.nodes[node_id]
        if not node.is_active:
            logger.warning(f"Inactive node {node_id} attempted to vote")
            return False

        # Create vote message
        vote = VoteMessage(
            node_id=node_id,
            block_hash=block_hash,
            round=self.current_round,
            timestamp=datetime.now(),
            signature=signature,
        )

        # Record vote
        self.votes[self.current_round][block_hash].append(vote)

        logger.debug(f"Vote recorded: {node_id} (w={node.voting_weight}) -> {block_hash[:8]}...")

        return True

    def check_consensus(self, round_num: Optional[int] = None) -> Optional[str]:
        """
        Check if consensus reached for a round.

        Args:
            round_num: Round to check (default: current)

        Returns:
            Winning block hash if consensus reached, None otherwise
        """
        round_num = round_num or self.current_round

        if round_num not in self.votes:
            return None

        # Calculate weighted votes for each block
        block_weights = {}
        for block_hash, votes in self.votes[round_num].items():
            total_weight = 0
            voters = set()

            for vote in votes:
                # Prevent double voting
                if vote.node_id not in voters:
                    node = self.nodes[vote.node_id]
                    total_weight += node.voting_weight
                    voters.add(vote.node_id)

            block_weights[block_hash] = total_weight

        # Check if any block has enough weight
        for block_hash, weight in block_weights.items():
            if weight >= self.threshold:
                logger.info(
                    f"✅ Consensus reached for block {block_hash[:8]}... "
                    f"with weight {weight}/{self.threshold}"
                )
                return block_hash

        # Log current status
        max_weight = max(block_weights.values()) if block_weights else 0
        logger.debug(f"No consensus yet. Max weight: {max_weight}/{self.threshold}")

        return None

    def finalize_round(self, winning_block: str):
        """
        Finalize a consensus round and distribute rewards.

        Args:
            winning_block: Hash of winning block
        """
        round_votes = self.votes.get(self.current_round, {})
        winning_votes = round_votes.get(winning_block, [])

        # Award credits to nodes that voted for winning block
        rewarded_nodes = set()
        for vote in winning_votes:
            if vote.node_id not in rewarded_nodes:
                node = self.nodes[vote.node_id]
                reward = node.award_credits()
                rewarded_nodes.add(vote.node_id)

        logger.info(
            f"Round {self.current_round} finalized. "
            f"{len(rewarded_nodes)} nodes rewarded for block {winning_block[:8]}..."
        )

        # Advance round
        self.current_round += 1

    def simulate_byzantine_nodes(self, byzantine_fraction: float = 0.2) -> Set[str]:
        """
        Simulate Byzantine nodes for testing.

        Args:
            byzantine_fraction: Fraction of weight to make Byzantine

        Returns:
            Set of Byzantine node IDs
        """
        byzantine_nodes = set()
        target_weight = int(self.total_weight * byzantine_fraction)
        current_weight = 0

        # Sort nodes by weight (prefer making high-weight nodes Byzantine for impact)
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: n.voting_weight, reverse=True)

        for node in sorted_nodes:
            if current_weight >= target_weight:
                break
            byzantine_nodes.add(node.node_id)
            current_weight += node.voting_weight

        logger.info(
            f"Simulated {len(byzantine_nodes)} Byzantine nodes "
            f"with total weight {current_weight}/{self.total_weight}"
        )

        return byzantine_nodes

    def get_weight_distribution(self) -> Dict[str, int]:
        """Get weight distribution statistics."""
        distribution = {
            "LIGHT_NS": 0,  # Light Non-Signer
            "LIGHT_TS": 0,  # Light Trusted Signatory
            "FULL_NS": 0,  # Full Non-Signer
            "FULL_TS": 0,  # Full Trusted Signatory
            "ARCHIVE_NS": 0,  # Archive Non-Signer
            "ARCHIVE_TS": 0,  # Archive Trusted Signatory
        }

        for node in self.nodes.values():
            if not node.is_active:
                continue

            # Build key from resource class name (uppercase) and signatory status
            key = f"{node.resource_class.name.upper()}_{('TS' if node.signatory_status.value > 0 else 'NS')}"
            if key in distribution:
                distribution[key] += node.voting_weight

        return distribution


def simulate_network(
    num_nodes: int = 20,
    hipaa_fraction: float = 0.3,
    resource_distribution: Dict[ResourceClass, float] = None,
) -> List[DualAxisNode]:
    """
    Simulate a network of mixed node types.

    Args:
        num_nodes: Total number of nodes
        hipaa_fraction: Fraction that are HIPAA entities
        resource_distribution: Distribution of resource classes

    Returns:
        List of simulated nodes
    """
    if resource_distribution is None:
        resource_distribution = {
            ResourceClass.LIGHT: 0.5,
            ResourceClass.FULL: 0.35,
            ResourceClass.ARCHIVE: 0.15,
        }

    nodes = []
    verifier = HIPAAVerifier()

    # Available NPIs for HIPAA nodes
    available_npis = list(verifier.cms_registry.keys())

    for i in range(num_nodes):
        # Determine resource class
        rand = random.random()
        cumulative = 0.0
        resource_class = ResourceClass.LIGHT

        for rc, prob in resource_distribution.items():
            cumulative += prob
            if rand < cumulative:
                resource_class = rc
                break

        # Create node
        node = DualAxisNode(
            node_id=f"node_{i:03d}", resource_class=resource_class, stake=random.uniform(100, 1000)
        )

        # Assign HIPAA status to some nodes
        if random.random() < hipaa_fraction and available_npis:
            node.npi_number = available_npis.pop(0)
            verifier.fast_track_verification(node)

        nodes.append(node)

    return nodes


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  GENOMEVAULT DUAL-AXIS WEIGHTED VOTING CONSENSUS")
    print("=" * 70)

    # Simulate network
    print("\n1. Simulating network...")
    nodes = simulate_network(num_nodes=20, hipaa_fraction=0.25)

    # Show node distribution
    print("\n2. Node distribution:")
    for node in nodes[:5]:  # Show first 5
        print(f"  {node}")
    print(f"  ... and {len(nodes)-5} more nodes")

    # Initialize consensus
    print("\n3. Initializing BFT consensus...")
    consensus = BFTConsensus(nodes, byzantine_ratio=0.3)

    # Verify safety
    is_safe, explanation = consensus.verify_safety()
    print(f"\n4. Safety verification: {explanation}")

    # Show weight distribution
    print("\n5. Weight distribution by type:")
    distribution = consensus.get_weight_distribution()
    for node_type, weight in distribution.items():
        if weight > 0:
            print(f"  {node_type}: {weight}")

    # Simulate voting
    print("\n6. Simulating consensus round...")

    # Create two competing blocks
    block_a = hashlib.sha256(b"block_a").hexdigest()
    block_b = hashlib.sha256(b"block_b").hexdigest()

    # Simulate Byzantine nodes (20% of weight)
    byzantine_nodes = consensus.simulate_byzantine_nodes(0.2)

    # Nodes vote
    for node_id, node in consensus.nodes.items():
        if node_id in byzantine_nodes:
            # Byzantine nodes vote for block B
            consensus.submit_vote(node_id, block_b)
        else:
            # Honest nodes vote for block A
            consensus.submit_vote(node_id, block_a)

    # Check consensus
    winning_block = consensus.check_consensus()
    if winning_block:
        print(f"\n✅ Consensus achieved on block: {winning_block[:8]}...")

        # Finalize and distribute rewards
        consensus.finalize_round(winning_block)

        # Show some rewards
        print("\n7. Sample rewards (first 3 nodes):")
        for node in list(nodes)[:3]:
            if node.credits > 0:
                print(f"  {node.node_id}: {node.credits:.1f} credits")
    else:
        print("\n❌ No consensus reached")

    # Test slashing
    print("\n8. Testing slashing mechanism...")
    test_node = nodes[0]
    print(f"  Before: {test_node.node_id} stake = {test_node.stake:.0f}")
    test_node.slash_stake(0.25)
    print(f"  After 25% slash: stake = {test_node.stake:.0f}")

    print("\n" + "=" * 70)
    print("  Dual-axis weighted voting system ready!")
    print("=" * 70)
