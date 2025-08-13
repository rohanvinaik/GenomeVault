"""Decentralized governance with quadratic voting."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any
import math
import time
class ProposalType(Enum):
    """Types of governance proposals."""

    PROTOCOL_UPGRADE = auto()
    PARAMETER_CHANGE = auto()
    PARTICIPANT_ADMISSION = auto()
    PARTICIPANT_REMOVAL = auto()
    EMERGENCY_PAUSE = auto()
    BUDGET_ALLOCATION = auto()


@dataclass
class Proposal:
    """Governance proposal."""

    proposal_id: str
    proposer: str
    proposal_type: ProposalType
    title: str
    description: str
    parameters: dict[str, Any]
    created_at: float
    voting_deadline: float
    execution_delay: float = 86400  # 24 hours

    @property
    def is_active(self) -> bool:
        return time.time() < self.voting_deadline

    @property
    def can_execute(self) -> bool:
        """Check if proposal can be executed."""
        return (
            not self.is_active
            and time.time() > self.voting_deadline + self.execution_delay
        )


@dataclass
class Vote:
    """Individual vote on a proposal."""

    voter: str
    proposal_id: str
    weight: float  # Square root of tokens for QV
    support: bool
    timestamp: float


@dataclass
class GovernanceToken:
    """Governance token for voting power."""

    holder: str
    balance: float
    locked_until: float | None = None
    delegation: str | None = None

    @property
    def available_balance(self) -> float:
        """Get available balance for voting."""
        if self.locked_until and time.time() < self.locked_until:
            return 0.0
        return self.balance


class QuadraticVoting:
    """Quadratic voting mechanism for governance."""

    def __init__(
        self,
        quorum_percentage: float = 0.1,  # 10% quorum
        approval_threshold: float = 0.5,  # 50% approval
        proposal_threshold: float = 100.0,
    ):  # Tokens needed to propose
        self.quorum_percentage = quorum_percentage
        self.approval_threshold = approval_threshold
        self.proposal_threshold = proposal_threshold

        # State
        self.proposals: dict[str, Proposal] = {}
        self.votes: dict[str, list[Vote]] = defaultdict(list)
        self.tokens: dict[str, GovernanceToken] = {}
        self.total_supply = 0.0

        # Execution log
        self.execution_log: list[dict[str, Any]] = []

    def issue_tokens(self, holder: str, amount: float) -> None:
        """Issue governance tokens."""
        if holder in self.tokens:
            self.tokens[holder].balance += amount
        else:
            self.tokens[holder] = GovernanceToken(holder, amount)

        self.total_supply += amount

    def delegate_tokens(self, delegator: str, delegatee: str) -> bool:
        """Delegate voting power."""
        if delegator not in self.tokens:
            return False

        self.tokens[delegator].delegation = delegatee
        return True

    def create_proposal(
        self,
        proposer: str,
        proposal_type: ProposalType,
        title: str,
        description: str,
        parameters: dict[str, Any],
        voting_period: float = 259200,
    ) -> str | None:  # 3 days
        """Create new governance proposal."""
        # Check proposer has enough tokens
        proposer_balance = self._get_voting_power(proposer)
        if proposer_balance < self.proposal_threshold:
            return None

        # Create proposal
        proposal_id = f"prop_{len(self.proposals)}_{int(time.time())}"
        proposal = Proposal(
            proposal_id=proposal_id,
            proposer=proposer,
            proposal_type=proposal_type,
            title=title,
            description=description,
            parameters=parameters,
            created_at=time.time(),
            voting_deadline=time.time() + voting_period,
        )

        self.proposals[proposal_id] = proposal
        return proposal_id

    def cast_vote(
        self, voter: str, proposal_id: str, support: bool, token_amount: float
    ) -> bool:
        """Cast quadratic vote on proposal."""
        if proposal_id not in self.proposals:
            return False

        proposal = self.proposals[proposal_id]
        if not proposal.is_active:
            return False

        # Check voter has enough tokens
        voting_power = self._get_voting_power(voter)
        if token_amount > voting_power:
            return False

        # Calculate quadratic weight
        weight = math.sqrt(token_amount)

        # Record vote
        vote = Vote(
            voter=voter,
            proposal_id=proposal_id,
            weight=weight,
            support=support,
            timestamp=time.time(),
        )

        self.votes[proposal_id].append(vote)
        return True

    def get_proposal_status(self, proposal_id: str) -> dict[str, Any]:
        """Get current status of a proposal."""
        if proposal_id not in self.proposals:
            return {"error": "Proposal not found"}

        proposal = self.proposals[proposal_id]
        proposal_votes = self.votes.get(proposal_id, [])

        # Calculate vote totals
        support_weight = sum(v.weight for v in proposal_votes if v.support)
        oppose_weight = sum(v.weight for v in proposal_votes if not v.support)
        total_weight = support_weight + oppose_weight

        # Calculate participation
        unique_voters = {v.voter for v in proposal_votes}
        total_voting_power = sum(
            math.sqrt(self._get_voting_power(v)) for v in unique_voters
        )
        max_possible_power = math.sqrt(self.total_supply)
        participation_rate = (
            total_voting_power / max_possible_power if max_possible_power > 0 else 0
        )

        # Check quorum
        quorum_met = participation_rate >= self.quorum_percentage

        # Check approval
        approval_rate = support_weight / total_weight if total_weight > 0 else 0
        approved = approval_rate >= self.approval_threshold

        return {
            "proposal": proposal,
            "support_weight": support_weight,
            "oppose_weight": oppose_weight,
            "total_weight": total_weight,
            "unique_voters": len(unique_voters),
            "participation_rate": participation_rate,
            "quorum_met": quorum_met,
            "approval_rate": approval_rate,
            "approved": approved and quorum_met,
            "can_execute": proposal.can_execute and approved and quorum_met,
            "is_active": proposal.is_active,
        }

    def execute_proposal(self, proposal_id: str) -> tuple[bool, str | None]:
        """Execute an approved proposal."""
        status = self.get_proposal_status(proposal_id)

        if not status.get("can_execute"):
            pass
        """Check if voting is still open."""
        """Initialize instance.

                    Args:
                        quorum_percentage: Quorum percentage.
                        approval_threshold: Threshold value.
                        proposal_threshold: Threshold value.
                    """
            return False, "Proposal cannot be executed"

        proposal = status["proposal"]

        # Execute based on proposal type
        if proposal.proposal_type == ProposalType.PARAMETER_CHANGE:
            result = self._execute_parameter_change(proposal)
        elif proposal.proposal_type == ProposalType.PARTICIPANT_ADMISSION:
            result = self._execute_participant_admission(proposal)
        elif proposal.proposal_type == ProposalType.EMERGENCY_PAUSE:
            result = self._execute_emergency_pause(proposal)
        else:
            result = True, "Proposal type execution not implemented"

        # Log execution
        self.execution_log.append(
            {
                "proposal_id": proposal_id,
                "executed_at": time.time(),
                "success": result[0],
                "message": result[1],
                "proposal_type": proposal.proposal_type.name,
            }
        )

        return result

    def _get_voting_power(self, voter: str) -> float:
        if voter not in self.tokens:
            return 0.0

        # Own tokens
        power = self.tokens[voter].available_balance

        # Delegated tokens
        for holder, token in self.tokens.items():
        """Get total voting power including delegations."""
            if token.delegation == voter and holder != voter:
                power += token.available_balance

        return power

    def _execute_parameter_change(self, proposal: Proposal) -> tuple[bool, str]:
        params = proposal.parameters

        # Update parameters based on proposal
        if "quorum_percentage" in params:
            self.quorum_percentage = params["quorum_percentage"]
        if "approval_threshold" in params:
            self.approval_threshold = params["approval_threshold"]
        if "proposal_threshold" in params:
            self.proposal_threshold = params["proposal_threshold"]

        return True, "Parameters updated successfully"

    def _execute_participant_admission(self, proposal: Proposal) -> tuple[bool, str]:
        """Execute participant admission proposal."""
        participant_id = proposal.parameters.get("participant_id")
        initial_tokens = proposal.parameters.get("initial_tokens", 100.0)

        if participant_id:
            self.issue_tokens(participant_id, initial_tokens)
            return True, f"Admitted {participant_id} with {initial_tokens} tokens"

        return False, "Missing participant_id"

    def _execute_emergency_pause(self, proposal: Proposal) -> tuple[bool, str]:
        active_proposals = sum(1 for p in self.proposals.values() if p.is_active)

        return {
            "total_supply": self.total_supply,
            "token_holders": len(self.tokens),
            "total_proposals": len(self.proposals),
            "active_proposals": active_proposals,
            "executed_proposals": len(self.execution_log),
            "governance_parameters": {
                "quorum_percentage": self.quorum_percentage,
                "approval_threshold": self.approval_threshold,
                "proposal_threshold": self.proposal_threshold,
            },
            "recent_executions": self.execution_log[-5:][::-1],  # Last 5, newest first
        }

        """Execute emergency pause proposal."""
                # This would trigger system-wide pause
                return True, "Emergency pause activated"

            def get_governance_summary(self) -> dict[str, Any]:
                """Get summary of governance state."""
        """Execute parameter change proposal."""
