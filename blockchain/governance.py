"""
GenomeVault Governance System

Implements the DAO governance framework with:
- Multi-stakeholder committees
- Quadratic voting mechanisms
- Proposal management
- HIPAA oracle for fast-track verification
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import json
from abc import ABC, abstractmethod

from ..utils import get_logger
from ..utils.logging import audit_logger
from ..core.constants import NodeType, SignatoryWeight

logger = get_logger(__name__)


class ProposalType(Enum):
    """Types of governance proposals"""
    PROTOCOL_UPDATE = "protocol_update"
    PARAMETER_CHANGE = "parameter_change"
    REFERENCE_UPDATE = "reference_update"
    ALGORITHM_CERTIFICATION = "algorithm_certification"
    TREASURY_ALLOCATION = "treasury_allocation"
    EMERGENCY_ACTION = "emergency_action"
    COMMITTEE_ELECTION = "committee_election"


class ProposalStatus(Enum):
    """Proposal lifecycle states"""
    DRAFT = "draft"
    ACTIVE = "active"
    PASSED = "passed"
    REJECTED = "rejected"
    EXECUTED = "executed"
    CANCELLED = "cancelled"


class CommitteeType(Enum):
    """Governance committee types"""
    SCIENTIFIC_ADVISORY = "scientific_advisory"
    ETHICS = "ethics"
    SECURITY = "security"
    USER_REPRESENTATIVES = "user_representatives"


@dataclass
class VoteRecord:
    """Individual vote record"""
    voter_id: str
    voting_power: int
    vote_weight: float  # For quadratic voting
    choice: str  # yes, no, abstain
    timestamp: datetime
    delegate_from: Optional[str] = None  # If delegated vote


@dataclass
class Proposal:
    """Governance proposal"""
    proposal_id: str
    proposal_type: ProposalType
    title: str
    description: str
    proposer: str
    created_at: datetime
    voting_start: datetime
    voting_end: datetime
    execution_delay: timedelta
    status: ProposalStatus
    
    # Voting data
    votes: List[VoteRecord] = field(default_factory=list)
    yes_votes: float = 0.0
    no_votes: float = 0.0
    abstain_votes: float = 0.0
    
    # Requirements
    quorum_required: float = 0.1  # 10% of voting power
    approval_threshold: float = 0.51  # 51% for normal, 67% for protocol
    
    # Execution
    execution_data: Optional[Dict[str, Any]] = None
    execution_timestamp: Optional[datetime] = None
    
    def add_vote(self, vote: VoteRecord):
        """Add a vote to the proposal"""
        self.votes.append(vote)
        
        if vote.choice == "yes":
            self.yes_votes += vote.vote_weight
        elif vote.choice == "no":
            self.no_votes += vote.vote_weight
        else:
            self.abstain_votes += vote.vote_weight
    
    def get_total_votes(self) -> float:
        """Get total vote weight"""
        return self.yes_votes + self.no_votes + self.abstain_votes
    
    def has_quorum(self, total_voting_power: float) -> bool:
        """Check if proposal has reached quorum"""
        return self.get_total_votes() >= (total_voting_power * self.quorum_required)
    
    def get_approval_rate(self) -> float:
        """Get approval rate (yes / (yes + no))"""
        total_deciding = self.yes_votes + self.no_votes
        if total_deciding == 0:
            return 0.0
        return self.yes_votes / total_deciding
    
    def has_passed(self) -> bool:
        """Check if proposal has passed"""
        return self.get_approval_rate() >= self.approval_threshold


@dataclass
class Committee:
    """Governance committee"""
    committee_type: CommitteeType
    members: Set[str]
    chair: Optional[str]
    term_end: datetime
    responsibilities: List[str]
    voting_weight_multiplier: float = 1.0
    
    def is_member(self, address: str) -> bool:
        """Check if address is committee member"""
        return address in self.members
    
    def add_member(self, address: str):
        """Add committee member"""
        self.members.add(address)
        logger.info(f"Added {address} to {self.committee_type.value} committee")
    
    def remove_member(self, address: str):
        """Remove committee member"""
        self.members.discard(address)
        logger.info(f"Removed {address} from {self.committee_type.value} committee")


class VotingMechanism(ABC):
    """Abstract base class for voting mechanisms"""
    
    @abstractmethod
    def calculate_vote_weight(self, voting_power: int, choice: str) -> float:
        """Calculate vote weight based on mechanism"""
        pass
    
    @abstractmethod
    def get_cost(self, voting_power: int, choice: str) -> int:
        """Get cost of vote (for quadratic voting)"""
        pass


class SimpleVoting(VotingMechanism):
    """Simple 1-token-1-vote mechanism"""
    
    def calculate_vote_weight(self, voting_power: int, choice: str) -> float:
        return float(voting_power)
    
    def get_cost(self, voting_power: int, choice: str) -> int:
        return 0  # No cost for simple voting


class QuadraticVoting(VotingMechanism):
    """Quadratic voting mechanism"""
    
    def calculate_vote_weight(self, voting_power: int, choice: str) -> float:
        # Vote weight is square root of tokens spent
        import math
        return math.sqrt(voting_power)
    
    def get_cost(self, voting_power: int, choice: str) -> int:
        # Cost is square of vote weight
        import math
        vote_weight = math.sqrt(voting_power)
        return int(vote_weight ** 2)


class DelegatedVoting:
    """Liquid democracy delegation system"""
    
    def __init__(self):
        self.delegations: Dict[str, str] = {}  # delegator -> delegate
        self.delegation_chains: Dict[str, List[str]] = {}  # Track chains
    
    def delegate(self, delegator: str, delegate: str):
        """Delegate voting power"""
        # Check for circular delegation
        if self._would_create_cycle(delegator, delegate):
            raise ValueError("Delegation would create a cycle")
        
        self.delegations[delegator] = delegate
        self._update_delegation_chains()
        
        logger.info(f"{delegator} delegated to {delegate}")
    
    def revoke_delegation(self, delegator: str):
        """Revoke delegation"""
        if delegator in self.delegations:
            del self.delegations[delegator]
            self._update_delegation_chains()
            logger.info(f"{delegator} revoked delegation")
    
    def get_final_delegate(self, voter: str) -> str:
        """Get final delegate after following chain"""
        current = voter
        seen = set()
        
        while current in self.delegations and current not in seen:
            seen.add(current)
            current = self.delegations[current]
        
        return current
    
    def _would_create_cycle(self, delegator: str, delegate: str) -> bool:
        """Check if delegation would create a cycle"""
        current = delegate
        seen = set()
        
        while current in self.delegations:
            if current == delegator or current in seen:
                return True
            seen.add(current)
            current = self.delegations[current]
        
        return False
    
    def _update_delegation_chains(self):
        """Update delegation chain cache"""
        self.delegation_chains.clear()
        
        for delegator in self.delegations:
            chain = []
            current = delegator
            
            while current in self.delegations:
                chain.append(current)
                current = self.delegations[current]
            
            self.delegation_chains[delegator] = chain


class GovernanceSystem:
    """
    Main governance system implementing DAO mechanics.
    """
    
    def __init__(self):
        """Initialize governance system"""
        self.proposals: Dict[str, Proposal] = {}
        self.committees: Dict[CommitteeType, Committee] = {}
        self.voting_mechanism = QuadraticVoting()
        self.delegation_system = DelegatedVoting()
        self.total_voting_power = 0
        
        # Initialize committees
        self._initialize_committees()
        
        # Voting parameters
        self.proposal_threshold = 100  # Min voting power to create proposal
        self.voting_period = timedelta(days=7)
        self.execution_delay = timedelta(days=2)
        
        logger.info("Governance system initialized")
    
    def _initialize_committees(self):
        """Initialize governance committees"""
        # Scientific Advisory Board
        self.committees[CommitteeType.SCIENTIFIC_ADVISORY] = Committee(
            committee_type=CommitteeType.SCIENTIFIC_ADVISORY,
            members=set(),
            chair=None,
            term_end=datetime.now() + timedelta(days=365),
            responsibilities=[
                "Algorithm validation",
                "Reference standard updates",
                "Research protocol approval",
                "Scientific integrity oversight"
            ],
            voting_weight_multiplier=1.5
        )
        
        # Ethics Committee
        self.committees[CommitteeType.ETHICS] = Committee(
            committee_type=CommitteeType.ETHICS,
            members=set(),
            chair=None,
            term_end=datetime.now() + timedelta(days=365),
            responsibilities=[
                "Privacy protection policies",
                "Consent framework updates",
                "Ethical research guidelines",
                "Vulnerable population protections"
            ],
            voting_weight_multiplier=1.3
        )
        
        # Security Council
        self.committees[CommitteeType.SECURITY] = Committee(
            committee_type=CommitteeType.SECURITY,
            members=set(),
            chair=None,
            term_end=datetime.now() + timedelta(days=180),
            responsibilities=[
                "Threat monitoring",
                "Incident response",
                "Security parameter updates",
                "Emergency actions"
            ],
            voting_weight_multiplier=2.0  # Higher for emergency actions
        )
        
        # User Representatives
        self.committees[CommitteeType.USER_REPRESENTATIVES] = Committee(
            committee_type=CommitteeType.USER_REPRESENTATIVES,
            members=set(),
            chair=None,
            term_end=datetime.now() + timedelta(days=365),
            responsibilities=[
                "User experience feedback",
                "Community concerns",
                "Accessibility requirements",
                "User education initiatives"
            ],
            voting_weight_multiplier=1.0
        )
    
    def create_proposal(self, proposer: str, proposal_type: ProposalType,
                       title: str, description: str,
                       execution_data: Optional[Dict[str, Any]] = None) -> Proposal:
        """
        Create a new governance proposal.
        
        Args:
            proposer: Address creating the proposal
            proposal_type: Type of proposal
            title: Proposal title
            description: Detailed description
            execution_data: Data for automatic execution
            
        Returns:
            Created proposal
        """
        # Check proposer has sufficient voting power
        proposer_power = self._get_voting_power(proposer)
        if proposer_power < self.proposal_threshold:
            raise ValueError(f"Insufficient voting power: {proposer_power} < {self.proposal_threshold}")
        
        # Generate proposal ID
        proposal_id = hashlib.sha256(
            f"{proposer}:{title}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Set requirements based on proposal type
        if proposal_type in [ProposalType.PROTOCOL_UPDATE, ProposalType.EMERGENCY_ACTION]:
            approval_threshold = 0.67  # 67% for critical changes
            quorum_required = 0.2  # 20% quorum
        else:
            approval_threshold = 0.51  # 51% for normal proposals
            quorum_required = 0.1  # 10% quorum
        
        # Create proposal
        proposal = Proposal(
            proposal_id=proposal_id,
            proposal_type=proposal_type,
            title=title,
            description=description,
            proposer=proposer,
            created_at=datetime.now(),
            voting_start=datetime.now() + timedelta(days=1),  # 1 day delay
            voting_end=datetime.now() + self.voting_period + timedelta(days=1),
            execution_delay=self.execution_delay,
            status=ProposalStatus.DRAFT,
            approval_threshold=approval_threshold,
            quorum_required=quorum_required,
            execution_data=execution_data
        )
        
        self.proposals[proposal_id] = proposal
        
        # Audit log
        audit_logger.log_event(
            event_type="governance_proposal",
            actor=proposer,
            action="create_proposal",
            resource=proposal_id,
            metadata={
                'proposal_type': proposal_type.value,
                'title': title
            }
        )
        
        logger.info(f"Proposal {proposal_id} created by {proposer}")
        
        return proposal
    
    def vote(self, proposal_id: str, voter: str, choice: str,
            voting_power: Optional[int] = None) -> VoteRecord:
        """
        Cast a vote on a proposal.
        
        Args:
            proposal_id: ID of proposal to vote on
            voter: Address casting the vote
            choice: Vote choice (yes, no, abstain)
            voting_power: Override voting power (for quadratic voting)
            
        Returns:
            Vote record
        """
        if proposal_id not in self.proposals:
            raise ValueError("Proposal not found")
        
        proposal = self.proposals[proposal_id]
        
        # Check proposal is active
        now = datetime.now()
        if now < proposal.voting_start:
            raise ValueError("Voting has not started")
        if now > proposal.voting_end:
            raise ValueError("Voting has ended")
        if proposal.status != ProposalStatus.ACTIVE:
            raise ValueError("Proposal is not active")
        
        # Check if already voted
        for vote in proposal.votes:
            if vote.voter_id == voter or vote.delegate_from == voter:
                raise ValueError("Already voted")
        
        # Get voting power
        if voting_power is None:
            voting_power = self._get_voting_power(voter)
        
        # Check for delegation
        final_voter = self.delegation_system.get_final_delegate(voter)
        delegate_from = voter if final_voter != voter else None
        
        # Apply committee multiplier if applicable
        multiplier = self._get_committee_multiplier(final_voter, proposal.proposal_type)
        
        # Calculate vote weight
        vote_weight = self.voting_mechanism.calculate_vote_weight(voting_power, choice)
        vote_weight *= multiplier
        
        # Create vote record
        vote_record = VoteRecord(
            voter_id=final_voter,
            voting_power=voting_power,
            vote_weight=vote_weight,
            choice=choice,
            timestamp=datetime.now(),
            delegate_from=delegate_from
        )
        
        # Add vote to proposal
        proposal.add_vote(vote_record)
        
        # Check if proposal outcome is determined
        self._check_proposal_outcome(proposal)
        
        logger.info(f"Vote cast on {proposal_id}: {choice} with weight {vote_weight}")
        
        return vote_record
    
    def _get_voting_power(self, address: str) -> int:
        """Get voting power for an address"""
        # In production, would query from blockchain state
        # For now, simulate based on node type
        base_power = 100
        
        # Add node class weight
        node_class_bonus = 50  # Placeholder
        
        # Add signatory bonus
        signatory_bonus = 200 if self._is_trusted_signatory(address) else 0
        
        return base_power + node_class_bonus + signatory_bonus
    
    def _is_trusted_signatory(self, address: str) -> bool:
        """Check if address is a trusted signatory"""
        # In production, would check on-chain status
        return address.startswith("ts_")
    
    def _get_committee_multiplier(self, voter: str, proposal_type: ProposalType) -> float:
        """Get voting multiplier based on committee membership"""
        multiplier = 1.0
        
        # Check committee memberships
        for committee_type, committee in self.committees.items():
            if committee.is_member(voter):
                # Apply multiplier for relevant proposals
                if self._is_committee_relevant(committee_type, proposal_type):
                    multiplier = max(multiplier, committee.voting_weight_multiplier)
        
        return multiplier
    
    def _is_committee_relevant(self, committee_type: CommitteeType,
                              proposal_type: ProposalType) -> bool:
        """Check if committee is relevant to proposal type"""
        relevance_map = {
            CommitteeType.SCIENTIFIC_ADVISORY: [
                ProposalType.ALGORITHM_CERTIFICATION,
                ProposalType.REFERENCE_UPDATE,
                ProposalType.PROTOCOL_UPDATE
            ],
            CommitteeType.ETHICS: [
                ProposalType.PROTOCOL_UPDATE,
                ProposalType.PARAMETER_CHANGE
            ],
            CommitteeType.SECURITY: [
                ProposalType.EMERGENCY_ACTION,
                ProposalType.PROTOCOL_UPDATE
            ],
            CommitteeType.USER_REPRESENTATIVES: [
                ProposalType.PARAMETER_CHANGE,
                ProposalType.TREASURY_ALLOCATION
            ]
        }
        
        return proposal_type in relevance_map.get(committee_type, [])
    
    def _check_proposal_outcome(self, proposal: Proposal):
        """Check if proposal outcome is determined"""
        # Update total voting power
        self.total_voting_power = self._calculate_total_voting_power()
        
        # Check if voting period ended
        if datetime.now() > proposal.voting_end:
            if proposal.has_quorum(self.total_voting_power) and proposal.has_passed():
                proposal.status = ProposalStatus.PASSED
                logger.info(f"Proposal {proposal.proposal_id} passed")
            else:
                proposal.status = ProposalStatus.REJECTED
                logger.info(f"Proposal {proposal.proposal_id} rejected")
    
    def _calculate_total_voting_power(self) -> float:
        """Calculate total voting power in the system"""
        # In production, would sum from all active participants
        return 10000.0  # Placeholder
    
    def execute_proposal(self, proposal_id: str) -> Dict[str, Any]:
        """
        Execute a passed proposal.
        
        Args:
            proposal_id: ID of proposal to execute
            
        Returns:
            Execution result
        """
        if proposal_id not in self.proposals:
            raise ValueError("Proposal not found")
        
        proposal = self.proposals[proposal_id]
        
        # Check proposal can be executed
        if proposal.status != ProposalStatus.PASSED:
            raise ValueError("Proposal has not passed")
        
        # Check execution delay
        time_since_end = datetime.now() - proposal.voting_end
        if time_since_end < proposal.execution_delay:
            raise ValueError("Execution delay not met")
        
        # Execute based on proposal type
        result = self._execute_proposal_action(proposal)
        
        # Update proposal status
        proposal.status = ProposalStatus.EXECUTED
        proposal.execution_timestamp = datetime.now()
        
        # Audit log
        audit_logger.log_event(
            event_type="governance_execution",
            actor="governance_system",
            action="execute_proposal",
            resource=proposal_id,
            metadata={
                'proposal_type': proposal.proposal_type.value,
                'result': result
            }
        )
        
        logger.info(f"Proposal {proposal_id} executed")
        
        return result
    
    def _execute_proposal_action(self, proposal: Proposal) -> Dict[str, Any]:
        """Execute the specific action for a proposal"""
        if proposal.proposal_type == ProposalType.PARAMETER_CHANGE:
            return self._execute_parameter_change(proposal)
        elif proposal.proposal_type == ProposalType.ALGORITHM_CERTIFICATION:
            return self._execute_algorithm_certification(proposal)
        elif proposal.proposal_type == ProposalType.TREASURY_ALLOCATION:
            return self._execute_treasury_allocation(proposal)
        elif proposal.proposal_type == ProposalType.COMMITTEE_ELECTION:
            return self._execute_committee_election(proposal)
        else:
            return {"status": "manual_execution_required"}
    
    def _execute_parameter_change(self, proposal: Proposal) -> Dict[str, Any]:
        """Execute parameter change proposal"""
        if not proposal.execution_data:
            return {"error": "No execution data"}
        
        parameter = proposal.execution_data.get("parameter")
        new_value = proposal.execution_data.get("new_value")
        
        # In production, would update on-chain parameters
        logger.info(f"Parameter {parameter} changed to {new_value}")
        
        return {
            "status": "success",
            "parameter": parameter,
            "new_value": new_value
        }
    
    def _execute_algorithm_certification(self, proposal: Proposal) -> Dict[str, Any]:
        """Execute algorithm certification proposal"""
        if not proposal.execution_data:
            return {"error": "No execution data"}
        
        algorithm_id = proposal.execution_data.get("algorithm_id")
        certification_level = proposal.execution_data.get("certification_level")
        
        # In production, would update algorithm registry
        logger.info(f"Algorithm {algorithm_id} certified at level {certification_level}")
        
        return {
            "status": "success",
            "algorithm_id": algorithm_id,
            "certification_level": certification_level
        }
    
    def _execute_treasury_allocation(self, proposal: Proposal) -> Dict[str, Any]:
        """Execute treasury allocation proposal"""
        if not proposal.execution_data:
            return {"error": "No execution data"}
        
        recipient = proposal.execution_data.get("recipient")
        amount = proposal.execution_data.get("amount")
        
        # In production, would transfer funds
        logger.info(f"Treasury allocation: {amount} to {recipient}")
        
        return {
            "status": "success",
            "recipient": recipient,
            "amount": amount
        }
    
    def _execute_committee_election(self, proposal: Proposal) -> Dict[str, Any]:
        """Execute committee election proposal"""
        if not proposal.execution_data:
            return {"error": "No execution data"}
        
        committee_type = CommitteeType(proposal.execution_data.get("committee_type"))
        new_members = proposal.execution_data.get("new_members", [])
        remove_members = proposal.execution_data.get("remove_members", [])
        
        committee = self.committees[committee_type]
        
        # Update committee membership
        for member in remove_members:
            committee.remove_member(member)
        
        for member in new_members:
            committee.add_member(member)
        
        return {
            "status": "success",
            "committee": committee_type.value,
            "added": new_members,
            "removed": remove_members
        }
    
    def get_active_proposals(self) -> List[Proposal]:
        """Get all active proposals"""
        now = datetime.now()
        active = []
        
        for proposal in self.proposals.values():
            if (proposal.status == ProposalStatus.ACTIVE or
                (proposal.status == ProposalStatus.DRAFT and 
                 now >= proposal.voting_start and 
                 now <= proposal.voting_end)):
                
                # Update status if needed
                if proposal.status == ProposalStatus.DRAFT:
                    proposal.status = ProposalStatus.ACTIVE
                
                active.append(proposal)
        
        return active
    
    def get_proposal_details(self, proposal_id: str) -> Dict[str, Any]:
        """Get detailed information about a proposal"""
        if proposal_id not in self.proposals:
            raise ValueError("Proposal not found")
        
        proposal = self.proposals[proposal_id]
        
        return {
            "proposal_id": proposal.proposal_id,
            "type": proposal.proposal_type.value,
            "title": proposal.title,
            "description": proposal.description,
            "proposer": proposal.proposer,
            "status": proposal.status.value,
            "voting_start": proposal.voting_start.isoformat(),
            "voting_end": proposal.voting_end.isoformat(),
            "votes": {
                "yes": proposal.yes_votes,
                "no": proposal.no_votes,
                "abstain": proposal.abstain_votes,
                "total": proposal.get_total_votes()
            },
            "requirements": {
                "quorum": proposal.quorum_required,
                "approval_threshold": proposal.approval_threshold,
                "has_quorum": proposal.has_quorum(self.total_voting_power),
                "current_approval": proposal.get_approval_rate()
            },
            "vote_count": len(proposal.votes)
        }


class HIPAAOracle:
    """
    Oracle for HIPAA fast-track verification.
    Verifies healthcare provider credentials on-chain.
    """
    
    def __init__(self):
        """Initialize HIPAA oracle"""
        self.verified_providers: Dict[str, Dict[str, Any]] = {}
        self.verification_cache: Dict[str, bool] = {}
        
        logger.info("HIPAA oracle initialized")
    
    async def verify_provider(self, npi: str, baa_hash: str,
                            risk_analysis_hash: str, hsm_serial: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify healthcare provider for trusted signatory status.
        
        Args:
            npi: National Provider Identifier
            baa_hash: Hash of Business Associate Agreement
            risk_analysis_hash: Hash of HIPAA risk analysis
            hsm_serial: Hardware Security Module serial number
            
        Returns:
            Tuple of (is_valid, verification_details)
        """
        # Check cache
        cache_key = f"{npi}:{baa_hash}:{risk_analysis_hash}:{hsm_serial}"
        if cache_key in self.verification_cache:
            return self.verification_cache[cache_key], self.verified_providers.get(npi, {})
        
        # Verify NPI format
        if not self._validate_npi_format(npi):
            return False, {"error": "Invalid NPI format"}
        
        # In production, would query CMS NPI registry
        # For now, simulate verification
        is_valid = await self._check_cms_registry(npi)
        
        if not is_valid:
            return False, {"error": "NPI not found in CMS registry"}
        
        # Verify other components
        if not all([baa_hash, risk_analysis_hash, hsm_serial]):
            return False, {"error": "Missing required components"}
        
        # Create verification record
        verification = {
            "npi": npi,
            "baa_hash": baa_hash,
            "risk_analysis_hash": risk_analysis_hash,
            "hsm_serial": hsm_serial,
            "verified_at": datetime.now().isoformat(),
            "signatory_weight": SignatoryWeight.TRUSTED_SIGNATORY.value,
            "honesty_probability": 0.98
        }
        
        # Store verification
        self.verified_providers[npi] = verification
        self.verification_cache[cache_key] = True
        
        # Audit log
        audit_logger.log_event(
            event_type="hipaa_verification",
            actor="hipaa_oracle",
            action="verify_provider",
            resource=npi,
            metadata={
                'hsm_serial': hsm_serial,
                'success': True
            }
        )
        
        logger.info(f"HIPAA provider {npi} verified as trusted signatory")
        
        return True, verification
    
    def _validate_npi_format(self, npi: str) -> bool:
        """Validate NPI format (10 digits with Luhn check)"""
        if not npi or len(npi) != 10 or not npi.isdigit():
            return False
        
        # Luhn algorithm check
        total = 0
        for i, digit in enumerate(npi[:-1]):
            d = int(digit)
            if i % 2 == 0:  # Double every other digit
                d *= 2
                if d > 9:
                    d = d - 9
            total += d
        
        check_digit = (10 - (total % 10)) % 10
        return int(npi[-1]) == check_digit
    
    async def _check_cms_registry(self, npi: str) -> bool:
        """Check NPI in CMS registry (simulated)"""
        # In production, would make actual API call to CMS NPPES
        # For simulation, accept NPIs starting with 1-9
        return npi[0] in '123456789'
    
    def get_provider_details(self, npi: str) -> Optional[Dict[str, Any]]:
        """Get verified provider details"""
        return self.verified_providers.get(npi)
    
    def revoke_verification(self, npi: str, reason: str):
        """Revoke provider verification"""
        if npi in self.verified_providers:
            provider = self.verified_providers[npi]
            provider['revoked'] = True
            provider['revoked_at'] = datetime.now().isoformat()
            provider['revocation_reason'] = reason
            
            # Clear cache
            for key in list(self.verification_cache.keys()):
                if key.startswith(f"{npi}:"):
                    del self.verification_cache[key]
            
            audit_logger.log_event(
                event_type="hipaa_revocation",
                actor="hipaa_oracle",
                action="revoke_verification",
                resource=npi,
                metadata={'reason': reason}
            )
            
            logger.info(f"HIPAA verification revoked for {npi}: {reason}")


# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Initialize governance system
    governance = GovernanceSystem()
    
    # Add some committee members
    governance.committees[CommitteeType.SCIENTIFIC_ADVISORY].add_member("scientist_1")
    governance.committees[CommitteeType.ETHICS].add_member("ethicist_1")
    governance.committees[CommitteeType.SECURITY].add_member("security_1")
    
    # Create a proposal
    print("Creating governance proposal...")
    proposal = governance.create_proposal(
        proposer="user_with_power",
        proposal_type=ProposalType.PARAMETER_CHANGE,
        title="Increase PIR query timeout",
        description="Increase the PIR query timeout from 30s to 45s to improve reliability",
        execution_data={
            "parameter": "pir_query_timeout_seconds",
            "old_value": 30,
            "new_value": 45
        }
    )
    print(f"Proposal created: {proposal.proposal_id}")
    
    # Simulate moving to active status
    proposal.status = ProposalStatus.ACTIVE
    
    # Cast some votes
    print("\nCasting votes...")
    governance.vote(proposal.proposal_id, "user_1", "yes", voting_power=100)
    governance.vote(proposal.proposal_id, "user_2", "no", voting_power=50)
    governance.vote(proposal.proposal_id, "scientist_1", "yes", voting_power=200)
    
    # Check proposal status
    details = governance.get_proposal_details(proposal.proposal_id)
    print(f"\nProposal details:")
    print(f"  Yes votes: {details['votes']['yes']}")
    print(f"  No votes: {details['votes']['no']}")
    print(f"  Approval rate: {details['requirements']['current_approval']:.2%}")
    
    # Test HIPAA Oracle
    print("\n\nTesting HIPAA Oracle...")
    oracle = HIPAAOracle()
    
    # Test valid NPI
    npi = "1234567893"  # Valid format with correct check digit
    result, details = asyncio.run(oracle.verify_provider(
        npi=npi,
        baa_hash="baa_hash_example",
        risk_analysis_hash="risk_hash_example",
        hsm_serial="HSM123456"
    ))
    
    print(f"Verification result: {result}")
    if result:
        print(f"Provider details: {details}")
    
    # Test delegation
    print("\n\nTesting delegation system...")
    delegation = DelegatedVoting()
    delegation.delegate("user_3", "user_4")
    delegation.delegate("user_4", "expert_1")
    
    print(f"Final delegate for user_3: {delegation.get_final_delegate('user_3')}")
    
    # Test circular delegation prevention
    try:
        delegation.delegate("expert_1", "user_3")
        print("ERROR: Circular delegation allowed!")
    except ValueError as e:
        print(f"Correctly prevented circular delegation: {e}")
