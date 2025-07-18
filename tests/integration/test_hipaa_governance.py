"""
Integration tests for HIPAA Fast-Track with Governance
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from genomevault.blockchain.hipaa import (
    HIPAAVerifier,
    HIPAACredentials,
    CMSNPIRegistry
)
from genomevault.blockchain.hipaa.integration import (
    HIPAANodeIntegration,
    HIPAAGovernanceIntegration
)
from genomevault.blockchain.governance import (
    GovernanceSystem,
    ProposalType,
    ProposalStatus,
    CommitteeType
)
from genomevault.blockchain.node import NodeType
from genomevault.core.constants import SignatoryWeight
from genomevault.core.exceptions import VerificationError


@pytest.mark.asyncio
class TestHIPAANodeIntegration:
    """Test HIPAA node integration"""
    
    async def test_provider_node_registration(self):
        """Test registering a provider as a blockchain node"""
        async with CMSNPIRegistry() as registry:
            verifier = HIPAAVerifier(npi_registry=registry)
            governance = GovernanceSystem()
            integration = HIPAANodeIntegration(verifier, governance)
            
            # Initial voting power
            initial_power = governance.total_voting_power
            
            # Register provider
            credentials = HIPAACredentials(
                npi="1234567893",
                baa_hash="a" * 64,
                risk_analysis_hash="b" * 64,
                hsm_serial="HSM-12345"
            )
            
            node_config = {
                'node_class': NodeType.FULL,
                'location': 'US-East'
            }
            
            node = await integration.register_provider_node(credentials, node_config)
            
            # Verify node properties
            assert node.node_id == f"hipaa_{credentials.npi}"
            assert node.node_type == NodeType.FULL
            assert node.signatory_status == SignatoryWeight.TRUSTED_SIGNATORY
            assert node.voting_power == 14  # Full (4) + TS (10)
            
            # Verify metadata
            assert node.metadata['npi'] == credentials.npi
            assert node.metadata['hsm_serial'] == credentials.hsm_serial
            assert node.metadata['honesty_probability'] == 0.98
            
            # Verify governance update
            assert governance.total_voting_power == initial_power + 14
            
            # Verify registry
            assert integration.get_provider_node(credentials.npi) == node
    
    async def test_multiple_provider_registration(self):
        """Test registering multiple providers"""
        async with CMSNPIRegistry() as registry:
            verifier = HIPAAVerifier(npi_registry=registry)
            governance = GovernanceSystem()
            integration = HIPAANodeIntegration(verifier, governance)
            
            providers = [
                ("1234567893", NodeType.LIGHT, 11),   # Light + TS
                ("1987654321", NodeType.FULL, 14),    # Full + TS
                ("2468135790", NodeType.ARCHIVE, 18), # Archive + TS
            ]
            
            total_power = 0
            
            for npi, node_type, expected_power in providers:
                credentials = HIPAACredentials(
                    npi=npi,
                    baa_hash="a" * 64,
                    risk_analysis_hash="b" * 64,
                    hsm_serial=f"HSM-{npi[-6:]}"
                )
                
                node = await integration.register_provider_node(
                    credentials,
                    {'node_class': node_type}
                )
                
                assert node.voting_power == expected_power
                total_power += expected_power
            
            # Verify total governance power
            assert governance.total_voting_power == total_power
    
    async def test_provider_node_revocation(self):
        """Test revoking a provider's node status"""
        async with CMSNPIRegistry() as registry:
            verifier = HIPAAVerifier(npi_registry=registry)
            governance = GovernanceSystem()
            integration = HIPAANodeIntegration(verifier, governance)
            
            # Register provider
            credentials = HIPAACredentials(
                npi="1234567893",
                baa_hash="a" * 64,
                risk_analysis_hash="b" * 64,
                hsm_serial="HSM-12345"
            )
            
            node = await integration.register_provider_node(
                credentials,
                {'node_class': NodeType.FULL}
            )
            
            initial_power = governance.total_voting_power
            
            # Revoke
            success = await integration.revoke_provider_node(
                credentials.npi,
                "Test revocation"
            )
            
            assert success
            assert integration.get_provider_node(credentials.npi) is None
            assert governance.total_voting_power == initial_power - node.voting_power
    
    async def test_verification_refresh(self):
        """Test refreshing verifications"""
        async with CMSNPIRegistry() as registry:
            verifier = HIPAAVerifier(npi_registry=registry)
            governance = GovernanceSystem()
            integration = HIPAANodeIntegration(verifier, governance)
            
            # Set short expiry for testing
            verifier.verification_expiry_days = 0
            
            # Register providers
            npis = ["1234567893", "1987654321"]
            
            for npi in npis:
                credentials = HIPAACredentials(
                    npi=npi,
                    baa_hash="a" * 64,
                    risk_analysis_hash="b" * 64,
                    hsm_serial=f"HSM-{npi[-6:]}"
                )
                
                await integration.register_provider_node(
                    credentials,
                    {'node_class': NodeType.LIGHT}
                )
            
            # Refresh should remove expired nodes
            results = await integration.refresh_verifications()
            
            assert results['expired_verifications'] == 2
            assert results['revoked_nodes'] == 2
            assert results['active_nodes'] == 0
            assert len(integration.node_registry) == 0


@pytest.mark.asyncio
class TestHIPAAGovernance:
    """Test HIPAA governance integration"""
    
    async def test_hipaa_committee_creation(self):
        """Test creating HIPAA committee"""
        governance = GovernanceSystem()
        
        # Create HIPAA committee
        HIPAAGovernanceIntegration.create_hipaa_committee(governance)
        
        # Verify committee exists
        committee = governance.committees[CommitteeType.SCIENTIFIC_ADVISORY]
        assert "Clinical data governance" in committee.responsibilities
        assert "HIPAA compliance oversight" in committee.responsibilities
        assert committee.voting_weight_multiplier == 1.5
    
    async def test_hipaa_proposal_types(self):
        """Test HIPAA-specific proposal types"""
        governance = GovernanceSystem()
        
        # Add HIPAA proposal types
        HIPAAGovernanceIntegration.add_hipaa_proposal_types(governance)
        
        # Verify proposal rules
        assert hasattr(governance, 'hipaa_proposal_rules')
        
        clinical_rules = governance.hipaa_proposal_rules['clinical_protocol_update']
        assert clinical_rules['approval_threshold'] == 0.60
        assert clinical_rules['quorum_required'] == 0.15
        assert clinical_rules['voting_period_days'] == 14
        assert clinical_rules['requires_hipaa_member'] == True
    
    async def test_hipaa_enhanced_voting(self):
        """Test enhanced voting power for HIPAA members"""
        async with CMSNPIRegistry() as registry:
            verifier = HIPAAVerifier(npi_registry=registry)
            governance = GovernanceSystem()
            integration = HIPAANodeIntegration(verifier, governance)
            
            # Setup committee
            HIPAAGovernanceIntegration.create_hipaa_committee(governance)
            
            # Register HIPAA provider
            credentials = HIPAACredentials(
                npi="1234567893",
                baa_hash="a" * 64,
                risk_analysis_hash="b" * 64,
                hsm_serial="HSM-12345"
            )
            
            node = await integration.register_provider_node(
                credentials,
                {'node_class': NodeType.FULL}
            )
            
            # Add to committee
            governance.committees[CommitteeType.SCIENTIFIC_ADVISORY].add_member(node.node_id)
            
            # Create proposal
            proposal = governance.create_proposal(
                proposer=node.node_id,
                proposal_type=ProposalType.PROTOCOL_UPDATE,
                title="Test Clinical Protocol",
                description="Test description"
            )
            
            # Activate proposal
            proposal.status = ProposalStatus.ACTIVE
            
            # Vote with committee membership
            vote = governance.vote(
                proposal_id=proposal.proposal_id,
                voter=node.node_id,
                choice="yes",
                voting_power=node.voting_power
            )
            
            # Verify enhanced weight applied
            base_weight = governance.voting_mechanism.calculate_vote_weight(node.voting_power, "yes")
            assert vote.vote_weight > base_weight  # Committee multiplier applied
    
    async def test_governance_participation_comparison(self):
        """Test voting power difference between HIPAA and regular nodes"""
        governance = GovernanceSystem()
        
        # Create proposal requiring votes
        proposal = governance.create_proposal(
            proposer="admin",
            proposal_type=ProposalType.PARAMETER_CHANGE,
            title="Test Parameter Change",
            description="Testing voting power differences"
        )
        
        proposal.status = ProposalStatus.ACTIVE
        
        # Simulate HIPAA node vote
        hipaa_node_id = "hipaa_1234567893"
        hipaa_voting_power = 14  # Full + TS
        
        hipaa_vote = governance.vote(
            proposal_id=proposal.proposal_id,
            voter=hipaa_node_id,
            choice="yes",
            voting_power=hipaa_voting_power
        )
        
        # Simulate regular node vote
        regular_node_id = "regular_node_1"
        regular_voting_power = 4  # Full only
        
        regular_vote = governance.vote(
            proposal_id=proposal.proposal_id,
            voter=regular_node_id,
            choice="yes",
            voting_power=regular_voting_power
        )
        
        # HIPAA node should have significantly more weight
        assert hipaa_vote.vote_weight > regular_vote.vote_weight
        
        # In quadratic voting, weight is sqrt of power
        import math
        assert hipaa_vote.vote_weight == pytest.approx(math.sqrt(hipaa_voting_power))
        assert regular_vote.vote_weight == pytest.approx(math.sqrt(regular_voting_power))


@pytest.mark.asyncio
class TestEndToEndFlow:
    """Test complete HIPAA fast-track flow"""
    
    async def test_complete_provider_journey(self):
        """Test complete journey from registration to governance participation"""
        async with CMSNPIRegistry() as registry:
            # Initialize system
            verifier = HIPAAVerifier(npi_registry=registry)
            governance = GovernanceSystem()
            integration = HIPAANodeIntegration(verifier, governance)
            
            # Setup governance
            HIPAAGovernanceIntegration.create_hipaa_committee(governance)
            HIPAAGovernanceIntegration.add_hipaa_proposal_types(governance)
            
            # Step 1: Provider registration
            provider_npi = "1234567893"
            credentials = HIPAACredentials(
                npi=provider_npi,
                baa_hash="baa_production_hash_" + "a" * 40,
                risk_analysis_hash="risk_analysis_hash_" + "b" * 40,
                hsm_serial="HSM-PROD-12345",
                provider_name="Metro General Hospital"
            )
            
            node = await integration.register_provider_node(
                credentials,
                {
                    'node_class': NodeType.ARCHIVE,
                    'location': 'US-East-1',
                    'bandwidth': 10000
                }
            )
            
            assert node.voting_power == 18  # Archive (8) + TS (10)
            
            # Step 2: Join committee
            governance.committees[CommitteeType.SCIENTIFIC_ADVISORY].add_member(node.node_id)
            
            # Step 3: Create clinical proposal
            proposal = governance.create_proposal(
                proposer=node.node_id,
                proposal_type=ProposalType.PROTOCOL_UPDATE,
                title="Standardize Diabetes Management Protocol",
                description="Implement standardized protocol for diabetes risk management",
                execution_data={
                    'protocol_version': '2.0',
                    'effective_date': (datetime.now() + timedelta(days=30)).isoformat()
                }
            )
            
            # Step 4: Voting phase
            proposal.status = ProposalStatus.ACTIVE
            
            # Provider votes yes
            provider_vote = governance.vote(
                proposal_id=proposal.proposal_id,
                voter=node.node_id,
                choice="yes"
            )
            
            # Add other votes to meet quorum
            for i in range(5):
                governance.vote(
                    proposal_id=proposal.proposal_id,
                    voter=f"supporter_{i}",
                    choice="yes",
                    voting_power=50
                )
            
            # Check proposal status
            details = governance.get_proposal_details(proposal.proposal_id)
            assert details['votes']['yes'] > 0
            assert details['requirements']['current_approval'] > 0.5
            
            # Step 5: Verify ongoing participation
            active_proposals = governance.get_active_proposals()
            assert proposal in active_proposals
            
            # Verify node remains active
            status = verifier.get_verification_status(provider_npi)
            assert status.is_active()
            
            # Verify enhanced rewards calculation
            # Credits per block = c + 2 (for TS)
            expected_credits = 8 + 2  # Archive + TS bonus
            assert expected_credits == 10


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
