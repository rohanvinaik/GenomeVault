"""
HIPAA Fast-Track Example

Demonstrates the complete HIPAA verification and governance integration flow.
"""

import asyncio

from genomevault.blockchain.governance import (
    CommitteeType,
    GovernanceSystem,
    ProposalStatus,
    ProposalType,
)
from genomevault.blockchain.hipaa import CMSNPIRegistry, HIPAACredentials, HIPAAVerifier
from genomevault.blockchain.hipaa.integration import (
    HIPAAGovernanceIntegration,
    HIPAANodeIntegration,
)
from genomevault.blockchain.node import NodeType
from genomevault.utils import get_logger

logger = get_logger(__name__)


async def demonstrate_hipaa_fasttrack():
    """
    Complete demonstration of HIPAA fast-track system.

    Shows:
    1. Healthcare provider verification
    2. Automatic trusted signatory status
    3. Enhanced voting power in governance
    4. Special committee participation
    """

    print("=" * 80)
    print("GenomeVault HIPAA Fast-Track Demonstration")
    print("=" * 80)

    # Initialize components
    async with CMSNPIRegistry() as registry:
        verifier = HIPAAVerifier(npi_registry=registry)
        governance = GovernanceSystem()
        integration = HIPAANodeIntegration(verifier, governance)

        # Setup HIPAA governance features
        HIPAAGovernanceIntegration.create_hipaa_committee(governance)
        HIPAAGovernanceIntegration.add_hipaa_proposal_types(governance)

        print("\n1. HEALTHCARE PROVIDER REGISTRATION")
        print("-" * 40)

        # Simulate three different healthcare providers
        providers = [
            {
                "name": "Metro General Hospital",
                "npi": "1234567893",
                "node_class": NodeType.ARCHIVE,  # Large hospital with archive node
                "description": "Major medical center with 500+ beds",
            },
            {
                "name": "Smith Family Practice",
                "npi": "1987654321",
                "node_class": NodeType.LIGHT,  # Small practice with light node
                "description": "Solo practitioner with Mac mini",
            },
            {
                "name": "Regional Medical Labs",
                "npi": "2468135790",
                "node_class": NodeType.FULL,  # Lab with standard server
                "description": "Clinical laboratory with 1U server",
            },
        ]

        registered_nodes = []

        for provider in providers:
            print("\nRegistering: {provider['name']}")
            print("  NPI: {provider['npi']}")
            print("  Type: {provider['description']}")

            credentials = HIPAACredentials(
                npi=provider["npi"],
                baa_hash="a" * 64,  # Simulated BAA hash
                risk_analysis_hash="b" * 64,  # Simulated risk analysis
                hsm_serial="HSM-{provider['npi'][-6:]}",
                provider_name=provider["name"],
            )

            node_config = {
                "node_class": provider["node_class"],
                "location": "US-East",
                "bandwidth": 1000 if provider["node_class"] != NodeType.LIGHT else 100,
            }

            try:
                node = await integration.register_provider_node(credentials, node_config)
                registered_nodes.append((provider, node))

                print("  ✓ Verification successful!")
                print("  ✓ Node ID: {node.node_id}")
                print("  ✓ Voting power: {node.voting_power}")

                # Add to HIPAA committee
                governance.committees[CommitteeType.SCIENTIFIC_ADVISORY].add_member(node.node_id)

            except Exception:
                logger.exception("Unhandled exception")
                print("  ✗ Registration failed: {e}")
                raise

        print("\n\nTotal governance voting power: {governance.total_voting_power}")

        print("\n\n2. GOVERNANCE PARTICIPATION")
        print("-" * 40)

        # Create a clinical protocol proposal
        if registered_nodes:
            proposer_node = registered_nodes[0][1]  # Metro General Hospital

            print("\n{registered_nodes[0][0]['name']} creating clinical data proposal...")

            proposal = governance.create_proposal(
                proposer=proposer_node.node_id,
                proposal_type=ProposalType.PROTOCOL_UPDATE,
                title="Standardize Clinical Data Format for Diabetes Pilot",
                description="""
                Proposal to standardize the clinical data format for the diabetes
                risk management pilot program. This will ensure interoperability
                between different healthcare providers and improve patient outcomes.
                
                Key changes:
                - Adopt FHIR R4 for all clinical data exchange
                - Require LOINC codes for lab results
                - Implement real-time glucose monitoring integration
                - Add HbA1c trending requirements
                """,
                execution_data={
                    "standard": "FHIR_R4",
                    "required_codes": ["LOINC", "SNOMED-CT"],
                    "pilot_duration_days": 180,
                },
            )

            print("  ✓ Proposal created: {proposal.proposal_id}")
            print("  Voting period: {proposal.voting_period.days} days")
            print("  Required quorum: {proposal.quorum_required:.0%}")
            print("  Approval threshold: {proposal.approval_threshold:.0%}")

            # Simulate voting
            print("\n\nSimulating governance votes...")
            proposal.status = ProposalStatus.ACTIVE  # Activate for demo

            # Healthcare providers vote
            for provider, node in registered_nodes:
                choice = (
                    "yes"
                    if "Hospital" in provider["name"] or "Lab" in provider["name"]
                    else "abstain"
                )

                governance.vote(
                    proposal_id=proposal.proposal_id,
                    voter=node.node_id,
                    choice=choice,
                    voting_power=node.voting_power,
                )

                print("\n{provider['name']} voted: {choice}")
                print("  Vote weight: {vote.vote_weight}")

                # Show enhanced weight for committee members
                if governance._get_committee_multiplier(node.node_id, proposal.proposal_type) > 1:
                    print("  ✓ Committee bonus applied!")

            # Add some non-HIPAA votes for comparison
            print("\n\nNon-HIPAA participants voting...")
            for i in range(3):
                voter_id = "regular_node_{i}"
                choice = "yes" if i < 2 else "no"

                governance.vote(
                    proposal_id=proposal.proposal_id,
                    voter=voter_id,
                    choice=choice,
                    voting_power=50,  # Lower voting power
                )

                print("Regular Node {i} voted: {choice} (weight: {vote.vote_weight})")

            # Check results
            print("\n\nProposal Results:")
            print("-" * 40)
            governance.get_proposal_details(proposal.proposal_id)

            print("Total votes cast: {details['vote_count']}")
            print("Yes votes: {details['votes']['yes']:.1f}")
            print("No votes: {details['votes']['no']:.1f}")
            print("Abstentions: {details['votes']['abstain']:.1f}")
            print("Current approval: {details['requirements']['current_approval']:.1%}")
            print("Has quorum: {details['requirements']['has_quorum']}")

            # Demonstrate the voting power difference
            print("\n\n3. VOTING POWER COMPARISON")
            print("-" * 40)
            print("\nHIPAA-Verified Nodes (Trusted Signatories):")
            for provider, node in registered_nodes:
                print(
                    "  {provider['name']:30} | Class: {node.node_type.name:7} | Power: {node.voting_power:3}"
                )

            print("\nRegular Nodes (Non-signatories):")
            print("  {'Regular Light Node':30} | Class: {'LIGHT':7} | Power: {1:3}")
            print("  {'Regular Full Node':30} | Class: {'FULL':7} | Power: {4:3}")
            print("  {'Regular Archive Node':30} | Class: {'ARCHIVE':7} | Power: {8:3}")

            print("\n✓ HIPAA providers receive +10 signatory weight!")

        print("\n\n4. BENEFITS SUMMARY")
        print("-" * 40)
        print(
            """
The HIPAA Fast-Track system provides:

1. STREAMLINED ONBOARDING
   - Submit: NPI + BAA hash + Risk Analysis hash + HSM serial
   - Automatic verification via CMS registry
   - Instant Trusted Signatory status

2. ENHANCED VOTING POWER
   - Base weight: Node class (1, 4, or 8)
   - Signatory bonus: +10
   - Total: 11-18 voting power (vs 1-8 for regular nodes)

3. GOVERNANCE BENEFITS
   - 3x credits per block for Light TS nodes
   - Committee membership eligibility
   - Enhanced weight for healthcare proposals

4. TRUST & COMPLIANCE
   - Higher honesty probability (0.98 vs 0.95)
   - Annual verification renewal
   - Automated compliance tracking
        """
        )

        # Cleanup demonstration
        print("\n\n5. VERIFICATION MANAGEMENT")
        print("-" * 40)

        print("\nRefreshing verifications...")
        await integration.refresh_verifications()
        print("  Active nodes: {refresh_results['active_nodes']}")
        print("  Expired verifications: {refresh_results['expired_verifications']}")
        print("  Revoked nodes: {refresh_results['revoked_nodes']}")

        # Demonstrate revocation
        if registered_nodes:
            test_npi = registered_nodes[-1][0]["npi"]
            print("\nDemonstrating revocation for NPI {test_npi}...")

            success = await integration.revoke_provider_node(
                npi=test_npi, reason="Demonstration of revocation process"
            )

            if success:
                print("  ✓ Verification revoked")
                print("  ✓ Node removed from network")
                print("  ✓ Voting power updated")

        print("\n" + "=" * 80)
        print("HIPAA Fast-Track Demonstration Complete!")
        print("=" * 80)


if __name__ == "__main__":
    # Configure logging
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run demonstration
    asyncio.run(demonstrate_hipaa_fasttrack())
