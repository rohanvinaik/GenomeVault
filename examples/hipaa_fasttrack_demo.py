"""
HIPAA Fast-Track Example

Demonstrates the complete HIPAA verification and governance integration flow.
"""

import asyncio
from datetime import datetime, timedelta

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
            print(f"\nRegistering: {provider['name']}")
            print(f"  NPI: {provider['npi']}")
            print(f"  Type: {provider['description']}")

            credentials = HIPAACredentials(
                npi=provider["npi"],
                baa_hash="a" * 64,  # Simulated BAA hash
                risk_analysis_hash="b" * 64,  # Simulated risk analysis
                hsm_serial=f"HSM-{provider['npi'][-6:]}",
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

                print(f"  ✓ Verification successful!")
                print(f"  ✓ Node ID: {node.node_id}")
                print(f"  ✓ Voting power: {node.voting_power}")

                # Add to HIPAA committee
                governance.committees[CommitteeType.SCIENTIFIC_ADVISORY].add_member(node.node_id)

            except Exception as e:
                print(f"  ✗ Registration failed: {e}")

        print(f"\n\nTotal governance voting power: {governance.total_voting_power}")

        print("\n\n2. GOVERNANCE PARTICIPATION")
        print("-" * 40)

        # Create a clinical protocol proposal
        if registered_nodes:
            proposer_node = registered_nodes[0][1]  # Metro General Hospital

            print(f"\n{registered_nodes[0][0]['name']} creating clinical data proposal...")

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

            print(f"  ✓ Proposal created: {proposal.proposal_id}")
            print(f"  Voting period: {proposal.voting_period.days} days")
            print(f"  Required quorum: {proposal.quorum_required:.0%}")
            print(f"  Approval threshold: {proposal.approval_threshold:.0%}")

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

                vote = governance.vote(
                    proposal_id=proposal.proposal_id,
                    voter=node.node_id,
                    choice=choice,
                    voting_power=node.voting_power,
                )

                print(f"\n{provider['name']} voted: {choice}")
                print(f"  Vote weight: {vote.vote_weight}")

                # Show enhanced weight for committee members
                if governance._get_committee_multiplier(node.node_id, proposal.proposal_type) > 1:
                    print(f"  ✓ Committee bonus applied!")

            # Add some non-HIPAA votes for comparison
            print("\n\nNon-HIPAA participants voting...")
            for i in range(3):
                voter_id = f"regular_node_{i}"
                choice = "yes" if i < 2 else "no"

                vote = governance.vote(
                    proposal_id=proposal.proposal_id,
                    voter=voter_id,
                    choice=choice,
                    voting_power=50,  # Lower voting power
                )

                print(f"Regular Node {i} voted: {choice} (weight: {vote.vote_weight})")

            # Check results
            print("\n\nProposal Results:")
            print("-" * 40)
            details = governance.get_proposal_details(proposal.proposal_id)

            print(f"Total votes cast: {details['vote_count']}")
            print(f"Yes votes: {details['votes']['yes']:.1f}")
            print(f"No votes: {details['votes']['no']:.1f}")
            print(f"Abstentions: {details['votes']['abstain']:.1f}")
            print(f"Current approval: {details['requirements']['current_approval']:.1%}")
            print(f"Has quorum: {details['requirements']['has_quorum']}")

            # Demonstrate the voting power difference
            print("\n\n3. VOTING POWER COMPARISON")
            print("-" * 40)
            print("\nHIPAA-Verified Nodes (Trusted Signatories):")
            for provider, node in registered_nodes:
                print(
                    f"  {provider['name']:30} | Class: {node.node_type.name:7} | Power: {node.voting_power:3}"
                )

            print("\nRegular Nodes (Non-signatories):")
            print(f"  {'Regular Light Node':30} | Class: {'LIGHT':7} | Power: {1:3}")
            print(f"  {'Regular Full Node':30} | Class: {'FULL':7} | Power: {4:3}")
            print(f"  {'Regular Archive Node':30} | Class: {'ARCHIVE':7} | Power: {8:3}")

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
        refresh_results = await integration.refresh_verifications()
        print(f"  Active nodes: {refresh_results['active_nodes']}")
        print(f"  Expired verifications: {refresh_results['expired_verifications']}")
        print(f"  Revoked nodes: {refresh_results['revoked_nodes']}")

        # Demonstrate revocation
        if registered_nodes:
            test_npi = registered_nodes[-1][0]["npi"]
            print(f"\nDemonstrating revocation for NPI {test_npi}...")

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
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run demonstration
    asyncio.run(demonstrate_hipaa_fasttrack())
