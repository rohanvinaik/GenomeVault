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

    logger.debug("=" * 80)
    logger.debug("GenomeVault HIPAA Fast-Track Demonstration")
    logger.debug("=" * 80)

    # Initialize components
    async with CMSNPIRegistry() as registry:
        verifier = HIPAAVerifier(npi_registry=registry)
        governance = GovernanceSystem()
        integration = HIPAANodeIntegration(verifier, governance)

        # Setup HIPAA governance features
        HIPAAGovernanceIntegration.create_hipaa_committee(governance)
        HIPAAGovernanceIntegration.add_hipaa_proposal_types(governance)

        logger.debug("\n1. HEALTHCARE PROVIDER REGISTRATION")
        logger.debug("-" * 40)

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
            logger.debug("\nRegistering: {provider['name']}")
            logger.debug("  NPI: {provider['npi']}")
            logger.debug("  Type: {provider['description']}")

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

                logger.info("  ✓ Verification successful!")
                logger.info("  ✓ Node ID: {node.node_id}")
                logger.info("  ✓ Voting power: {node.voting_power}")

                # Add to HIPAA committee
                governance.committees[CommitteeType.SCIENTIFIC_ADVISORY].add_member(node.node_id)

            except Exception:
                logger.exception("Unhandled exception")
                logger.error("  ✗ Registration failed: {e}")
                raise

        logger.debug("\n\nTotal governance voting power: {governance.total_voting_power}")

        logger.debug("\n\n2. GOVERNANCE PARTICIPATION")
        logger.debug("-" * 40)

        # Create a clinical protocol proposal
        if registered_nodes:
            proposer_node = registered_nodes[0][1]  # Metro General Hospital

            logger.debug("\n{registered_nodes[0][0]['name']} creating clinical data proposal...")

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

            logger.info("  ✓ Proposal created: {proposal.proposal_id}")
            logger.debug("  Voting period: {proposal.voting_period.days} days")
            logger.debug("  Required quorum: {proposal.quorum_required:.0%}")
            logger.debug("  Approval threshold: {proposal.approval_threshold:.0%}")

            # Simulate voting
            logger.debug("\n\nSimulating governance votes...")
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

                logger.debug("\n{provider['name']} voted: {choice}")
                logger.debug("  Vote weight: {vote.vote_weight}")

                # Show enhanced weight for committee members
                if governance._get_committee_multiplier(node.node_id, proposal.proposal_type) > 1:
                    logger.info("  ✓ Committee bonus applied!")

            # Add some non-HIPAA votes for comparison
            logger.debug("\n\nNon-HIPAA participants voting...")
            for i in range(3):
                voter_id = "regular_node_{i}"
                choice = "yes" if i < 2 else "no"

                governance.vote(
                    proposal_id=proposal.proposal_id,
                    voter=voter_id,
                    choice=choice,
                    voting_power=50,  # Lower voting power
                )

                logger.debug("Regular Node {i} voted: {choice} (weight: {vote.vote_weight})")

            # Check results
            logger.debug("\n\nProposal Results:")
            logger.debug("-" * 40)
            governance.get_proposal_details(proposal.proposal_id)

            logger.debug("Total votes cast: {details['vote_count']}")
            logger.debug("Yes votes: {details['votes']['yes']:.1f}")
            logger.debug("No votes: {details['votes']['no']:.1f}")
            logger.debug("Abstentions: {details['votes']['abstain']:.1f}")
            logger.debug("Current approval: {details['requirements']['current_approval']:.1%}")
            logger.debug("Has quorum: {details['requirements']['has_quorum']}")

            # Demonstrate the voting power difference
            logger.debug("\n\n3. VOTING POWER COMPARISON")
            logger.debug("-" * 40)
            logger.debug("\nHIPAA-Verified Nodes (Trusted Signatories):")
            for provider, node in registered_nodes:
                print(
                    "  {provider['name']:30} | Class: {node.node_type.name:7} | Power: {node.voting_power:3}"
                )

            logger.debug("\nRegular Nodes (Non-signatories):")
            logger.debug("  {'Regular Light Node':30} | Class: {'LIGHT':7} | Power: {1:3}")
            logger.debug("  {'Regular Full Node':30} | Class: {'FULL':7} | Power: {4:3}")
            logger.debug("  {'Regular Archive Node':30} | Class: {'ARCHIVE':7} | Power: {8:3}")

            logger.info("\n✓ HIPAA providers receive +10 signatory weight!")

        logger.debug("\n\n4. BENEFITS SUMMARY")
        logger.debug("-" * 40)
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
        logger.debug("\n\n5. VERIFICATION MANAGEMENT")
        logger.debug("-" * 40)

        logger.debug("\nRefreshing verifications...")
        await integration.refresh_verifications()
        logger.debug("  Active nodes: {refresh_results['active_nodes']}")
        logger.debug("  Expired verifications: {refresh_results['expired_verifications']}")
        logger.debug("  Revoked nodes: {refresh_results['revoked_nodes']}")

        # Demonstrate revocation
        if registered_nodes:
            test_npi = registered_nodes[-1][0]["npi"]
            logger.debug("\nDemonstrating revocation for NPI {test_npi}...")

            success = await integration.revoke_provider_node(
                npi=test_npi, reason="Demonstration of revocation process"
            )

            if success:
                logger.info("  ✓ Verification revoked")
                logger.info("  ✓ Node removed from network")
                logger.info("  ✓ Voting power updated")

        logger.debug("\n" + "=" * 80)
        logger.info("HIPAA Fast-Track Demonstration Complete!")
        logger.debug("=" * 80)


if __name__ == "__main__":
    # Configure logging
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run demonstration
    asyncio.run(demonstrate_hipaa_fasttrack())
