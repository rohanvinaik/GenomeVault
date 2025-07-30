from genomevault.observability.logging import configure_logging

logger = configure_logging()
"""
Example: Diabetes Risk Assessment with GenomeVault ZK Proofs
Demonstrates privacy-preserving medical risk evaluation.
"""

import asyncio

from genomevault_zk_integration import GenomeVaultZKSystem


async def main():
    # Initialize ZK system
    zk_system = GenomeVaultZKSystem(max_workers=2)
    await zk_system.start()

    try:
        # Patient data (stays private)
        glucose_reading = 140.0  # mg/dL
        genetic_risk_score = 0.82  # From PRS calculation

        # Clinical thresholds (public)
        glucose_threshold = 126.0  # Diabetes threshold
        risk_threshold = 0.75  # High genetic risk threshold

        logger.info("🏥 Diabetes Risk Assessment Demo")
        logger.info(f"Patient glucose: {glucose_reading} mg/dL (private)")
        logger.info(f"Genetic risk: {genetic_risk_score} (private)")
        logger.info("Alert triggers if both exceed thresholds")

        # Generate ZK proof
        result = await zk_system.zk_integration.prove_diabetes_risk_alert(
            glucose_reading=glucose_reading,
            risk_score=genetic_risk_score,
            glucose_threshold=glucose_threshold,
            risk_threshold=risk_threshold,
        )

        if result.success:
            logger.info("\n✅ ZK Proof Generated:")
            logger.info(f"   Proof size: {len(result.proof.proof_bytes)} bytes")
            logger.info(f"   Generation time: {result.generation_time:.3f}s")
            logger.info(f"   Verification time: {result.verification_time:.3f}s")
            logger.info("   Alert condition: BOTH values exceed thresholds")
            logger.info("   Privacy: Actual values never revealed")
        else:
            logger.info(f"❌ Proof generation failed: {result.error_message}")

    finally:
        await zk_system.stop()


if __name__ == "__main__":
    asyncio.run(main())
