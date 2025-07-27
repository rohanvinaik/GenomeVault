from typing import Any, Dict

"""
Example: Diabetes Risk Assessment with GenomeVault ZK Proofs
Demonstrates privacy-preserving medical risk evaluation.
"""

import asyncio

from genomevault_zk_integration import GenomeVaultZKSystem


async def main() -> None:
    """TODO: Add docstring for main"""
    """TODO: Add docstring for main"""
        """TODO: Add docstring for main"""
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

        print("🏥 Diabetes Risk Assessment Demo")
        print(f"Patient glucose: {glucose_reading} mg/dL (private)")
        print(f"Genetic risk: {genetic_risk_score} (private)")
        print(f"Alert triggers if both exceed thresholds")

        # Generate ZK proof
        result = await zk_system.zk_integration.prove_diabetes_risk_alert(
            glucose_reading=glucose_reading,
            risk_score=genetic_risk_score,
            glucose_threshold=glucose_threshold,
            risk_threshold=risk_threshold,
        )

        if result.success:
            print(f"\n✅ ZK Proof Generated:")
            print(f"   Proof size: {len(result.proof.proof_bytes)} bytes")
            print(f"   Generation time: {result.generation_time:.3f}s")
            print(f"   Verification time: {result.verification_time:.3f}s")
            print(f"   Alert condition: BOTH values exceed thresholds")
            print(f"   Privacy: Actual values never revealed")
        else:
            print(f"❌ Proof generation failed: {result.error_message}")

    finally:
        await zk_system.stop()


if __name__ == "__main__":
    asyncio.run(main())
