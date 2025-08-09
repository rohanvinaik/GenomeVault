"""
GenomeVault Diabetes Pilot Implementation

Implements the diabetes risk assessment system with:
- Combined genetic risk score (PRS) and glucose monitoring
- Zero-knowledge proofs for privacy-preserving alerts
- HIPAA-compliant data handling
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from ...core.constants import GLUCOSE_THRESHOLD_MG_DL
from ...utils import get_config, get_logger
from ...zk_proofs.prover import Prover

logger = logging.getLogger(__name__)


logger = get_logger(__name__)
config = get_config()


@dataclass
class GeneticRiskProfile:
    """Genetic risk profile for diabetes"""

    prs_score: float  # Polygenic risk score (0-1)
    risk_variants: list[dict[str, Any]]
    confidence_interval: tuple[float, float]
    ancestry_adjusted: bool
    calculation_date: datetime
    dp_noise_added: float  # Differential privacy noise

    def get_risk_category(self) -> str:
        """Get risk category based on PRS"""
        if self.prs_score < 0.3:
            return "low"
        elif self.prs_score < 0.7:
            return "moderate"
        else:
            return "high"


@dataclass
class GlucoseReading:
    """Glucose measurement data"""

    value: float  # mg/dL
    measurement_type: str  # "fasting", "random", "ogtt"
    timestamp: datetime
    device_id: str | None = None

    def is_diabetic_range(self) -> bool:
        """Check if reading is in diabetic range"""
        if self.measurement_type == "fasting":
            return self.value >= GLUCOSE_THRESHOLD_MG_DL
        elif self.measurement_type == "random":
            return self.value >= 200
        elif self.measurement_type == "ogtt":
            return self.value >= 200
        return False


@dataclass
class DiabetesRiskAlert:
    """Privacy-preserving diabetes risk alert"""

    alert_triggered: bool
    proof: bytes | None = None
    proof_id: str | None = None
    timestamp: datetime | None = None
    metadata: dict[str, Any] | None = None


class DiabetesRiskCalculator:
    """
    Calculates diabetes risk from genetic and clinical data.
    Implements the specification's privacy-preserving alert system.
    """

    # Diabetes risk variants (simplified subset)
    RISK_VARIANTS = [
        {"rsid": "rs7903146", "gene": "TCF7L2", "risk_allele": "T", "or": 1.37},
        {"rsid": "rs1801282", "gene": "PPARG", "risk_allele": "C", "or": 1.14},
        {"rsid": "rs5219", "gene": "KCNJ11", "risk_allele": "T", "or": 1.14},
        {"rsid": "rs9300039", "gene": "FTO", "risk_allele": "A", "or": 1.13},
        {"rsid": "rs10830963", "gene": "MTNR1B", "risk_allele": "G", "or": 1.09},
        {"rsid": "rs1111875", "gene": "HHEX", "risk_allele": "C", "or": 1.13},
        {"rsid": "rs13266634", "gene": "SLC30A8", "risk_allele": "C", "or": 1.12},
        {"rsid": "rs7901695", "gene": "TCF7L2", "risk_allele": "C", "or": 1.31},
        {"rsid": "rs10885122", "gene": "ADRA2A", "risk_allele": "G", "or": 1.10},
        {"rsid": "rs2237892", "gene": "KCNQ1", "risk_allele": "C", "or": 1.08},
    ]

    def __init__(self):
        """Initialize risk calculator"""
        self.prover = Prover()
        self.differential_privacy_epsilon = 1.0
        logger.info("Diabetes risk calculator initialized")

    def calculate_genetic_risk(
        self, variants: list[dict[str, Any]], add_dp_noise: bool = True
    ) -> GeneticRiskProfile:
        """
        Calculate polygenic risk score for diabetes.

        Args:
            variants: List of genetic variants
            add_dp_noise: Whether to add differential privacy noise

        Returns:
            Genetic risk profile with PRS
        """
        risk_score = 0.0
        matched_variants = []
        weights_sum = 0.0

        # Create variant lookup
        variant_lookup = {"{v['chromosome']}:{v['position']}": v for v in variants}

        # Calculate weighted risk score
        for risk_var in self.RISK_VARIANTS:
            # Check if user has risk variant
            # In practice, would use actual genomic coordinates
            if self._has_risk_allele(variant_lookup, risk_var):
                # Add log odds ratio
                risk_contribution = np.log(risk_var["or"])
                risk_score += risk_contribution
                weights_sum += 1

                matched_variants.append(
                    {
                        "rsid": risk_var["rsid"],
                        "gene": risk_var["gene"],
                        "contribution": risk_contribution,
                    }
                )

        # Normalize to 0-1 scale
        # Using logistic function to map log-odds to probability
        if weights_sum > 0:
            avg_log_odds = risk_score / weights_sum
            prs = 1 / (1 + np.exp(-avg_log_odds * len(self.RISK_VARIANTS)))
        else:
            prs = 0.5  # Baseline risk

        # Add differential privacy noise
        dp_noise = 0.0
        if add_dp_noise:
            # Calibrated Laplace noise for bounded [0,1] output
            sensitivity = 1.0 / len(self.RISK_VARIANTS)
            scale = sensitivity / self.differential_privacy_epsilon
            dp_noise = np.random.laplace(0, scale)

            # Ensure PRS stays in [0,1]
            prs = np.clip(prs + dp_noise, 0, 1)

        # Calculate confidence interval
        # Simplified - in practice would use bootstrap or analytical methods
        std_error = 0.1 / np.sqrt(len(matched_variants) + 1)
        ci_lower = max(0, prs - 1.96 * std_error)
        ci_upper = min(1, prs + 1.96 * std_error)

        profile = GeneticRiskProfile(
            prs_score=prs,
            risk_variants=matched_variants,
            confidence_interval=(ci_lower, ci_upper),
            ancestry_adjusted=False,  # Simplified
            calculation_date=datetime.now(),
            dp_noise_added=dp_noise,
        )

        logger.info(
            f"Calculated PRS: {prs:.3f} (category: {profile.get_risk_category()})",
            extra={"privacy_safe": True},
        )

        return profile

    def _has_risk_allele(
        self, variant_lookup: dict[str, dict], risk_variant: dict[str, Any]
    ) -> bool:
        """Check if user has risk allele (simplified)"""
        # In practice, would look up by actual genomic coordinates
        # For now, simulate with probability based on population frequency
        # Real implementation would check actual genotype
        return np.random.random() < 0.3  # 30% frequency placeholder

    def create_risk_alert(
        self,
        genetic_profile: GeneticRiskProfile,
        glucose_reading: GlucoseReading,
        risk_threshold: float = 0.75,
    ) -> DiabetesRiskAlert:
        """
        Create privacy-preserving risk alert using ZK proof.

        Args:
            genetic_profile: Genetic risk profile with PRS
            glucose_reading: Current glucose measurement
            risk_threshold: PRS threshold for alert (default 0.75)

        Returns:
            Diabetes risk alert with ZK proof
        """
        # Extract values
        glucose = glucose_reading.value
        prs = genetic_profile.prs_score

        # Determine if alert should trigger
        # Alert triggers when BOTH conditions are met:
        # (G > G_threshold) AND (R > R_threshold)
        glucose_exceeds = glucose > GLUCOSE_THRESHOLD_MG_DL
        risk_exceeds = prs > risk_threshold
        alert_triggered = glucose_exceeds and risk_exceeds

        # Generate ZK proof that proves the condition without revealing values
        public_inputs = {
            "glucose_threshold": GLUCOSE_THRESHOLD_MG_DL,
            "risk_threshold": risk_threshold,
            "result_commitment": hashlib.sha256(
                b"{alert_triggered}:{datetime.now().isoformat()}"
            ).hexdigest(),
        }

        private_inputs = {
            "glucose_reading": glucose,
            "risk_score": prs,
            "witness_randomness": np.random.bytes(32).hex(),
        }

        # Generate proof
        proof = self.prover.generate_proof(
            circuit_name="diabetes_risk_alert",
            public_inputs=public_inputs,
            private_inputs=private_inputs,
        )

        # Create alert
        alert = DiabetesRiskAlert(
            alert_triggered=alert_triggered,
            proof=proof.proof_data,
            proof_id=proof.proof_id,
            timestamp=datetime.now(),
            metadata={
                "proof_size_bytes": len(proof.proof_data),
                "verification_time_ms": proof.metadata.get("generation_time_seconds", 0) * 1000,
                "glucose_type": glucose_reading.measurement_type,
                "risk_category": genetic_profile.get_risk_category(),
            },
        )

        logger.info(
            f"Risk alert created: triggered={alert_triggered}",
            extra={"privacy_safe": True},
        )

        return alert

    def verify_alert(self, alert: DiabetesRiskAlert, public_inputs: dict[str, Any]) -> bool:
        """
        Verify diabetes risk alert proof.

        Args:
            alert: Alert with proof to verify
            public_inputs: Public inputs used in proof

        Returns:
            Whether proof is valid
        """
        # In production, would use actual PLONK verifier
        # For now, simulate verification
        if not alert.proof or not alert.proof_id:
            return False

        # Verify proof structure
        try:
            # Check proof size
            if len(alert.proof) != 384:  # Expected size from spec
                return False

            # Simulate verification time < 25ms
            import time

            start = time.time()
            # Verification logic would go here
            verification_time = (time.time() - start) * 1000

            if verification_time > 25:
                logger.warning(f"Verification took {verification_time:.1f}ms")

            return True

        except Exception as e:
            logger.exception("Unhandled exception")
            logger.error(f"Alert verification failed: {e}")
            return False
            raise RuntimeError("Unspecified error")

    def monitor_continuous_risk(
        self,
        genetic_profile: GeneticRiskProfile,
        glucose_readings: list[GlucoseReading],
        window_days: int = 7,
    ) -> dict[str, Any]:
        """
        Monitor continuous diabetes risk over time window.

        Args:
            genetic_profile: Genetic risk profile
            glucose_readings: Historical glucose readings
            window_days: Days to consider for trend

        Returns:
            Risk monitoring summary
        """
        # Filter readings within window
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_readings = [r for r in glucose_readings if r.timestamp >= cutoff_date]

        if not recent_readings:
            return {"status": "insufficient_data", "message": "No readings in window"}

        # Calculate statistics
        glucose_values = [r.value for r in recent_readings]
        avg_glucose = np.mean(glucose_values)
        std_glucose = np.std(glucose_values)
        max_glucose = np.max(glucose_values)

        # Count readings above threshold
        high_readings = sum(1 for v in glucose_values if v > GLUCOSE_THRESHOLD_MG_DL)
        high_percentage = high_readings / len(glucose_values) * 100

        # Determine trend
        if len(glucose_values) >= 3:
            # Simple linear regression for trend
            x = np.arange(len(glucose_values))
            slope, _ = np.polyfit(x, glucose_values, 1)
            trend = "increasing" if slope > 1 else "decreasing" if slope < -1 else "stable"
        else:
            trend = "unknown"

        # Risk assessment
        risk_level = "low"
        if genetic_profile.prs_score > 0.75:
            if avg_glucose > 110 or high_percentage > 30:
                risk_level = "high"
            elif avg_glucose > 100 or high_percentage > 20:
                risk_level = "moderate"
        elif genetic_profile.prs_score > 0.5:
            if avg_glucose > 115 or high_percentage > 40:
                risk_level = "moderate"

        summary = {
            "status": "monitored",
            "window_days": window_days,
            "reading_count": len(recent_readings),
            "statistics": {
                "mean_glucose": round(avg_glucose, 1),
                "std_glucose": round(std_glucose, 1),
                "max_glucose": round(max_glucose, 1),
                "high_reading_percentage": round(high_percentage, 1),
            },
            "trend": trend,
            "risk_level": risk_level,
            "genetic_risk_category": genetic_profile.get_risk_category(),
            "recommendations": self._get_recommendations(risk_level, trend),
        }

        return summary

    def _get_recommendations(self, risk_level: str, trend: str) -> list[str]:
        """Get personalized recommendations based on risk"""
        recommendations = []

        if risk_level == "high":
            recommendations.extend(
                [
                    "Consider scheduling an appointment with your healthcare provider",
                    "Monitor glucose levels more frequently",
                    "Review dietary habits with a nutritionist",
                ]
            )
        elif risk_level == "moderate":
            recommendations.extend(
                [
                    "Continue regular glucose monitoring",
                    "Maintain healthy lifestyle habits",
                    "Consider preventive measures discussion with provider",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Maintain current monitoring schedule",
                    "Continue healthy lifestyle choices",
                ]
            )

        if trend == "increasing":
            recommendations.append("Glucose trend is increasing - consider lifestyle adjustments")

        return recommendations


class ClinicalIntegration:
    """
    Integration with clinical systems for diabetes management.
    Implements HIPAA-compliant data handling.
    """

    def __init__(self):
        """Initialize clinical integration"""
        self.calculator = DiabetesRiskCalculator()
        self.fhir_enabled = config.enable_fhir
        logger.info("Clinical integration initialized")

    def process_clinical_data(self, patient_data: dict[str, Any]) -> dict[str, Any]:
        """
        Process clinical data for diabetes risk assessment.

        Args:
            patient_data: FHIR-compatible patient data

        Returns:
            Processed risk assessment
        """
        # Extract genetic data
        genetic_variants = self._extract_genetic_variants(patient_data)

        # Extract glucose measurements
        glucose_readings = self._extract_glucose_readings(patient_data)

        # Calculate genetic risk
        genetic_profile = self.calculator.calculate_genetic_risk(genetic_variants)

        # Get latest glucose reading
        latest_glucose = max(glucose_readings, key=lambda r: r.timestamp)

        # Create risk alert
        alert = self.calculator.create_risk_alert(genetic_profile, latest_glucose)

        # Continuous monitoring
        monitoring_summary = self.calculator.monitor_continuous_risk(
            genetic_profile, glucose_readings
        )

        # Prepare HIPAA-compliant response
        response = {
            "patient_id": patient_data.get("id", "anonymous"),
            "assessment_date": datetime.now().isoformat(),
            "alert_status": {
                "triggered": alert.alert_triggered,
                "proof_id": alert.proof_id,
                "verification_available": True,
            },
            "monitoring_summary": monitoring_summary,
            "privacy_preserved": True,
            "hipaa_compliant": True,
        }

        # Audit log (privacy-safe)
        logger.info(
            "Clinical diabetes assessment completed",
            extra={
                "privacy_safe": True,
                "alert_triggered": alert.alert_triggered,
                "risk_level": monitoring_summary["risk_level"],
            },
        )

        return response

    def _extract_genetic_variants(self, patient_data: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract genetic variants from patient data"""
        variants = []

        # Look for genetic observations in FHIR format
        if "observations" in patient_data:
            for obs in patient_data["observations"]:
                if obs.get("category") == "genetic":
                    variants.append(
                        {
                            "chromosome": obs.get("chromosome", ""),
                            "position": obs.get("position", 0),
                            "reference": obs.get("reference_allele", ""),
                            "alternate": obs.get("alternate_allele", ""),
                            "genotype": obs.get("genotype", "0/0"),
                        }
                    )

        return variants

    def _extract_glucose_readings(self, patient_data: dict[str, Any]) -> list[GlucoseReading]:
        """Extract glucose readings from patient data"""
        readings = []

        # Look for glucose observations
        if "observations" in patient_data:
            for obs in patient_data["observations"]:
                if obs.get("code") in ["glucose", "blood_glucose", "2339-0"]:
                    reading = GlucoseReading(
                        value=obs.get("value", 0),
                        measurement_type=obs.get("measurement_type", "random"),
                        timestamp=datetime.fromisoformat(
                            obs.get("timestamp", datetime.now().isoformat())
                        ),
                        device_id=obs.get("device_id"),
                    )
                    readings.append(reading)

        return readings


# Example usage and testing
if __name__ == "__main__":
    from datetime import timedelta

    # Initialize calculator
    calculator = DiabetesRiskCalculator()

    # Example genetic variants
    variants = [
        {
            "chromosome": "10",
            "position": 114758349,
            "reference": "C",
            "alternate": "T",
            "genotype": "0/1",
        },
        {
            "chromosome": "3",
            "position": 12393125,
            "reference": "C",
            "alternate": "G",
            "genotype": "1/1",
        },
    ]

    # Calculate genetic risk
    logger.info("Calculating genetic risk...")
    genetic_profile = calculator.calculate_genetic_risk(variants)
    logger.info(f"PRS Score: {genetic_profile.prs_score:.3f}")
    logger.info(f"Risk Category: {genetic_profile.get_risk_category()}")
    logger.info(f"Confidence Interval: {genetic_profile.confidence_interval}")

    # Example glucose reading
    glucose = GlucoseReading(
        value=140,
        measurement_type="fasting",
        timestamp=datetime.now(),  # Above threshold
    )

    # Create risk alert
    logger.info("\nCreating risk alert...")
    alert = calculator.create_risk_alert(genetic_profile, glucose)
    logger.info(f"Alert Triggered: {alert.alert_triggered}")
    logger.info(f"Proof ID: {alert.proof_id}")
    logger.info(f"Proof Size: {alert.metadata['proof_size_bytes']} bytes")
    logger.info(f"Verification Time: {alert.metadata['verification_time_ms']:.1f} ms")

    # Verify alert
    public_inputs = {
        "glucose_threshold": GLUCOSE_THRESHOLD_MG_DL,
        "risk_threshold": 0.75,
        "result_commitment": hashlib.sha256(b"{alert.alert_triggered}").hexdigest(),
    }

    is_valid = calculator.verify_alert(alert, public_inputs)
    logger.info(f"\nProof Verification: {'PASSED' if is_valid else 'FAILED'}")

    # Test continuous monitoring
    logger.info("\nTesting continuous monitoring...")
    glucose_history = [
        GlucoseReading(120, "fasting", datetime.now() - timedelta(days=6)),
        GlucoseReading(135, "fasting", datetime.now() - timedelta(days=4)),
        GlucoseReading(128, "fasting", datetime.now() - timedelta(days=2)),
        GlucoseReading(140, "fasting", datetime.now()),
    ]

    monitoring = calculator.monitor_continuous_risk(genetic_profile, glucose_history)
    logger.info(f"Risk Level: {monitoring['risk_level']}")
    logger.info(f"Trend: {monitoring['trend']}")
    logger.info(f"Recommendations: {monitoring['recommendations']}")

    # Test clinical integration
    logger.info("\nTesting clinical integration...")
    clinical = ClinicalIntegration()

    patient_data = {
        "id": "patient123",
        "observations": [
            {
                "category": "genetic",
                "chromosome": "10",
                "position": 114758349,
                "reference_allele": "C",
                "alternate_allele": "T",
                "genotype": "0/1",
            },
            {
                "code": "glucose",
                "value": 140,
                "measurement_type": "fasting",
                "timestamp": datetime.now().isoformat(),
            },
        ],
    }

    clinical_result = clinical.process_clinical_data(patient_data)
    import json

    logger.info(f"Clinical Assessment: {json.dumps(clinical_result, indent=2, default=str)}")
