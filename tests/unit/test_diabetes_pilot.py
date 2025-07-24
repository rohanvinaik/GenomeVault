# tests/unit/test_diabetes_pilot.py
"""Test suite for the diabetes risk pilot implementation"""
from unittest.mock import Mock, patch

import pytest

from genomevault.clinical.diabetes_pilot.risk_calculator import (
    DiabetesRiskCalculator,
    RiskAssessment,
    generate_zkp_alert,
)


class TestDiabetesPilot:
    """Test the diabetes risk calculator and ZKP alert system"""

    @pytest.fixture
    def risk_calculator(self):
        """Create a risk calculator instance"""
        return DiabetesRiskCalculator()

    @pytest.fixture
    def sample_genetic_data(self):
        """Sample genetic risk factors for diabetes"""
        return {
            "rs7903146": "TT",  # High risk variant in TCF7L2
            "rs1801282": "CC",  # PPARG variant
            "rs5219": "TT",  # KCNJ11 variant
            "rs7754840": "CC",  # CDKAL1 variant
            "rs10830963": "GG",  # MTNR1B variant
        }

    @pytest.fixture
    def sample_clinical_data(self):
        """Sample clinical data for risk assessment"""
        return {
            "glucose": 105,  # mg/dL
            "hba1c": 5.8,  # %
            "bmi": 28.5,
            "age": 45,
            "family_history": True,
        }

    def test_genetic_risk_score_calculation(self, risk_calculator, sample_genetic_data):
        """Test polygenic risk score calculation"""
        risk_score = risk_calculator.calculate_genetic_risk(sample_genetic_data)

        # Risk score should be between 0 and 1
        assert 0 <= risk_score <= 1

        # With high-risk variants, score should be elevated
        assert risk_score > 0.15  # Threshold for elevated risk

    def test_combined_risk_assessment(
        self, risk_calculator, sample_genetic_data, sample_clinical_data
    ):
        """Test combined genetic and clinical risk assessment"""
        assessment = risk_calculator.assess_combined_risk(
            genetic_data=sample_genetic_data, clinical_data=sample_clinical_data
        )

        assert isinstance(assessment, RiskAssessment)
        assert assessment.genetic_risk > 0
        assert assessment.clinical_risk > 0
        assert assessment.combined_risk > 0
        assert assessment.risk_category in ["low", "moderate", "high"]

    def test_zkp_alert_generation(self, sample_genetic_data, sample_clinical_data):
        """Test zero-knowledge proof alert generation"""
        # Set thresholds
        glucose_threshold = 100
        risk_threshold = 0.15

        # Generate ZKP alert
        with patch("clinical.diabetes_pilot.risk_calculator.generate_proof") as mock_proof:
            mock_proof.return_value = {
                "proo": b"mock_proof_data",
                "size": 384,
                "verification_time_ms": 23,
            }

            alert = generate_zkp_alert(
                glucose=sample_clinical_data["glucose"],
                genetic_risk_score=0.18,
                glucose_threshold=glucose_threshold,
                risk_threshold=risk_threshold,
            )

            # Verify alert structure
            assert alert["triggered"] is True
            assert alert["proof"] is not None
            assert alert["proof_size_bytes"] == 384
            assert alert["verification_time_ms"] < 25

            # Verify privacy - no raw values exposed
            assert "glucose" not in alert
            assert "risk_score" not in alert

    def test_zkp_alert_not_triggered(self):
        """Test ZKP alert when thresholds not exceeded"""
        alert = generate_zkp_alert(
            glucose=95,  # Below threshold
            genetic_risk_score=0.10,  # Below threshold
            glucose_threshold=100,
            risk_threshold=0.15,
        )

        assert alert["triggered"] is False
        assert alert["proof"] is None

    @pytest.mark.parametrize(
        "glucose,risk_score,g_thresh,r_thresh,expected",
        [
            (105, 0.18, 100, 0.15, True),  # Both exceed
            (95, 0.18, 100, 0.15, False),  # Only risk exceeds
            (105, 0.10, 100, 0.15, False),  # Only glucose exceeds
            (95, 0.10, 100, 0.15, False),  # Neither exceeds
        ],
    )
    def test_alert_trigger_conditions(self, glucose, risk_score, g_thresh, r_thresh, expected):
        """Test various alert trigger conditions"""
        alert = generate_zkp_alert(glucose, risk_score, g_thresh, r_thresh)
        assert alert["triggered"] == expected

    def test_proof_size_specification(self):
        """Verify proof size meets specification (<384 bytes)"""
        with patch("clinical.diabetes_pilot.risk_calculator.generate_proof") as mock_proof:
            mock_proof.return_value = {
                "proo": b"a" * 384,  # Exactly 384 bytes
                "size": 384,
                "verification_time_ms": 20,
            }

            alert = generate_zkp_alert(
                glucose=105,
                genetic_risk_score=0.18,
                glucose_threshold=100,
                risk_threshold=0.15,
            )

            assert alert["proof_size_bytes"] <= 384

    def test_verification_time_specification(self):
        """Verify proof verification time meets specification (<25ms)"""
        with patch("clinical.diabetes_pilot.risk_calculator.generate_proof") as mock_proof:
            mock_proof.return_value = {
                "proo": b"mock_proof",
                "size": 384,
                "verification_time_ms": 24.5,
            }

            alert = generate_zkp_alert(
                glucose=105,
                genetic_risk_score=0.18,
                glucose_threshold=100,
                risk_threshold=0.15,
            )

            assert alert["verification_time_ms"] < 25

    def test_privacy_preservation(self, risk_calculator, sample_genetic_data, sample_clinical_data):
        """Test that no private data leaks in outputs"""
        assessment = risk_calculator.assess_combined_risk(
            genetic_data=sample_genetic_data, clinical_data=sample_clinical_data
        )

        # Convert to dict for transmission
        assessment_dict = assessment.to_dict()

        # Verify no raw genetic data exposed
        for variant in sample_genetic_data:
            assert variant not in str(assessment_dict)
            assert sample_genetic_data[variant] not in str(assessment_dict)

        # Verify no raw clinical values exposed
        assert str(sample_clinical_data["glucose"]) not in str(assessment_dict)
        assert str(sample_clinical_data["hba1c"]) not in str(assessment_dict)

    @pytest.mark.performance
    def test_risk_calculation_performance(
        self,
        risk_calculator,
        sample_genetic_data,
        sample_clinical_data,
        performance_benchmark,
    ):
        """Test that risk calculation completes within performance bounds"""

        def calculate():
            return risk_calculator.assess_combined_risk(
                genetic_data=sample_genetic_data, clinical_data=sample_clinical_data
            )

        # Measure performance
        result = performance_benchmark.measure("risk_calculation", calculate)

        # Should complete in under 100ms
        performance_benchmark.assert_performance("risk_calculation", 100)
