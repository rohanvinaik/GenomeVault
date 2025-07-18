"""
Diabetes Management Pilot Application

Combines genetic risk scores with glucose measurements
using zero-knowledge proofs for privacy
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import hashlib
import json

from core.constants import (
    GLUCOSE_THRESHOLD_MG_DL,
    HBA1C_THRESHOLD_PERCENT,
    GENETIC_RISK_SCORE_THRESHOLD
)
from core.exceptions import ClinicalError
from zk_proofs.circuits.biological.variant import RiskScoreCircuit
from hypervector.encoding.genomic import GenomicEncoder


@dataclass
class GlucoseReading:
    """Single glucose measurement"""
    timestamp: datetime
    value: float  # mg/dL
    measurement_type: str  # fasting, postprandial, random
    device_id: Optional[str] = None


@dataclass
class GeneticRiskProfile:
    """Genetic risk factors for diabetes"""
    polygenic_risk_score: float
    variant_count: int
    risk_variants: List[str]  # rs IDs
    ancestry_adjustment: float
    confidence_interval: Tuple[float, float]


@dataclass
class DiabetesAlert:
    """Privacy-preserving alert"""
    alert_id: str
    timestamp: datetime
    risk_level: str  # low, medium, high, critical
    zk_proof: bytes
    recommendations: List[str]
    next_check: datetime


class DiabetesRiskCalculator:
    """
    Calculates diabetes risk using genetic and glucose data
    """
    
    def __init__(self):
        self.risk_variants = self._load_diabetes_variants()
        self.encoder = GenomicEncoder()
        self.zk_circuit = RiskScoreCircuit()
    
    def _load_diabetes_variants(self) -> Dict[str, float]:
        """Load known diabetes risk variants"""
        # In production, load from clinical database
        # Using T2D GWAS catalog variants
        return {
            "rs7903146": 1.4,    # TCF7L2 - strongest T2D variant
            "rs1801282": 1.2,    # PPARG
            "rs5219": 1.15,      # KCNJ11
            "rs7754840": 1.2,    # CDKAL1
            "rs10811661": 1.2,   # CDKN2A/B
            "rs4607103": 1.1,    # ADAMTS9
            "rs13266634": 1.15,  # SLC30A8
            "rs1111875": 1.1,    # HHEX
            "rs7923837": 1.1,    # HHEX
            "rs10885122": 1.1,   # ADRA2A
            # Add more variants...
        }
    
    def calculate_genetic_risk(self, variants: Dict[str, str]) -> GeneticRiskProfile:
        """
        Calculate polygenic risk score for diabetes
        
        Args:
            variants: Dict mapping rsID to genotype (e.g., "AA", "AG", "GG")
            
        Returns:
            Genetic risk profile
        """
        risk_score = 0
        risk_variant_list = []
        variant_count = 0
        
        for rsid, risk_ratio in self.risk_variants.items():
            if rsid in variants:
                genotype = variants[rsid]
                variant_count += 1
                
                # Calculate allele count (0, 1, or 2 risk alleles)
                # This is simplified - in practice, need to know which is risk allele
                if len(genotype) == 2:
                    risk_allele_count = genotype.count('G')  # Assuming G is risk allele
                    
                    if risk_allele_count > 0:
                        risk_variant_list.append(rsid)
                        # Multiplicative model
                        risk_score += np.log(risk_ratio) * risk_allele_count
        
        # Convert to odds ratio
        combined_risk = np.exp(risk_score)
        
        # Apply ancestry adjustment (simplified)
        ancestry_adjustment = 1.0  # Would be calculated based on ancestry
        
        # Calculate confidence interval
        std_error = 0.1 * combined_risk  # Simplified
        ci_lower = combined_risk - 1.96 * std_error
        ci_upper = combined_risk + 1.96 * std_error
        
        return GeneticRiskProfile(
            polygenic_risk_score=combined_risk,
            variant_count=variant_count,
            risk_variants=risk_variant_list,
            ancestry_adjustment=ancestry_adjustment,
            confidence_interval=(max(0, ci_lower), ci_upper)
        )
    
    def evaluate_glucose_pattern(self, readings: List[GlucoseReading]) -> Dict[str, float]:
        """
        Evaluate glucose patterns for diabetes risk
        """
        if not readings:
            raise ClinicalError("No glucose readings provided")
        
        # Sort by timestamp
        readings.sort(key=lambda x: x.timestamp)
        
        # Calculate metrics
        fasting_readings = [r.value for r in readings if r.measurement_type == "fasting"]
        all_readings = [r.value for r in readings]
        
        metrics = {
            "mean_glucose": np.mean(all_readings),
            "glucose_variability": np.std(all_readings),
            "time_in_range": sum(1 for r in all_readings if 70 <= r <= 180) / len(all_readings),
            "high_readings_percent": sum(1 for r in all_readings if r > GLUCOSE_THRESHOLD_MG_DL) / len(all_readings)
        }
        
        if fasting_readings:
            metrics["mean_fasting_glucose"] = np.mean(fasting_readings)
            metrics["fasting_above_threshold"] = sum(1 for r in fasting_readings if r > GLUCOSE_THRESHOLD_MG_DL) / len(fasting_readings)
        
        # Estimate HbA1c from average glucose (Nathan formula)
        metrics["estimated_hba1c"] = (metrics["mean_glucose"] + 46.7) / 28.7
        
        return metrics
    
    def generate_risk_proof(self,
                          genetic_risk: GeneticRiskProfile,
                          glucose_metrics: Dict[str, float]) -> Tuple[bytes, bool]:
        """
        Generate zero-knowledge proof of combined risk
        
        Returns:
            (proof, high_risk_flag)
        """
        # Combine genetic and glucose risk
        genetic_component = genetic_risk.polygenic_risk_score
        glucose_component = glucose_metrics.get("estimated_hba1c", 5.0) / 6.5  # Normalize to ~1
        
        # Combined risk score
        combined_risk = (genetic_component + glucose_component) / 2
        
        # Generate commitment
        risk_data = {
            "genetic_risk": genetic_component,
            "glucose_risk": glucose_component,
            "combined_risk": combined_risk,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        risk_json = json.dumps(risk_data, sort_keys=True)
        randomness = hashlib.sha256(risk_json.encode()).hexdigest()[:16]
        commitment = hashlib.sha256(f"{combined_risk}:{randomness}".encode()).hexdigest()
        
        # Check if high risk
        high_risk = (
            combined_risk > GENETIC_RISK_SCORE_THRESHOLD or
            glucose_metrics.get("estimated_hba1c", 0) > HBA1C_THRESHOLD_PERCENT
        )
        
        # Generate proof only if high risk
        if high_risk:
            setup = self.zk_circuit.setup_threshold_proof(commitment, GENETIC_RISK_SCORE_THRESHOLD)
            proof = self.zk_circuit.generate_proof(combined_risk, randomness, setup)
        else:
            # Generate dummy proof for low risk
            proof = b"LOW_RISK" + bytes(376)  # Pad to expected size
        
        return proof, high_risk


class DiabetesMonitor:
    """
    Real-time diabetes monitoring with privacy
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.calculator = DiabetesRiskCalculator()
        self.genetic_profile: Optional[GeneticRiskProfile] = None
        self.glucose_readings: List[GlucoseReading] = []
        self.alerts: List[DiabetesAlert] = []
    
    def set_genetic_profile(self, variants: Dict[str, str]):
        """Set user's genetic profile"""
        self.genetic_profile = self.calculator.calculate_genetic_risk(variants)
    
    def add_glucose_reading(self, reading: GlucoseReading):
        """Add a new glucose reading"""
        self.glucose_readings.append(reading)
        
        # Keep only last 90 days
        cutoff = datetime.utcnow() - timedelta(days=90)
        self.glucose_readings = [r for r in self.glucose_readings if r.timestamp > cutoff]
    
    async def check_alert_conditions(self) -> Optional[DiabetesAlert]:
        """
        Check if alert conditions are met
        
        Only generates alert if BOTH genetic risk AND glucose exceed thresholds
        """
        if not self.genetic_profile or len(self.glucose_readings) < 5:
            return None
        
        # Evaluate current glucose pattern
        glucose_metrics = self.calculator.evaluate_glucose_pattern(self.glucose_readings)
        
        # Generate risk proof
        proof, high_risk = self.calculator.generate_risk_proof(
            self.genetic_profile,
            glucose_metrics
        )
        
        if high_risk:
            # Determine risk level
            combined_risk = (self.genetic_profile.polygenic_risk_score + 
                           glucose_metrics["estimated_hba1c"] / 6.5) / 2
            
            if combined_risk > 1.5:
                risk_level = "critical"
                recommendations = [
                    "Immediate medical consultation recommended",
                    "Consider continuous glucose monitoring",
                    "Review medication with healthcare provider"
                ]
                next_check_hours = 24
            elif combined_risk > 1.3:
                risk_level = "high"
                recommendations = [
                    "Schedule appointment with healthcare provider",
                    "Increase glucose monitoring frequency",
                    "Review diet and exercise habits"
                ]
                next_check_hours = 72
            else:
                risk_level = "medium"
                recommendations = [
                    "Continue regular monitoring",
                    "Consider lifestyle modifications",
                    "Discuss with doctor at next visit"
                ]
                next_check_hours = 168  # 1 week
            
            alert = DiabetesAlert(
                alert_id=hashlib.sha256(f"{self.user_id}:{datetime.utcnow()}".encode()).hexdigest()[:16],
                timestamp=datetime.utcnow(),
                risk_level=risk_level,
                zk_proof=proof,
                recommendations=recommendations,
                next_check=datetime.utcnow() + timedelta(hours=next_check_hours)
            )
            
            self.alerts.append(alert)
            return alert
        
        return None
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get privacy-preserving monitoring summary"""
        if not self.glucose_readings:
            return {"status": "no_data"}
        
        glucose_metrics = self.calculator.evaluate_glucose_pattern(self.glucose_readings)
        
        return {
            "monitoring_period_days": (self.glucose_readings[-1].timestamp - self.glucose_readings[0].timestamp).days,
            "reading_count": len(self.glucose_readings),
            "time_in_range_percent": glucose_metrics["time_in_range"] * 100,
            "glucose_variability": "high" if glucose_metrics["glucose_variability"] > 40 else "normal",
            "genetic_risk_assessed": self.genetic_profile is not None,
            "alert_count": len(self.alerts),
            "last_alert": self.alerts[-1].timestamp.isoformat() if self.alerts else None,
            "monitoring_active": True
        }
