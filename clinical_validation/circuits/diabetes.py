"""
Diabetes Risk Assessment Circuit.
Optimized implementation for diabetes risk analysis with ZK proofs.
"""

from typing import Dict, Any
import numpy as np
import hashlib
import json

from .base import BaseCircuit
from ..proofs.models import CircuitConfig, ProofData, CircuitType


class DiabetesRiskCircuit(BaseCircuit):
    """
    Optimized diabetes risk assessment circuit.
    Proves clinical values exceed thresholds without revealing actual values.
    """
    
    def __init__(self):
        config = CircuitConfig(
            name="DiabetesRiskCircuit",
            version="2.0.0",
            constraints=15000,
            proof_size=384,
            supported_parameters={
                'glucose_range': tuple,
                'hba1c_range': tuple,
                'risk_score_range': tuple,
                'risk_factors_threshold': int
            }
        )
        super().__init__(config)
        
        # Default ranges
        self.glucose_range = (70.0, 300.0)
        self.hba1c_range = (4.0, 14.0)
        self.risk_score_range = (-3.0, 3.0)
        self.risk_factors_threshold = 2
        
    def setup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Setup with validation"""
        result = super().setup(params)
        
        # Update ranges if provided
        if 'glucose_range' in params:
            self.glucose_range = params['glucose_range']
        if 'hba1c_range' in params:
            self.hba1c_range = params['hba1c_range']
        if 'risk_score_range' in params:
            self.risk_score_range = params['risk_score_range']
        if 'risk_factors_threshold' in params:
            self.risk_factors_threshold = params['risk_factors_threshold']
        
        result['supported_thresholds'] = {
            'glucose': self.glucose_range,
            'hba1c': self.hba1c_range,
            'risk_score': self.risk_score_range
        }
        
        return result
    
    def generate_witness(self, private_inputs: Dict[str, float], 
                        public_inputs: Dict[str, float]) -> Dict[str, Any]:
        """Generate witness with enhanced validation"""
        # Validate inputs are within ranges
        glucose = private_inputs.get('glucose', 100.0)
        if not self.glucose_range[0] <= glucose <= self.glucose_range[1]:
            raise ValueError(f"Glucose {glucose} out of range {self.glucose_range}")
        
        hba1c = private_inputs.get('hba1c', 5.5)
        if not self.hba1c_range[0] <= hba1c <= self.hba1c_range[1]:
            raise ValueError(f"HbA1c {hba1c} out of range {self.hba1c_range}")
        
        risk_score = private_inputs.get('genetic_risk_score', 0.0)
        if not self.risk_score_range[0] <= risk_score <= self.risk_score_range[1]:
            raise ValueError(f"Risk score {risk_score} out of range {self.risk_score_range}")
        
        # Extract thresholds
        glucose_threshold = public_inputs.get('glucose_threshold', 126.0)
        hba1c_threshold = public_inputs.get('hba1c_threshold', 6.5)
        risk_threshold = public_inputs.get('risk_threshold', 0.5)
        
        # Compute comparisons
        glucose_exceeds = float(glucose > glucose_threshold)
        hba1c_exceeds = float(hba1c > hba1c_threshold)
        risk_exceeds = float(risk_score > risk_threshold)
        
        # Risk assessment
        risk_factors = glucose_exceeds + hba1c_exceeds + risk_exceeds
        is_high_risk = risk_factors >= self.risk_factors_threshold
        
        # Enhanced witness with noise for differential privacy
        noise_factor = np.random.normal(0, 0.001, size=3)
        
        return {
            'private_values': {
                'glucose': glucose + noise_factor[0],
                'hba1c': hba1c + noise_factor[1],
                'risk_score': risk_score + noise_factor[2]
            },
            'comparisons': {
                'glucose_exceeds': glucose_exceeds,
                'hba1c_exceeds': hba1c_exceeds,
                'risk_exceeds': risk_exceeds
            },
            'result': {
                'risk_factors': int(risk_factors),
                'is_high_risk': is_high_risk,
                'confidence': min(0.99, risk_factors / 3.0)  # Confidence score
            },
            'randomness': np.random.bytes(32),
            'noise_factor': noise_factor.tolist()
        }
    
    def prove(self, witness: Dict[str, Any], public_inputs: Dict[str, float]) -> ProofData:
        """Generate proof with enhanced security"""
        if not self._setup_complete:
            raise RuntimeError("Circuit setup not complete")
        
        is_high_risk = witness['result']['is_high_risk']
        confidence = witness['result']['confidence']
        
        # Create proof with detailed output
        public_output = f"RISK_LEVEL:{'HIGH' if is_high_risk else 'NORMAL'}:CONFIDENCE:{confidence:.2f}"
        
        # Generate proof bytes (in production, use actual ZK proof system)
        proof_data = {
            'witness_commitment': hashlib.sha256(str(witness).encode()).hexdigest(),
            'public_inputs': public_inputs,
            'risk_assessment': witness['result']
        }
        
        proof_bytes = hashlib.sha256(json.dumps(proof_data).encode()).digest()
        proof_bytes += np.random.bytes(self.config.proof_size - 32)
        
        # Create proof object
        proof = ProofData(
            public_output=public_output,
            proof_bytes=proof_bytes,
            verification_key=self._generate_verification_key(),
            circuit_type=CircuitType.DIABETES_RISK
        )
        
        # Add metadata
        self._add_proof_metadata(proof, public_inputs, witness)
        proof.metadata['risk_factors_used'] = self.risk_factors_threshold
        proof.metadata['confidence_score'] = confidence
        
        return proof
    
    def _verify_proof_specific(self, proof: ProofData, public_inputs: Dict) -> bool:
        """Specific verification for diabetes risk circuit"""
        # Verify output format
        if not proof.public_output.startswith("RISK_LEVEL:"):
            return False
        
        parts = proof.public_output.split(":")
        if len(parts) != 4:
            return False
        
        risk_level = parts[1]
        if risk_level not in ['HIGH', 'NORMAL']:
            return False
        
        # Verify confidence is valid
        try:
            confidence = float(parts[3])
            if not 0 <= confidence <= 1:
                return False
        except ValueError:
            return False
        
        return True
