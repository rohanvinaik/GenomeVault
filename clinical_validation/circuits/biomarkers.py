"""
Clinical Biomarker Circuits.
Generic implementation for various biomarker threshold proofs.
"""

from typing import Dict, Any
import numpy as np
import hashlib

from .base import BaseCircuit
from ..proofs.models import CircuitConfig, ProofData, CircuitType, ComparisonType


class ClinicalBiomarkerCircuit(BaseCircuit):
    """
    Generic circuit for clinical biomarker threshold proofs.
    Supports multiple comparison types and multi-threshold checks.
    """
    
    def __init__(self, biomarker_name: str = "generic"):
        config = CircuitConfig(
            name=f"ClinicalBiomarker_{biomarker_name}_Circuit",
            version="2.0.0",
            constraints=5000,
            proof_size=256,
            supported_parameters={
                'value_range': tuple,
                'comparison_types': list,
                'precision': float
            }
        )
        super().__init__(config)
        
        self.biomarker_name = biomarker_name
        self.value_range = (0.0, 1000.0)
        self.precision = 0.001
        self.supported_comparisons = [ComparisonType.GREATER, ComparisonType.LESS, ComparisonType.EQUAL]
    
    def setup(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Setup with biomarker-specific configuration"""
        result = super().setup(params)
        
        if 'value_range' in params:
            self.value_range = params['value_range']
        if 'precision' in params:
            self.precision = params['precision']
        if 'comparison_types' in params:
            self.supported_comparisons = [ComparisonType(ct) for ct in params['comparison_types']]
        
        result['biomarker_config'] = {
            'name': self.biomarker_name,
            'value_range': self.value_range,
            'precision': self.precision,
            'supported_comparisons': [ct.value for ct in self.supported_comparisons]
        }
        
        return result
    
    def generate_witness(self, private_inputs: Dict[str, float], 
                        public_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate witness for biomarker comparison"""
        value = private_inputs.get('value', 0.0)
        
        # Validate value is in range
        if not self.value_range[0] <= value <= self.value_range[1]:
            raise ValueError(f"Value {value} out of range {self.value_range}")
        
        threshold = public_inputs.get('threshold', 0.0)
        comparison_type = ComparisonType(public_inputs.get('comparison', 'greater'))
        
        if comparison_type not in self.supported_comparisons:
            raise ValueError(f"Comparison type {comparison_type} not supported")
        
        # Perform comparison
        if comparison_type == ComparisonType.GREATER:
            result = value > threshold
        elif comparison_type == ComparisonType.LESS:
            result = value < threshold
        elif comparison_type == ComparisonType.EQUAL:
            result = abs(value - threshold) < self.precision
        else:
            result = False
        
        # Support for range comparisons
        if comparison_type == ComparisonType.RANGE and 'threshold_high' in public_inputs:
            threshold_high = public_inputs['threshold_high']
            result = threshold <= value <= threshold_high
        
        # Add noise for privacy
        noise = np.random.normal(0, self.precision / 10)
        
        return {
            'private_value': value + noise,
            'public_threshold': threshold,
            'comparison_type': comparison_type.value,
            'result': result,
            'margin': abs(value - threshold),  # How far from threshold
            'randomness': np.random.bytes(32),
            'noise': noise
        }
    
    def prove(self, witness: Dict[str, Any], public_inputs: Dict[str, float]) -> ProofData:
        """Generate proof for biomarker threshold"""
        if not self._setup_complete:
            raise RuntimeError("Circuit setup not complete")
        
        result = witness['result']
        margin = witness['margin']
        
        # Create detailed output
        status = 'EXCEEDS' if result else 'NORMAL'
        confidence = min(0.99, margin / abs(witness['public_threshold'] + 0.01))
        
        public_output = f"{self.biomarker_name}:{status}:MARGIN:{margin:.3f}:CONFIDENCE:{confidence:.2f}"
        
        # Generate proof
        proof_bytes = hashlib.sha256(str(witness).encode()).digest()
        proof_bytes += np.random.bytes(self.config.proof_size - 32)
        
        proof = ProofData(
            public_output=public_output,
            proof_bytes=proof_bytes,
            verification_key=self._generate_verification_key(),
            circuit_type=CircuitType.BIOMARKER_THRESHOLD
        )
        
        self._add_proof_metadata(proof, public_inputs, witness)
        proof.metadata['biomarker'] = self.biomarker_name
        proof.metadata['comparison_type'] = witness['comparison_type']
        
        return proof
    
    def _verify_proof_specific(self, proof: ProofData, public_inputs: Dict) -> bool:
        """Specific verification for biomarker circuit"""
        # Verify output format
        parts = proof.public_output.split(":")
        if len(parts) < 6:
            return False
        
        if parts[0] != self.biomarker_name:
            return False
        
        if parts[1] not in ['EXCEEDS', 'NORMAL']:
            return False
        
        # Verify numeric values
        try:
            margin = float(parts[3])
            confidence = float(parts[5])
            if not 0 <= confidence <= 1:
                return False
        except ValueError:
            return False
        
        return True
