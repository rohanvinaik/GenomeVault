"""
Circuit manager for Zero-Knowledge proof system.

Handles circuit selection, optimization, and management for 
genomic privacy-preserving proofs.
"""
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.config import config
from utils.logging import logger, performance_logger

from .circuits.base_circuits import BaseCircuit
from .circuits.biological.multi_omics import (
    ClinicalTrialEligibilityCircuit,
    GenotypePhenotypeAssociationCircuit,
    MultiOmicsCorrelationCircuit,
    RareVariantBurdenCircuit,
)
from .circuits.biological.variant import (
    DiabetesRiskCircuit,
    PathwayEnrichmentCircuit,
    PharmacogenomicCircuit,
    PolygenenicRiskScoreCircuit,
    VariantPresenceCircuit,
)


@dataclass
class CircuitMetadata:
    """Metadata for a circuit implementation."""
    name: str
    circuit_class: type
    constraint_count: int
    proof_size_bytes: int
    verification_time_ms: float
    security_level: int = 128  # bits
    post_quantum: bool = False
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'constraint_count': self.constraint_count,
            'proof_size_bytes': self.proof_size_bytes,
            'verification_time_ms': self.verification_time_ms,
            'security_level': self.security_level,
            'post_quantum': self.post_quantum,
            'parameters': self.parameters
        }


class CircuitManager:
    """
    Manages ZK circuit selection and optimization.
    """
    
    def __init__(self):
        """Initialize circuit manager."""
        self.circuits = self._initialize_circuits()
        self.optimization_cache = {}
        self.performance_stats = {}
        
        logger.info("Circuit manager initialized", 
                   extra={'circuit_count': len(self.circuits)})
    
    def _initialize_circuits(self) -> Dict[str, CircuitMetadata]:
        """Initialize available circuits with metadata."""
        circuits = {
            'variant_presence': CircuitMetadata(
                name='variant_presence',
                circuit_class=VariantPresenceCircuit,
                constraint_count=5000,
                proof_size_bytes=192,
                verification_time_ms=10.0,
                parameters={'merkle_depth': 20}
            ),
            'polygenic_risk_score': CircuitMetadata(
                name='polygenic_risk_score',
                circuit_class=PolygenenicRiskScoreCircuit,
                constraint_count=20000,
                proof_size_bytes=384,
                verification_time_ms=25.0,
                parameters={'max_variants': 1000, 'precision_bits': 16}
            ),
            'diabetes_risk_alert': CircuitMetadata(
                name='diabetes_risk_alert',
                circuit_class=DiabetesRiskCircuit,
                constraint_count=15000,
                proof_size_bytes=384,
                verification_time_ms=25.0,
                parameters={'clinical_pilot': True}
            ),
            'pharmacogenomic': CircuitMetadata(
                name='pharmacogenomic',
                circuit_class=PharmacogenomicCircuit,
                constraint_count=10000,
                proof_size_bytes=320,
                verification_time_ms=20.0,
                parameters={'genes': ['CYP2C19', 'CYP2D6', 'CYP2C9', 'VKORC1', 'TPMT']}
            ),
            'pathway_enrichment': CircuitMetadata(
                name='pathway_enrichment',
                circuit_class=PathwayEnrichmentCircuit,
                constraint_count=25000,
                proof_size_bytes=512,
                verification_time_ms=30.0,
                parameters={'max_genes': 20000, 'permutations': 1000}
            ),
            'multi_omics_correlation': CircuitMetadata(
                name='multi_omics_correlation',
                circuit_class=MultiOmicsCorrelationCircuit,
                constraint_count=30000,
                proof_size_bytes=640,
                verification_time_ms=40.0,
                parameters={'max_dimensions': 1000}
            ),
            'genotype_phenotype': CircuitMetadata(
                name='genotype_phenotype',
                circuit_class=GenotypePhenotypeAssociationCircuit,
                constraint_count=40000,
                proof_size_bytes=768,
                verification_time_ms=50.0,
                parameters={'max_samples': 10000}
            ),
            'clinical_trial_eligibility': CircuitMetadata(
                name='clinical_trial_eligibility',
                circuit_class=ClinicalTrialEligibilityCircuit,
                constraint_count=20000,
                proof_size_bytes=448,
                verification_time_ms=30.0,
                parameters={'criteria_types': ['genomic', 'clinical', 'demographic']}
            ),
            'rare_variant_burden': CircuitMetadata(
                name='rare_variant_burden',
                circuit_class=RareVariantBurdenCircuit,
                constraint_count=15000,
                proof_size_bytes=384,
                verification_time_ms=25.0,
                parameters={'max_variants_per_gene': 100}
            )
        }
        
        return circuits
    
    def get_circuit(self, circuit_name: str) -> BaseCircuit:
        """
        Get circuit instance by name.
        
        Args:
            circuit_name: Name of the circuit
            
        Returns:
            Circuit instance
        """
        if circuit_name not in self.circuits:
            raise ValueError(f"Unknown circuit: {circuit_name}")
        
        metadata = self.circuits[circuit_name]
        
        # Create circuit instance with parameters
        if circuit_name == 'variant_presence':
            return metadata.circuit_class(
                merkle_depth=metadata.parameters.get('merkle_depth', 20)
            )
        elif circuit_name == 'polygenic_risk_score':
            return metadata.circuit_class(
                max_variants=metadata.parameters.get('max_variants', 1000)
            )
        elif circuit_name == 'pharmacogenomic':
            return metadata.circuit_class(
                max_star_alleles=metadata.parameters.get('max_star_alleles', 50)
            )
        elif circuit_name == 'pathway_enrichment':
            return metadata.circuit_class(
                max_genes=metadata.parameters.get('max_genes', 20000)
            )
        elif circuit_name == 'multi_omics_correlation':
            return metadata.circuit_class(
                max_dimensions=metadata.parameters.get('max_dimensions', 1000)
            )
        elif circuit_name == 'genotype_phenotype':
            return metadata.circuit_class(
                max_samples=metadata.parameters.get('max_samples', 10000)
            )
        elif circuit_name == 'rare_variant_burden':
            return metadata.circuit_class(
                max_variants_per_gene=metadata.parameters.get('max_variants_per_gene', 100)
            )
        else:
            return metadata.circuit_class()
    
    def get_circuit_metadata(self, circuit_name: str) -> CircuitMetadata:
        """Get metadata for a circuit."""
        if circuit_name not in self.circuits:
            raise ValueError(f"Unknown circuit: {circuit_name}")
        
        return self.circuits[circuit_name]
    
    def list_circuits(self) -> List[Dict[str, Any]]:
        """List all available circuits with metadata."""
        return [
            {
                'name': name,
                **metadata.to_dict()
            }
            for name, metadata in self.circuits.items()
        ]
    
    def select_optimal_circuit(self, analysis_type: str, 
                             data_characteristics: Dict[str, Any]) -> str:
        """
        Select optimal circuit based on analysis type and data.
        
        Args:
            analysis_type: Type of analysis needed
            data_characteristics: Properties of the data
            
        Returns:
            Name of optimal circuit
        """
        # Map analysis types to circuits
        analysis_circuit_map = {
            'variant_verification': 'variant_presence',
            'risk_score': 'polygenic_risk_score',
            'diabetes_screening': 'diabetes_risk_alert',
            'drug_response': 'pharmacogenomic',
            'pathway_analysis': 'pathway_enrichment',
            'multi_omics': 'multi_omics_correlation',
            'gwas': 'genotype_phenotype',
            'trial_matching': 'clinical_trial_eligibility',
            'rare_disease': 'rare_variant_burden'
        }
        
        # Get base circuit
        base_circuit = analysis_circuit_map.get(analysis_type)
        
        if not base_circuit:
            # Try to infer from data characteristics
            if 'variants' in data_characteristics:
                if data_characteristics.get('variant_count', 0) > 100:
                    base_circuit = 'polygenic_risk_score'
                else:
                    base_circuit = 'variant_presence'
            elif 'expression' in data_characteristics:
                base_circuit = 'pathway_enrichment'
            elif 'multi_layer' in data_characteristics:
                base_circuit = 'multi_omics_correlation'
            else:
                base_circuit = 'variant_presence'  # Default
        
        # Check if optimization is needed
        if self._needs_optimization(base_circuit, data_characteristics):
            base_circuit = self._optimize_circuit_selection(
                base_circuit, data_characteristics
            )
        
        logger.info(f"Selected circuit: {base_circuit} for {analysis_type}")
        
        return base_circuit
    
    def _needs_optimization(self, circuit_name: str, 
                          data_characteristics: Dict[str, Any]) -> bool:
        """Check if circuit selection needs optimization."""
        metadata = self.circuits[circuit_name]
        
        # Check data size constraints
        if 'variant_count' in data_characteristics:
            max_variants = metadata.parameters.get('max_variants', float('inf'))
            if data_characteristics['variant_count'] > max_variants:
                return True
        
        # Check performance requirements
        if 'max_proof_time_ms' in data_characteristics:
            if metadata.verification_time_ms > data_characteristics['max_proof_time_ms']:
                return True
        
        return False
    
    def _optimize_circuit_selection(self, base_circuit: str,
                                  data_characteristics: Dict[str, Any]) -> str:
        """Optimize circuit selection based on constraints."""
        # Cache key for optimization
        cache_key = f"{base_circuit}:{json.dumps(data_characteristics, sort_keys=True)}"
        
        if cache_key in self.optimization_cache:
            return self.optimization_cache[cache_key]
        
        # Find circuits that meet constraints
        candidates = []
        
        for name, metadata in self.circuits.items():
            # Check if circuit can handle the data
            if self._circuit_meets_requirements(metadata, data_characteristics):
                candidates.append((name, metadata))
        
        # Sort by efficiency (constraints/proof_size ratio)
        candidates.sort(
            key=lambda x: x[1].constraint_count / x[1].proof_size_bytes
        )
        
        # Select most efficient
        if candidates:
            optimal = candidates[0][0]
        else:
            optimal = base_circuit  # Fallback to original
        
        # Cache result
        self.optimization_cache[cache_key] = optimal
        
        return optimal
    
    def _circuit_meets_requirements(self, metadata: CircuitMetadata,
                                  requirements: Dict[str, Any]) -> bool:
        """Check if circuit meets requirements."""
        # Check proof size
        if 'max_proof_size' in requirements:
            if metadata.proof_size_bytes > requirements['max_proof_size']:
                return False
        
        # Check verification time
        if 'max_verification_ms' in requirements:
            if metadata.verification_time_ms > requirements['max_verification_ms']:
                return False
        
        # Check security level
        if 'min_security_bits' in requirements:
            if metadata.security_level < requirements['min_security_bits']:
                return False
        
        # Check post-quantum requirement
        if requirements.get('require_post_quantum', False):
            if not metadata.post_quantum:
                return False
        
        return True
    
    @performance_logger.log_operation("optimize_circuit_parameters")
    def optimize_circuit_parameters(self, circuit_name: str,
                                  target_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize circuit parameters for target metrics.
        
        Args:
            circuit_name: Circuit to optimize
            target_metrics: Target performance metrics
            
        Returns:
            Optimized parameters
        """
        metadata = self.circuits[circuit_name]
        optimized_params = metadata.parameters.copy()
        
        # Optimize based on targets
        if 'target_proof_size' in target_metrics:
            # Reduce constraint count if possible
            if circuit_name == 'polygenic_risk_score':
                max_size = target_metrics['target_proof_size']
                # Approximate: 0.2 bytes per variant
                max_variants = int(max_size / 0.2)
                optimized_params['max_variants'] = min(
                    max_variants,
                    optimized_params.get('max_variants', 1000)
                )
        
        if 'target_verification_time' in target_metrics:
            # Adjust complexity parameters
            if circuit_name == 'pathway_enrichment':
                target_time = target_metrics['target_verification_time']
                # Reduce permutations for faster verification
                if target_time < 20:
                    optimized_params['permutations'] = 100
                elif target_time < 30:
                    optimized_params['permutations'] = 500
        
        logger.info(f"Optimized parameters for {circuit_name}: {optimized_params}")
        
        return optimized_params
    
    def estimate_proof_generation_time(self, circuit_name: str,
                                     data_size: Dict[str, int]) -> float:
        """
        Estimate proof generation time.
        
        Args:
            circuit_name: Circuit name
            data_size: Size characteristics of input data
            
        Returns:
            Estimated time in seconds
        """
        metadata = self.circuits[circuit_name]
        
        # Base time estimate
        base_time = metadata.constraint_count / 5000  # 5k constraints/second
        
        # Adjust for data size
        if 'variant_count' in data_size:
            variant_factor = data_size['variant_count'] / 1000
            base_time *= max(1, variant_factor)
        
        if 'sample_count' in data_size:
            sample_factor = data_size['sample_count'] / 1000
            base_time *= max(1, sample_factor ** 0.5)  # Sub-linear scaling
        
        # Add overhead
        overhead = 0.5  # Setup overhead
        
        return base_time + overhead
    
    def get_circuit_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for circuits."""
        stats = {
            'total_circuits': len(self.circuits),
            'circuits': {},
            'optimization_cache_size': len(self.optimization_cache),
            'performance_records': len(self.performance_stats)
        }
        
        for name, metadata in self.circuits.items():
            stats['circuits'][name] = {
                'constraint_count': metadata.constraint_count,
                'proof_size': metadata.proof_size_bytes,
                'verification_time': metadata.verification_time_ms,
                'usage_count': self.performance_stats.get(name, {}).get('count', 0)
            }
        
        return stats
    
    def validate_circuit_inputs(self, circuit_name: str,
                              public_inputs: Dict[str, Any],
                              private_inputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate inputs for a circuit.
        
        Args:
            circuit_name: Circuit to validate for
            public_inputs: Public inputs
            private_inputs: Private inputs
            
        Returns:
            (is_valid, error_messages)
        """
        errors = []
        
        # Get expected inputs based on circuit
        expected_inputs = self._get_expected_inputs(circuit_name)
        
        # Check public inputs
        for required in expected_inputs['public']:
            if required not in public_inputs:
                errors.append(f"Missing public input: {required}")
        
        # Check private inputs
        for required in expected_inputs['private']:
            if required not in private_inputs:
                errors.append(f"Missing private input: {required}")
        
        # Circuit-specific validation
        if circuit_name == 'diabetes_risk_alert':
            # Validate glucose threshold
            if 'glucose_threshold' in public_inputs:
                threshold = public_inputs['glucose_threshold']
                if not (50 <= threshold <= 300):
                    errors.append("Glucose threshold out of valid range (50-300)")
            
            # Validate risk threshold
            if 'risk_threshold' in public_inputs:
                threshold = public_inputs['risk_threshold']
                if not (0 <= threshold <= 1):
                    errors.append("Risk threshold must be between 0 and 1")
        
        elif circuit_name == 'polygenic_risk_score':
            # Check variant and weight counts match
            if 'variants' in private_inputs and 'weights' in private_inputs:
                if len(private_inputs['variants']) != len(private_inputs['weights']):
                    errors.append("Variant and weight counts must match")
        
        is_valid = len(errors) == 0
        
        return is_valid, errors
    
    def _get_expected_inputs(self, circuit_name: str) -> Dict[str, List[str]]:
        """Get expected inputs for a circuit."""
        expected = {
            'variant_presence': {
                'public': ['variant_hash', 'reference_hash', 'commitment_root'],
                'private': ['variant_data', 'merkle_proof', 'witness_randomness']
            },
            'polygenic_risk_score': {
                'public': ['prs_model', 'score_range', 'result_commitment', 'genome_commitment'],
                'private': ['variants', 'weights', 'merkle_proofs', 'witness_randomness']
            },
            'diabetes_risk_alert': {
                'public': ['glucose_threshold', 'risk_threshold', 'result_commitment'],
                'private': ['glucose_reading', 'risk_score', 'witness_randomness']
            },
            'pharmacogenomic': {
                'public': ['medication_id', 'response_category', 'model_version'],
                'private': ['star_alleles', 'variant_genotypes', 'activity_scores', 'witness_randomness']
            },
            'pathway_enrichment': {
                'public': ['pathway_id', 'enrichment_score', 'significance'],
                'private': ['expression_values', 'gene_sets', 'permutation_seeds', 'witness_randomness']
            },
            'multi_omics_correlation': {
                'public': ['correlation_coefficient', 'modality_1', 'modality_2', 'significance_threshold'],
                'private': ['data_1', 'data_2', 'sample_size', 'witness_randomness']
            },
            'genotype_phenotype': {
                'public': ['phenotype_id', 'association_strength', 'p_value_commitment', 'study_size'],
                'private': ['genotypes', 'phenotypes', 'covariates', 'witness_randomness']
            },
            'clinical_trial_eligibility': {
                'public': ['trial_id', 'eligibility_result', 'criteria_hash'],
                'private': ['genomic_features', 'clinical_features', 'demographic_features', 'witness_randomness']
            },
            'rare_variant_burden': {
                'public': ['gene_id', 'burden_score', 'max_allele_frequency'],
                'private': ['variants', 'allele_frequencies', 'functional_scores', 'witness_randomness']
            }
        }
        
        return expected.get(circuit_name, {'public': [], 'private': []})
    
    def benchmark_circuit(self, circuit_name: str, 
                         test_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Benchmark a circuit's performance.
        
        Args:
            circuit_name: Circuit to benchmark
            test_data: Test data for benchmarking
            
        Returns:
            Performance metrics
        """
        start_time = time.time()
        
        # Get circuit
        circuit = self.get_circuit(circuit_name)
        
        # Generate test data if not provided
        if not test_data:
            test_data = self._generate_test_data(circuit_name)
        
        # Setup circuit
        setup_start = time.time()
        circuit.setup(test_data['public_inputs'], test_data['private_inputs'])
        setup_time = time.time() - setup_start
        
        # Generate constraints
        constraint_start = time.time()
        circuit.generate_constraints()
        constraint_time = time.time() - constraint_start
        
        # Total time
        total_time = time.time() - start_time
        
        # Record stats
        if circuit_name not in self.performance_stats:
            self.performance_stats[circuit_name] = {
                'count': 0,
                'total_time': 0,
                'avg_setup_time': 0,
                'avg_constraint_time': 0
            }
        
        stats = self.performance_stats[circuit_name]
        stats['count'] += 1
        stats['total_time'] += total_time
        stats['avg_setup_time'] = (
            (stats['avg_setup_time'] * (stats['count'] - 1) + setup_time) / 
            stats['count']
        )
        stats['avg_constraint_time'] = (
            (stats['avg_constraint_time'] * (stats['count'] - 1) + constraint_time) / 
            stats['count']
        )
        
        return {
            'setup_time': setup_time,
            'constraint_time': constraint_time,
            'total_time': total_time,
            'constraint_count': len(circuit.constraints)
        }
    
    def _generate_test_data(self, circuit_name: str) -> Dict[str, Any]:
        """Generate test data for benchmarking."""
        import numpy as np
        
        if circuit_name == 'variant_presence':
            return {
                'public_inputs': {
                    'variant_hash': hashlib.sha256(b"test_variant").hexdigest(),
                    'reference_hash': hashlib.sha256(b"GRCh38").hexdigest(),
                    'commitment_root': hashlib.sha256(b"test_root").hexdigest()
                },
                'private_inputs': {
                    'variant_data': {
                        'chr': 'chr1',
                        'pos': 12345,
                        'ref': 'A',
                        'alt': 'G'
                    },
                    'merkle_proof': {
                        'path': [hashlib.sha256(f"node_{i}".encode()).hexdigest() for i in range(20)],
                        'indices': [i % 2 for i in range(20)]
                    },
                    'witness_randomness': np.random.bytes(32).hex()
                }
            }
        
        elif circuit_name == 'diabetes_risk_alert':
            return {
                'public_inputs': {
                    'glucose_threshold': 126,
                    'risk_threshold': 0.75,
                    'result_commitment': hashlib.sha256(b"test_result").hexdigest()
                },
                'private_inputs': {
                    'glucose_reading': 140,
                    'risk_score': 0.82,
                    'witness_randomness': np.random.bytes(32).hex()
                }
            }
        
        else:
            # Generic test data
            return {
                'public_inputs': {},
                'private_inputs': {
                    'witness_randomness': np.random.bytes(32).hex()
                }
            }
