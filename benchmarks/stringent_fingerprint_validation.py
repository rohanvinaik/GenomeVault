#!/usr/bin/env python3
"""
Stringent Fingerprint Validation with Large-Scale Testing
Validates the AUC and EER results with much larger cohorts and harder test conditions
"""

import numpy as np
import hashlib
import time
import json
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import stats
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig
from genomevault.core.constants import OmicsType

@dataclass
class StringentTestConfig:
    """Configuration for stringent validation"""
    dimension: int
    sparsity: float
    subjects: int = 500  # 10x larger cohort
    samples_per_subject: int = 10  # More samples per subject
    num_features: int = 50000  # Even more features for harder test
    intra_subject_noise: float = 0.05  # Slightly more noise (5% vs 2%)
    inter_subject_overlap: float = 0.3  # 30% shared features between subjects (harder)
    seed: int = 42

@dataclass
class ValidationResult:
    """Comprehensive validation results"""
    config: StringentTestConfig
    far: float
    frr: float  
    eer: float
    auc: float
    auc_ci: Tuple[float, float]
    d_prime: float
    storage_kb: float
    genuine_scores: np.ndarray
    impostor_scores: np.ndarray
    genuine_count: int
    impostor_count: int
    p50_genuine: float
    p95_genuine: float
    p50_impostor: float
    p95_impostor: float
    separation_ratio: float  # min(genuine)/max(impostor)

class StringentFingerprintValidator:
    """Stringent validator with harder test conditions"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self.encoder = None
        self.results = []
        
    def setup_encoder(self, config: StringentTestConfig):
        """Create encoder with fixed seed"""
        if self.encoder is None:
            hv_config = HypervectorConfig(
                dimension=config.dimension,
                seed=42,  # Fixed seed
                normalize=True,
                use_metal=True
            )
            self.encoder = HypervectorEncoder(config=hv_config)
        return self.encoder
    
    def generate_challenging_genomic_profile(self, subject_id: int, config: StringentTestConfig) -> np.ndarray:
        """Generate more challenging test data with higher inter-subject overlap"""
        
        # Cryptographic PRF for subject
        subject_seed = hashlib.sha256(
            f"subject_{subject_id}_{self.seed}".encode()
        ).digest()
        rng = np.random.RandomState(int.from_bytes(subject_seed[:4], 'big'))
        
        num_features = config.num_features
        
        # 1. Population-wide common variants (higher overlap - harder test)
        common_variants = np.zeros(num_features)
        num_common = int(num_features * config.inter_subject_overlap)
        
        # Use global seed for common features (shared across subjects)
        common_rng = np.random.RandomState(self.seed)
        common_indices = common_rng.choice(num_features, num_common, replace=False)
        
        # Add some subject-specific variation to common variants
        for idx in common_indices:
            if rng.random() < 0.8:  # 80% chance of having common variant
                common_variants[idx] = common_rng.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        
        # 2. Rare variants (subject-specific but fewer to make discrimination harder)
        rare_variants = np.zeros(num_features)
        num_rare = rng.poisson(30)  # Fewer rare variants
        if num_rare > 0:
            rare_indices = rng.choice(num_features, min(num_rare, num_features), replace=False)
            rare_variants[rare_indices] = rng.choice([1, 2], len(rare_indices), p=[0.8, 0.2])
        
        # 3. Family-based patterns (siblings/relatives have similar patterns)
        family_group = subject_id // 10  # Every 10 subjects are "related"
        family_seed = hashlib.sha256(f"family_{family_group}".encode()).digest()
        family_rng = np.random.RandomState(int.from_bytes(family_seed[:4], 'big'))
        
        family_variants = np.zeros(num_features)
        num_family = int(num_features * 0.1)  # 10% family-specific
        family_indices = family_rng.choice(num_features, num_family, replace=False)
        family_variants[family_indices] = family_rng.choice([0, 1], num_family)
        
        # 4. Complex LD patterns
        ld_blocks = np.zeros(num_features)
        num_blocks = 50  # More LD blocks
        block_size = 200
        for _ in range(num_blocks):
            if num_features > block_size:
                block_start = rng.randint(0, num_features - block_size)
                # Correlated pattern within block
                base_pattern = rng.randn(block_size) * 0.3
                decay = np.exp(-np.linspace(0, 5, block_size))
                ld_blocks[block_start:block_start + block_size] += base_pattern * decay
        
        # 5. Population stratification (ethnic groups)
        ethnic_group = (subject_id // 50) % 5  # 5 ethnic groups
        ethnic_seed = hashlib.sha256(f"ethnic_{ethnic_group}".encode()).digest()
        ethnic_rng = np.random.RandomState(int.from_bytes(ethnic_seed[:4], 'big'))
        
        ethnic_variants = np.zeros(num_features)
        num_ethnic = int(num_features * 0.15)
        ethnic_indices = ethnic_rng.choice(num_features, num_ethnic, replace=False)
        ethnic_variants[ethnic_indices] = ethnic_rng.random(num_ethnic) * 0.5
        
        # 6. Gene expression with batch effects
        batch = subject_id % 3  # 3 batches
        expression = rng.lognormal(0, 1, num_features) * 0.1
        batch_effect = np.ones(num_features) * (1 + batch * 0.1)  # 10% batch effect
        expression *= batch_effect
        
        # Combine all components
        genomic_profile = (
            common_variants * 0.5 +  # Reduce weight of common
            rare_variants * 3 +      # Increase weight of rare
            family_variants * 1.5 +   # Family patterns
            ethnic_variants +         # Population structure
            ld_blocks +              # LD patterns
            expression               # Expression data
        )
        
        # Add technical noise (sequencing errors, etc.)
        technical_noise = rng.randn(num_features) * 0.01
        genomic_profile += technical_noise
        
        # Ensure non-negative
        genomic_profile = np.abs(genomic_profile)
        
        # Normalize
        genomic_profile = genomic_profile / (np.max(genomic_profile) + 1e-10)
        
        return genomic_profile.astype(np.float32)
    
    def add_realistic_sample_variation(self, base_profile: np.ndarray, sample_id: int, 
                                      config: StringentTestConfig) -> np.ndarray:
        """Add realistic within-subject variation"""
        np.random.seed(sample_id * 9999 + self.seed)
        
        # Different types of technical variation
        # 1. Sampling noise (which cells/DNA molecules were captured)
        sampling_noise = np.random.randn(len(base_profile)) * config.intra_subject_noise
        
        # 2. Batch effects (different days/runs)
        batch_effect = 1.0 + (sample_id % 3) * 0.01  # 1% batch effect
        
        # 3. Temporal variation (samples taken at different times)
        temporal_drift = np.sin(sample_id * 0.5) * 0.01  # Sinusoidal drift
        
        # 4. Technical artifacts (specific to sample processing)
        artifact_mask = np.random.random(len(base_profile)) > 0.99  # 1% features affected
        artifacts = np.zeros_like(base_profile)
        artifacts[artifact_mask] = np.random.randn(np.sum(artifact_mask)) * 0.1
        
        # Apply variations
        varied_profile = base_profile * batch_effect + sampling_noise + temporal_drift + artifacts
        
        # Ensure non-negative
        varied_profile = np.abs(varied_profile)
        
        return varied_profile.astype(np.float32)
    
    def compute_hdc_similarity(self, hv1: np.ndarray, hv2: np.ndarray) -> float:
        """Compute HDC similarity"""
        
        # Handle tensor types
        if hasattr(hv1, 'numpy'):
            hv1 = hv1.numpy()
        elif hasattr(hv1, 'cpu'):
            hv1 = hv1.cpu().numpy()
            
        if hasattr(hv2, 'numpy'):
            hv2 = hv2.numpy()
        elif hasattr(hv2, 'cpu'):
            hv2 = hv2.cpu().numpy()
        
        # Active components
        threshold = 1e-10
        active1 = np.abs(hv1) > threshold
        active2 = np.abs(hv2) > threshold
        
        # Jaccard similarity
        intersection = np.sum(active1 & active2)
        union = np.sum(active1 | active2)
        
        if union == 0:
            return 0.0
        
        structural_sim = intersection / union
        
        # Cosine similarity on active components
        active_both = active1 & active2
        if np.sum(active_both) > 0:
            v1_active = hv1[active_both]
            v2_active = hv2[active_both]
            
            dot_product = np.dot(v1_active, v2_active)
            norm1 = np.linalg.norm(v1_active)
            norm2 = np.linalg.norm(v2_active)
            
            if norm1 > 0 and norm2 > 0:
                magnitude_sim = (dot_product / (norm1 * norm2) + 1) / 2
            else:
                magnitude_sim = 0.0
        else:
            magnitude_sim = 0.0
        
        # Weighted combination
        similarity = 0.3 * structural_sim + 0.7 * magnitude_sim
        
        return similarity
    
    def run_stringent_validation(self):
        """Run comprehensive validation with stringent conditions"""
        print("="*80)
        print("STRINGENT FINGERPRINT VALIDATION")
        print("Large-scale testing with challenging conditions")
        print("="*80)
        
        # Test multiple configurations
        configs = [
            # Best performer from previous test
            StringentTestConfig(
                dimension=4096,
                sparsity=0.4,
                subjects=500,  # 10x larger
                samples_per_subject=10,  # More samples
                num_features=50000,  # 5x more features
                intra_subject_noise=0.05,  # More noise
                inter_subject_overlap=0.3  # Higher overlap
            ),
            # Medium dimension
            StringentTestConfig(
                dimension=8192,
                sparsity=0.5,
                subjects=300,
                samples_per_subject=8,
                num_features=30000,
                intra_subject_noise=0.04,
                inter_subject_overlap=0.25
            ),
            # High dimension
            StringentTestConfig(
                dimension=16384,
                sparsity=0.6,
                subjects=200,
                samples_per_subject=5,
                num_features=20000,
                intra_subject_noise=0.03,
                inter_subject_overlap=0.2
            ),
        ]
        
        for config in configs:
            print(f"\n{'='*60}")
            print(f"Testing: {config.dimension}D @ {config.sparsity*100:.0f}% sparsity")
            print(f"Subjects: {config.subjects}, Samples/subject: {config.samples_per_subject}")
            print(f"Features: {config.num_features}, Noise: {config.intra_subject_noise*100:.1f}%")
            print(f"Inter-subject overlap: {config.inter_subject_overlap*100:.0f}%")
            print(f"{'='*60}")
            
            result = self.validate_configuration(config)
            self.results.append(result)
            
            # Print detailed results
            print(f"\n📊 Results:")
            print(f"  EER: {result.eer:.4f}")
            print(f"  AUC: {result.auc:.4f} (95% CI: [{result.auc_ci[0]:.4f}, {result.auc_ci[1]:.4f}])")
            print(f"  D-prime: {result.d_prime:.2f}")
            print(f"  FAR: {result.far:.4f}, FRR: {result.frr:.4f}")
            
            print(f"\n📈 Score distributions:")
            print(f"  Genuine: P50={result.p50_genuine:.4f}, P95={result.p95_genuine:.4f}")
            print(f"  Impostor: P50={result.p50_impostor:.4f}, P95={result.p95_impostor:.4f}")
            print(f"  Separation ratio: {result.separation_ratio:.2f}")
            
            print(f"\n📦 Statistics:")
            print(f"  Genuine comparisons: {result.genuine_count:,}")
            print(f"  Impostor comparisons: {result.impostor_count:,}")
            print(f"  Storage: {result.storage_kb:.1f} KB")
        
        self.save_validation_results()
    
    def validate_configuration(self, config: StringentTestConfig) -> ValidationResult:
        """Validate a single configuration"""
        
        # Setup encoder
        encoder = self.setup_encoder(config)
        
        # Generate large cohort
        print(f"  Generating {config.subjects} subjects...")
        fingerprints = {}
        
        for subject_id in range(config.subjects):
            if subject_id % 100 == 0:
                print(f"    Generated {subject_id}/{config.subjects} subjects...")
            
            # Generate challenging base profile
            base_profile = self.generate_challenging_genomic_profile(subject_id, config)
            
            for sample_id in range(config.samples_per_subject):
                # Add realistic variation
                sample_profile = self.add_realistic_sample_variation(base_profile, sample_id, config)
                
                # Encode
                hv = encoder.encode(sample_profile, OmicsType.GENOMIC)
                
                # Sparsify
                if config.sparsity > 0:
                    if hasattr(hv, 'numpy'):
                        hv_np = hv.numpy()
                    elif hasattr(hv, 'cpu'):
                        hv_np = hv.cpu().numpy()
                    else:
                        hv_np = np.array(hv)
                    
                    threshold = np.percentile(np.abs(hv_np), config.sparsity * 100)
                    hv_np[np.abs(hv_np) < threshold] = 0
                    hv = hv_np
                
                fingerprints[(subject_id, sample_id)] = hv
        
        print(f"  Computing genuine scores...")
        # Genuine scores (same subject)
        genuine_scores = []
        for subject_id in range(config.subjects):
            for i in range(config.samples_per_subject):
                for j in range(i+1, config.samples_per_subject):
                    fp1 = fingerprints[(subject_id, i)]
                    fp2 = fingerprints[(subject_id, j)]
                    score = self.compute_hdc_similarity(fp1, fp2)
                    genuine_scores.append(score)
        
        print(f"  Computing impostor scores...")
        # Impostor scores (different subjects) - test more pairs
        impostor_scores = []
        max_impostor = min(50000, config.subjects * (config.subjects - 1) // 2)
        
        # Test especially challenging pairs (family members, same ethnic group)
        for i in range(config.subjects):
            for j in range(i+1, config.subjects):
                if len(impostor_scores) >= max_impostor:
                    break
                    
                # Test multiple sample pairs for robustness
                for si in range(min(2, config.samples_per_subject)):
                    for sj in range(min(2, config.samples_per_subject)):
                        fp1 = fingerprints[(i, si)]
                        fp2 = fingerprints[(j, sj)]
                        score = self.compute_hdc_similarity(fp1, fp2)
                        impostor_scores.append(score)
                        
                        if len(impostor_scores) >= max_impostor:
                            break
                    if len(impostor_scores) >= max_impostor:
                        break
            if len(impostor_scores) >= max_impostor:
                break
        
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        
        print(f"  Analyzing results...")
        
        # Calculate metrics
        genuine_mean = np.mean(genuine_scores)
        genuine_std = np.std(genuine_scores)
        impostor_mean = np.mean(impostor_scores)
        impostor_std = np.std(impostor_scores)
        
        pooled_std = np.sqrt((genuine_std**2 + impostor_std**2) / 2)
        if pooled_std > 0:
            d_prime = (genuine_mean - impostor_mean) / pooled_std
        else:
            d_prime = 0.0
        
        # ROC analysis
        labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        scores = np.concatenate([genuine_scores, impostor_scores])
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # EER
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        
        # Bootstrap CI
        n_bootstrap = 200  # More bootstrap iterations
        auc_scores = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(scores), len(scores), replace=True)
            boot_labels = labels[idx]
            boot_scores = scores[idx]
            try:
                boot_fpr, boot_tpr, _ = roc_curve(boot_labels, boot_scores)
                auc_scores.append(auc(boot_fpr, boot_tpr))
            except:
                continue
        
        auc_ci = (np.percentile(auc_scores, 2.5), np.percentile(auc_scores, 97.5))
        
        # Percentiles
        p50_genuine = np.percentile(genuine_scores, 50)
        p95_genuine = np.percentile(genuine_scores, 95)
        p50_impostor = np.percentile(impostor_scores, 50)
        p95_impostor = np.percentile(impostor_scores, 95)
        
        # Separation ratio
        min_genuine = np.min(genuine_scores)
        max_impostor = np.max(impostor_scores)
        if max_impostor > 0:
            separation_ratio = min_genuine / max_impostor
        else:
            separation_ratio = float('inf')
        
        # Storage
        sample_fp = fingerprints[(0, 0)]
        if hasattr(sample_fp, 'numpy'):
            sample_fp = sample_fp.numpy()
        elif hasattr(sample_fp, 'cpu'):
            sample_fp = sample_fp.cpu().numpy()
        
        non_zero = np.count_nonzero(sample_fp)
        storage_kb = (non_zero * 4) / 1024
        
        return ValidationResult(
            config=config,
            far=fpr[eer_idx],
            frr=fnr[eer_idx],
            eer=eer,
            auc=roc_auc,
            auc_ci=auc_ci,
            d_prime=d_prime,
            storage_kb=storage_kb,
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores,
            genuine_count=len(genuine_scores),
            impostor_count=len(impostor_scores),
            p50_genuine=p50_genuine,
            p95_genuine=p95_genuine,
            p50_impostor=p50_impostor,
            p95_impostor=p95_impostor,
            separation_ratio=separation_ratio
        )
    
    def save_validation_results(self):
        """Save comprehensive validation results"""
        os.makedirs("benchmark_results", exist_ok=True)
        
        results_data = []
        for r in self.results:
            results_data.append({
                'dimension': r.config.dimension,
                'sparsity': r.config.sparsity,
                'subjects': r.config.subjects,
                'samples_per_subject': r.config.samples_per_subject,
                'num_features': r.config.num_features,
                'intra_subject_noise': r.config.intra_subject_noise,
                'inter_subject_overlap': r.config.inter_subject_overlap,
                'storage_kb': r.storage_kb,
                'eer': float(r.eer),
                'far': float(r.far),
                'frr': float(r.frr),
                'auc': float(r.auc),
                'auc_ci_lower': float(r.auc_ci[0]),
                'auc_ci_upper': float(r.auc_ci[1]),
                'd_prime': float(r.d_prime),
                'genuine_count': r.genuine_count,
                'impostor_count': r.impostor_count,
                'p50_genuine': float(r.p50_genuine),
                'p95_genuine': float(r.p95_genuine),
                'p50_impostor': float(r.p50_impostor),
                'p95_impostor': float(r.p95_impostor),
                'separation_ratio': float(r.separation_ratio),
                'genuine_mean': float(np.mean(r.genuine_scores)),
                'genuine_std': float(np.std(r.genuine_scores)),
                'impostor_mean': float(np.mean(r.impostor_scores)),
                'impostor_std': float(np.std(r.impostor_scores))
            })
        
        with open('benchmark_results/stringent_fingerprint_validation.json', 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_type': 'Stringent Fingerprint Validation',
                'test_conditions': {
                    'cohort_size': '200-500 subjects',
                    'samples_per_subject': '5-10',
                    'feature_dimension': '20,000-50,000',
                    'intra_subject_noise': '3-5%',
                    'inter_subject_overlap': '20-30%',
                    'includes_family_structure': True,
                    'includes_population_stratification': True,
                    'includes_batch_effects': True
                },
                'results': results_data
            }, f, indent=2)
        
        print("\n" + "="*80)
        print("VALIDATION COMPLETE")
        print("="*80)
        print(f"✅ Results saved to benchmark_results/stringent_fingerprint_validation.json")

def main():
    """Run stringent validation"""
    validator = StringentFingerprintValidator(seed=42)
    validator.run_stringent_validation()

if __name__ == "__main__":
    main()