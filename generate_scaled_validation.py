#!/usr/bin/env python3
"""
Generate scaled validation data with target sample sizes for defensive statistics
Target: ≥20K impostor pairs, ≥2K genuine pairs per split
"""

import numpy as np
import json
from pathlib import Path
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarks.secure_fingerprint_evaluation import ExperimentConfig, RigorousFingerprintEvaluator

def calculate_required_subjects(target_genuine_pairs=2000, target_impostor_pairs=20000, samples_per_subject=5):
    """
    Calculate required number of subjects to achieve target pair counts
    
    For n subjects with s samples each:
    - Genuine pairs: n * (s * (s-1)) / 2
    - Impostor pairs: (n * (n-1)) / 2 * s^2
    """
    
    # Genuine pairs per subject: s * (s-1) / 2
    genuine_per_subject = samples_per_subject * (samples_per_subject - 1) // 2
    min_subjects_genuine = max(1, target_genuine_pairs // genuine_per_subject)
    
    # For impostor pairs: n * (n-1) / 2 * s^2 >= target
    # Solving: n^2 - n >= 2 * target / s^2
    # Using quadratic formula: n >= (1 + sqrt(1 + 8*target/s^2)) / 2
    s_squared = samples_per_subject ** 2
    discriminant = 1 + 8 * target_impostor_pairs / s_squared
    min_subjects_impostor = int(np.ceil((1 + np.sqrt(discriminant)) / 2))
    
    required_subjects = max(min_subjects_genuine, min_subjects_impostor)
    
    # Calculate actual pairs we'll get
    actual_genuine = required_subjects * genuine_per_subject
    actual_impostor = (required_subjects * (required_subjects - 1) // 2) * s_squared
    
    print(f"Target: {target_genuine_pairs:,} genuine, {target_impostor_pairs:,} impostor")
    print(f"Required subjects: {required_subjects} (from genuine: {min_subjects_genuine}, from impostor: {min_subjects_impostor})")
    print(f"Actual pairs: {actual_genuine:,} genuine, {actual_impostor:,} impostor")
    
    return required_subjects

def generate_scaled_validation():
    """Generate validation with proper sample sizes"""
    
    # Calculate required subjects for target sample sizes
    required_subjects = calculate_required_subjects(
        target_genuine_pairs=2000,
        target_impostor_pairs=20000,
        samples_per_subject=5
    )
    
    # Create configurations for each split type
    strategies = ['subject_disjoint', 'LFamO', 'LBxO']
    
    for strategy in strategies:
        print(f"\n🚀 Generating {strategy} validation...")
        
        # Configure for adequate sample sizes
        config = ExperimentConfig(
            n_subjects=required_subjects,
            n_families=required_subjects // 5,  # 5 subjects per family
            n_batches=min(20, required_subjects // 10),  # ~10 subjects per batch
            samples_per_subject=5,
            split_type=strategy,
            n_folds=5,
            output_dir=f"benchmark_results/fingerprint_{strategy}"
        )
        
        print(f"  Config: {config.n_subjects} subjects, {config.n_families} families")
        
        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run evaluation
        evaluator = RigorousFingerprintEvaluator(config)
        results = evaluator.run_validation()
        
        # Check actual sample sizes
        total_genuine = sum(r.n_genuine_pairs for r in results)
        total_impostor = sum(r.n_impostor_pairs for r in results)
        
        print(f"  ✅ Generated: {total_genuine:,} genuine, {total_impostor:,} impostor pairs")
        print(f"  📊 Rule-of-three bounds: genuine ≤{3/total_genuine*100:.3f}%, impostor ≤{3/total_impostor*100:.3f}%")
        
        # Check if we meet defensible thresholds
        genuine_bound = 3/total_genuine if total_genuine > 0 else 1.0
        impostor_bound = 3/total_impostor if total_impostor > 0 else 1.0
        
        if genuine_bound <= 0.01 and impostor_bound <= 0.01:  # ≤1% bounds
            print(f"  🎯 DEFENSIBLE: Both error bounds ≤1%")
        elif genuine_bound <= 0.05 and impostor_bound <= 0.05:  # ≤5% bounds
            print(f"  ⚠️  MARGINAL: Error bounds ≤5% but >1%")
        else:
            print(f"  ❌ UNDERPOWERED: Error bounds too loose")

if __name__ == "__main__":
    generate_scaled_validation()