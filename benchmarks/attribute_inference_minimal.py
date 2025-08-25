#!/usr/bin/env python3
"""Minimal attribute inference experiment for benchmark bundles."""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Simplified test without rate limiting for bundle generation
def run_minimal_attribute_inference():
    """Run minimal attribute inference test."""
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 200
    n_features = 1000
    hypervector_dim = 8192
    
    # Create synthetic hypervectors (simulating encoded genomes)
    # No protection scenario
    X_no_protection = np.random.randn(n_samples, hypervector_dim)
    
    # Add structure for ancestry (3 populations)
    ancestry_labels = np.random.choice([0, 1, 2], n_samples)
    for pop in range(3):
        mask = ancestry_labels == pop
        # Add tiny population signal
        X_no_protection[mask, pop*100:(pop+1)*100] += 0.05
    
    # With randomization (orthogonal transform simulation)
    from scipy.stats import ortho_group
    R = ortho_group.rvs(hypervector_dim)
    X_with_random = X_no_protection @ R
    
    # With noise
    noise = np.random.normal(0, 0.001, X_no_protection.shape)
    X_with_noise = np.sign(X_no_protection + noise)
    
    # Full protection
    X_full_protection = np.sign(X_with_random + noise)
    
    results = []
    
    for name, X in [
        ("no_protection", X_no_protection),
        ("with_randomization", X_with_random),
        ("with_noise", X_with_noise),
        ("full_protection", X_full_protection),
    ]:
        # Train/test split
        split_idx = int(0.7 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = ancestry_labels[:split_idx], ancestry_labels[split_idx:]
        
        # Attack
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        baseline = 1.0 / 3  # Random guessing for 3 classes
        
        results.append({
            "configuration": name,
            "attribute": "ancestry",
            "accuracy": float(accuracy),
            "baseline": float(baseline),
            "improvement": float(accuracy - baseline),
            "n_samples": n_samples,
            "hypervector_dim": hypervector_dim,
        })
    
    # Calculate mitigation effectiveness
    no_prot = next(r for r in results if r["configuration"] == "no_protection")
    full_prot = next(r for r in results if r["configuration"] == "full_protection")
    
    reduction = (no_prot["improvement"] - full_prot["improvement"]) / max(no_prot["improvement"], 0.001) * 100
    
    summary = {
        "experiment": "attribute_inference",
        "timestamp": str(np.datetime64('now')),
        "results": results,
        "mitigation_effectiveness": {
            "no_protection_accuracy": no_prot["accuracy"],
            "full_protection_accuracy": full_prot["accuracy"],
            "attack_success_reduction_percent": float(reduction),
        },
        "security_assessment": {
            "worst_case_accuracy": max(r["accuracy"] for r in results),
            "best_case_accuracy": min(r["accuracy"] for r in results),
            "randomization_effective": results[1]["accuracy"] < results[0]["accuracy"],
            "noise_effective": results[2]["accuracy"] < results[0]["accuracy"],
        }
    }
    
    return summary


if __name__ == "__main__":
    # Run experiment
    results = run_minimal_attribute_inference()
    
    # Save to bundle location
    output_dir = Path("benchmark_results/attribute_inference")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "minimal_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("ATTRIBUTE INFERENCE EXPERIMENT RESULTS")
    print("=" * 40)
    print(f"Configurations tested: {len(results['results'])}")
    print(f"Worst case accuracy: {results['security_assessment']['worst_case_accuracy']:.3f}")
    print(f"Best case accuracy: {results['security_assessment']['best_case_accuracy']:.3f}")
    print(f"Attack reduction with full protection: {results['mitigation_effectiveness']['attack_success_reduction_percent']:.1f}%")
    print(f"\nResults saved to: {output_dir / 'minimal_results.json'}")