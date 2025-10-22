#!/usr/bin/env python3
"""Attribute inference experiment to test information leakage in hypervectors.

This experiment tests whether sensitive attributes (e.g., ancestry, disease status)
can be inferred from hypervectors using various attack strategies.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import logging

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from genomevault.hypervector_transform.secure_encoder import (
    SecureHypervectorEncoder,
    SessionConfig,
    RateLimitConfig,
    HypervectorConfig,
)
from genomevault.core.constants import OmicsType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttributeInferenceAttack:
    """Simulates attribute inference attacks on hypervectors."""
    
    def __init__(self, n_samples: int = 1000, n_features: int = 400000):
        """Initialize attack simulation.
        
        Args:
            n_samples: Number of synthetic genomes to generate
            n_features: Number of genomic features (variants)
        """
        self.n_samples = n_samples
        self.n_features = n_features
        
        # Generate synthetic genomic data with known attributes
        self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Generate synthetic genomic data with correlated attributes."""
        np.random.seed(42)  # For reproducibility
        
        # Generate base genomic data (random variants)
        self.genomic_data = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        
        # Add structure for different attributes
        # Attribute 1: Ancestry (3 populations)
        self.ancestry_labels = np.random.choice([0, 1, 2], self.n_samples)
        
        # Add population-specific signals (very subtle)
        for pop in range(3):
            mask = self.ancestry_labels == pop
            # Add small population-specific bias to certain variants
            population_variants = np.random.choice(self.n_features, 100, replace=False)
            self.genomic_data[mask][:, population_variants] += np.random.randn(mask.sum(), 100) * 0.1
        
        # Attribute 2: Disease status (binary)
        self.disease_labels = np.random.choice([0, 1], self.n_samples, p=[0.7, 0.3])
        
        # Add disease-associated variants
        disease_variants = np.random.choice(self.n_features, 50, replace=False)
        disease_mask = self.disease_labels == 1
        self.genomic_data[disease_mask][:, disease_variants] += np.random.randn(disease_mask.sum(), 50) * 0.15
        
        # Attribute 3: Sex (binary)
        self.sex_labels = np.random.choice([0, 1], self.n_samples, p=[0.5, 0.5])
        
        # Add sex-specific signals (e.g., X/Y chromosome)
        sex_variants = np.random.choice(self.n_features, 200, replace=False)
        sex_mask = self.sex_labels == 1
        self.genomic_data[sex_mask][:, sex_variants] += np.random.randn(sex_mask.sum(), 200) * 0.08
        
        logger.info(f"Generated {self.n_samples} synthetic genomes with {self.n_features} features")
        logger.info(f"Ancestry distribution: {np.bincount(self.ancestry_labels)}")
        logger.info(f"Disease prevalence: {self.disease_labels.mean():.1%}")
        logger.info(f"Sex distribution: {np.bincount(self.sex_labels)}")
    
    def run_attack(
        self,
        encoder: SecureHypervectorEncoder,
        attribute: str = "ancestry",
        attack_model: str = "logistic",
    ) -> Dict[str, float]:
        """Run attribute inference attack.
        
        Args:
            encoder: Hypervector encoder to attack
            attribute: Which attribute to try to infer
            attack_model: Type of attack model to use
            
        Returns:
            Dictionary of attack metrics
        """
        # Select target labels
        if attribute == "ancestry":
            labels = self.ancestry_labels
            multiclass = True
        elif attribute == "disease":
            labels = self.disease_labels
            multiclass = False
        elif attribute == "sex":
            labels = self.sex_labels
            multiclass = False
        else:
            raise ValueError(f"Unknown attribute: {attribute}")
        
        logger.info(f"Running {attribute} inference attack with {attack_model} model")
        
        # Encode all samples
        encoded_vectors = []
        for i in range(self.n_samples):
            try:
                encoded, _ = encoder.encode_secure(
                    self.genomic_data[i],
                    OmicsType.GENOMIC,
                    client_id=f"attacker_{i % 10}",  # Simulate 10 attackers
                )
                encoded_vectors.append(encoded)
            except PermissionError:
                # Rate limited - use previous encoding
                if encoded_vectors:
                    encoded_vectors.append(encoded_vectors[-1])
                else:
                    encoded_vectors.append(np.zeros(encoder.base_encoder.config.dimension))
        
        X = np.array(encoded_vectors)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, labels, test_size=0.3, random_state=42, stratify=labels
        )
        
        # Train attack model
        if attack_model == "logistic":
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif attack_model == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown attack model: {attack_model}")
        
        model.fit(X_train, y_train)
        
        # Evaluate attack
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate AUC for binary or use accuracy for multiclass
        if not multiclass:
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = accuracy  # Use accuracy as proxy for multiclass
        
        # Calculate baseline (random guessing)
        if multiclass:
            baseline = 1.0 / len(np.unique(labels))
        else:
            baseline = max(labels.mean(), 1 - labels.mean())
        
        # Information leakage estimate
        # Using mutual information approximation
        from sklearn.metrics import mutual_info_score
        mi = mutual_info_score(y_test, y_pred)
        
        results = {
            "attribute": attribute,
            "attack_model": attack_model,
            "accuracy": accuracy,
            "auc": auc,
            "baseline_accuracy": baseline,
            "improvement_over_baseline": accuracy - baseline,
            "mutual_information_bits": mi,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }
        
        logger.info(f"Attack accuracy: {accuracy:.3f} (baseline: {baseline:.3f})")
        logger.info(f"Improvement over baseline: {accuracy - baseline:.3f}")
        logger.info(f"Estimated MI: {mi:.4f} bits")
        
        return results


def run_comprehensive_experiment(output_dir: Path):
    """Run comprehensive attribute inference experiment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test configurations
    configs = [
        {
            "name": "no_protection",
            "session_config": SessionConfig(
                enable_randomization=False,
                noise_sigma=0.0,
            ),
        },
        {
            "name": "with_randomization",
            "session_config": SessionConfig(
                enable_randomization=True,
                noise_sigma=0.0,
            ),
        },
        {
            "name": "with_noise",
            "session_config": SessionConfig(
                enable_randomization=False,
                noise_sigma=0.001,
            ),
        },
        {
            "name": "full_protection",
            "session_config": SessionConfig(
                enable_randomization=True,
                noise_sigma=0.001,
            ),
        },
    ]
    
    attributes = ["ancestry", "disease", "sex"]
    attack_models = ["logistic", "random_forest"]
    
    all_results = []
    
    for config in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing configuration: {config['name']}")
        logger.info(f"{'='*60}")
        
        # Create encoder with specific config
        encoder = SecureHypervectorEncoder(
            base_config=HypervectorConfig(dimension=8192)
    # Note: Use create_backend_encoder(backend='auto') to leverage hardware acceleration,
            rate_limit_config=RateLimitConfig(
                max_queries_per_day=10000,  # High limit for testing
                max_queries_per_hour=5000,
                max_queries_per_minute=100,
            ),
            session_config=config["session_config"],
        )
        
        # Create attacker
        attacker = AttributeInferenceAttack(n_samples=500, n_features=10000)  # Smaller for speed
        
        for attribute in attributes:
            for attack_model in attack_models:
                logger.info(f"\nAttribute: {attribute}, Model: {attack_model}")
                
                results = attacker.run_attack(encoder, attribute, attack_model)
                results["configuration"] = config["name"]
                all_results.append(results)
    
    # Save results
    results_file = output_dir / "attribute_inference_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary report
    generate_report(all_results, output_dir)
    
    logger.info(f"\nResults saved to {results_file}")
    
    return all_results


def generate_report(results: List[Dict], output_dir: Path):
    """Generate markdown report from results."""
    report_file = output_dir / "attribute_inference_report.md"
    
    with open(report_file, "w") as f:
        f.write("# Attribute Inference Experiment Results\n\n")
        f.write("## Summary\n\n")
        
        # Group by configuration
        configs = {}
        for r in results:
            config = r["configuration"]
            if config not in configs:
                configs[config] = []
            configs[config].append(r)
        
        f.write("| Configuration | Avg Accuracy | Avg Improvement | Max MI (bits) |\n")
        f.write("|--------------|--------------|-----------------|---------------|\n")
        
        for config, config_results in configs.items():
            avg_acc = np.mean([r["accuracy"] for r in config_results])
            avg_imp = np.mean([r["improvement_over_baseline"] for r in config_results])
            max_mi = max([r["mutual_information_bits"] for r in config_results])
            
            f.write(f"| {config} | {avg_acc:.3f} | {avg_imp:+.3f} | {max_mi:.4f} |\n")
        
        f.write("\n## Detailed Results\n\n")
        
        # Detailed table
        f.write("| Config | Attribute | Model | Accuracy | Baseline | Improvement | MI (bits) |\n")
        f.write("|--------|-----------|-------|----------|----------|-------------|----------|\n")
        
        for r in results:
            f.write(f"| {r['configuration']} | {r['attribute']} | {r['attack_model']} | "
                   f"{r['accuracy']:.3f} | {r['baseline_accuracy']:.3f} | "
                   f"{r['improvement_over_baseline']:+.3f} | {r['mutual_information_bits']:.4f} |\n")
        
        f.write("\n## Security Assessment\n\n")
        
        # Find worst case
        worst_case = max(results, key=lambda x: x["improvement_over_baseline"])
        
        f.write(f"**Worst Case Leakage:**\n")
        f.write(f"- Configuration: {worst_case['configuration']}\n")
        f.write(f"- Attribute: {worst_case['attribute']}\n")
        f.write(f"- Attack Model: {worst_case['attack_model']}\n")
        f.write(f"- Accuracy: {worst_case['accuracy']:.3f}\n")
        f.write(f"- Improvement over baseline: {worst_case['improvement_over_baseline']:.3f}\n")
        f.write(f"- Information leakage: {worst_case['mutual_information_bits']:.4f} bits\n\n")
        
        # Check if mitigations work
        no_protection_avg = np.mean([r["accuracy"] for r in results if r["configuration"] == "no_protection"])
        full_protection_avg = np.mean([r["accuracy"] for r in results if r["configuration"] == "full_protection"])
        reduction = (no_protection_avg - full_protection_avg) / no_protection_avg * 100
        
        f.write(f"**Mitigation Effectiveness:**\n")
        f.write(f"- No protection accuracy: {no_protection_avg:.3f}\n")
        f.write(f"- Full protection accuracy: {full_protection_avg:.3f}\n")
        f.write(f"- Attack success reduction: {reduction:.1f}%\n")
    
    logger.info(f"Report saved to {report_file}")


if __name__ == "__main__":
    output_dir = Path("benchmark_results/attribute_inference")
    results = run_comprehensive_experiment(output_dir)
    
    print("\n" + "="*60)
    print("ATTRIBUTE INFERENCE EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"Number of experiments: {len(results)}")
    
    # Print quick summary
    avg_accuracy = np.mean([r["accuracy"] for r in results])
    max_mi = max([r["mutual_information_bits"] for r in results])
    
    print(f"Average attack accuracy: {avg_accuracy:.3f}")
    print(f"Maximum information leakage: {max_mi:.4f} bits")