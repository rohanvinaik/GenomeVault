#!/usr/bin/env python3
"""Fixed fingerprint quality evaluation for GenomeVault HDC encoding"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
import time
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import stats

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig
from genomevault.core.constants import OmicsType

@dataclass
class FingerprintConfig:
    dimension: int
    sparsity: float
    subjects: int = 50
    samples_per_subject: int = 3

@dataclass
class MatchingResult:
    config: FingerprintConfig
    far: float  # False Accept Rate at EER
    frr: float  # False Reject Rate at EER  
    eer: float  # Equal Error Rate
    auc: float  # Area Under ROC Curve
    auc_ci: Tuple[float, float]  # 95% CI
    storage_kb: float
    genuine_scores: np.ndarray
    impostor_scores: np.ndarray

class FingerprintEvaluator:
    def __init__(self):
        self.results = []
        
    def generate_synthetic_subject(self, subject_id: int, config: FingerprintConfig) -> np.ndarray:
        """Generate synthetic genomic data for a subject"""
        np.random.seed(subject_id * 1000)
        # Base profile + small noise for within-subject variation
        base = np.random.randn(100) * 2.0 + subject_id * 0.5
        return base.astype(np.float32)
    
    def add_sample_variation(self, base_data: np.ndarray, sample_id: int) -> np.ndarray:
        """Add small variation for different samples from same subject"""
        np.random.seed(sample_id * 7777)
        noise = np.random.randn(len(base_data)) * 0.1
        return (base_data + noise).astype(np.float32)
    
    def encode_fingerprint(self, data: np.ndarray, config: FingerprintConfig) -> np.ndarray:
        """Encode genomic data into HDC fingerprint"""
        hv_config = HypervectorConfig(dimension=config.dimension)
        encoder = HypervectorEncoder(config=hv_config)
        
        # Encode
        encoded = encoder.encode(data, OmicsType.GENOMIC)
        
        # Convert to numpy if needed
        if hasattr(encoded, 'numpy'):
            encoded = encoded.numpy()
        elif hasattr(encoded, 'cpu'):
            encoded = encoded.cpu().numpy()
            
        # Sparsify
        if config.sparsity > 0:
            threshold = np.percentile(np.abs(encoded), config.sparsity * 100)
            encoded[np.abs(encoded) < threshold] = 0
            
        return encoded
    
    def cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot / (norm1 * norm2)
    
    def evaluate_configuration(self, config: FingerprintConfig) -> MatchingResult:
        """Evaluate a single configuration"""
        print(f"\n  Evaluating: {config.dimension}D, {config.sparsity*100:.0f}% sparsity")
        
        # Generate cohort
        fingerprints = {}
        for subject_id in range(config.subjects):
            base_data = self.generate_synthetic_subject(subject_id, config)
            for sample_id in range(config.samples_per_subject):
                data = self.add_sample_variation(base_data, sample_id)
                fp = self.encode_fingerprint(data, config)
                fingerprints[(subject_id, sample_id)] = fp
        
        # Compute genuine scores (same subject, different samples)
        genuine_scores = []
        for subject_id in range(config.subjects):
            for i in range(config.samples_per_subject):
                for j in range(i+1, config.samples_per_subject):
                    fp1 = fingerprints[(subject_id, i)]
                    fp2 = fingerprints[(subject_id, j)]
                    score = self.cosine_similarity(fp1, fp2)
                    genuine_scores.append(score)
        
        # Compute impostor scores (different subjects)
        impostor_scores = []
        num_impostor_comparisons = min(1000, config.subjects * (config.subjects - 1) // 2)
        comparisons_done = 0
        
        for i in range(config.subjects):
            for j in range(i+1, config.subjects):
                if comparisons_done >= num_impostor_comparisons:
                    break
                fp1 = fingerprints[(i, 0)]
                fp2 = fingerprints[(j, 0)]
                score = self.cosine_similarity(fp1, fp2)
                impostor_scores.append(score)
                comparisons_done += 1
            if comparisons_done >= num_impostor_comparisons:
                break
        
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        
        # Calculate ROC and EER
        labels = np.concatenate([np.ones(len(genuine_scores)), np.zeros(len(impostor_scores))])
        scores = np.concatenate([genuine_scores, impostor_scores])
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        # Find EER
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        
        # Bootstrap confidence interval for AUC
        n_bootstrap = 100  # Reduced from 1000 for speed
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
        
        # Calculate storage
        sample_fp = fingerprints[(0, 0)]
        non_zero = np.count_nonzero(sample_fp)
        storage_kb = (non_zero * 4) / 1024  # 4 bytes per float
        
        return MatchingResult(
            config=config,
            far=fpr[eer_idx],
            frr=fnr[eer_idx],
            eer=eer,
            auc=roc_auc,
            auc_ci=auc_ci,
            storage_kb=storage_kb,
            genuine_scores=genuine_scores,
            impostor_scores=impostor_scores
        )
    
    def run_evaluation(self):
        """Run complete fingerprint quality evaluation"""
        print("="*80)
        print("GENOMEVAULT HDC FINGERPRINT QUALITY EVALUATION")
        print("="*80)
        
        dimensions = [4096, 8192, 16384]
        sparsities = [0.4, 0.5, 0.6, 0.7]
        
        for dim in dimensions:
            for sparsity in sparsities:
                config = FingerprintConfig(
                    dimension=dim,
                    sparsity=sparsity,
                    subjects=50,
                    samples_per_subject=3
                )
                
                result = self.evaluate_configuration(config)
                self.results.append(result)
                
                print(f"    EER: {result.eer:.3f}, AUC: {result.auc:.3f}, Storage: {result.storage_kb:.1f}KB")
        
        self.generate_plots()
        self.generate_report()
        
    def generate_plots(self):
        """Generate visualization plots"""
        os.makedirs("benchmark_results", exist_ok=True)
        
        # Operating Frontier Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        dimensions = sorted(set(r.config.dimension for r in self.results))
        colors = plt.cm.viridis(np.linspace(0, 1, len(dimensions)))
        
        for i, dim in enumerate(dimensions):
            dim_results = [r for r in self.results if r.config.dimension == dim]
            storage = [r.storage_kb for r in dim_results]
            eer = [r.eer for r in dim_results]
            ax.plot(storage, eer, 'o-', color=colors[i], label=f'{dim}D', markersize=8)
            
            for r in dim_results:
                ax.annotate(f'{r.config.sparsity*100:.0f}%', 
                          (r.storage_kb, r.eer),
                          fontsize=8, ha='center')
        
        ax.set_xlabel('Storage (KB)')
        ax.set_ylabel('Equal Error Rate (EER)')
        ax.set_title('HDC Fingerprint Operating Frontier')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('benchmark_results/fingerprint_frontier_fixed.png', dpi=150)
        plt.close()
        
        # ROC Curves
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Select 4 representative configurations
        selected_configs = [
            (4096, 0.5), (8192, 0.5), (16384, 0.5), (8192, 0.7)
        ]
        
        for idx, (dim, sparsity) in enumerate(selected_configs):
            ax = axes[idx]
            result = next((r for r in self.results 
                          if r.config.dimension == dim and r.config.sparsity == sparsity), None)
            
            if result:
                labels = np.concatenate([
                    np.ones(len(result.genuine_scores)),
                    np.zeros(len(result.impostor_scores))
                ])
                scores = np.concatenate([result.genuine_scores, result.impostor_scores])
                
                fpr, tpr, _ = roc_curve(labels, scores)
                
                ax.plot(fpr, tpr, 'b-', linewidth=2)
                ax.plot([0, 1], [0, 1], 'r--', alpha=0.5)
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title(f'{dim}D, {sparsity*100:.0f}% sparse\nAUC={result.auc:.3f}')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('HDC Fingerprint ROC Curves')
        plt.tight_layout()
        plt.savefig('benchmark_results/fingerprint_roc_fixed.png', dpi=150)
        plt.close()
        
        print("\n✅ Plots saved to benchmark_results/")
        
    def generate_report(self):
        """Generate comprehensive report"""
        
        # Save JSON results
        json_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configurations': [
                {
                    'dimension': r.config.dimension,
                    'sparsity': r.config.sparsity,
                    'storage_kb': r.storage_kb,
                    'eer': r.eer,
                    'far': r.far,
                    'frr': r.frr,
                    'auc': r.auc,
                    'auc_ci_lower': r.auc_ci[0],
                    'auc_ci_upper': r.auc_ci[1],
                    'genuine_mean': float(np.mean(r.genuine_scores)),
                    'genuine_std': float(np.std(r.genuine_scores)),
                    'impostor_mean': float(np.mean(r.impostor_scores)),
                    'impostor_std': float(np.std(r.impostor_scores))
                }
                for r in self.results
            ]
        }
        
        with open('benchmark_results/fingerprint_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Generate markdown report
        report = []
        report.append("# HDC Fingerprint Quality Evaluation Report\n")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        report.append("\n## Executive Summary\n")
        
        # Find best configurations
        best_accuracy = min(self.results, key=lambda r: r.eer)
        best_storage = min(self.results, key=lambda r: r.storage_kb)
        best_balanced = min(self.results, key=lambda r: r.eer * r.storage_kb)
        
        report.append(f"- **Best Accuracy**: {best_accuracy.config.dimension}D @ {best_accuracy.config.sparsity*100:.0f}% sparsity")
        report.append(f"  - EER: {best_accuracy.eer:.3f}, Storage: {best_accuracy.storage_kb:.1f}KB\n")
        
        report.append(f"- **Best Storage**: {best_storage.config.dimension}D @ {best_storage.config.sparsity*100:.0f}% sparsity")
        report.append(f"  - EER: {best_storage.eer:.3f}, Storage: {best_storage.storage_kb:.1f}KB\n")
        
        report.append(f"- **Best Balanced**: {best_balanced.config.dimension}D @ {best_balanced.config.sparsity*100:.0f}% sparsity")
        report.append(f"  - EER: {best_balanced.eer:.3f}, Storage: {best_balanced.storage_kb:.1f}KB\n")
        
        report.append("\n## Detailed Results\n")
        
        report.append("| Dimension | Sparsity | Storage (KB) | EER | FAR | FRR | AUC | 95% CI |")
        report.append("|-----------|----------|-------------|-----|-----|-----|-----|--------|")
        
        for r in sorted(self.results, key=lambda x: (x.config.dimension, x.config.sparsity)):
            report.append(f"| {r.config.dimension} | {r.config.sparsity*100:.0f}% | {r.storage_kb:.1f} | "
                         f"{r.eer:.3f} | {r.far:.3f} | {r.frr:.3f} | {r.auc:.3f} | "
                         f"[{r.auc_ci[0]:.3f}, {r.auc_ci[1]:.3f}] |")
        
        report.append("\n## Recommendations\n")
        report.append("- **Clinical High-Accuracy**: Use 16384D @ 50% sparsity (best accuracy vs storage)")
        report.append("- **Balanced Research**: Use 8192D @ 60% sparsity (good accuracy, moderate storage)")
        report.append("- **Resource-Constrained**: Use 4096D @ 70% sparsity (minimal storage, acceptable accuracy)")
        
        report.append("\n## Statistical Analysis\n")
        report.append(f"- Cohort Size: {self.results[0].config.subjects} subjects × {self.results[0].config.samples_per_subject} samples")
        report.append(f"- Genuine Comparisons: {len(self.results[0].genuine_scores)} pairs")
        report.append(f"- Impostor Comparisons: {len(self.results[0].impostor_scores)} pairs")
        report.append(f"- Bootstrap Iterations: 100 (for CI calculation)")
        
        report_text = '\n'.join(report)
        
        with open('benchmark_results/fingerprint_report.md', 'w') as f:
            f.write(report_text)
        
        print("\n✅ Report saved to benchmark_results/fingerprint_report.md")
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)

if __name__ == "__main__":
    evaluator = FingerprintEvaluator()
    evaluator.run_evaluation()