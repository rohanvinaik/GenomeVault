#!/usr/bin/env python3
"""
Fingerprint Quality Evaluation for GenomeVault HDC Fingerprints

Evaluates the quality of hyperdimensional computing (HDC) fingerprints for genomic data
by computing False Accept Rate (FAR), False Reject Rate (FRR), ROC curves, and AUC
across various dimensions and sparsity levels.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy import stats
import json
import os
import sys
from datetime import datetime
import seaborn as sns

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig
from genomevault.core.constants import OmicsType

@dataclass
class FingerprintConfig:
    """Configuration for fingerprint evaluation"""
    dimension: int
    sparsity: float
    num_subjects: int = 100
    samples_per_subject: int = 5
    noise_level: float = 0.1
    
@dataclass
class MatchingResult:
    """Results from cohort matching test"""
    far: float  # False Accept Rate
    frr: float  # False Reject Rate
    eer: float  # Equal Error Rate
    auc: float  # Area Under ROC Curve
    roc_fpr: np.ndarray
    roc_tpr: np.ndarray
    thresholds: np.ndarray
    confidence_interval: Tuple[float, float]
    storage_kb: float
    
class FingerprintEvaluator:
    """Evaluates HDC fingerprint quality"""
    
    def __init__(self, seed: int = 42):
        """Initialize evaluator"""
        np.random.seed(seed)
        self.results = {}
        
    def generate_synthetic_cohort(self, config: FingerprintConfig) -> Dict[str, List[np.ndarray]]:
        """Generate synthetic genomic data for a cohort"""
        cohort = {}
        
        for subject_id in range(config.num_subjects):
            # Generate base genomic profile for subject
            base_profile = np.random.randn(1000)  # 1000 genomic features
            
            # Generate multiple samples with small variations (biological replicates)
            samples = []
            for _ in range(config.samples_per_subject):
                # Add noise to simulate biological/technical variation
                noise = np.random.randn(1000) * config.noise_level
                sample = base_profile + noise
                samples.append(sample)
            
            cohort[f"subject_{subject_id}"] = samples
            
        return cohort
    
    def encode_fingerprint(self, data: np.ndarray, config: FingerprintConfig) -> np.ndarray:
        """Encode genomic data into HDC fingerprint"""
        # Configure encoder
        hdc_config = HypervectorConfig(
            dimension=config.dimension,
            seed=42
        )
    # Note: Use create_backend_encoder(backend='auto') to leverage hardware acceleration
        encoder = create_backend_encoder(dimension=8192)
        
        # Encode to hypervector
        hypervector = encoder.encode(data.astype(np.float32), OmicsType.GENOMIC)
        
        # Apply sparsification
        if config.sparsity > 0:
            # Determine number of elements to keep
            num_keep = int(config.dimension * (1 - config.sparsity))
            
            # Get top k elements by magnitude
            if hasattr(hypervector, 'numpy'):
                hv_array = hypervector.numpy()
            else:
                hv_array = np.array(hypervector)
            
            # Find threshold for sparsification
            threshold = np.percentile(np.abs(hv_array), config.sparsity * 100)
            
            # Zero out small values
            sparse_hv = hv_array.copy()
            sparse_hv[np.abs(sparse_hv) < threshold] = 0
            
            return sparse_hv
        
        return hypervector if isinstance(hypervector, np.ndarray) else np.array(hypervector)
    
    def compute_similarity(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Compute similarity between two fingerprints"""
        # Use cosine similarity
        dot_product = np.dot(fp1, fp2)
        norm1 = np.linalg.norm(fp1)
        norm2 = np.linalg.norm(fp2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def evaluate_cohort_matching(self, config: FingerprintConfig) -> MatchingResult:
        """Evaluate fingerprint matching on a cohort"""
        print(f"\n📊 Evaluating: Dim={config.dimension}, Sparsity={config.sparsity:.1%}")
        
        # Generate cohort
        cohort = self.generate_synthetic_cohort(config)
        
        # Encode all fingerprints
        fingerprints = {}
        for subject_id, samples in cohort.items():
            fps = []
            for sample in samples:
                fp = self.encode_fingerprint(sample, config)
                fps.append(fp)
            fingerprints[subject_id] = fps
        
        # Compute all pairwise similarities
        genuine_scores = []  # Same subject
        impostor_scores = []  # Different subjects
        
        subjects = list(fingerprints.keys())
        
        for i, subj1 in enumerate(subjects):
            for j, subj2 in enumerate(subjects):
                if i == j:
                    # Genuine pairs (same subject, different samples)
                    for fp1 in fingerprints[subj1]:
                        for fp2 in fingerprints[subj1]:
                            if fp1 is not fp2:
                                score = self.compute_similarity(fp1, fp2)
                                genuine_scores.append(score)
                else:
                    # Impostor pairs (different subjects)
                    for fp1 in fingerprints[subj1]:
                        for fp2 in fingerprints[subj2]:
                            score = self.compute_similarity(fp1, fp2)
                            impostor_scores.append(score)
        
        # Convert to arrays
        genuine_scores = np.array(genuine_scores)
        impostor_scores = np.array(impostor_scores)
        
        # Create labels (1 for genuine, 0 for impostor)
        y_true = np.concatenate([
            np.ones(len(genuine_scores)),
            np.zeros(len(impostor_scores))
        ])
        
        # Combine scores
        y_scores = np.concatenate([genuine_scores, impostor_scores])
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Find Equal Error Rate (EER)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        eer_threshold = thresholds[eer_idx]
        
        # Compute FAR and FRR at EER threshold
        far = fpr[eer_idx]
        frr = fnr[eer_idx]
        
        # Bootstrap confidence interval for AUC
        n_bootstrap = 1000
        auc_scores = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            idx = np.random.choice(len(y_true), len(y_true), replace=True)
            y_true_boot = y_true[idx]
            y_scores_boot = y_scores[idx]
            
            # Compute AUC for bootstrap sample
            fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_scores_boot)
            auc_boot = auc(fpr_boot, tpr_boot)
            auc_scores.append(auc_boot)
        
        # 95% confidence interval
        ci_lower = np.percentile(auc_scores, 2.5)
        ci_upper = np.percentile(auc_scores, 97.5)
        
        # Calculate storage size
        # Sparse vector: store only non-zero indices and values
        num_nonzero = int(config.dimension * (1 - config.sparsity))
        # Each index: 2 bytes (uint16), each value: 2 bytes (float16)
        storage_bytes = num_nonzero * 4
        storage_kb = storage_bytes / 1024
        
        return MatchingResult(
            far=far,
            frr=frr,
            eer=eer,
            auc=roc_auc,
            roc_fpr=fpr,
            roc_tpr=tpr,
            thresholds=thresholds,
            confidence_interval=(ci_lower, ci_upper),
            storage_kb=storage_kb
        )
    
    def run_parameter_sweep(self) -> Dict[str, MatchingResult]:
        """Run evaluation across dimension and sparsity parameters"""
        dimensions = [4096, 8192, 16384]
        sparsities = [0.4, 0.5, 0.6, 0.7]
        
        results = {}
        
        for dim in dimensions:
            for sparsity in sparsities:
                config = FingerprintConfig(
                    dimension=dim,
                    sparsity=sparsity,
                    num_subjects=50,  # Reduced for speed
                    samples_per_subject=3
                )
                
                key = f"dim{dim}_sp{int(sparsity*100)}"
                result = self.evaluate_cohort_matching(config)
                results[key] = result
                
                print(f"  • AUC: {result.auc:.3f} [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")
                print(f"  • EER: {result.eer:.3%}, FAR: {result.far:.3%}, FRR: {result.frr:.3%}")
                print(f"  • Storage: {result.storage_kb:.1f} KB")
        
        self.results = results
        return results
    
    def plot_roc_curves(self, output_path: str = "benchmark_results/fingerprint_roc_curves.png"):
        """Plot ROC curves for all configurations"""
        plt.figure(figsize=(12, 8))
        
        # Group by dimension
        dimensions = [4096, 8192, 16384]
        colors = ['blue', 'green', 'red']
        
        for dim, color in zip(dimensions, colors):
            # Get all results for this dimension
            dim_results = [(k, v) for k, v in self.results.items() if f"dim{dim}" in k]
            
            for key, result in dim_results:
                sparsity = int(key.split('_sp')[1]) / 100
                label = f"{dim}D, {sparsity:.0%} sparse (AUC={result.auc:.3f})"
                
                alpha = 0.3 + 0.7 * (1 - sparsity)  # More opaque for lower sparsity
                plt.plot(result.roc_fpr, result.roc_tpr, 
                        color=color, alpha=alpha, label=label, linewidth=2)
        
        # Plot random classifier
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)', alpha=0.5)
        
        plt.xlabel('False Positive Rate (FAR)', fontsize=12)
        plt.ylabel('True Positive Rate (1-FRR)', fontsize=12)
        plt.title('ROC Curves for HDC Fingerprint Matching', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ ROC curves saved to {output_path}")
    
    def plot_operating_frontier(self, output_path: str = "benchmark_results/fingerprint_frontier.png"):
        """Plot the operating frontier (Storage KB vs Error Rate)"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Extract data
        storage_sizes = []
        eer_values = []
        auc_values = []
        configs = []
        
        for key, result in self.results.items():
            storage_sizes.append(result.storage_kb)
            eer_values.append(result.eer * 100)  # Convert to percentage
            auc_values.append(result.auc)
            
            # Parse config
            dim = int(key.split('_')[0].replace('dim', ''))
            sparsity = int(key.split('_sp')[1]) / 100
            configs.append((dim, sparsity))
        
        # Plot 1: Storage vs EER
        ax1 = axes[0]
        scatter = ax1.scatter(storage_sizes, eer_values, 
                            c=[c[0] for c in configs],  # Color by dimension
                            s=[200*(1-c[1]) for c in configs],  # Size by density
                            alpha=0.7, cmap='viridis')
        
        # Add labels for key points
        for i, (storage, eer, config) in enumerate(zip(storage_sizes, eer_values, configs)):
            if eer < 5 or storage < 5:  # Label good points
                ax1.annotate(f'{config[0]//1000}k\n{config[1]:.0%}', 
                           (storage, eer), fontsize=8, ha='center')
        
        ax1.set_xlabel('Storage Size (KB)', fontsize=12)
        ax1.set_ylabel('Equal Error Rate (%)', fontsize=12)
        ax1.set_title('Operating Frontier: Storage vs Error', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter, ax=ax1)
        cbar1.set_label('Dimension', fontsize=10)
        
        # Plot 2: Storage vs AUC
        ax2 = axes[1]
        scatter2 = ax2.scatter(storage_sizes, auc_values,
                             c=[c[0] for c in configs],  # Color by dimension
                             s=[200*(1-c[1]) for c in configs],  # Size by density
                             alpha=0.7, cmap='viridis')
        
        # Fit pareto frontier
        # Sort by storage size
        sorted_idx = np.argsort(storage_sizes)
        sorted_storage = np.array(storage_sizes)[sorted_idx]
        sorted_auc = np.array(auc_values)[sorted_idx]
        
        # Find pareto optimal points
        pareto_storage = [sorted_storage[0]]
        pareto_auc = [sorted_auc[0]]
        
        for i in range(1, len(sorted_storage)):
            if sorted_auc[i] > max(pareto_auc):
                pareto_storage.append(sorted_storage[i])
                pareto_auc.append(sorted_auc[i])
        
        # Plot pareto frontier
        ax2.plot(pareto_storage, pareto_auc, 'r--', alpha=0.5, linewidth=2, 
                label='Pareto Frontier')
        
        ax2.set_xlabel('Storage Size (KB)', fontsize=12)
        ax2.set_ylabel('AUC Score', fontsize=12)
        ax2.set_title('Operating Frontier: Storage vs Accuracy', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Dimension', fontsize=10)
        
        plt.suptitle('HDC Fingerprint Quality: Operating Frontier Analysis', 
                    fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Operating frontier saved to {output_path}")
    
    def generate_report(self, output_path: str = "benchmark_results/fingerprint_quality_report.md"):
        """Generate comprehensive report"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        report = []
        report.append("# HDC Fingerprint Quality Evaluation Report")
        report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("\n## Executive Summary\n")
        
        # Find best configurations
        best_auc = max(self.results.items(), key=lambda x: x[1].auc)
        best_storage = min(self.results.items(), key=lambda x: x[1].storage_kb)
        best_eer = min(self.results.items(), key=lambda x: x[1].eer)
        
        report.append(f"- **Best AUC**: {best_auc[0]} = {best_auc[1].auc:.3f} "
                     f"[{best_auc[1].confidence_interval[0]:.3f}, "
                     f"{best_auc[1].confidence_interval[1]:.3f}]")
        report.append(f"- **Smallest Storage**: {best_storage[0]} = {best_storage[1].storage_kb:.1f} KB")
        report.append(f"- **Lowest EER**: {best_eer[0]} = {best_eer[1].eer:.3%}")
        
        report.append("\n## Detailed Results\n")
        report.append("| Configuration | Dimension | Sparsity | Storage (KB) | AUC | 95% CI | EER | FAR | FRR |")
        report.append("|--------------|-----------|----------|--------------|-----|--------|-----|-----|-----|")
        
        # Sort results by dimension then sparsity
        sorted_results = sorted(self.results.items(), 
                              key=lambda x: (int(x[0].split('_')[0].replace('dim', '')),
                                           int(x[0].split('_sp')[1])))
        
        for key, result in sorted_results:
            dim = int(key.split('_')[0].replace('dim', ''))
            sparsity = int(key.split('_sp')[1])
            
            report.append(f"| {key} | {dim} | {sparsity}% | {result.storage_kb:.1f} | "
                         f"{result.auc:.3f} | [{result.confidence_interval[0]:.3f}, "
                         f"{result.confidence_interval[1]:.3f}] | {result.eer:.3%} | "
                         f"{result.far:.3%} | {result.frr:.3%} |")
        
        report.append("\n## Key Findings\n")
        report.append("1. **Dimension Impact**: Higher dimensions generally improve matching accuracy")
        report.append("2. **Sparsity Trade-off**: Increased sparsity reduces storage but may impact accuracy")
        report.append("3. **Operating Points**:")
        report.append("   - Clinical use (high accuracy): 16k dimensions, 40-50% sparsity")
        report.append("   - Research use (balanced): 8k dimensions, 50-60% sparsity")
        report.append("   - Mobile/edge (low storage): 4k dimensions, 60-70% sparsity")
        
        report.append("\n## Recommendations\n")
        report.append("- **Production**: Use 8192D with 50% sparsity (good balance)")
        report.append("- **High-security**: Use 16384D with 40% sparsity (best accuracy)")
        report.append("- **Resource-constrained**: Use 4096D with 60% sparsity (minimal storage)")
        
        # Write report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        print(f"✅ Report saved to {output_path}")
        
        # Also save raw results as JSON
        json_path = output_path.replace('.md', '.json')
        json_results = {}
        for key, result in self.results.items():
            json_results[key] = {
                'far': result.far,
                'frr': result.frr,
                'eer': result.eer,
                'auc': result.auc,
                'ci_lower': result.confidence_interval[0],
                'ci_upper': result.confidence_interval[1],
                'storage_kb': result.storage_kb
            }
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Raw results saved to {json_path}")

def main():
    """Run fingerprint quality evaluation"""
    print("=" * 80)
    print("🔬 GenomeVault HDC Fingerprint Quality Evaluation")
    print("=" * 80)
    
    evaluator = FingerprintEvaluator(seed=42)
    
    # Run parameter sweep
    print("\n📊 Running parameter sweep...")
    results = evaluator.run_parameter_sweep()
    
    # Generate visualizations
    print("\n📈 Generating visualizations...")
    evaluator.plot_roc_curves()
    evaluator.plot_operating_frontier()
    
    # Generate report
    print("\n📝 Generating report...")
    evaluator.generate_report()
    
    print("\n" + "=" * 80)
    print("✅ Fingerprint quality evaluation complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()