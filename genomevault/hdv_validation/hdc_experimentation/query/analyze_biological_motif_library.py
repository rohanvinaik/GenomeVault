#!/usr/bin/env python3
"""
Deep-Dive Analysis for BIOLOGICAL Motif Library

Runs the same three analyses as analyze_motif_library_deepdive.py,
but on REAL biological motifs extracted from chr22 genomic data.

Purpose: Calibrate "corrective lenses" based on interactions between
actual biological sequences (TATA_BOX, CAAT_BOX, etc.) and the HDC system.

Author: Phase 1 Week 3-4
Date: November 22, 2025
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_biological_motif_data(json_path: str):
    """Load biological motif ground truth data."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_features_and_labels(ground_truth):
    """
    Extract feature matrix and labels from biological motif ground truth.

    Args:
        ground_truth: Dict with motif names as keys, each containing:
            - 'chunks': list of samples with 'signals' and 'composition'
            - 'consensus': consensus sequence string
            - 'description': motif description

    Returns:
        X: Feature matrix (n_samples, 23)
        y: Labels (n_samples,)
        feature_names: List of feature names
        motif_names: List of motif names per sample
    """
    samples = []
    labels = []
    motif_names = []

    # First pass: compute mean bank3 values per motif
    motif_bank3_means = {}
    for motif_name, motif_data in ground_truth.items():
        bank3_pos_vals = [chunk['signals']['bank3_pos_mag'] for chunk in motif_data['chunks']]
        bank3_neg_vals = [chunk['signals']['bank3_neg_mag'] for chunk in motif_data['chunks']]
        motif_bank3_means[motif_name] = {
            'bank3_pos_mean': np.mean(bank3_pos_vals),
            'bank3_neg_mean': np.mean(bank3_neg_vals),
        }

    # Second pass: extract features
    for motif_name, motif_data in ground_truth.items():
        for chunk in motif_data['chunks']:
            # Extract all 23 features
            signals = chunk['signals']
            comp = chunk['composition']

            at_content = comp['A_pct'] + comp['T_pct']
            gc_content = comp['G_pct'] + comp['C_pct']

            at_pathway_mag = signals['bank1_pos_mag'] + signals['bank1_neg_mag']
            gc_pathway_mag = signals['bank2_pos_mag'] + signals['bank2_neg_mag']
            total_transition_signal = signals['bank3_pos_mag'] + signals['bank3_neg_mag']

            features = {
                # Bank magnitudes (6)
                'bank1_pos': signals['bank1_pos_mag'],
                'bank1_neg': signals['bank1_neg_mag'],
                'bank2_pos': signals['bank2_pos_mag'],
                'bank2_neg': signals['bank2_neg_mag'],
                'bank3_pos': signals['bank3_pos_mag'],
                'bank3_neg': signals['bank3_neg_mag'],

                # Composition (2)
                'at_content': at_content,
                'gc_content': gc_content,

                # Pathway magnitudes (2)
                'at_pathway_mag': at_pathway_mag,
                'gc_pathway_mag': gc_pathway_mag,

                # Transition signal (1)
                'total_transition_signal': total_transition_signal,

                # Ratios and products (2)
                'bank1_bank2_ratio': signals['bank1_pos_mag'] / (signals['bank2_pos_mag'] + 1e-6),
                'bank1_bank2_product': signals['bank1_pos_mag'] * signals['bank2_pos_mag'],

                # Bank3 asymmetry (1)
                'yr_ry_asymmetry': (signals['bank3_pos_mag'] - signals['bank3_neg_mag']) / (total_transition_signal + 1e-6),

                # Bank3 deviations (2)
                'bank3_pos_deviation': signals['bank3_pos_mag'] - motif_bank3_means[motif_name]['bank3_pos_mean'],
                'bank3_neg_deviation': signals['bank3_neg_mag'] - motif_bank3_means[motif_name]['bank3_neg_mean'],

                # Compositional imbalance (1)
                'compositional_imbalance': abs(at_content - gc_content),
            }

            samples.append(list(features.values()))
            labels.append(motif_name)
            motif_names.append(motif_name)

    X = np.array(samples)
    y = np.array(labels)
    feature_names = list(features.keys())

    return X, y, feature_names, motif_names


def run_analysis_1_confusion_matrix(X, y, feature_names, output_dir):
    """
    Analysis 1: Confusion Matrix Investigation

    Which biological motifs get confused with each other?
    Are there symmetric or asymmetric confusion patterns?
    """
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS 1: CONFUSION MATRIX INVESTIGATION")
    logger.info("="*80)

    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    clf.fit(X, y)

    # Predict on training data (to see confusion patterns)
    y_pred = clf.predict(X)

    # Confusion matrix
    motif_labels = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=motif_labels)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[m.replace('_', '\n') for m in motif_labels],
                yticklabels=[m.replace('_', '\n') for m in motif_labels])
    plt.title(f'Biological Motif Classification Confusion Matrix (n={len(motif_labels)}×100 samples)', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    confusion_matrix_path = output_dir / "biological_confusion_matrix.png"
    plt.savefig(confusion_matrix_path, dpi=150)
    plt.close()

    logger.info(f"✓ Confusion matrix saved to {confusion_matrix_path}")

    # Classification report
    report = classification_report(y, y_pred)
    logger.info("\n=== Classification Report ===")
    logger.info(f"\n{report}")

    # Analyze confusion patterns
    logger.info("\n=== Confusion Pattern Analysis ===")
    n_per_class = len(X) // len(motif_labels)

    for i, true_label in enumerate(motif_labels):
        for j, pred_label in enumerate(motif_labels):
            if i != j and cm[i, j] > 0:
                pct = (cm[i, j] / n_per_class) * 100
                logger.info(f"  {true_label} → {pred_label}: {cm[i, j]} samples ({pct:.1f}%)")

    # Symmetry analysis
    logger.info("\n=== Symmetry Analysis ===")
    for i in range(len(motif_labels)):
        for j in range(i+1, len(motif_labels)):
            if cm[i, j] > 0 or cm[j, i] > 0:
                asymmetry = abs(cm[i, j] - cm[j, i]) / (cm[i, j] + cm[j, i] + 1e-6) * 100
                logger.info(f"  {motif_labels[i]} ↔ {motif_labels[j]}: {cm[i, j]} vs {cm[j, i]} (asymmetry: {asymmetry:.1f}%)")


def run_analysis_2_feature_interactions(X, y, feature_names, output_dir):
    """
    Analysis 2: Feature Interaction Terms

    Do Bank3 × composition interactions improve classification?
    """
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS 2: FEATURE INTERACTION TERMS")
    logger.info("="*80)

    # Baseline: no interactions
    clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    scores_baseline = cross_val_score(clf_baseline, X, y, cv=5)

    logger.info("\n=== Baseline (17 features) ===")
    logger.info(f"Accuracy: {scores_baseline.mean():.4f} ± {scores_baseline.std():.4f}")

    # Augmented: add interactions
    # Interaction features:
    # - yr_ry_asymmetry × at_content
    # - yr_ry_asymmetry × gc_content
    # - yr_ry_asymmetry × compositional_imbalance
    # - bank3_pos × at_content
    # - bank3_pos × gc_content

    yr_ry_asymmetry_idx = feature_names.index('yr_ry_asymmetry')
    at_content_idx = feature_names.index('at_content')
    gc_content_idx = feature_names.index('gc_content')
    comp_imbalance_idx = feature_names.index('compositional_imbalance')
    bank3_pos_idx = feature_names.index('bank3_pos')

    interaction_features = np.column_stack([
        X[:, yr_ry_asymmetry_idx] * X[:, at_content_idx],  # yr_ry_asym × at
        X[:, yr_ry_asymmetry_idx] * X[:, gc_content_idx],  # yr_ry_asym × gc
        X[:, yr_ry_asymmetry_idx] * X[:, comp_imbalance_idx],  # yr_ry_asym × imbalance
        X[:, bank3_pos_idx] * X[:, at_content_idx],  # bank3_pos × at
        X[:, bank3_pos_idx] * X[:, gc_content_idx],  # bank3_pos × gc
    ])

    X_augmented = np.hstack([X, interaction_features])

    augmented_feature_names = feature_names + [
        'yr_ry_asym_x_at',
        'yr_ry_asym_x_gc',
        'yr_ry_asym_x_imbalance',
        'bank3_pos_x_at',
        'bank3_pos_x_gc',
    ]

    clf_augmented = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    scores_augmented = cross_val_score(clf_augmented, X_augmented, y, cv=5)

    logger.info("\n=== Augmented ({} features with interactions) ===".format(X_augmented.shape[1]))
    logger.info(f"Accuracy: {scores_augmented.mean():.4f} ± {scores_augmented.std():.4f}")

    improvement_abs = scores_augmented.mean() - scores_baseline.mean()
    improvement_pct = (improvement_abs / scores_baseline.mean()) * 100

    logger.info("\n=== Improvement ===")
    logger.info(f"Absolute: {improvement_abs:+.4f}")
    logger.info(f"Relative: {improvement_pct:+.2f}%")

    if improvement_abs > 0.01:
        logger.info("✓ Significant improvement - Interactions help!")
    else:
        logger.info("✗ No significant improvement - Interactions may not help")

    # Feature importance comparison
    clf_augmented.fit(X_augmented, y)
    importances = clf_augmented.feature_importances_

    top_k = 15
    top_indices = np.argsort(importances)[-top_k:][::-1]

    logger.info(f"\n=== Top {top_k} Features (Augmented Model) ===")
    for i, idx in enumerate(top_indices, 1):
        logger.info(f"  {i}. {augmented_feature_names[idx]}: {importances[idx]:.4f}")

    # Plot feature importance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Baseline
    clf_baseline.fit(X, y)
    baseline_importances = clf_baseline.feature_importances_
    top_baseline_indices = np.argsort(baseline_importances)[-15:]
    ax1.barh(range(15), baseline_importances[top_baseline_indices])
    ax1.set_yticks(range(15))
    ax1.set_yticklabels([feature_names[i] for i in top_baseline_indices])
    ax1.set_xlabel('Importance')
    ax1.set_title(f'Baseline ({len(feature_names)} features)\nAccuracy = {scores_baseline.mean():.3f} ± {scores_baseline.std():.3f}')

    # Augmented
    top_augmented_indices = np.argsort(importances)[-15:]
    ax2.barh(range(15), importances[top_augmented_indices])
    ax2.set_yticks(range(15))
    ax2.set_yticklabels([augmented_feature_names[i] for i in top_augmented_indices])
    ax2.set_xlabel('Importance')
    ax2.set_title(f'Augmented ({len(augmented_feature_names)} features)\nAccuracy = {scores_augmented.mean():.3f} ± {scores_augmented.std():.3f}')

    plt.tight_layout()

    importance_path = output_dir / "biological_interaction_feature_importance.png"
    plt.savefig(importance_path, dpi=150)
    plt.close()

    logger.info(f"\n✓ Feature importance comparison saved to {importance_path}")


def run_analysis_3_hierarchical_classifier(X, y, feature_names, output_dir):
    """
    Analysis 3: Threshold-Based Hierarchical Classifier

    Can simple bank magnitude thresholds classify biological motifs?
    """
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS 3: THRESHOLD-BASED HIERARCHICAL CLASSIFIER")
    logger.info("="*80)

    # Define thresholds (these are arbitrary - biological motifs don't have
    # simple threshold rules like synthetic motifs)
    logger.info("\n=== Heuristic Thresholds (Biological Motifs) ===")
    logger.info("  Note: Biological motifs don't follow simple threshold rules")
    logger.info("  This analysis is expected to perform poorly")

    # Extract features
    bank1_pos_idx = feature_names.index('bank1_pos')
    bank2_pos_idx = feature_names.index('bank2_pos')
    yr_ry_asymmetry_idx = feature_names.index('yr_ry_asymmetry')

    bank1_pos = X[:, bank1_pos_idx]
    bank2_pos = X[:, bank2_pos_idx]
    yr_ry_asym = X[:, yr_ry_asymmetry_idx]

    # Predict using simple heuristics
    y_pred = np.full(len(y), 'UNKNOWN', dtype='<U30')

    # These heuristics are intentionally naive (biological motifs are complex)
    for i in range(len(X)):
        if bank2_pos[i] > bank1_pos[i]:
            y_pred[i] = 'GC_BOX'  # GC-rich
        elif bank1_pos[i] > bank2_pos[i]:
            y_pred[i] = 'TATA_BOX'  # AT-rich
        elif yr_ry_asym[i] > 0.1:
            y_pred[i] = 'ALU_CONSENSUS_5'
        elif yr_ry_asym[i] < -0.1:
            y_pred[i] = 'LINE1_5'
        else:
            y_pred[i] = 'CAAT_BOX'  # Default

    # Compute accuracy
    accuracy = np.mean(y_pred == y)

    logger.info(f"\n=== Hierarchical Classifier Accuracy ===")
    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Confusion matrix
    motif_labels = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=motif_labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=[m.replace('_', '\n') for m in motif_labels],
                yticklabels=[m.replace('_', '\n') for m in motif_labels])
    plt.title(f'Hierarchical Classifier Confusion Matrix\nAccuracy: {accuracy:.3f}', fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    hierarchical_cm_path = output_dir / "biological_hierarchical_confusion_matrix.png"
    plt.savefig(hierarchical_cm_path, dpi=150)
    plt.close()

    logger.info(f"✓ Hierarchical confusion matrix saved to {hierarchical_cm_path}")

    # Classification report
    report = classification_report(y, y_pred, zero_division=0)
    logger.info("\n=== Classification Report ===")
    logger.info(f"\n{report}")

    # Compare to Random Forest
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_scores = cross_val_score(clf_rf, X, y, cv=5)

    logger.info("\n=== Comparison ===")
    logger.info(f"Hierarchical:  {accuracy:.4f}")
    logger.info(f"Random Forest: {rf_scores.mean():.4f}")

    if rf_scores.mean() > accuracy:
        diff = rf_scores.mean() - accuracy
        logger.info(f"✗ Random Forest is {diff:.4f} better ({diff*100:.2f}% difference)")
    else:
        logger.info("✓ Hierarchical is competitive")


if __name__ == '__main__':
    # Paths
    ground_truth_path = "genomevault/hdv_validation/hdc_experimentation/output/biological_motif_ground_truth.json"
    output_dir = Path("genomevault/hdv_validation/hdc_experimentation/output/biological_motif_deep_dive")
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("\n" + "="*80)
    logger.info("LOADING BIOLOGICAL MOTIF GROUND TRUTH DATA")
    logger.info("="*80)

    # Load data
    ground_truth = load_biological_motif_data(ground_truth_path)
    X, y, feature_names, motif_names = extract_features_and_labels(ground_truth)

    logger.info(f"\nLoaded {len(X)} samples with {len(feature_names)} features")
    motif_counts = {motif: np.sum(y == motif) for motif in set(y)}
    logger.info(f"Motif distribution: {motif_counts}")

    # Run analyses
    run_analysis_1_confusion_matrix(X, y, feature_names, output_dir)
    run_analysis_2_feature_interactions(X, y, feature_names, output_dir)
    run_analysis_3_hierarchical_classifier(X, y, feature_names, output_dir)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("BIOLOGICAL MOTIF DEEP-DIVE ANALYSIS COMPLETE!")
    logger.info("="*80)

    logger.info(f"\nArtifacts saved to: {output_dir}")
    logger.info("="*80)
