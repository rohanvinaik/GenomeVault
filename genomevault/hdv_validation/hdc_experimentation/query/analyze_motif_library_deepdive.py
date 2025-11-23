"""
Deep-Dive Analysis of Production Motif Library v2

Executes three critical investigations:
1. Confusion matrix - which motifs are misclassified?
2. Feature interaction terms - Bank3 × composition synergy?
3. Hierarchical classifier - threshold-based decision tree

Author: Phase 1 Week 3-4
Date: November 22, 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_motif_profiles(profile_path: str):
    """Load motif profiles from production library v2."""
    with open(profile_path, 'r') as f:
        profiles = json.load(f)
    return profiles


def extract_features_and_labels(ground_truth):
    """
    Extract feature matrix and labels from ground truth data.

    Args:
        ground_truth: Dict with motif names as keys, each containing 'chunks' list

    Returns:
        X: Feature matrix (n_samples, 23)
        y: Labels (n_samples,)
        feature_names: List of feature names
        motif_names: List of motif names per sample
    """
    samples = []
    labels = []
    motif_names = []

    # First pass: collect all samples to compute mean bank3 values per motif
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
            features = {
                # Bank magnitudes (6)
                'bank1_pos': chunk['signals']['bank1_pos_mag'],
                'bank1_neg': chunk['signals']['bank1_neg_mag'],
                'bank2_pos': chunk['signals']['bank2_pos_mag'],
                'bank2_neg': chunk['signals']['bank2_neg_mag'],
                'bank3_pos': chunk['signals']['bank3_pos_mag'],
                'bank3_neg': chunk['signals']['bank3_neg_mag'],

                # Derived (17)
                'at_content': chunk['composition']['A_pct'] + chunk['composition']['T_pct'],
                'gc_content': chunk['composition']['G_pct'] + chunk['composition']['C_pct'],
                'at_pathway_mag': chunk['signals']['bank1_pos_mag'] + chunk['signals']['bank1_neg_mag'],
                'gc_pathway_mag': chunk['signals']['bank2_pos_mag'] + chunk['signals']['bank2_neg_mag'],
                'total_transition_signal': chunk['signals']['bank3_pos_mag'] + chunk['signals']['bank3_neg_mag'],
                'bank1_bank2_ratio': chunk['signals']['bank1_pos_mag'] / (chunk['signals']['bank2_pos_mag'] + 1e-6),
                'bank1_bank2_product': chunk['signals']['bank1_pos_mag'] * chunk['signals']['bank2_pos_mag'],
                'yr_ry_asymmetry': (chunk['signals']['bank3_pos_mag'] - chunk['signals']['bank3_neg_mag']) / (chunk['signals']['bank3_pos_mag'] + chunk['signals']['bank3_neg_mag'] + 1e-6),
                'bank3_pos_deviation': chunk['signals']['bank3_pos_mag'] - motif_bank3_means[motif_name]['bank3_pos_mean'],
                'bank3_neg_deviation': chunk['signals']['bank3_neg_mag'] - motif_bank3_means[motif_name]['bank3_neg_mean'],
                'compositional_imbalance': abs((chunk['composition']['A_pct'] + chunk['composition']['T_pct']) - (chunk['composition']['G_pct'] + chunk['composition']['C_pct'])),
            }

            samples.append(list(features.values()))
            labels.append(motif_name)
            motif_names.append(motif_name)

    X = np.array(samples)
    y = np.array(labels)
    feature_names = list(features.keys())

    return X, y, feature_names, motif_names


def add_interaction_features(X, feature_names):
    """
    Add Bank3 × composition interaction terms.

    Hypothesis: Bank3's discriminative power varies with composition.
    """
    # Extract relevant features by index
    feature_idx = {name: i for i, name in enumerate(feature_names)}

    bank3_pos_idx = feature_idx['bank3_pos']
    bank3_neg_idx = feature_idx['bank3_neg']
    yr_ry_asym_idx = feature_idx['yr_ry_asymmetry']
    at_content_idx = feature_idx['at_content']
    gc_content_idx = feature_idx['gc_content']
    comp_imbalance_idx = feature_idx['compositional_imbalance']

    # Create 4 interaction features
    interactions = np.zeros((X.shape[0], 4))
    interactions[:, 0] = X[:, bank3_pos_idx] * X[:, at_content_idx]  # bank3_pos × AT%
    interactions[:, 1] = X[:, yr_ry_asym_idx] * X[:, at_content_idx]  # yr_ry_asym × AT%
    interactions[:, 2] = X[:, bank3_pos_idx] * X[:, gc_content_idx]  # bank3_pos × GC%
    interactions[:, 3] = X[:, yr_ry_asym_idx] * X[:, comp_imbalance_idx]  # yr_ry_asym × imbalance

    # Concatenate
    X_augmented = np.hstack([X, interactions])

    new_feature_names = feature_names + [
        'bank3_pos_x_at',
        'yr_ry_asym_x_at',
        'bank3_pos_x_gc',
        'yr_ry_asym_x_imbalance'
    ]

    return X_augmented, new_feature_names


def analyze_confusion_matrix(X, y, output_dir):
    """
    Analysis 1: Confusion Matrix Investigation

    Questions:
    - Which motif pairs are most confused?
    - Are errors symmetric or asymmetric?
    - Do misclassifications happen at compositional boundaries?
    """
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS 1: CONFUSION MATRIX INVESTIGATION")
    logger.info("="*80)

    # Train RF and get cross-validated predictions
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_pred = cross_val_predict(rf, X_scaled, y, cv=5)

    # Compute confusion matrix
    motif_labels = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=motif_labels)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['GC_SUP', 'AT_SUP', 'B3_POS', 'B3_NEG', 'BAL'],
                yticklabels=['GC_SUP', 'AT_SUP', 'B3_POS', 'B3_NEG', 'BAL'])
    plt.title('Motif Classification Confusion Matrix (n=50 per class)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
    logger.info(f"✓ Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")

    # Detailed classification report
    logger.info("\n=== Classification Report ===")
    report = classification_report(y, y_pred, labels=motif_labels, target_names=motif_labels)
    print(report)

    # Analyze confusion patterns
    logger.info("\n=== Confusion Pattern Analysis ===")

    # Off-diagonal elements (misclassifications)
    for i, true_label in enumerate(motif_labels):
        for j, pred_label in enumerate(motif_labels):
            if i != j and cm[i, j] > 0:
                confusion_rate = cm[i, j] / cm[i].sum() * 100
                logger.info(f"  {true_label} → {pred_label}: {cm[i, j]} samples ({confusion_rate:.1f}%)")

    # Symmetry analysis
    logger.info("\n=== Symmetry Analysis ===")
    for i in range(len(motif_labels)):
        for j in range(i+1, len(motif_labels)):
            forward = cm[i, j]
            backward = cm[j, i]
            if forward + backward > 0:
                asymmetry = abs(forward - backward) / (forward + backward) * 100
                logger.info(f"  {motif_labels[i]} ↔ {motif_labels[j]}: {forward} vs {backward} (asymmetry: {asymmetry:.1f}%)")

    return cm, report


def analyze_interaction_features(X, y, feature_names, output_dir):
    """
    Analysis 2: Feature Interaction Terms

    Test if Bank3 × composition interactions improve accuracy.
    """
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS 2: FEATURE INTERACTION TERMS")
    logger.info("="*80)

    # Baseline accuracy (23 features)
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    baseline_scores = cross_val_score(rf_baseline, X_scaled, y, cv=5, scoring='accuracy')
    baseline_mean = baseline_scores.mean()
    baseline_std = baseline_scores.std()

    logger.info(f"\n=== Baseline (23 features) ===")
    logger.info(f"Accuracy: {baseline_mean:.4f} ± {baseline_std:.4f}")

    # Augmented accuracy (27 features with interactions)
    X_aug, feature_names_aug = add_interaction_features(X, feature_names)
    X_aug_scaled = scaler.fit_transform(X_aug)

    rf_augmented = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    augmented_scores = cross_val_score(rf_augmented, X_aug_scaled, y, cv=5, scoring='accuracy')
    augmented_mean = augmented_scores.mean()
    augmented_std = augmented_scores.std()

    logger.info(f"\n=== Augmented (27 features with interactions) ===")
    logger.info(f"Accuracy: {augmented_mean:.4f} ± {augmented_std:.4f}")

    improvement = (augmented_mean - baseline_mean) * 100
    logger.info(f"\n=== Improvement ===")
    logger.info(f"Absolute: {augmented_mean - baseline_mean:.4f}")
    logger.info(f"Relative: {improvement:.2f}%")

    if augmented_mean > baseline_mean + 2 * baseline_std:
        logger.info("✓ SIGNIFICANT IMPROVEMENT - Interactions are valuable!")
    else:
        logger.info("✗ No significant improvement - Interactions may not help")

    # Feature importance for augmented model
    rf_augmented.fit(X_aug_scaled, y)
    importances = rf_augmented.feature_importances_

    # Sort and display top 15
    indices = np.argsort(importances)[::-1][:15]
    logger.info("\n=== Top 15 Features (Augmented Model) ===")
    for rank, idx in enumerate(indices, 1):
        logger.info(f"  {rank}. {feature_names_aug[idx]}: {importances[idx]:.4f}")

    # Plot feature importance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Baseline
    rf_baseline.fit(X_scaled, y)
    baseline_imp = rf_baseline.feature_importances_
    indices_base = np.argsort(baseline_imp)[::-1][:15]
    ax1.barh(range(15), baseline_imp[indices_base])
    ax1.set_yticks(range(15))
    ax1.set_yticklabels([feature_names[i] for i in indices_base])
    ax1.set_xlabel('Importance')
    ax1.set_title(f'Baseline (23 features)\nAccuracy: {baseline_mean:.3f} ± {baseline_std:.3f}')
    ax1.invert_yaxis()

    # Augmented
    ax2.barh(range(15), importances[indices])
    ax2.set_yticks(range(15))
    ax2.set_yticklabels([feature_names_aug[i] for i in indices])
    ax2.set_xlabel('Importance')
    ax2.set_title(f'Augmented (27 features)\nAccuracy: {augmented_mean:.3f} ± {augmented_std:.3f}')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_dir / 'interaction_feature_importance.png', dpi=300)
    logger.info(f"\n✓ Feature importance comparison saved to {output_dir / 'interaction_feature_importance.png'}")

    return baseline_mean, augmented_mean, feature_names_aug


def analyze_hierarchical_classifier(X, y, feature_names, output_dir):
    """
    Analysis 3: Threshold-Based Hierarchical Classifier

    Exploit Bank3 dominance with a decision tree strategy.
    """
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS 3: THRESHOLD-BASED HIERARCHICAL CLASSIFIER")
    logger.info("="*80)

    # Extract feature indices
    feature_idx = {name: i for i, name in enumerate(feature_names)}
    bank1_pos_idx = feature_idx['bank1_pos']
    bank2_pos_idx = feature_idx['bank2_pos']
    yr_ry_asym_idx = feature_idx['yr_ry_asymmetry']

    # Threshold tuning (from production recommendations)
    thresholds = {
        'gc_suppress': {'bank2_pos_min': 250, 'bank1_pos_max': 250},
        'at_suppress': {'bank1_pos_min': 250, 'bank2_pos_max': 250},
        'bank3_extreme_pos': {'yr_ry_asym_min': 0.15},
        'bank3_extreme_neg': {'yr_ry_asym_max': -0.05},
    }

    logger.info("\n=== Threshold Configuration ===")
    for motif, thresh in thresholds.items():
        logger.info(f"  {motif}: {thresh}")

    # Apply hierarchical classifier
    y_pred_hierarchical = []

    for i in range(X.shape[0]):
        bank1_pos = X[i, bank1_pos_idx]
        bank2_pos = X[i, bank2_pos_idx]
        yr_ry_asym = X[i, yr_ry_asym_idx]

        # Stage 1: Composition extremes
        if bank2_pos > thresholds['gc_suppress']['bank2_pos_min'] and bank1_pos < thresholds['gc_suppress']['bank1_pos_max']:
            y_pred_hierarchical.append('GC_SUPPRESS')
        elif bank1_pos > thresholds['at_suppress']['bank1_pos_min'] and bank2_pos < thresholds['at_suppress']['bank2_pos_max']:
            y_pred_hierarchical.append('AT_SUPPRESS')

        # Stage 2: Bank3 structural extremes
        elif yr_ry_asym > thresholds['bank3_extreme_pos']['yr_ry_asym_min']:
            y_pred_hierarchical.append('BANK3_EXTREME_POS')
        elif yr_ry_asym < thresholds['bank3_extreme_neg']['yr_ry_asym_max']:
            y_pred_hierarchical.append('BANK3_EXTREME_NEG')

        # Stage 3: Default to BALANCED
        else:
            y_pred_hierarchical.append('BALANCED')

    y_pred_hierarchical = np.array(y_pred_hierarchical)

    # Compute accuracy
    accuracy = (y_pred_hierarchical == y).mean()
    logger.info(f"\n=== Hierarchical Classifier Accuracy ===")
    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Confusion matrix
    motif_labels = sorted(set(y))
    cm_hierarchical = confusion_matrix(y, y_pred_hierarchical, labels=motif_labels)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_hierarchical, annot=True, fmt='d', cmap='Greens',
                xticklabels=['GC_SUP', 'AT_SUP', 'B3_POS', 'B3_NEG', 'BAL'],
                yticklabels=['GC_SUP', 'AT_SUP', 'B3_POS', 'B3_NEG', 'BAL'])
    plt.title(f'Hierarchical Classifier Confusion Matrix\nAccuracy: {accuracy:.3f}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'hierarchical_confusion_matrix.png', dpi=300)
    logger.info(f"✓ Hierarchical confusion matrix saved to {output_dir / 'hierarchical_confusion_matrix.png'}")

    # Detailed report
    logger.info("\n=== Classification Report ===")
    report = classification_report(y, y_pred_hierarchical, labels=motif_labels, target_names=motif_labels)
    print(report)

    # Compare with Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
    rf_mean = rf_scores.mean()

    logger.info(f"\n=== Comparison ===")
    logger.info(f"Hierarchical:  {accuracy:.4f}")
    logger.info(f"Random Forest: {rf_mean:.4f}")

    if accuracy > rf_mean:
        logger.info("✓ Hierarchical classifier OUTPERFORMS Random Forest!")
    else:
        delta = rf_mean - accuracy
        logger.info(f"✗ Random Forest is {delta:.4f} better ({delta*100:.2f}% difference)")

    return accuracy, cm_hierarchical


if __name__ == '__main__':
    # Load motif ground truth data
    ground_truth_path = "genomevault/hdv_validation/hdc_experimentation/output/motif_ground_truth_split_binary.json"
    output_dir = Path("genomevault/hdv_validation/hdc_experimentation/output/production_motif_library_v2/deep_dive")
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("\n" + "="*80)
    logger.info("LOADING MOTIF GROUND TRUTH DATA")
    logger.info("="*80)

    ground_truth = load_motif_profiles(ground_truth_path)  # Function name is generic, works for both
    X, y, feature_names, motif_names = extract_features_and_labels(ground_truth)

    logger.info(f"\nLoaded {X.shape[0]} samples with {X.shape[1]} features")
    logger.info(f"Motif distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Analysis 1: Confusion Matrix
    cm, report = analyze_confusion_matrix(X, y, output_dir)

    # Analysis 2: Feature Interactions
    baseline_acc, augmented_acc, feature_names_aug = analyze_interaction_features(X, y, feature_names, output_dir)

    # Analysis 3: Hierarchical Classifier
    hierarchical_acc, cm_hierarchical = analyze_hierarchical_classifier(X, y, feature_names, output_dir)

    # Summary
    logger.info("\n" + "="*80)
    logger.info("DEEP-DIVE ANALYSIS COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nSummary:")
    logger.info(f"  1. Random Forest baseline:      {baseline_acc:.4f}")
    logger.info(f"  2. RF with interactions:         {augmented_acc:.4f} (Δ = {(augmented_acc - baseline_acc)*100:+.2f}%)")
    logger.info(f"  3. Hierarchical threshold:       {hierarchical_acc:.4f} (Δ = {(hierarchical_acc - baseline_acc)*100:+.2f}%)")
    logger.info(f"\nArtifacts saved to: {output_dir}")
    logger.info("="*80)
