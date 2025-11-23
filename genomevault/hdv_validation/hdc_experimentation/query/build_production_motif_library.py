"""
Production Motif Library Builder - Complete 6-Phase Pipeline

Phase 1: Expand motif dataset to n≥50-200 per motif
Phase 2: Feature engineering (20+ dimensions)
Phase 3: Statistical modeling per motif
Phase 4: Dimensionality reduction & visualization
Phase 5: Discriminative feature analysis
Phase 6: Generate production query thresholds

Author: Phase 1 Week 3-4 - Production Query Optimization
Date: November 22, 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 2: Feature Engineering
# ============================================================================

def extract_motif_features(chunk_data):
    """
    Extract comprehensive 20+ dimensional feature vector for motif fingerprinting.

    Returns features that are:
    - Raw magnitudes (absolute signal strength)
    - Pathway magnitudes (combined AT vs GC response)
    - Relative magnitudes (composition-invariant - CRITICAL!)
    - Bank3 deviations (structural constraint violations)
    - Compositional features (confounders to control for)
    """
    features = {}

    # 1. Raw bank magnitudes (6 features)
    features['bank1_pos'] = chunk_data['signals']['bank1_pos_mag']
    features['bank1_neg'] = chunk_data['signals']['bank1_neg_mag']
    features['bank2_pos'] = chunk_data['signals']['bank2_pos_mag']
    features['bank2_neg'] = chunk_data['signals']['bank2_neg_mag']
    features['bank3_pos'] = chunk_data['signals']['bank3_pos_mag']
    features['bank3_neg'] = chunk_data['signals']['bank3_neg_mag']

    # 2. Pathway magnitudes (2 features)
    features['at_pathway_mag'] = np.sqrt(
        features['bank1_pos']**2 + features['bank1_neg']**2
    )
    features['gc_pathway_mag'] = np.sqrt(
        features['bank2_pos']**2 + features['bank2_neg']**2
    )

    # 3. Relative magnitudes - KEY for motif discrimination (5 features)
    # These are composition-invariant!
    eps = 1e-6  # Avoid division by zero
    features['bank2_bank1_ratio'] = features['gc_pathway_mag'] / (features['at_pathway_mag'] + eps)
    features['bank3_pos_rel'] = features['bank3_pos'] / (features['at_pathway_mag'] + eps)
    features['bank3_neg_rel'] = features['bank3_neg'] / (features['gc_pathway_mag'] + eps)
    features['pos_neg_asymmetry_bank1'] = (features['bank1_pos'] - features['bank1_neg']) / (features['at_pathway_mag'] + eps)
    features['pos_neg_asymmetry_bank2'] = (features['bank2_pos'] - features['bank2_neg']) / (features['gc_pathway_mag'] + eps)

    # 4. Bank3 structural signal (2 features)
    # Deviation from the "normal composition" median
    BANK3_NORMAL_MEDIAN = 14.5
    features['bank3_pos_deviation'] = abs(features['bank3_pos'] - BANK3_NORMAL_MEDIAN)
    features['bank3_neg_deviation'] = abs(features['bank3_neg'] - BANK3_NORMAL_MEDIAN)

    # 5. Compositional features (5 features - for confounder analysis)
    features['gc_content'] = chunk_data['composition']['G_pct'] + chunk_data['composition']['C_pct']
    features['at_content'] = chunk_data['composition']['A_pct'] + chunk_data['composition']['T_pct']
    features['purine_content'] = chunk_data['composition']['A_pct'] + chunk_data['composition']['G_pct']
    features['pyrimidine_content'] = chunk_data['composition']['C_pct'] + chunk_data['composition']['T_pct']

    if features['gc_content'] > 0:
        features['gc_skew'] = (chunk_data['composition']['G_pct'] - chunk_data['composition']['C_pct']) / features['gc_content']
    else:
        features['gc_skew'] = 0.0

    return features


# ============================================================================
# PHASE 3: Statistical Modeling Per Motif
# ============================================================================

def build_motif_profile(motif_name, chunk_features):
    """
    Build statistical fingerprint for a motif type.

    Returns profile with mean, std, percentiles, confidence intervals for each feature.

    Key insight: Motifs with LOW variance in a feature = discriminative signal!
    """
    if len(chunk_features) == 0:
        logger.warning(f"No features for motif {motif_name}")
        return None

    df = pd.DataFrame(chunk_features)

    profile = {
        'motif': motif_name,
        'n_samples': len(df),
        'features': {}
    }

    for col in df.columns:
        n = len(df)
        std = df[col].std()
        mean = df[col].mean()

        profile['features'][col] = {
            'mean': float(mean),
            'median': float(df[col].median()),
            'std': float(std),
            'q25': float(df[col].quantile(0.25)),
            'q75': float(df[col].quantile(0.75)),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            # Confidence intervals (95%)
            'ci_lower': float(mean - 1.96 * std / np.sqrt(n)) if n > 0 else 0,
            'ci_upper': float(mean + 1.96 * std / np.sqrt(n)) if n > 0 else 0,
            # Coefficient of variation (normalized std)
            'cv': float(std / abs(mean)) if abs(mean) > 1e-6 else 0,
        }

    return profile


# ============================================================================
# PHASE 4: Dimensionality Reduction & Visualization
# ============================================================================

def visualize_motif_clusters(all_features, labels, feature_names, output_dir):
    """
    PCA to visualize how motifs separate in bank profile space.

    Expected insights:
    - TATA_BOX vs GC_BOX should separate on PC1 (bank2/bank1 ratio)
    - ALU elements cluster separately (unique dinucleotide patterns - Bank3)
    - POLY_A vs POLY_T separate on pos/neg asymmetry features
    """
    if len(all_features) < 2:
        logger.warning("Not enough samples for PCA")
        return None, None

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(all_features)

    # PCA to 2D
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features_scaled)

    # Plot
    plt.figure(figsize=(14, 10))

    motif_types = sorted(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, len(motif_types)))

    for idx, motif in enumerate(motif_types):
        mask = labels == motif
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   label=f'{motif} (n={mask.sum()})',
                   alpha=0.6, s=50, c=[colors[idx]])

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Motif Clustering in Bank Profile Space (20+ Dimensional Feature Space)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_dir) / 'motif_pca_clustering.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved PCA visualization to {output_path}")

    # Print feature loadings
    logger.info("\n=== Top features contributing to PC1 ===")
    pc1_loadings = sorted(zip(feature_names, pca.components_[0]),
                         key=lambda x: abs(x[1]), reverse=True)[:5]
    for feat, loading in pc1_loadings:
        logger.info(f"  {feat}: {loading:.3f}")

    logger.info("\n=== Top features contributing to PC2 ===")
    pc2_loadings = sorted(zip(feature_names, pca.components_[1]),
                         key=lambda x: abs(x[1]), reverse=True)[:5]
    for feat, loading in pc2_loadings:
        logger.info(f"  {feat}: {loading:.3f}")

    return pca, scaler


# ============================================================================
# PHASE 5: Discriminative Feature Analysis
# ============================================================================

def identify_discriminative_features(features, labels, feature_names):
    """
    Use Random Forest to identify which bank features best predict motif type.

    Hypothesis:
    - bank2_bank1_ratio will be top (separates AT-rich from GC-rich)
    - bank3_*_deviation will be high for structured motifs (ALU, TATA) vs simple repeats (POLY_A)
    - pos_neg_asymmetry features will separate strand-specific motifs
    """
    if len(set(labels)) < 2:
        logger.warning("Need at least 2 motif types for classification")
        return None, None

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

    # Cross-validation accuracy
    cv_scores = cross_val_score(rf, features, labels, cv=min(5, len(set(labels))))
    logger.info(f"\n=== Motif Classification Accuracy ===")
    logger.info(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    if cv_scores.mean() > 0.9:
        logger.info("✅ EXCELLENT! >90% accuracy means encoding is information-rich!")
    elif cv_scores.mean() > 0.7:
        logger.info("✓ Good accuracy - motifs are distinguishable")
    else:
        logger.info("⚠️ Moderate accuracy - motifs may overlap in feature space")

    # Train on full dataset for feature importances
    rf.fit(features, labels)

    # Feature importance ranking
    importances = sorted(zip(feature_names, rf.feature_importances_),
                        key=lambda x: x[1], reverse=True)

    logger.info("\n=== Top 10 Discriminative Features ===")
    for feat, importance in importances[:10]:
        logger.info(f"  {feat}: {importance:.4f}")

    return rf, importances


# ============================================================================
# PHASE 6: Generate Production Query Thresholds
# ============================================================================

def generate_query_thresholds(motif_profile, confidence_level=0.95):
    """
    Generate bank thresholds for selective motif retrieval.

    Uses 95% confidence intervals by default.
    Critical for production queries!
    """
    if motif_profile is None:
        return None

    thresholds = {
        'motif': motif_profile['motif'],
        'n_samples': motif_profile['n_samples'],
        'bank_thresholds': {},
        'feature_thresholds': {},
    }

    for feature, stats in motif_profile['features'].items():
        # Bank thresholds (for primary filtering)
        if 'bank' in feature and 'deviation' not in feature and 'ratio' not in feature and 'asymmetry' not in feature:
            thresholds['bank_thresholds'][feature] = {
                'min': stats['ci_lower'],
                'max': stats['ci_upper'],
                'median': stats['median'],
                'tight_min': stats['q25'],  # Tighter threshold (50% of samples)
                'tight_max': stats['q75'],
            }

        # All feature thresholds (for advanced filtering)
        thresholds['feature_thresholds'][feature] = {
            'mean': stats['mean'],
            'std': stats['std'],
            'ci_lower': stats['ci_lower'],
            'ci_upper': stats['ci_upper'],
            'cv': stats['cv'],  # Coefficient of variation - LOW = discriminative!
        }

    return thresholds


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_production_library(
    data_file: str,
    output_dir: str,
    min_samples_per_motif: int = 5,
):
    """
    Complete 6-phase pipeline to build production motif library.

    Args:
        data_file: Path to motif ground truth JSON (from identify_extreme_motifs.py)
        output_dir: Where to save outputs
        min_samples_per_motif: Skip motifs with fewer samples
    """
    logger.info("\n" + "="*80)
    logger.info("PRODUCTION MOTIF LIBRARY BUILDER - 6-PHASE PIPELINE")
    logger.info("="*80)

    # Load motif ground truth
    with open(data_file, 'r') as f:
        data = json.load(f)

    logger.info(f"\nLoaded {len(data)} motif groups")

    # Group chunks by motif type
    motif_groups = {}
    for motif_name, motif_info in data.items():
        chunks = motif_info.get('chunks', [])
        if len(chunks) >= min_samples_per_motif:
            motif_groups[motif_name] = chunks
            logger.info(f"  {motif_name}: {len(chunks)} chunks")
        else:
            logger.info(f"  {motif_name}: {len(chunks)} chunks (SKIPPED - below threshold)")

    if len(motif_groups) == 0:
        logger.error("No motifs with sufficient samples!")
        return

    # ========================================================================
    # PHASE 2: Feature Engineering
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: FEATURE ENGINEERING")
    logger.info("="*80)

    all_features_list = []
    all_labels = []
    motif_feature_sets = {}

    for motif_name, chunks in motif_groups.items():
        motif_features = []
        for chunk in chunks:
            features = extract_motif_features(chunk)
            motif_features.append(features)
            all_features_list.append(features)
            all_labels.append(motif_name)

        motif_feature_sets[motif_name] = motif_features
        logger.info(f"{motif_name}: Extracted {len(features)} features per chunk")

    # Convert to arrays
    feature_names = list(all_features_list[0].keys())
    all_features_array = np.array([[f[name] for name in feature_names] for f in all_features_list])
    all_labels_array = np.array(all_labels)

    logger.info(f"\nTotal samples: {len(all_features_array)}")
    logger.info(f"Feature dimensions: {len(feature_names)}")

    # ========================================================================
    # PHASE 3: Statistical Modeling Per Motif
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: STATISTICAL MODELING PER MOTIF")
    logger.info("="*80)

    motif_profiles = {}
    for motif_name, features in motif_feature_sets.items():
        profile = build_motif_profile(motif_name, features)
        if profile:
            motif_profiles[motif_name] = profile

            # Report low-variance features (discriminative!)
            low_var_features = []
            for feat, stats in profile['features'].items():
                if stats['cv'] < 0.1 and 'bank' in feat:  # CV < 10% = very consistent!
                    low_var_features.append((feat, stats['cv']))

            if low_var_features:
                low_var_features.sort(key=lambda x: x[1])
                logger.info(f"\n{motif_name} - Low variance features (discriminative!):")
                for feat, cv in low_var_features[:3]:
                    logger.info(f"  {feat}: CV={cv:.3f} (mean={profile['features'][feat]['mean']:.2f})")

    # ========================================================================
    # PHASE 4: Dimensionality Reduction & Visualization
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: DIMENSIONALITY REDUCTION & VISUALIZATION")
    logger.info("="*80)

    pca, scaler = visualize_motif_clusters(all_features_array, all_labels_array, feature_names, output_dir)

    # ========================================================================
    # PHASE 5: Discriminative Feature Analysis
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 5: DISCRIMINATIVE FEATURE ANALYSIS")
    logger.info("="*80)

    rf, importances = identify_discriminative_features(all_features_array, all_labels_array, feature_names)

    # ========================================================================
    # PHASE 6: Generate Production Query Thresholds
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 6: PRODUCTION QUERY THRESHOLDS")
    logger.info("="*80)

    query_library = {
        'metadata': {
            'total_motifs': len(motif_profiles),
            'total_samples': len(all_features_array),
            'feature_dimensions': len(feature_names),
            'feature_names': feature_names,
        },
        'motifs': {},
        'feature_importances': {feat: float(imp) for feat, imp in importances} if importances else {},
    }

    for motif_name, profile in motif_profiles.items():
        thresholds = generate_query_thresholds(profile)
        query_library['motifs'][motif_name] = thresholds

        logger.info(f"\n{motif_name} Query Thresholds:")
        for bank, thresh in thresholds['bank_thresholds'].items():
            logger.info(f"  {bank}: {thresh['min']:.2f} - {thresh['max']:.2f} (median={thresh['median']:.2f})")

    # Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save full profiles
    with open(output_path / 'motif_profiles_full.json', 'w') as f:
        json.dump(motif_profiles, f, indent=2)
    logger.info(f"\nSaved full profiles to {output_path / 'motif_profiles_full.json'}")

    # Save production query library
    with open(output_path / 'production_query_library.json', 'w') as f:
        json.dump(query_library, f, indent=2)
    logger.info(f"Saved production library to {output_path / 'production_query_library.json'}")

    # Save feature importance plot
    if importances:
        plt.figure(figsize=(10, 8))
        top_features = importances[:15]
        features, importance_values = zip(*top_features)
        plt.barh(range(len(features)), importance_values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Discriminative Features for Motif Classification')
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {output_path / 'feature_importance.png'}")

    logger.info("\n" + "="*80)
    logger.info("✅ PRODUCTION MOTIF LIBRARY BUILD COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nOutputs saved to: {output_path}")
    logger.info(f"  - motif_profiles_full.json: Detailed statistical profiles")
    logger.info(f"  - production_query_library.json: Query thresholds for production")
    logger.info(f"  - motif_pca_clustering.png: PCA visualization")
    logger.info(f"  - feature_importance.png: Discriminative feature ranking")

    return query_library, motif_profiles


if __name__ == '__main__':
    # Input: Motif ground truth from identify_extreme_motifs.py
    data_file = "genomevault/hdv_validation/hdc_experimentation/output/extreme_motifs_ground_truth.json"

    # Output directory
    output_dir = "genomevault/hdv_validation/hdc_experimentation/output/production_motif_library"

    # Build library
    library, profiles = build_production_library(
        data_file=data_file,
        output_dir=output_dir,
        min_samples_per_motif=5,  # Require at least 5 samples per motif
    )
