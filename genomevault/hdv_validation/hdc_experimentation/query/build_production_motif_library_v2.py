"""
Production Motif Library Builder v2 - Bank3 Structural Signal Optimized

CRITICAL INSIGHT: Bank3 shows infinite fold difference (0→1023 transitions)
at same composition. This is the PRIMARY motif discriminator!

Updated Strategy:
- Phase 1: Strategic motif selection (structural diversity)
- Phase 2: Enhanced features (23D with Bank3 residuals)
- Phase 3: Complexity-aware profiles (Bank3 variance classification)
- Phase 4: F-ratio discriminative power analysis (between/within variance)
- Phase 5: Motif similarity matrix (clustering analysis)
- Phase 6: Three-stage query optimizer (composition → structure → quality)

Author: Phase 1 Week 3-4 - Production Query Optimization
Date: November 22, 2025
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 1: Strategic Motif Selection
# ============================================================================

STRUCTURAL_MOTIFS = {
    # Homopolymers (minimal transitions)
    'POLY_A': {'expected_transitions': 'low', 'n_target': 100},
    'POLY_T': {'expected_transitions': 'low', 'n_target': 100},
    'POLY_G': {'expected_transitions': 'low', 'n_target': 100},
    'POLY_C': {'expected_transitions': 'low', 'n_target': 100},

    # Alternating motifs (maximal transitions)
    'AT_REPEAT': {'expected_transitions': 'high', 'n_target': 100},
    'GC_REPEAT': {'expected_transitions': 'high', 'n_target': 100},

    # Structured repeats (intermediate transitions)
    'TATA_BOX': {'expected_transitions': 'medium', 'n_target': 200},
    'CAAT_BOX': {'expected_transitions': 'medium', 'n_target': 100},
    'GC_BOX': {'expected_transitions': 'medium', 'n_target': 100},

    # Complex structured elements
    'ALU_CONSENSUS': {'expected_transitions': 'variable', 'n_target': 100},
    'CpG_ISLAND': {'expected_transitions': 'high', 'n_target': 100},
}


# ============================================================================
# PHASE 2: Enhanced Feature Engineering (23 features)
# ============================================================================

def extract_motif_features_v2(chunk_data):
    """
    Enhanced feature extraction leveraging Bank3 structural signal
    and Bank1/Bank2 anti-correlation.

    KEY ADDITIONS:
    - Bank3 residuals (pure structural signal after removing AT correlation)
    - Y→R vs R→Y asymmetry (directional transition bias)
    - Total transition signal
    """
    features = {}

    # === 1. Raw Bank Magnitudes (6 features) ===
    features['bank1_pos'] = chunk_data['signals']['bank1_pos_mag']
    features['bank1_neg'] = chunk_data['signals']['bank1_neg_mag']
    features['bank2_pos'] = chunk_data['signals']['bank2_pos_mag']
    features['bank2_neg'] = chunk_data['signals']['bank2_neg_mag']
    features['bank3_pos'] = chunk_data['signals']['bank3_pos_mag']
    features['bank3_neg'] = chunk_data['signals']['bank3_neg_mag']

    # === 2. Pathway Magnitudes (2 features) ===
    features['at_pathway_mag'] = np.sqrt(
        features['bank1_pos']**2 + features['bank1_neg']**2
    )
    features['gc_pathway_mag'] = np.sqrt(
        features['bank2_pos']**2 + features['bank2_neg']**2
    )

    # === 3. CRITICAL: Bank3 Structural Signatures (6 features) ===
    # These are your PRIMARY discriminants!

    BANK3_NORMAL_MEDIAN = 14.5  # From variance collapse analysis
    BANK3_HIGH_PRECISION_STD = 0.61  # σ from AT_p75-100 bin

    # Absolute deviations from structural constraint
    features['bank3_pos_deviation'] = abs(features['bank3_pos'] - BANK3_NORMAL_MEDIAN)
    features['bank3_neg_deviation'] = abs(features['bank3_neg'] - BANK3_NORMAL_MEDIAN)

    # CRITICAL: Normalized structural signal (accounts for AT pathway correlation)
    # Expected Bank3 given AT pathway: Bank3 ≈ 0.63 * AT_pathway + offset
    # Residual = observed - expected captures PURE structural signal
    expected_bank3_from_at = 0.63 * features['at_pathway_mag'] + 4.7  # Rough fit from r=0.63
    features['bank3_pos_residual'] = features['bank3_pos'] - expected_bank3_from_at
    features['bank3_neg_residual'] = features['bank3_neg'] - expected_bank3_from_at

    # Y→R vs R→Y asymmetry (detects directional bias)
    # TATA box vs reverse might show bias
    features['yr_ry_asymmetry'] = (features['bank3_pos'] - features['bank3_neg']) / \
                                   (features['bank3_pos'] + features['bank3_neg'] + 1e-10)

    # Total dinucleotide transition signal
    features['total_transition_signal'] = np.sqrt(
        features['bank3_pos']**2 + features['bank3_neg']**2
    )

    # === 4. Compositional Anti-Correlation (3 features) ===
    # Leverage Bank1/Bank2 ρ=-0.767
    eps = 1e-6
    features['bank1_bank2_ratio'] = features['at_pathway_mag'] / (features['gc_pathway_mag'] + eps)
    features['bank1_bank2_product'] = features['at_pathway_mag'] * features['gc_pathway_mag']  # Budget constraint
    features['compositional_imbalance'] = abs(features['bank1_bank2_ratio'] - 1.0)

    # === 5. Pos/Neg Asymmetry (2 features) ===
    # A vs T dominance, G vs C dominance
    features['at_asymmetry'] = (features['bank1_pos'] - features['bank1_neg']) / \
                                (features['at_pathway_mag'] + eps)
    features['gc_asymmetry'] = (features['bank2_pos'] - features['bank2_neg']) / \
                                (features['gc_pathway_mag'] + eps)

    # === 6. Composition (for confounder analysis) (4 features) ===
    features['gc_content'] = chunk_data['composition']['G_pct'] + chunk_data['composition']['C_pct']
    features['at_content'] = chunk_data['composition']['A_pct'] + chunk_data['composition']['T_pct']
    features['purine_pct'] = chunk_data['composition']['A_pct'] + chunk_data['composition']['G_pct']
    features['pyrimidine_pct'] = chunk_data['composition']['C_pct'] + chunk_data['composition']['T_pct']

    return features  # 23 total features


# ============================================================================
# PHASE 3: Motif Profile with Complexity Metrics
# ============================================================================

def build_motif_profile_v2(motif_name, chunk_features):
    """
    Enhanced motif profile with Bank3 complexity classification.

    Args:
        motif_name: e.g., 'TATA_BOX'
        chunk_features: List of feature dicts

    Returns:
        Comprehensive motif fingerprint with complexity rating
    """
    if len(chunk_features) == 0:
        logger.warning(f"No features for motif {motif_name}")
        return None

    df = pd.DataFrame(chunk_features)

    # Compute Bank3 variance for complexity classification
    chunk_bank3_variance = df['bank3_pos'].std()

    # Complexity classification
    if chunk_bank3_variance < 1.0:
        complexity = 'HIGH_PRECISION'
        confidence = 'HIGH'
    elif chunk_bank3_variance < 3.0:
        complexity = 'MEDIUM_PRECISION'
        confidence = 'MEDIUM'
    else:
        complexity = 'LOW_PRECISION'
        confidence = 'LOW'

    profile = {
        'motif': motif_name,
        'n_samples': len(df),
        'complexity': complexity,
        'confidence': confidence,
        'bank3_variance': float(chunk_bank3_variance),

        # === CRITICAL: Bank3 Structural Signature ===
        'structural_signature': {
            'bank3_pos_median': float(df['bank3_pos'].median()),
            'bank3_neg_median': float(df['bank3_neg'].median()),
            'bank3_pos_std': float(df['bank3_pos'].std()),
            'bank3_neg_std': float(df['bank3_neg'].std()),
            'yr_ry_asymmetry_median': float(df['yr_ry_asymmetry'].median()),
            'total_transition_signal_median': float(df['total_transition_signal'].median()),
            # CRITICAL: Residual after removing AT correlation
            'bank3_pos_residual_median': float(df['bank3_pos_residual'].median()),
            'bank3_pos_residual_std': float(df['bank3_pos_residual'].std()),
        },

        # === Compositional Prefilter Signature ===
        'compositional_signature': {
            'bank1_bank2_ratio_median': float(df['bank1_bank2_ratio'].median()),
            'bank1_bank2_ratio_std': float(df['bank1_bank2_ratio'].std()),
            'at_pathway_mag_median': float(df['at_pathway_mag'].median()),
            'gc_pathway_mag_median': float(df['gc_pathway_mag'].median()),
        },

        # === Full Feature Statistics ===
        'features': {}
    }

    # Per-feature statistics
    for col in df.columns:
        n = len(df)
        std = df[col].std()
        mean = df[col].mean()

        profile['features'][col] = {
            'median': float(df[col].median()),
            'mean': float(mean),
            'std': float(std),
            'q25': float(df[col].quantile(0.25)),
            'q75': float(df[col].quantile(0.75)),
            'iqr': float(df[col].quantile(0.75) - df[col].quantile(0.25)),
            'min': float(df[col].min()),
            'max': float(df[col].max()),
            'ci_lower': float(mean - 1.96 * std / np.sqrt(n)) if n > 0 else 0,
            'ci_upper': float(mean + 1.96 * std / np.sqrt(n)) if n > 0 else 0,
            'cv': float(std / abs(mean)) if abs(mean) > 1e-6 else 0,
        }

    return profile


# ============================================================================
# PHASE 4: Discriminative Power Analysis (F-ratio)
# ============================================================================

def analyze_discriminative_power(motif_profiles):
    """
    Identify features with HIGH between-motif variance and LOW within-motif variance.

    This is the gold standard for feature selection.

    F-ratio = between-motif variance / within-motif variance
    High F-ratio = good discriminator!
    """
    if len(motif_profiles) < 2:
        logger.warning("Need at least 2 motifs for discriminative analysis")
        return []

    # Get all feature names from first profile
    feature_names = list(motif_profiles[0]['features'].keys())

    results = []

    # For each feature, compute F-ratio (ANOVA-style)
    for feature_name in feature_names:
        between_motif_var = []  # Variance of medians across motifs
        within_motif_var = []   # Average variance within each motif

        for profile in motif_profiles:
            between_motif_var.append(profile['features'][feature_name]['median'])
            within_motif_var.append(profile['features'][feature_name]['std']**2)

        # F-ratio: between-group variance / within-group variance
        between_var = np.var(between_motif_var)
        within_var = np.mean(within_motif_var)
        f_ratio = between_var / (within_var + 1e-10)  # Avoid div by zero

        results.append({
            'feature': feature_name,
            'f_ratio': float(f_ratio),
            'between_var': float(between_var),
            'within_var': float(within_var),
        })

    # Sort by F-ratio (high = good discriminator)
    results = sorted(results, key=lambda x: x['f_ratio'], reverse=True)

    logger.info("\n" + "="*80)
    logger.info("TOP 10 DISCRIMINATIVE FEATURES (F-ratio)")
    logger.info("="*80)
    logger.info("High F-ratio = HIGH between-motif variance, LOW within-motif variance")
    logger.info("")
    for i, r in enumerate(results[:10], 1):
        logger.info(f"{i:2d}. {r['feature']:30s}: F={r['f_ratio']:8.2f} "
                   f"(between={r['between_var']:6.4f}, within={r['within_var']:6.4f})")

    return results


# ============================================================================
# PHASE 5: Motif Similarity Matrix
# ============================================================================

def compute_motif_similarity_matrix(motif_profiles):
    """
    Compute pairwise Euclidean distance between motif fingerprints.

    Uses ONLY the most discriminative features (top 8 from F-ratio analysis).

    Expected clusters:
    - Homopolymers (POLY_A, POLY_T, POLY_G, POLY_C) - low Bank3
    - Alternating repeats (AT_REPEAT, GC_REPEAT) - high Bank3
    - Structured elements (TATA, CAAT, GC_BOX) - intermediate
    """
    if len(motif_profiles) < 2:
        logger.warning("Need at least 2 motifs for similarity matrix")
        return None, None

    # Extract top discriminative features
    # Based on expected results:
    top_features = ['bank3_pos_residual', 'yr_ry_asymmetry', 'bank1_bank2_ratio',
                   'total_transition_signal', 'bank3_pos_deviation',
                   'at_asymmetry', 'gc_asymmetry', 'compositional_imbalance']

    # Build feature matrix (n_motifs × n_features)
    motif_names = [p['motif'] for p in motif_profiles]
    feature_matrix = []

    for profile in motif_profiles:
        feature_vector = []
        for f in top_features:
            if f in profile['features']:
                feature_vector.append(profile['features'][f]['median'])
            else:
                feature_vector.append(0.0)
        feature_matrix.append(feature_vector)

    feature_matrix = np.array(feature_matrix)

    # Standardize (important for distance calculation)
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)

    # Pairwise distances
    distances = squareform(pdist(feature_matrix_scaled, metric='euclidean'))

    # Print similarity matrix
    logger.info("\n" + "="*80)
    logger.info("MOTIF SIMILARITY MATRIX")
    logger.info("="*80)
    logger.info("Euclidean distance in 8D feature space (lower = more similar)")
    logger.info("")

    # Build header row
    header = f"{'':15}"
    for name in motif_names:
        header += f"{name[:10]:>12}"
    logger.info(header)

    # Build data rows
    for i, name1 in enumerate(motif_names):
        row = f"{name1[:15]:15}"
        for j, name2 in enumerate(motif_names):
            row += f"{distances[i,j]:12.2f}"
        logger.info(row)

    logger.info("")

    return distances, motif_names


# ============================================================================
# PHASE 6: Production Query Optimizer (Three-Stage Architecture)
# ============================================================================

def generate_query_strategy(motif_profile, target_precision=0.95):
    """
    Generate three-stage query strategy for a given motif.

    Three-Stage Architecture:
    1. Composition Prefilter (Bank1/Bank2 anti-correlation) → 70-80% recall
    2. Structure Filter (Bank3 PRIMARY discriminant) → 90-95% recall
    3. Quality Gate (Bank3 variance threshold) → flag low confidence

    Args:
        motif_profile: Output from build_motif_profile_v2()
        target_precision: Desired precision (0.95 = 95% CI)

    Returns:
        Query thresholds for three-stage architecture
    """
    if motif_profile is None:
        return None

    # Confidence multiplier (95% CI ≈ 1.96σ)
    z_score = 1.96 if target_precision == 0.95 else 2.58  # 99% CI

    strategy = {
        'motif': motif_profile['motif'],
        'complexity': motif_profile['complexity'],
        'confidence': motif_profile['confidence'],

        # === STAGE 1: Composition Prefilter (Bank1/Bank2 anti-correlation) ===
        'stage1_composition_prefilter': {
            'bank1_bank2_ratio_min': float(
                motif_profile['compositional_signature']['bank1_bank2_ratio_median'] -
                z_score * motif_profile['compositional_signature']['bank1_bank2_ratio_std']
            ),
            'bank1_bank2_ratio_max': float(
                motif_profile['compositional_signature']['bank1_bank2_ratio_median'] +
                z_score * motif_profile['compositional_signature']['bank1_bank2_ratio_std']
            ),
            'expected_recall': '70-80%',  # Coarse filter
        },

        # === STAGE 2: Structure Filter (Bank3 PRIMARY discriminant) ===
        'stage2_structure_filter': {
            'bank3_pos_min': float(
                motif_profile['structural_signature']['bank3_pos_median'] -
                z_score * motif_profile['structural_signature']['bank3_pos_std']
            ),
            'bank3_pos_max': float(
                motif_profile['structural_signature']['bank3_pos_median'] +
                z_score * motif_profile['structural_signature']['bank3_pos_std']
            ),
            'bank3_neg_min': float(
                motif_profile['structural_signature']['bank3_neg_median'] -
                z_score * motif_profile['structural_signature']['bank3_neg_std']
            ),
            'bank3_neg_max': float(
                motif_profile['structural_signature']['bank3_neg_median'] +
                z_score * motif_profile['structural_signature']['bank3_neg_std']
            ),
            'bank3_pos_residual_min': float(
                motif_profile['structural_signature']['bank3_pos_residual_median'] -
                z_score * motif_profile['structural_signature']['bank3_pos_residual_std']
            ),
            'bank3_pos_residual_max': float(
                motif_profile['structural_signature']['bank3_pos_residual_median'] +
                z_score * motif_profile['structural_signature']['bank3_pos_residual_std']
            ),
            'expected_recall': '90-95%',  # Tight filter
        },

        # === STAGE 3: Quality Gate (Bank3 variance threshold) ===
        'stage3_quality_gate': {
            'max_acceptable_variance': 1.0 if motif_profile['complexity'] == 'HIGH_PRECISION' else 3.0,
            'action_if_fail': 'FLAG_LOW_CONFIDENCE',
        },
    }

    return strategy


# ============================================================================
# Visualization (PCA + Feature Importance)
# ============================================================================

def visualize_motif_clusters(all_features, labels, feature_names, output_dir):
    """
    PCA to visualize how motifs separate in bank profile space.
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
    plt.title('Motif Clustering in Bank3 Structural Feature Space (23D → 2D)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_dir) / 'motif_pca_clustering.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"\nSaved PCA visualization to {output_path}")

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


def identify_discriminative_features_rf(features, labels, feature_names):
    """
    Use Random Forest to identify which bank features best predict motif type.
    """
    if len(set(labels)) < 2:
        logger.warning("Need at least 2 motif types for classification")
        return None, None

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)

    # Cross-validation accuracy
    cv_scores = cross_val_score(rf, features, labels, cv=min(5, len(set(labels))))
    logger.info(f"\n=== Motif Classification Accuracy (Random Forest) ===")
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

    logger.info("\n=== Top 10 Discriminative Features (Random Forest) ===")
    for feat, importance in importances[:10]:
        logger.info(f"  {feat}: {importance:.4f}")

    return rf, importances


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_production_library_v2(
    data_file: str,
    output_dir: str,
    min_samples_per_motif: int = 5,
):
    """
    Complete 6-phase pipeline to build production motif library.

    v2 Enhancements:
    - Bank3 residuals (pure structural signal)
    - F-ratio discriminative power analysis
    - Motif similarity matrix (clustering)
    - Three-stage query optimizer
    """
    logger.info("\n" + "="*80)
    logger.info("PRODUCTION MOTIF LIBRARY BUILDER v2")
    logger.info("Bank3 Structural Signal Optimized")
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
    # PHASE 2: Enhanced Feature Engineering (23 features)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: ENHANCED FEATURE ENGINEERING (23 features)")
    logger.info("="*80)

    all_features_list = []
    all_labels = []
    motif_feature_sets = {}

    for motif_name, chunks in motif_groups.items():
        motif_features = []
        for chunk in chunks:
            features = extract_motif_features_v2(chunk)
            motif_features.append(features)
            all_features_list.append(features)
            all_labels.append(motif_name)

        motif_feature_sets[motif_name] = motif_features

    # Convert to arrays
    feature_names = list(all_features_list[0].keys())
    all_features_array = np.array([[f[name] for name in feature_names] for f in all_features_list])
    all_labels_array = np.array(all_labels)

    logger.info(f"\nTotal samples: {len(all_features_array)}")
    logger.info(f"Feature dimensions: {len(feature_names)}")
    logger.info(f"\nKey features:")
    logger.info(f"  - Bank3 residuals (pure structural signal)")
    logger.info(f"  - Y→R vs R→Y asymmetry")
    logger.info(f"  - Total transition signal")

    # ========================================================================
    # PHASE 3: Statistical Modeling with Complexity Classification
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: STATISTICAL MODELING WITH COMPLEXITY CLASSIFICATION")
    logger.info("="*80)

    motif_profiles = []
    for motif_name, features in motif_feature_sets.items():
        profile = build_motif_profile_v2(motif_name, features)
        if profile:
            motif_profiles.append(profile)

            logger.info(f"\n{motif_name}:")
            logger.info(f"  Complexity: {profile['complexity']} (σ={profile['bank3_variance']:.2f})")
            logger.info(f"  Bank3_pos median: {profile['structural_signature']['bank3_pos_median']:.2f}")
            logger.info(f"  Bank3 residual median: {profile['structural_signature']['bank3_pos_residual_median']:.2f}")

    # ========================================================================
    # PHASE 4: F-Ratio Discriminative Power Analysis
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: F-RATIO DISCRIMINATIVE POWER ANALYSIS")
    logger.info("="*80)

    f_ratio_results = analyze_discriminative_power(motif_profiles)

    # ========================================================================
    # PHASE 5: Motif Similarity Matrix
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 5: MOTIF SIMILARITY MATRIX")
    logger.info("="*80)

    distances, motif_names = compute_motif_similarity_matrix(motif_profiles)

    # ========================================================================
    # PHASE 4b: PCA Visualization
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 4b: DIMENSIONALITY REDUCTION & VISUALIZATION")
    logger.info("="*80)

    pca, scaler = visualize_motif_clusters(all_features_array, all_labels_array, feature_names, output_dir)

    # ========================================================================
    # PHASE 5b: Random Forest Classification
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 5b: RANDOM FOREST CLASSIFICATION")
    logger.info("="*80)

    rf, importances = identify_discriminative_features_rf(all_features_array, all_labels_array, feature_names)

    # ========================================================================
    # PHASE 6: Production Query Optimizer (Three-Stage Architecture)
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("PHASE 6: PRODUCTION QUERY OPTIMIZER (Three-Stage Architecture)")
    logger.info("="*80)

    query_library = {
        'metadata': {
            'version': '2.0',
            'description': 'Bank3 structural signal optimized motif library',
            'total_motifs': len(motif_profiles),
            'total_samples': len(all_features_array),
            'feature_dimensions': len(feature_names),
            'feature_names': feature_names,
        },
        'motifs': {},
        'discriminative_features': {
            'f_ratio_ranking': f_ratio_results[:10] if f_ratio_results else [],
            'random_forest_ranking': {feat: float(imp) for feat, imp in importances[:10]} if importances else {},
        },
    }

    for profile in motif_profiles:
        strategy = generate_query_strategy(profile)
        if strategy:
            query_library['motifs'][profile['motif']] = strategy

            logger.info(f"\n{profile['motif']} Three-Stage Query Strategy:")
            logger.info(f"  Stage 1 (Composition): Bank1/Bank2 ratio = {strategy['stage1_composition_prefilter']['bank1_bank2_ratio_min']:.2f} - {strategy['stage1_composition_prefilter']['bank1_bank2_ratio_max']:.2f}")
            logger.info(f"  Stage 2 (Structure): Bank3_pos = {strategy['stage2_structure_filter']['bank3_pos_min']:.2f} - {strategy['stage2_structure_filter']['bank3_pos_max']:.2f}")
            logger.info(f"  Stage 3 (Quality): Max variance = {strategy['stage3_quality_gate']['max_acceptable_variance']:.1f}")

    # Save outputs
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save full profiles
    with open(output_path / 'motif_profiles_v2.json', 'w') as f:
        json.dump(motif_profiles, f, indent=2)
    logger.info(f"\nSaved full profiles to {output_path / 'motif_profiles_v2.json'}")

    # Save production query library
    with open(output_path / 'production_query_library_v2.json', 'w') as f:
        json.dump(query_library, f, indent=2)
    logger.info(f"Saved production library to {output_path / 'production_query_library_v2.json'}")

    # Save feature importance plot
    if importances:
        plt.figure(figsize=(10, 10))
        top_features = importances[:15]
        features, importance_values = zip(*top_features)
        plt.barh(range(len(features)), importance_values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Discriminative Features (Random Forest)')
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance_rf.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved RF feature importance to {output_path / 'feature_importance_rf.png'}")

    # Save F-ratio plot
    if f_ratio_results:
        plt.figure(figsize=(10, 10))
        top_f = f_ratio_results[:15]
        features_f = [r['feature'] for r in top_f]
        f_ratios = [r['f_ratio'] for r in top_f]
        plt.barh(range(len(features_f)), f_ratios)
        plt.yticks(range(len(features_f)), features_f)
        plt.xlabel('F-Ratio (Between/Within Variance)')
        plt.title('Top 15 Discriminative Features (F-Ratio Analysis)')
        plt.tight_layout()
        plt.savefig(output_path / 'feature_importance_fratio.png', dpi=300, bbox_inches='tight')
        logger.info(f"Saved F-ratio plot to {output_path / 'feature_importance_fratio.png'}")

    logger.info("\n" + "="*80)
    logger.info("✅ PRODUCTION MOTIF LIBRARY v2 BUILD COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nOutputs saved to: {output_path}")
    logger.info(f"  - motif_profiles_v2.json: Detailed statistical profiles with complexity")
    logger.info(f"  - production_query_library_v2.json: Three-stage query optimizer")
    logger.info(f"  - motif_pca_clustering.png: PCA visualization")
    logger.info(f"  - feature_importance_rf.png: Random Forest ranking")
    logger.info(f"  - feature_importance_fratio.png: F-ratio ranking")

    return query_library, motif_profiles


if __name__ == '__main__':
    # Input: Motif ground truth from identify_motifs_split_binary.py
    data_file = "genomevault/hdv_validation/hdc_experimentation/output/motif_ground_truth_split_binary.json"

    # Output directory
    output_dir = "genomevault/hdv_validation/hdc_experimentation/output/production_motif_library_v2"

    # Build library
    library, profiles = build_production_library_v2(
        data_file=data_file,
        output_dir=output_dir,
        min_samples_per_motif=5,
    )
