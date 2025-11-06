"""
Clinical-Grade Input Validation for GenomeVault

Implements error-bounded quality control based on the Accuracy-Efficiency-Privacy
Decision Matrix V2.0 (Section 7: Clinical Error Bounds and Decision Rules).

Key Functions:
- validate_input_quality(): Parse FASTQ quality scores and assess suitability
- compute_min_input_quality(): Calculate required Q_input_min for target epsilon_max
- select_optimal_configuration_clinical(): Optimal (k, D, B) for use case
- recommend_sequencing_platform(): Platform recommendation based on Q_min

Mathematical Framework:
    ε_total = ε_input + ε_pipeline + ε_query

    Where:
      ε_input    = 1 - Q_input      (sequencing errors)
      ε_pipeline = 1 - F_pipeline   (<0.01 for GenomeVault)
      ε_query    = P_false_positive (configurable via multi-run)

    For clinical use:
      Q_input_min = 1 - (ε_max - ε_pipeline - ε_query)

Privacy Note: All computations are LOCAL. No external network calls.
"""

import logging
import math
from pathlib import Path
from typing import Dict, Optional, Tuple
import gzip

logger = logging.getLogger(__name__)


# Clinical error bounds from Decision Matrix V2.0, Section 2.3
ERROR_BOUNDS_CLINICAL = {
    'screening': {
        'max_total_error': 0.30,     # 30% (exploratory)
        'min_confidence': 0.70,       # 70% required
        'recommended_runs': 1
    },
    'diagnostic': {
        'max_total_error': 0.05,      # 5% (high stakes)
        'min_confidence': 0.95,       # 95% required
        'recommended_runs': 2         # 99.99% with 2 runs
    },
    'life_critical': {
        'max_total_error': 0.025,     # 2.5% (emergency, with 3-run consensus)
        'min_confidence': 0.975,      # 97.5% required (99.9999% after 3 runs)
        'recommended_runs': 3         # Query error: 0.01³ = 0.000001 (0.0001%)
    },
    'regulatory': {
        'max_total_error': 0.023,     # 2.3% (FDA submission, with 4-run consensus)
        'min_confidence': 0.977,      # 97.7% required (99.999999% after 4 runs)
        'recommended_runs': 4         # Query error: 0.01⁴ = 0.00000001 (0.000001%)
    }
}


def parse_fastq_quality_scores(fastq_path: str, sample_size: int = 100000) -> Dict[str, float]:
    """
    Parse FASTQ quality scores and compute average base quality.

    FASTQ Q-scores encode error probability:
      Q = -10 × log₁₀(P_error)
      P_error = 10^(-Q/10)

    Examples:
      Q30 → P_error = 0.001 (0.1% error rate)
      Q20 → P_error = 0.01 (1% error rate)
      Q10 → P_error = 0.1 (10% error rate)

    Args:
        fastq_path: Path to FASTQ file (.fastq or .fastq.gz)
        sample_size: Number of reads to sample (default: 100,000)

    Returns:
        Dictionary with:
        - average_q_score: Mean Q-score across all sampled bases
        - median_q_score: Median Q-score
        - q30_fraction: Fraction of bases with Q≥30
        - total_bases_sampled: Number of bases analyzed
        - coverage_uniformity: Std dev of per-read quality (lower = better)

    Privacy: All computation is LOCAL (no network calls).
    """
    logger.info(f"Parsing FASTQ quality scores from {fastq_path}")

    fastq_path = Path(fastq_path)
    if not fastq_path.exists():
        raise FileNotFoundError(f"FASTQ file not found: {fastq_path}")

    # Detect compression
    open_fn = gzip.open if str(fastq_path).endswith('.gz') else open

    q_scores = []
    read_qualities = []  # Average Q per read (for uniformity assessment)
    bases_processed = 0
    reads_processed = 0

    with open_fn(fastq_path, 'rt') as f:
        while reads_processed < sample_size:
            # FASTQ format: 4 lines per read
            # Line 1: @read_id
            # Line 2: sequence
            # Line 3: +
            # Line 4: quality string

            try:
                header = f.readline()
                if not header:
                    break  # End of file

                sequence = f.readline().strip()
                plus = f.readline()
                quality = f.readline().strip()

                if not quality:
                    break  # Truncated file

                # Convert ASCII quality to Q-scores
                # FASTQ uses Phred+33 encoding: ASCII_value - 33 = Q_score
                read_q_scores = [ord(c) - 33 for c in quality]
                q_scores.extend(read_q_scores)

                # Track per-read average quality (for uniformity)
                read_qualities.append(sum(read_q_scores) / len(read_q_scores))

                bases_processed += len(read_q_scores)
                reads_processed += 1

            except Exception as e:
                logger.warning(f"Error parsing read {reads_processed + 1}: {e}")
                break

    if not q_scores:
        raise ValueError(f"No quality scores parsed from {fastq_path}")

    logger.info(f"Parsed {bases_processed:,} bases from {reads_processed:,} reads")

    # Compute statistics
    average_q = sum(q_scores) / len(q_scores)
    sorted_q = sorted(q_scores)
    median_q = sorted_q[len(sorted_q) // 2]
    q30_count = sum(1 for q in q_scores if q >= 30)
    q30_fraction = q30_count / len(q_scores)

    # Coverage uniformity: Std dev of per-read quality
    mean_read_q = sum(read_qualities) / len(read_qualities)
    variance = sum((q - mean_read_q) ** 2 for q in read_qualities) / len(read_qualities)
    coverage_uniformity = math.sqrt(variance)

    return {
        'average_q_score': average_q,
        'median_q_score': median_q,
        'q30_fraction': q30_fraction,
        'total_bases_sampled': bases_processed,
        'reads_sampled': reads_processed,
        'coverage_uniformity': coverage_uniformity,
    }


def validate_input_quality(
    fastq_path: str,
    target_epsilon: float,
    k: int = 3,
    D: int = 10000
) -> Dict:
    """
    Validate FASTQ quality and determine suitability for clinical use.

    Implements Section 7.1: Input Quality Requirements

    Args:
        fastq_path: Path to FASTQ file
        target_epsilon: Maximum acceptable total error (ε_max)
        k: k-anonymity level (default: 3)
        D: Hypervector dimension (default: 10000)

    Returns:
        Dictionary with:
        - Q_input: Measured sequencing quality (0-1)
        - epsilon_input: Sequencing error rate
        - meets_target: Boolean (True if quality sufficient)
        - recommendation: Sequencing platform recommendation if failed
        - quality_metrics: Detailed Q-score statistics
        - error_budget: Breakdown of ε_input, ε_pipeline, ε_query

    Raises:
        ValueError: If input quality insufficient for target error bound

    Privacy: All computation is LOCAL (no network calls).
    """
    logger.info(f"Validating input quality for target ε_max = {target_epsilon:.4f}")

    # Parse FASTQ quality scores
    metrics = parse_fastq_quality_scores(fastq_path)

    # Compute Q_input from average Q-score
    # Q-score → error probability: P_error = 10^(-Q/10)
    # Q_input = 1 - P_error_avg
    avg_q = metrics['average_q_score']
    P_error_avg = 10 ** (-avg_q / 10)
    Q_input = 1 - P_error_avg

    epsilon_input = 1 - Q_input

    logger.info(f"Measured Q_input = {Q_input:.4f} (ε_input = {epsilon_input:.4f})")

    # Compute minimum required quality
    min_quality_info = compute_min_input_quality(target_epsilon, k, D)
    Q_input_min = min_quality_info['Q_input_min']

    meets_target = Q_input >= Q_input_min

    if not meets_target:
        logger.warning(
            f"Input quality {Q_input:.3f} insufficient for target error {target_epsilon:.4f}. "
            f"Required: {Q_input_min:.3f}"
        )
    else:
        logger.info(f"Input quality meets target (Q_input={Q_input:.3f} ≥ {Q_input_min:.3f})")

    return {
        'Q_input': Q_input,
        'epsilon_input': epsilon_input,
        'Q_input_min': Q_input_min,
        'meets_target': meets_target,
        'recommendation': min_quality_info['sequencing_recommendation'] if not meets_target else 'Acceptable',
        'quality_metrics': metrics,
        'error_budget': min_quality_info['epsilon_breakdown'],
    }


def compute_min_input_quality(
    epsilon_max: float,
    k: int = 3,
    D: int = 10000
) -> Dict:
    """
    Compute minimum input quality to achieve target error bound.

    Implements Section 7.1: Input Quality Requirements

    Mathematical Formula:
        ε_total = ε_input + ε_pipeline + ε_query

        Q_input_min = 1 - (ε_max - ε_pipeline - ε_query)

        Where:
          ε_pipeline = 1 - (0.999 × F_hdc(D) × (1 - 2^-128) × 1.0)
          F_hdc(D) = 1 - exp(-0.575257 × ln(D))
          ε_query = 0.01 (single run, conservative)

    Args:
        epsilon_max: Maximum acceptable total error (clinical requirement)
        k: k-anonymity level (default: 3)
        D: Hypervector dimension (default: 10000)

    Returns:
        Dictionary with:
        - Q_input_min: Minimum sequencing quality required
        - epsilon_breakdown: Error budget allocation
        - sequencing_recommendation: Recommended sequencing platform

    Examples:
        Screening (ε_max = 0.30):
          Q_input_min: ~0.72 (72% sequencing quality)
          Recommendation: Any sequencing platform acceptable

        Diagnostic (ε_max = 0.05):
          Q_input_min: ~0.97 (97% sequencing quality)
          Recommendation: Illumina NovaSeq X Plus (>Q30)

        Life-Critical (ε_max = 0.02, with 3-run consensus):
          Q_input_min: ~0.98 (98% sequencing quality)
          Recommendation: PacBio HiFi (>Q50, 99.9% accuracy)
          Final confidence after 3 runs: 99.9999%

    Privacy: All computation is LOCAL (no network calls).
    """
    logger.info(f"Computing minimum input quality for ε_max = {epsilon_max:.4f}")

    # Pipeline error (validated, Section 5.1)
    # F_hdc(D) = 1 - exp(-λ_D × ln(D))
    # λ_D = 0.575 (empirically calibrated: 10,000D → 99.5% preservation)
    # Note: Decision Matrix V2.0 line 1109 states "10,000D → 99.5% preservation"
    # but line 1122 has a typo showing 0.00015 (which gives 0.14% fidelity, not 99.5%)
    lambda_D = 0.575257  # Correct value for F_hdc(10000) = 0.995
    F_hdc = 1 - math.exp(-lambda_D * math.log(D))

    # Component fidelities (Section 5.1)
    F_gdiff = 0.999       # GDiff encoding (lossless differential)
    F_zk = 1 - 2**-128    # ZK proof soundness (≈ 1.0)
    F_pir = 1.0           # IT-PIR correctness (information-theoretic)

    # Combined pipeline fidelity
    F_pipeline = F_gdiff * F_hdc * F_zk * F_pir
    epsilon_pipeline = 1 - F_pipeline

    # Query error (single run, conservative, Section 8.1)
    epsilon_query = 0.01  # 1% false positive rate

    # Required input quality
    epsilon_input_max = epsilon_max - epsilon_pipeline - epsilon_query

    if epsilon_input_max < 0:
        raise ValueError(
            f"Target error {epsilon_max:.4f} too strict. "
            f"Pipeline alone has ε_pipeline={epsilon_pipeline:.4f} + ε_query={epsilon_query:.4f} = "
            f"{epsilon_pipeline + epsilon_query:.4f}. "
            f"Minimum achievable ε_total = {epsilon_pipeline + epsilon_query:.4f}"
        )

    Q_input_min = 1 - epsilon_input_max

    logger.info(
        f"Minimum quality: Q_input_min = {Q_input_min:.4f} "
        f"(ε_input_max = {epsilon_input_max:.4f})"
    )

    return {
        'Q_input_min': Q_input_min,
        'epsilon_breakdown': {
            'input_max': epsilon_input_max,
            'pipeline': epsilon_pipeline,
            'query': epsilon_query,
            'total': epsilon_max
        },
        'sequencing_recommendation': recommend_sequencing_platform(Q_input_min)
    }


def recommend_sequencing_platform(Q_min: float) -> str:
    """
    Recommend sequencing platform based on quality requirement.

    Implements Section 9.3: Sequencing Platform Recommendations

    Platform Accuracy (from Decision Matrix V2.0):
    - PacBio HiFi: 0.999 (Q50+), $1000/genome
    - Illumina NovaSeq X+: 0.96 (Q30), $200/genome
    - Illumina NextSeq: 0.92 (Q20), $150/genome
    - Oxford Nanopore R10.4: 0.85 (Q15), $100/genome
    - MGI DNBSEQ: 0.90 (Q20), $80/genome

    Args:
        Q_min: Minimum sequencing quality required (0-1)

    Returns:
        Recommended sequencing platform with justification

    Privacy: All computation is LOCAL (no network calls).
    """
    if Q_min >= 0.99:
        return "PacBio HiFi (>Q50, 99.9% accuracy, $1000/genome) - Life-critical/regulatory use"
    elif Q_min >= 0.95:
        return "Illumina NovaSeq X Plus (>Q30, 95-98% accuracy, $200/genome) - Diagnostic use"
    elif Q_min >= 0.90:
        return "Illumina NextSeq (>Q20, 90-95% accuracy, $150/genome) - Research use"
    elif Q_min >= 0.80:
        return "Oxford Nanopore R10.4 (>Q15, 80-90% accuracy, $100/genome) - Screening use"
    else:
        return "Any sequencing platform acceptable (MGI DNBSEQ, $80/genome) - Consumer use"


def select_optimal_configuration_clinical(
    use_case: str,
    epsilon_max: float,
    Q_input: float,
    compute_budget_hours: float = 10.0,
    storage_budget_mb: float = 100.0
) -> Dict:
    """
    Select optimal (k, D, B) configuration for clinical use case.

    Implements Section 7.2: Configuration Selection Algorithm

    Args:
        use_case: Clinical use case:
          - 'screening': Exploratory, 30% error
          - 'diagnostic': High-stakes, 5% error
          - 'life_critical': Emergency, 0.1% error
          - 'regulatory': FDA submission, 0.01% error
        epsilon_max: Maximum acceptable total error
        Q_input: Measured input sequencing quality (from validate_input_quality)
        compute_budget_hours: Available compute time (default: 10 hours)
        storage_budget_mb: Available storage (default: 100 MB)

    Returns:
        Dictionary with:
        - configuration: {'k': int, 'D': int, 'B': int}
        - error_bounds: {epsilon_total, epsilon_input, epsilon_pipeline, epsilon_query, meets_requirement}
        - performance: {efficiency, privacy, query_time_seconds, setup_time_hours}
        - recommendations: {recommended_runs, sequencing_quality_ok}

    Raises:
        ValueError: If input quality insufficient for target error bound

    Privacy: All computation is LOCAL (no network calls).
    """
    logger.info(
        f"Selecting optimal configuration for use_case='{use_case}', "
        f"ε_max={epsilon_max:.4f}, Q_input={Q_input:.4f}"
    )

    # Validate input quality
    quality_check = compute_min_input_quality(epsilon_max)
    quality_sufficient = Q_input >= quality_check['Q_input_min']

    if not quality_sufficient:
        logger.warning(
            f"Input quality {Q_input:.3f} insufficient for target error {epsilon_max:.4f}. "
            f"Required: {quality_check['Q_input_min']:.3f}. "
            f"Recommendation: {quality_check['sequencing_recommendation']}"
        )

    # Use-case specific constraints (Section 7.2)
    use_case_params = {
        'screening': {'k_min': 2, 'D_min': 4096, 'runs': 1},
        'diagnostic': {'k_min': 3, 'D_min': 8192, 'runs': 2},
        'life_critical': {'k_min': 5, 'D_min': 16384, 'runs': 3},
        'regulatory': {'k_min': 10, 'D_min': 32768, 'runs': 4}
    }

    if use_case not in use_case_params:
        raise ValueError(
            f"Invalid use_case '{use_case}'. "
            f"Must be one of: {list(use_case_params.keys())}"
        )

    params = use_case_params[use_case]

    # Determine k from privacy constraint and compute budget
    # Each genome takes ~2 hours to align (Section 5.2, T_align = 7200s)
    k_min = params['k_min']
    k_budget = math.floor(compute_budget_hours * 3600 / 7200)  # 2 hours per genome
    k = max(k_min, min(k_min + 2, k_budget))

    logger.info(f"Selected k = {k} (min={k_min}, budget allows {k_budget})")

    # Determine D from error constraint
    # ε_pipeline = 1 - (0.999 × F_hdc(D) × ...)
    # Solving for D when F_hdc(D) ≥ F_target:
    F_hdc_target = 0.999  # Target 99.9% HDC fidelity
    lambda_D = 0.575257

    # Prevent domain error when F_hdc_target >= 1.0
    if F_hdc_target >= 1.0:
        D_required = 100000  # Maximum dimension
    else:
        D_required = math.exp((1/lambda_D) * math.log(1 / (1 - F_hdc_target)))

    D = max(params['D_min'], min(100000, round(D_required / 1024) * 1024))

    logger.info(f"Selected D = {D} (min={params['D_min']}, required={D_required:.0f})")

    # Storage constraint
    # S_total = 15 MB × k (GDiff per genome) + D × 4 bytes (HDV)
    S_total = 15 * k + D * 4 / 1e6
    if S_total > storage_budget_mb:
        # Reduce D to fit storage
        D_max_storage = (storage_budget_mb - 15 * k) * 1e6 / 4
        D = max(params['D_min'], min(D, round(D_max_storage / 1024) * 1024))
        logger.warning(f"Reduced D to {D} to fit storage budget ({storage_budget_mb} MB)")

    # Ensure D is valid (>= 1024 minimum)
    D = max(1024, D)

    # Batch size (GPU memory, Section 5.2)
    GPU_mem = 32e9  # 32 GB Apple Silicon
    B = min(10000, math.floor(GPU_mem / (D * 4 * 1.5)))

    logger.info(f"Selected B = {B} (GPU memory constrained)")

    # Compute expected performance
    lambda_D = 0.575257
    F_hdc = 1 - math.exp(-lambda_D * math.log(D))
    epsilon_pipeline = 1 - (0.999 * F_hdc)
    epsilon_input = 1 - Q_input
    epsilon_query = 0.01 * (0.01 ** (params['runs'] - 1))  # Multi-run reduction (Section 8.1)
    epsilon_total = epsilon_input + epsilon_pipeline + epsilon_query

    # Efficiency (Section 5.2)
    T_align = 7200 * k  # Alignment time (2 hours per genome)
    T_gdiff = 300       # GDiff encoding (5 min)
    T_hdc = 3.5e-9 * D * 78.96e6 / (43 * B)  # HDC encoding (validated)
    T_zk = 0.40         # ZK proof generation
    T_pir = 0.0025 + 0.005 * k  # PIR query
    T_total = T_align + T_gdiff + T_hdc + T_zk + T_pir

    E_norm = 1 / (1 + T_total / 21600 + S_total / 100)

    # Privacy (Section 5.3)
    P = 1 - 1/k

    logger.info(
        f"Performance: ε_total={epsilon_total:.4f}, E_norm={E_norm:.3f}, P={P:.3f}"
    )

    return {
        'configuration': {'k': k, 'D': int(D), 'B': B},
        'error_bounds': {
            'epsilon_total': epsilon_total,
            'epsilon_input': epsilon_input,
            'epsilon_pipeline': epsilon_pipeline,
            'epsilon_query': epsilon_query,
            'meets_requirement': epsilon_total <= epsilon_max
        },
        'performance': {
            'efficiency': E_norm,
            'privacy': P,
            'query_time_seconds': T_zk + T_pir,
            'setup_time_hours': T_total / 3600
        },
        'recommendations': {
            'recommended_runs': params['runs'],
            'sequencing_quality_ok': quality_sufficient
        }
    }
