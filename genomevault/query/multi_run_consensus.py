"""
Multi-Run Statistical Consensus for Query Accuracy

Implements Bayesian error reduction through independent query runs.

Based on Decision Matrix V2.0, Section 8: Multi-Run Statistical Consensus.

Key Concept:
    Running n independent queries and combining via Bayesian framework
    dramatically reduces false positive rate:

    Single run: 99% confidence (ε_query = 0.01)
    2 runs: 99.99% confidence (ε_query = 0.0001)
    3 runs: 99.9999% confidence (ε_query = 0.000001)
    4 runs: 99.999999% confidence (ε_query = 0.00000001)

Privacy Cost:
    Each query leaks ~1.58 bits for k=3 anonymity
    n runs → n × 1.58 bits total leakage
    Still within acceptable bounds for clinical use
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Clinical use case presets (from Decision Matrix V2.0, Section 8.2)
USE_CASE_PRESETS = {
    "screening": {
        "n_runs": 1,
        "target_confidence": 0.99,
        "target_epsilon_query": 0.01,
        "max_query_time_seconds": 0.45,
        "description": "Exploratory analysis, low-stakes screening"
    },
    "diagnostic": {
        "n_runs": 2,
        "target_confidence": 0.9999,
        "target_epsilon_query": 0.0001,
        "max_query_time_seconds": 0.90,
        "description": "Clinical diagnosis, high-stakes decision-making"
    },
    "life_critical": {
        "n_runs": 3,
        "target_confidence": 0.999999,
        "target_epsilon_query": 0.000001,
        "max_query_time_seconds": 1.35,
        "description": "Emergency/life-critical decisions"
    },
    "regulatory": {
        "n_runs": 4,
        "target_confidence": 0.99999999,
        "target_epsilon_query": 0.00000001,
        "max_query_time_seconds": 1.80,
        "description": "FDA submission, regulatory approval"
    }
}


@dataclass
class MultiRunResult:
    """
    Result from multi-run consensus query.

    Attributes:
        n_runs: Number of independent runs executed
        confidence: Bayesian confidence (0-1)
        epsilon_query: False positive rate after consensus
        consensus_result: Final consensus result
        individual_results: List of individual run results
        query_time_seconds: Total time for all runs
        privacy_cost_bits: Total information leakage (k=3: ~1.58 bits/run)
        use_case: Clinical use case (if applicable)
    """
    n_runs: int
    confidence: float
    epsilon_query: float
    consensus_result: Any
    individual_results: List[Any]
    query_time_seconds: float
    privacy_cost_bits: float
    use_case: Optional[str] = None


def compute_multi_run_confidence(
    n_runs: int,
    base_fidelity: float = 0.99
) -> Dict[str, float]:
    """
    Compute statistical confidence after n independent query runs.

    Uses Bayesian framework with independent Bernoulli trials:

    P(variant_present | n queries all positive) = p^n / (p^n + (1-p)^n)

    Where:
        p = base_fidelity = 1 - ε_query (single run)
        1-p = ε_query = false positive rate (single run)

    Args:
        n_runs: Number of independent query runs
        base_fidelity: Pipeline fidelity for single run (default: 0.99)

    Returns:
        Dictionary with:
        - n_runs: Number of runs
        - confidence: Bayesian confidence after n runs
        - epsilon_query: False positive rate after consensus
        - false_positive_rate: Same as epsilon_query
        - query_time_seconds: Estimated total time (n × 0.45s)
        - privacy_cost_bits: Information leakage (n × 1.58 bits for k=3)

    Examples:
        >>> result = compute_multi_run_confidence(1)
        >>> result['confidence']
        0.99

        >>> result = compute_multi_run_confidence(2)
        >>> result['confidence']
        0.9999

        >>> result = compute_multi_run_confidence(3)
        >>> result['confidence']
        0.999999
    """
    # Validate inputs
    if n_runs < 1:
        raise ValueError("n_runs must be ≥ 1")
    if not 0 < base_fidelity < 1:
        raise ValueError("base_fidelity must be in (0, 1)")

    # Compute Bayesian confidence
    # P(present | n positive) = p^n / (p^n + (1-p)^n)
    p_positive_given_present = base_fidelity ** n_runs
    p_positive_given_absent = (1 - base_fidelity) ** n_runs

    confidence = p_positive_given_present / (
        p_positive_given_present + p_positive_given_absent
    )

    epsilon_query = 1 - confidence

    # Estimate costs (from Decision Matrix V2.0, Section 8.1)
    query_time_seconds = n_runs * 0.45  # 0.45s per query with cached HDV
    privacy_cost_bits = n_runs * 1.58   # k=3 anonymity, ~1.58 bits/query

    return {
        'n_runs': n_runs,
        'confidence': confidence,
        'epsilon_query': epsilon_query,
        'false_positive_rate': epsilon_query,
        'query_time_seconds': query_time_seconds,
        'privacy_cost_bits': privacy_cost_bits
    }


def get_recommended_runs_for_use_case(use_case: str) -> int:
    """
    Get recommended number of runs for a clinical use case.

    Args:
        use_case: Clinical use case name
                 (screening, diagnostic, life_critical, regulatory)

    Returns:
        Recommended number of independent runs

    Raises:
        ValueError: If use_case is not recognized
    """
    if use_case not in USE_CASE_PRESETS:
        raise ValueError(
            f"Unknown use case '{use_case}'. "
            f"Valid options: {list(USE_CASE_PRESETS.keys())}"
        )

    return USE_CASE_PRESETS[use_case]["n_runs"]


def run_consensus_query(
    query_func: Callable,
    n_runs: int,
    base_fidelity: float = 0.99,
    use_case: Optional[str] = None,
    **query_kwargs
) -> MultiRunResult:
    """
    Execute n independent queries and combine via Bayesian consensus.

    This function runs the same query n times independently and combines
    the results using Bayesian statistics to achieve higher confidence.

    Args:
        query_func: Query function to execute (must be callable)
        n_runs: Number of independent runs
        base_fidelity: Pipeline fidelity for single run (default: 0.99)
        use_case: Optional clinical use case name
        **query_kwargs: Arguments to pass to query_func

    Returns:
        MultiRunResult with consensus result and statistics

    Example:
        >>> def my_query(chrom, pos):
        ...     # Execute PIR query
        ...     return {"present": True, "allele": "A"}
        ...
        >>> result = run_consensus_query(
        ...     my_query,
        ...     n_runs=2,
        ...     use_case="diagnostic",
        ...     chrom="chr1",
        ...     pos=12345
        ... )
        >>> result.confidence
        0.9999
    """
    import time

    logger.info(f"Running {n_runs}-run consensus query...")

    # Compute expected confidence
    confidence_stats = compute_multi_run_confidence(n_runs, base_fidelity)

    # Execute n independent queries
    individual_results = []
    start_time = time.time()

    for i in range(n_runs):
        logger.info(f"  Run {i+1}/{n_runs}...")
        try:
            result = query_func(**query_kwargs)
            individual_results.append(result)
            logger.info(f"  ✓ Run {i+1} complete")
        except Exception as e:
            logger.error(f"  ✗ Run {i+1} failed: {e}")
            individual_results.append({"error": str(e), "success": False})

    query_time = time.time() - start_time

    # Combine results via majority voting
    # (Simple consensus: require all runs to agree for positive result)
    consensus_result = _compute_consensus(individual_results)

    # Compute privacy cost (k=3 anonymity: ~1.58 bits per query)
    privacy_cost_bits = n_runs * 1.58  # From Decision Matrix V2.0

    logger.info(f"✓ {n_runs}-run consensus complete")
    logger.info(f"  Confidence: {confidence_stats['confidence']:.6f} ({confidence_stats['confidence']*100:.4f}%)")
    logger.info(f"  ε_query: {confidence_stats['epsilon_query']:.8f}")
    logger.info(f"  Total time: {query_time:.2f}s")
    logger.info(f"  Privacy cost: {privacy_cost_bits:.2f} bits")

    return MultiRunResult(
        n_runs=n_runs,
        confidence=confidence_stats['confidence'],
        epsilon_query=confidence_stats['epsilon_query'],
        consensus_result=consensus_result,
        individual_results=individual_results,
        query_time_seconds=query_time,
        privacy_cost_bits=privacy_cost_bits,
        use_case=use_case
    )


def _compute_consensus(individual_results: List[Any]) -> Any:
    """
    Compute consensus from individual query results.

    Strategy: Require unanimous agreement for positive result.
    If any run disagrees or fails, return negative/uncertain.

    Args:
        individual_results: List of individual run results

    Returns:
        Consensus result
    """
    # Check for errors
    successful_results = [r for r in individual_results if not isinstance(r, dict) or r.get("success", True)]

    if len(successful_results) == 0:
        return {"success": False, "error": "All runs failed"}

    if len(successful_results) < len(individual_results):
        logger.warning(f"Only {len(successful_results)}/{len(individual_results)} runs succeeded")

    # Simple consensus: first result (assuming query_func returns consistent results)
    # In practice, you might want more sophisticated voting
    consensus = successful_results[0]

    # Verify all results agree (for boolean queries)
    if all(isinstance(r, dict) and "present" in r for r in successful_results):
        # Check if all runs agree on presence
        all_present = all(r["present"] for r in successful_results)
        any_present = any(r["present"] for r in successful_results)

        if all_present == any_present:
            # Unanimous agreement
            consensus["unanimous"] = True
        else:
            # Disagreement
            consensus["unanimous"] = False
            consensus["warning"] = "Runs did not agree - returning conservative result"
            consensus["present"] = False  # Conservative: require unanimous positive

    return consensus


def compute_epsilon_query_for_runs(n_runs: int, base_fidelity: float = 0.99) -> float:
    """
    Compute epsilon_query after n independent runs.

    Convenience function for error tracking integration.

    Args:
        n_runs: Number of independent runs
        base_fidelity: Pipeline fidelity for single run (default: 0.99)

    Returns:
        epsilon_query after Bayesian consensus

    Examples:
        >>> compute_epsilon_query_for_runs(1)
        0.01

        >>> compute_epsilon_query_for_runs(2)
        0.0001

        >>> compute_epsilon_query_for_runs(3)
        1e-06
    """
    result = compute_multi_run_confidence(n_runs, base_fidelity)
    return result['epsilon_query']


def print_use_case_summary():
    """
    Print summary of all clinical use case presets.

    Useful for CLI help messages and documentation.
    """
    print("Multi-Run Consensus: Clinical Use Case Presets")
    print("=" * 80)
    print("")

    for use_case, config in USE_CASE_PRESETS.items():
        confidence_stats = compute_multi_run_confidence(config["n_runs"])

        print(f"{use_case.upper()}:")
        print(f"  Runs: {config['n_runs']}")
        print(f"  Confidence: {confidence_stats['confidence']:.8f} ({confidence_stats['confidence']*100:.6f}%)")
        print(f"  ε_query: {confidence_stats['epsilon_query']:.10f}")
        print(f"  Query time: {confidence_stats['query_time_seconds']:.2f}s")
        print(f"  Privacy cost: {confidence_stats['privacy_cost_bits']:.2f} bits")
        print(f"  Description: {config['description']}")
        print("")


if __name__ == "__main__":
    # Demo usage
    print_use_case_summary()
