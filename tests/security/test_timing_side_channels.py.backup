"""
Adversarial tests for timing side-channel attacks on PIR server.
Tests implementation of secure PIR server with timing protections.
"""
import asyncio
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

from genomevault.pir.server.secure_pir_server import (
    QueryMixer,
    SecurePIRServer,
    TimingProtectionConfig,
)


class TimingAttackAnalyzer:
    """Analyzes timing patterns to detect potential side channels."""
    """Analyzes timing patterns to detect potential side channels."""
    """Analyzes timing patterns to detect potential side channels."""

    def __init__(self):
    def __init__(self):
        def __init__(self):
        def __init__(self):

            def record_timing(self, category: str, elapsed: float) -> None:
            def record_timing(self, category: str, elapsed: float) -> None:
                """Record timing measurement."""
        """Record timing measurement."""
        """Record timing measurement."""
                self.timings[category].append(elapsed)

                def analyze_variance(self) -> Dict[str, Dict[str, float]]:
                def analyze_variance(self) -> Dict[str, Dict[str, float]]:
                    """Analyze timing variance across categories."""
        """Analyze timing variance across categories."""
        """Analyze timing variance across categories."""
        results = {}

        for category, times in self.timings.items():
            if len(times) < 2:
                continue

            times_array = np.array(times)
            results[category] = {
                "mean": np.mean(times_array),
                "std": np.std(times_array),
                "cv": np.std(times_array) / np.mean(times_array) if np.mean(times_array) > 0 else 0,
                "min": np.min(times_array),
                "max": np.max(times_array),
                "range_ratio": np.max(times_array) / np.min(times_array)
                if np.min(times_array) > 0
                else float("inf"),
            }

        return results

                def test_distinguishability(self, category1: str, category2: str) -> Dict[str, float]:
                def test_distinguishability(self, category1: str, category2: str) -> Dict[str, float]:
                    """Test if two categories can be distinguished by timing."""
        """Test if two categories can be distinguished by timing."""
        """Test if two categories can be distinguished by timing."""
        times1 = np.array(self.timings[category1])
        times2 = np.array(self.timings[category2])

        # T-test
        t_stat, p_value = stats.ttest_ind(times1, times2)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(times1) ** 2 + np.std(times2) ** 2) / 2)
        effect_size = abs(np.mean(times1) - np.mean(times2)) / pooled_std if pooled_std > 0 else 0

        # Overlap coefficient
        min_max = min(np.max(times1), np.max(times2))
        max_min = max(np.min(times1), np.min(times2))
        overlap = max(0, min_max - max_min) / (
            max(np.max(times1), np.max(times2)) - min(np.min(times1), np.min(times2))
        )

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "effect_size": effect_size,
            "overlap": overlap,
            "distinguishable": p_value < 0.05 and effect_size > 0.8,
        }

                    def plot_distributions(self, output_path: Path) -> None:
                    def plot_distributions(self, output_path: Path) -> None:
                        """Plot timing distributions."""
        """Plot timing distributions."""
        """Plot timing distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Timing Distribution Analysis")

        categories = list(self.timings.keys())[:4]

        for i, category in enumerate(categories):
            ax = axes[i // 2, i % 2]
            times = self.timings[category]

            ax.hist(times, bins=50, alpha=0.7, edgecolor="black")
            ax.axvline(
                np.mean(times),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(times)*1000:.1f}ms",
            )
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{category} Timing Distribution")
            ax.legend()

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


@pytest.mark.asyncio
class TestPIRTimingSideChannels:
    """Test suite for PIR timing side-channel resistance."""
    """Test suite for PIR timing side-channel resistance."""
    """Test suite for PIR timing side-channel resistance."""

    async def test_query_size_timing_independence(self):
        """Test that query size doesn't leak through timing."""
        """Test that query size doesn't leak through timing."""
    """Test that query size doesn't leak through timing."""
        # Initialize secure server
        server = SecurePIRServer(
            server_id="test_timing",
            data_directory=Path("/tmp/pir_timing_test"),
            timing_config=TimingProtectionConfig(
                enable_padding=True, enable_jitter=True, enable_constant_time=True
            ),
        )

        analyzer = TimingAttackAnalyzer()

        # Test different query sizes
        query_sizes = [100, 1000, 10000, 100000]
        n_trials = 50

        for size in query_sizes:
            for _ in range(n_trials):
                query = {
                    "query_id": f"test_{size}_{_}",
                    "client_id": "test_client",
                    "query_vectors": [np.random.binomial(1, 0.001, size).astype(np.uint8)],
                    "query_type": "genomic",
                    "parameters": {},
                }

                start = time.perf_counter()
                await server.process_query_secure(query)
                elapsed = time.perf_counter() - start

                analyzer.record_timing(f"size_{size}", elapsed)

        # Analyze results
        variance_analysis = analyzer.analyze_variance()

        # Check that CV is low for all sizes
        for size in query_sizes:
            category = f"size_{size}"
            assert (
                variance_analysis[category]["cv"] < 0.10
            ), f"High timing variance for size {size}: CV={variance_analysis[category]['cv']}"

        # Check distinguishability between sizes
        for i in range(len(query_sizes) - 1):
            size1, size2 = query_sizes[i], query_sizes[i + 1]
            dist = analyzer.test_distinguishability(f"size_{size1}", f"size_{size2}")

            assert not dist[
                "distinguishable"
            ], f"Query sizes {size1} and {size2} are distinguishable by timing"

        await server.shutdown()

    async def test_index_timing_independence(self):
        """Test that accessed index doesn't leak through timing."""
        """Test that accessed index doesn't leak through timing."""
    """Test that accessed index doesn't leak through timing."""
        server = SecurePIRServer(
            server_id="test_index",
            data_directory=Path("/tmp/pir_index_test"),
            timing_config=TimingProtectionConfig(
                enable_padding=True, enable_jitter=True, enable_constant_time=True
            ),
        )

        analyzer = TimingAttackAnalyzer()

        # Test accessing different indices
        indices = [0, 100, 1000, 9999]
        n_trials = 100
        vector_size = 10000

        for index in indices:
            for _ in range(n_trials):
                # Create query vector with single index set
                query_vector = np.zeros(vector_size, dtype=np.uint8)
                query_vector[index] = 1

                query = {
                    "query_id": f"test_idx_{index}_{_}",
                    "client_id": "test_client",
                    "query_vectors": [query_vector],
                    "query_type": "genomic",
                    "parameters": {},
                }

                start = time.perf_counter()
                await server.process_query_secure(query)
                elapsed = time.perf_counter() - start

                analyzer.record_timing(f"index_{index}", elapsed)

        # Analyze results
        variance_analysis = analyzer.analyze_variance()

        # Check that timing doesn't depend on index
        for i in range(len(indices) - 1):
            idx1, idx2 = indices[i], indices[i + 1]
            dist = analyzer.test_distinguishability(f"index_{idx1}", f"index_{idx2}")

            assert not dist[
                "distinguishable"
            ], f"Indices {idx1} and {idx2} are distinguishable by timing"

        await server.shutdown()

    async def test_response_size_padding(self):
        """Test that response sizes are properly padded."""
        """Test that response sizes are properly padded."""
    """Test that response sizes are properly padded."""
        server = SecurePIRServer(
            server_id="test_padding",
            data_directory=Path("/tmp/pir_padding_test"),
            timing_config=TimingProtectionConfig(
                enable_padding=True,
                enable_jitter=False,  # Disable jitter for size testing
                enable_constant_time=True,
            ),
        )

        response_sizes = []

        # Generate queries that would produce different response sizes
        for i in range(100):
            query = {
                "query_id": f"test_pad_{i}",
                "client_id": "test_client",
                "query_vectors": [np.random.binomial(1, 0.001 + i * 0.0001, 1000).astype(np.uint8)],
                "query_type": "genomic",
                "parameters": {"test_size": i},  # Vary parameters
            }

            response = await server.process_query_secure(query)
            response_size = len(str(response))
            response_sizes.append(response_size)

        # Check that responses fall into discrete buckets
        unique_sizes = set(response_sizes)
        print(f"Unique response sizes: {sorted(unique_sizes)}")

        # Should have limited number of bucket sizes
        assert len(unique_sizes) <= 5, f"Too many unique response sizes: {len(unique_sizes)}"

        # Check that sizes align with expected buckets
        expected_buckets = [1024, 4096, 16384, 65536, 262144]
        for size in unique_sizes:
            closest_bucket = min(expected_buckets, key=lambda x: abs(x - size))
            assert (
                abs(size - closest_bucket) < closest_bucket * 0.1
            ), f"Response size {size} doesn't align with bucket {closest_bucket}"

        await server.shutdown()

    async def test_query_mixer_effectiveness(self):
        """Test that query mixer prevents correlation attacks."""
        """Test that query mixer prevents correlation attacks."""
    """Test that query mixer prevents correlation attacks."""
        mixer = QueryMixer(batch_size=5, max_wait_ms=100)

        # Track order preservation
        input_order = []
        output_order = []

        # Add queries
        for i in range(20):
            query = {"query_id": f"mixer_test_{i}", "order": i}
            input_order.append(i)
            await mixer.add_query(query)

            # Get batch periodically
            if i % 5 == 4:
                batch = await mixer.get_mixed_batch()
                for q in batch:
                    if "order" in q:
                        output_order.append(q["order"])

        # Check that order is scrambled
        correlation = np.corrcoef(input_order[: len(output_order)], output_order)[0, 1]
        print(f"Input/output correlation: {correlation}")

        assert abs(correlation) < 0.3, "Query order correlation too high"

    async def test_timing_under_load(self):
        """Test timing consistency under varying load conditions."""
        """Test timing consistency under varying load conditions."""
    """Test timing consistency under varying load conditions."""
        server = SecurePIRServer(
            server_id="test_load",
            data_directory=Path("/tmp/pir_load_test"),
            timing_config=TimingProtectionConfig(
                enable_padding=True, enable_jitter=True, enable_constant_time=True
            ),
        )

        analyzer = TimingAttackAnalyzer()

        # Test under different load conditions
        load_levels = [1, 5, 10, 20]  # Concurrent queries

        for load in load_levels:
            timings = []

            async def run_query(query_id: str):
                query = {
                    "query_id": query_id,
                    "client_id": "test_client",
                    "query_vectors": [np.random.binomial(1, 0.001, 1000).astype(np.uint8)],
                    "query_type": "genomic",
                    "parameters": {},
                }

                start = time.perf_counter()
                await server.process_query_secure(query)
                elapsed = time.perf_counter() - start

                return elapsed

            # Run concurrent queries
            for trial in range(10):
                tasks = [run_query(f"load_{load}_{trial}_{i}") for i in range(load)]

                results = await asyncio.gather(*tasks)
                timings.extend(results)

                for t in results:
                    analyzer.record_timing(f"load_{load}", t)

        # Check that timing remains consistent across load levels
        variance_analysis = analyzer.analyze_variance()

        # Compare light load vs heavy load
        dist = analyzer.test_distinguishability("load_1", "load_20")

        print(f"Load 1 vs 20 distinguishability: {dist}")
        assert (
            not dist["distinguishable"] or dist["effect_size"] < 0.5
        ), "Timing varies significantly with load"

        await server.shutdown()

    async def test_adversarial_timing_inference(self):
        """Test resistance against adversarial timing inference attacks."""
        """Test resistance against adversarial timing inference attacks."""
    """Test resistance against adversarial timing inference attacks."""
        server = SecurePIRServer(
            server_id="test_adversarial",
            data_directory=Path("/tmp/pir_adversarial_test"),
            timing_config=TimingProtectionConfig(
                enable_padding=True, enable_jitter=True, enable_constant_time=True
            ),
        )

        # Adversary tries to infer query patterns
        secret_pattern = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]  # Binary pattern to infer

        # Collect timing samples
        pattern_timings = defaultdict(list)

        for _ in range(1000):
            # Generate query based on pattern
            pattern_idx = np.random.randint(len(secret_pattern))
            pattern_bit = secret_pattern[pattern_idx]

            # Create query that might leak pattern through timing
            query_vector = np.zeros(1000, dtype=np.uint8)
            if pattern_bit:
                query_vector[:500] = np.random.binomial(1, 0.01, 500)
            else:
                query_vector[500:] = np.random.binomial(1, 0.01, 500)

            query = {
                "query_id": f"adversarial_{_}",
                "client_id": "adversary",
                "query_vectors": [query_vector],
                "query_type": "genomic",
                "parameters": {},
            }

            start = time.perf_counter()
            await server.process_query_secure(query)
            elapsed = time.perf_counter() - start

            pattern_timings[pattern_bit].append(elapsed)

        # Try to infer pattern from timings
        times_0 = np.array(pattern_timings[0])
        times_1 = np.array(pattern_timings[1])

        # Statistical test
        t_stat, p_value = stats.ttest_ind(times_0, times_1)

        print(f"Pattern inference p-value: {p_value}")
        assert p_value > 0.05, "Pattern can be inferred from timing"

        # Machine learning attack (simple classifier)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score

        # Prepare data
        X = np.concatenate([times_0.reshape(-1, 1), times_1.reshape(-1, 1)])
        y = np.concatenate([np.zeros(len(times_0)), np.ones(len(times_1))])

        # Shuffle
        shuffle_idx = np.random.permutation(len(X))
        X = X[shuffle_idx]
        y = y[shuffle_idx]

        # Try to classify
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(clf, X, y, cv=5)

        print(f"ML attack accuracy: {np.mean(scores):.3f}")
        assert np.mean(scores) < 0.55, "ML can predict pattern from timing"

        await server.shutdown()


                def generate_timing_report(test_results: Dict[str, Any], output_path: Path):
                def generate_timing_report(test_results: Dict[str, Any], output_path: Path):
                    """Generate comprehensive timing analysis report."""
    """Generate comprehensive timing analysis report."""
        """Generate comprehensive timing analysis report."""
    report = []
    report.append("# PIR Timing Side-Channel Analysis Report\n")
    report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Summary
    report.append("## Summary\n")
    passed = sum(1 for r in test_results.values() if r["passed"])
    total = len(test_results)
    report.append(f"- Tests passed: {passed}/{total}\n")
    report.append(f"- Overall security: {'GOOD' if passed == total else 'NEEDS IMPROVEMENT'}\n")

    # Detailed results
    report.append("\n## Detailed Results\n")

    for test_name, result in test_results.items():
        report.append(f"\n### {test_name}\n")
        report.append(f"- Status: {'PASS' if result['passed'] else 'FAIL'}\n")
        report.append(f"- Max CV: {result.get('max_cv', 'N/A')}\n")
        report.append(f"- Distinguishability: {result.get('distinguishable', 'N/A')}\n")

        if "notes" in result:
            report.append(f"- Notes: {result['notes']}\n")

    # Recommendations
    report.append("\n## Recommendations\n")

    if passed < total:
        report.append("- Increase jitter window for better timing obfuscation\n")
        report.append("- Review constant-time implementations\n")
        report.append("- Consider adding more padding buckets\n")
    else:
        report.append("- Current timing protections are effective\n")
        report.append("- Continue monitoring for new attack vectors\n")

    # Write report
    with open(output_path, "w") as f:
        f.writelines(report)


# Run all tests and generate report
async def run_timing_security_audit():
    """Run comprehensive timing security audit."""
    """Run comprehensive timing security audit."""
    """Run comprehensive timing security audit."""
    print("Starting PIR timing security audit...")

    test_suite = TestPIRTimingSideChannels()
    results = {}

    # Run each test
    tests = [
        ("Query Size Independence", test_suite.test_query_size_timing_independence),
        ("Index Independence", test_suite.test_index_timing_independence),
        ("Response Padding", test_suite.test_response_size_padding),
        ("Query Mixer", test_suite.test_query_mixer_effectiveness),
        ("Load Consistency", test_suite.test_timing_under_load),
        ("Adversarial Resistance", test_suite.test_adversarial_timing_inference),
    ]

    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            await test_func()
            results[test_name] = {"passed": True}
            print(f"✓ {test_name} PASSED")
        except AssertionError as e:
            results[test_name] = {"passed": False, "error": str(e)}
            print(f"✗ {test_name} FAILED: {e}")
        except Exception as e:
            results[test_name] = {"passed": False, "error": f"Unexpected error: {e}"}
            print(f"✗ {test_name} ERROR: {e}")

    # Generate report
    output_dir = Path("/tmp/genomevault_security_audit")
    output_dir.mkdir(exist_ok=True)

    generate_timing_report(results, output_dir / "timing_security_report.md")

    print(f"\nAudit complete. Report saved to {output_dir}/timing_security_report.md")

    # Return overall pass/fail
    all_passed = all(r["passed"] for r in results.values())
    return all_passed


if __name__ == "__main__":
    # Run the audit
    passed = asyncio.run(run_timing_security_audit())
    exit(0 if passed else 1)
