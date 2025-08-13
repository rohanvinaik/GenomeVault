"""
Tests for tailchasing chromatin-inspired reporting functionality.
"""
from tailchasing.core.reporting import (
from typing import Dict, List
import json
import pytest

import numpy as np
    HiCHeatmapGenerator,
    PolymerMetricsReport,
    TAD,
    ThrashCluster,
    generate_comparative_matrices,
    integrate_chromatin_analysis,
)


class TestTAD:
    """Test TAD data structure."""

    def test_tad_creation(self):
        """Test TAD object creation."""
        tad = TAD(start=10, end=50, name="test_tad", activity_level=0.8)

        assert tad.start == 10
        assert tad.end == 50
        assert tad.name == "test_tad"
        assert tad.activity_level == 0.8

    def test_tad_size(self):
        """Test TAD size calculation."""
        tad = TAD(start=10, end=50, name="test_tad")
        assert tad.size() == 40

    def test_tad_contains(self):
        """Test TAD position containment."""
        tad = TAD(start=10, end=50, name="test_tad")

        assert tad.contains(25) is True
        assert tad.contains(10) is True  # Boundary
        assert tad.contains(50) is True  # Boundary
        assert tad.contains(5) is False
        assert tad.contains(55) is False


class TestThrashCluster:
    """Test ThrashCluster data structure."""

    def test_cluster_creation(self):
        """Test cluster creation."""
        cluster = ThrashCluster(
            positions=[10, 12, 15], risk_score=0.8, frequency=5, avg_latency=100.0
        )

        assert cluster.positions == [10, 12, 15]
        assert cluster.risk_score == 0.8
        assert cluster.frequency == 5
        assert cluster.avg_latency == 100.0

    def test_cluster_center(self):
        """Test cluster center calculation."""
        cluster = ThrashCluster(
            positions=[10, 20, 30], risk_score=0.5, frequency=3, avg_latency=50.0
        )

        assert cluster.center() == 20.0

    def test_empty_cluster_center(self):
        """Test center calculation for empty cluster."""
        cluster = ThrashCluster(positions=[], risk_score=0.0, frequency=0, avg_latency=0.0)

        assert cluster.center() == 0.0


class TestHiCHeatmapGenerator:
    """Test Hi-C heatmap generation functionality."""

    @pytest.fixture
    def generator(self):
        """Create HiCHeatmapGenerator instance."""
        return HiCHeatmapGenerator()

    @pytest.fixture
    def sample_matrix(self):
        """Create sample contact matrix."""
        matrix = np.array(
            [
                [1.0, 0.8, 0.3, 0.1],
                [0.8, 1.0, 0.6, 0.2],
                [0.3, 0.6, 1.0, 0.4],
                [0.1, 0.2, 0.4, 1.0],
            ]
        )
        return matrix

    @pytest.fixture
    def sample_tads(self):
        """Create sample TADs."""
        return [
            TAD(start=0, end=25, name="TAD1", activity_level=0.8),
            TAD(start=25, end=75, name="TAD2", activity_level=0.6),
            TAD(start=75, end=100, name="TAD3", activity_level=0.9),
        ]

    def test_empty_matrix_handling(self, generator):
        """Test handling of empty contact matrix."""
        empty_matrix = np.array([])
        result = generator.generate_contact_heatmap(empty_matrix)
        assert "Empty contact matrix" in result

    def test_contact_heatmap_generation(self, generator, sample_matrix):
        """Test basic heatmap generation."""
        result = generator.generate_contact_heatmap(sample_matrix)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Performance Contact Matrix" in result

    def test_heatmap_with_tads(self, generator, sample_matrix, sample_tads):
        """Test heatmap generation with TAD boundaries."""
        result = generator.generate_contact_heatmap(sample_matrix, sample_tads)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_normalize_matrix(self, generator):
        """Test matrix normalization."""
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        normalized = generator._normalize_matrix(matrix)

        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert normalized.shape == matrix.shape

    def test_normalize_constant_matrix(self, generator):
        """Test normalization of constant matrix."""
        matrix = np.array([[5, 5], [5, 5]])
        normalized = generator._normalize_matrix(matrix)

        assert np.allclose(normalized, 0.0)

    def test_matrix_to_ascii(self, generator):
        """Test ASCII conversion."""
        normalized = np.array([[0.0, 0.5], [0.8, 1.0]])
        ascii_lines = generator._matrix_to_ascii(normalized)

        assert len(ascii_lines) == 2
        assert len(ascii_lines[0]) == 2
        assert len(ascii_lines[1]) == 2

    def test_thrash_cluster_highlighting(self, generator, sample_matrix):
        """Test thrash cluster highlighting."""
        risk_scores = {(0, 0): 0.9, (1, 1): 0.8, (2, 2): 0.3}

        result = generator.highlight_thrash_clusters(sample_matrix, risk_scores)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_matrix_thrash_highlighting(self, generator):
        """Test thrash highlighting with empty matrix."""
        empty_matrix = np.array([])
        risk_scores = {}

        result = generator.highlight_thrash_clusters(empty_matrix, risk_scores)
        assert "Empty matrix" in result

    def test_tad_boundary_visualization(self, generator, sample_matrix, sample_tads):
        """Test TAD boundary visualization."""
        tad_map = {tad.name: tad for tad in sample_tads}

        result = generator.show_tad_boundaries(sample_matrix, tad_map)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "TAD1" in result
        assert "TAD2" in result
        assert "TAD3" in result

    def test_empty_matrix_tad_boundaries(self, generator):
        """Test TAD boundaries with empty matrix."""
        empty_matrix = np.array([])
        tad_map = {}

        result = generator.show_tad_boundaries(empty_matrix, tad_map)
        assert "Empty matrix" in result


class TestPolymerMetricsReport:
    """Test polymer metrics reporting functionality."""

    @pytest.fixture
    def reporter(self):
        """Create PolymerMetricsReport instance."""
        return PolymerMetricsReport()

    @pytest.fixture
    def sample_tads(self):
        """Create sample TADs."""
        return [
            TAD(start=0, end=30, name="TAD1", activity_level=0.8),
            TAD(start=30, end=70, name="TAD2", activity_level=0.6),
            TAD(start=70, end=100, name="TAD3", activity_level=0.9),
        ]

    @pytest.fixture
    def sample_interactions(self):
        """Create sample interaction data."""
        return [
            (10, 20, 0.8),  # Intra-TAD1
            (15, 25, 0.6),  # Intra-TAD1
            (40, 50, 0.7),  # Intra-TAD2
            (20, 45, 0.4),  # Inter-TAD (TAD1-TAD2)
            (50, 80, 0.5),  # Inter-TAD (TAD2-TAD3)
        ]

    @pytest.fixture
    def sample_fix_strategies(self):
        """Create sample fix strategies."""
        return [
            {
                "name": "caching",
                "impact_score": 0.8,
                "complexity": 0.3,
                "confidence": 0.9,
            },
            {
                "name": "refactoring",
                "impact_score": 0.6,
                "complexity": 0.7,
                "confidence": 0.7,
            },
            {
                "name": "optimization",
                "impact_score": 0.9,
                "complexity": 0.5,
                "confidence": 0.8,
            },
        ]

    @pytest.fixture
    def sample_timeline(self):
        """Create sample timeline data."""
        return [
            {
                "timestamp": 0.0,
                "name": "init",
                "duration": 10.0,
                "impact": 0.2,
                "status": "completed",
            },
            {
                "timestamp": 10.0,
                "name": "processing",
                "duration": 50.0,
                "impact": 0.8,
                "status": "completed",
            },
            {
                "timestamp": 60.0,
                "name": "cleanup",
                "duration": 5.0,
                "impact": 0.1,
                "status": "pending",
            },
        ]

    def test_polymer_distance_calculation(self, reporter, sample_tads, sample_interactions):
        """Test polymer distance calculation."""
        metrics = reporter.calculate_polymer_distances(sample_tads, sample_interactions)

        assert "intra_tad_distances" in metrics
        assert "inter_tad_distances" in metrics
        assert "global_metrics" in metrics

        # Check that we have intra-TAD metrics
        assert "TAD1" in metrics["intra_tad_distances"]
        assert "TAD2" in metrics["intra_tad_distances"]

        # Check metric structure
        tad1_metrics = metrics["intra_tad_distances"]["TAD1"]
        assert "mean" in tad1_metrics
        assert "std" in tad1_metrics
        assert "median" in tad1_metrics
        assert "count" in tad1_metrics

    def test_empty_interactions(self, reporter, sample_tads):
        """Test handling of empty interactions."""
        metrics = reporter.calculate_polymer_distances(sample_tads, [])

        assert metrics["intra_tad_distances"] == {}
        assert metrics["inter_tad_distances"] == {}
        assert metrics["global_metrics"] == {}

    def test_contact_probability_calculation(self, reporter, sample_interactions):
        """Test contact probability calculation."""
        probabilities = reporter.calculate_contact_probabilities(sample_interactions)

        assert "distance_bins" in probabilities
        assert "probabilities" in probabilities
        assert "statistics" in probabilities

        # Check statistics
        stats = probabilities["statistics"]
        assert "mean_contact_distance" in stats
        assert "contact_decay_rate" in stats
        assert "short_range_fraction" in stats
        assert "long_range_fraction" in stats

    def test_empty_contact_probabilities(self, reporter):
        """Test contact probabilities with empty data."""
        probabilities = reporter.calculate_contact_probabilities([])

        assert len(probabilities["distance_bins"]) > 0
        assert len(probabilities["probabilities"]) > 0
        assert all(p == 0 for p in probabilities["probabilities"])

    def test_thrash_reduction_prediction(self, reporter, sample_fix_strategies):
        """Test thrash reduction prediction."""
        predictions = reporter.predict_thrash_reduction(sample_fix_strategies)

        assert len(predictions) == 3
        assert "caching" in predictions
        assert "refactoring" in predictions
        assert "optimization" in predictions

        # Check prediction structure
        caching_pred = predictions["caching"]
        assert "estimated_reduction" in caching_pred
        assert "implementation_risk" in caching_pred
        assert "roi_score" in caching_pred
        assert "recommended_priority" in caching_pred

        # Check priority values
        assert caching_pred["recommended_priority"] in ["High", "Medium", "Low"]

    def test_empty_fix_strategies(self, reporter):
        """Test prediction with empty strategies."""
        predictions = reporter.predict_thrash_reduction([])

        assert predictions == {}

    def test_replication_timing_visualization(self, reporter, sample_timeline):
        """Test replication timing visualization."""
        result = reporter.visualize_replication_timing(sample_timeline)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "Replication Timing Schedule" in result
        assert "init" in result
        assert "processing" in result
        assert "cleanup" in result

    def test_empty_timeline_visualization(self, reporter):
        """Test visualization with empty timeline."""
        result = reporter.visualize_replication_timing([])

        assert "No timeline data available" in result

    def test_comprehensive_report_generation(
        self,
        reporter,
        sample_tads,
        sample_interactions,
        sample_fix_strategies,
        sample_timeline,
    ):
        """Test comprehensive report generation."""
        report = reporter.generate_comprehensive_report(
            sample_tads, sample_interactions, sample_fix_strategies, sample_timeline
        )

        assert "polymer_distances" in report
        assert "contact_probabilities" in report
        assert "thrash_predictions" in report
        assert "timeline_analysis" in report
        assert "summary_metrics" in report

        # Check summary metrics
        summary = report["summary_metrics"]
        assert "overall_health_score" in summary
        assert "optimization_potential" in summary
        assert "stability_index" in summary

        # Verify health score is in valid range
        assert 0.0 <= summary["overall_health_score"] <= 1.0
        assert 0.0 <= summary["stability_index"] <= 1.0


class TestIntegrationFunctions:
    """Test integration and utility functions."""

    @pytest.fixture
    def sample_matrix(self):
        """Create sample contact matrix."""
        return np.array([[1.0, 0.8, 0.3], [0.8, 1.0, 0.6], [0.3, 0.6, 1.0]])

    @pytest.fixture
    def sample_tads(self):
        """Create sample TADs."""
        return [
            TAD(start=0, end=50, name="TAD1", activity_level=0.8),
            TAD(start=50, end=100, name="TAD2", activity_level=0.6),
        ]

    @pytest.fixture
    def sample_interactions(self):
        """Create sample interactions."""
        return [(10, 20, 0.8), (30, 40, 0.6), (15, 35, 0.4)]

    @pytest.fixture
    def sample_fix_strategies(self):
        """Create sample fix strategies."""
        return [
            {
                "name": "optimization",
                "impact_score": 0.8,
                "complexity": 0.4,
                "confidence": 0.9,
            }
        ]

    @pytest.fixture
    def sample_timeline(self):
        """Create sample timeline."""
        return [
            {
                "timestamp": 0.0,
                "name": "test_event",
                "duration": 10.0,
                "impact": 0.5,
                "status": "completed",
            }
        ]

    def test_chromatin_analysis_integration(
        self,
        sample_matrix,
        sample_tads,
        sample_interactions,
        sample_fix_strategies,
        sample_timeline,
    ):
        """Test integration of chromatin analysis into existing reports."""
        existing_report = {"existing_key": "existing_value"}

        enhanced_report = integrate_chromatin_analysis(
            existing_report,
            sample_matrix,
            sample_tads,
            sample_interactions,
            sample_fix_strategies,
            sample_timeline,
        )

        # Check original data preserved
        assert enhanced_report["existing_key"] == "existing_value"

        # Check chromatin analysis added
        assert "chromatin_analysis" in enhanced_report

        chromatin = enhanced_report["chromatin_analysis"]
        assert "contact_matrix_summary" in chromatin
        assert "tad_analysis" in chromatin
        assert "polymer_metrics" in chromatin
        assert "visualization_data" in chromatin
        assert "risk_analysis" in chromatin

        # Check contact matrix summary
        matrix_summary = chromatin["contact_matrix_summary"]
        assert matrix_summary["dimensions"] == sample_matrix.shape
        assert matrix_summary["total_contacts"] == int(np.sum(sample_matrix))

        # Check TAD analysis
        tad_analysis = chromatin["tad_analysis"]
        assert tad_analysis["total_tads"] == len(sample_tads)
        assert len(tad_analysis["tad_details"]) == len(sample_tads)

    def test_comparative_matrix_generation(self, sample_matrix, sample_tads):
        """Test comparative before/after matrix analysis."""
        # Create "after" matrix with some improvements
        after_matrix = sample_matrix * 0.8  # 20% reduction

        comparative = generate_comparative_matrices(
            sample_matrix, after_matrix, sample_tads, "test_optimization"
        )

        assert "strategy_name" in comparative
        assert comparative["strategy_name"] == "test_optimization"

        assert "metrics" in comparative
        metrics = comparative["metrics"]
        assert "total_contacts_before" in metrics
        assert "total_contacts_after" in metrics
        assert "reduction_percentage" in metrics
        assert "improvement_score" in metrics

        # Check that reduction is calculated correctly
        expected_reduction = 20.0  # 20% reduction
        assert abs(metrics["reduction_percentage"] - expected_reduction) < 1.0

        assert "visualizations" in comparative
        viz = comparative["visualizations"]
        assert "before_heatmap" in viz
        assert "after_heatmap" in viz
        assert "difference_heatmap" in viz

        assert "tad_specific_improvements" in comparative

    def test_comparative_matrices_no_change(self, sample_matrix, sample_tads):
        """Test comparative analysis with no changes."""
        comparative = generate_comparative_matrices(
            sample_matrix, sample_matrix, sample_tads, "no_change"
        )

        metrics = comparative["metrics"]
        assert metrics["reduction_percentage"] == 0.0
        assert metrics["improvement_score"] == 0.0
        assert metrics["absolute_reduction"] == 0

    def test_comparative_matrices_negative_change(self, sample_matrix, sample_tads):
        """Test comparative analysis with performance degradation."""
        worse_matrix = sample_matrix * 1.2  # 20% increase (worse performance)

        comparative = generate_comparative_matrices(
            sample_matrix, worse_matrix, sample_tads, "degradation"
        )

        metrics = comparative["metrics"]
        assert metrics["reduction_percentage"] < 0  # Negative reduction = increase
        assert metrics["improvement_score"] == 0.0  # No improvement


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_element_matrix(self):
        """Test with 1x1 matrix."""
        generator = HiCHeatmapGenerator()
        matrix = np.array([[1.0]])

        result = generator.generate_contact_heatmap(matrix)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_zero_matrix(self):
        """Test with all-zero matrix."""
        generator = HiCHeatmapGenerator()
        matrix = np.zeros((3, 3))

        result = generator.generate_contact_heatmap(matrix)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_invalid_tad_ranges(self):
        """Test with TADs that have invalid ranges."""
        reporter = PolymerMetricsReport()

        # TAD with start > end
        invalid_tad = TAD(start=50, end=30, name="invalid")

        # Should not crash
        metrics = reporter.calculate_polymer_distances([invalid_tad], [(40, 45, 0.5)])
        assert isinstance(metrics, dict)

    def test_large_matrix_performance(self):
        """Test with larger matrix to ensure reasonable performance."""
        generator = HiCHeatmapGenerator()

        # Create 100x100 matrix
        large_matrix = np.random.rand(100, 100)

        result = generator.generate_contact_heatmap(large_matrix)
        assert isinstance(result, str)
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__])
