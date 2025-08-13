"""
Tests for tailchasing calibration functionality.
"""
from tailchasing.calibrate import (
from unittest.mock import patch, MagicMock
import pytest

import numpy as np
    CalibrationTool,
    CodebaseMetrics,
    ThrashEvent,
    create_sample_codebase_metrics,
    create_sample_thrash_events,
)
from tailchasing.config import CalibrationResult, DistanceWeights, PolymerConfig


class TestThrashEvent:
    """Test ThrashEvent data structure."""

    def test_thrash_event_creation(self):
        """Test ThrashEvent creation."""
        event = ThrashEvent(
            file_path="test.py",
            line_number=50,
            timestamp=1000.0,
            severity=0.8,
            frequency=5,
            latency_impact=100.0,
            related_files=["related.py"],
        )

        assert event.file_path == "test.py"
        assert event.line_number == 50
        assert event.timestamp == 1000.0
        assert event.severity == 0.8
        assert event.frequency == 5
        assert event.latency_impact == 100.0
        assert event.related_files == ["related.py"]

    def test_thrash_event_default_related_files(self):
        """Test ThrashEvent with default related_files."""
        event = ThrashEvent(
            file_path="test.py",
            line_number=50,
            timestamp=1000.0,
            severity=0.8,
            frequency=5,
            latency_impact=100.0,
        )

        assert event.related_files == []


class TestCodebaseMetrics:
    """Test CodebaseMetrics data structure."""

    def test_codebase_metrics_creation(self):
        """Test CodebaseMetrics creation."""
        metrics = CodebaseMetrics(
            total_files=10,
            total_lines=1000,
            package_structure={"pkg1": 5, "pkg2": 3},
            dependency_graph={"file1.py": ["file2.py"]},
            ast_complexity={"file1.py": 2.5},
            git_activity={"file1.py": 0.8},
        )

        assert metrics.total_files == 10
        assert metrics.total_lines == 1000
        assert metrics.package_structure == {"pkg1": 5, "pkg2": 3}
        assert metrics.dependency_graph == {"file1.py": ["file2.py"]}
        assert metrics.ast_complexity == {"file1.py": 2.5}
        assert metrics.git_activity == {"file1.py": 0.8}


class TestCalibrationTool:
    """Test CalibrationTool functionality."""

    @pytest.fixture
    def sample_events(self):
        """Create sample thrash events for testing."""
        return [
            ThrashEvent(
                file_path="genomevault/api/handlers.py",
                line_number=50,
                timestamp=1000.0,
                severity=0.8,
                frequency=3,
                latency_impact=50.0,
                related_files=["genomevault/core/engine.py"],
            ),
            ThrashEvent(
                file_path="genomevault/core/engine.py",
                line_number=100,
                timestamp=2000.0,
                severity=0.6,
                frequency=2,
                latency_impact=30.0,
                related_files=["genomevault/db/models.py"],
            ),
            ThrashEvent(
                file_path="genomevault/db/models.py",
                line_number=25,
                timestamp=3000.0,
                severity=0.9,
                frequency=5,
                latency_impact=100.0,
                related_files=[],
            ),
        ]

    @pytest.fixture
    def sample_codebase(self):
        """Create sample codebase metrics for testing."""
        return CodebaseMetrics(
            total_files=5,
            total_lines=2000,
            package_structure={"genomevault": 5},
            dependency_graph={
                "genomevault/api/handlers.py": ["genomevault/core/engine.py"],
                "genomevault/core/engine.py": ["genomevault/db/models.py"],
                "genomevault/db/models.py": [],
                "genomevault/ui/components.py": ["genomevault/api/handlers.py"],
                "genomevault/utils/helpers.py": [],
            },
            ast_complexity={
                "genomevault/api/handlers.py": 3.0,
                "genomevault/core/engine.py": 2.5,
                "genomevault/db/models.py": 1.8,
                "genomevault/ui/components.py": 2.2,
                "genomevault/utils/helpers.py": 1.5,
            },
            git_activity={
                "genomevault/api/handlers.py": 0.8,
                "genomevault/core/engine.py": 0.6,
                "genomevault/db/models.py": 0.9,
                "genomevault/ui/components.py": 0.4,
                "genomevault/utils/helpers.py": 0.2,
            },
        )

    def test_calibration_tool_creation(self):
        """Test CalibrationTool creation."""
        tool = CalibrationTool()

        assert isinstance(tool.config, PolymerConfig)
        assert tool.config.alpha == 1.2  # Default

    def test_calibration_tool_with_config(self):
        """Test CalibrationTool with custom config."""
        config = PolymerConfig(alpha=1.3)
        tool = CalibrationTool(config)

        assert tool.config.alpha == 1.3

    def test_prepare_training_data(self, sample_events, sample_codebase):
        """Test training data preparation."""
        tool = CalibrationTool()
        X, y = tool._prepare_training_data(sample_events, sample_codebase)

        assert X.shape[0] == len(sample_events)  # One row per event
        assert X.shape[1] == 7  # Expected feature count
        assert y.shape[0] == len(sample_events)
        assert len(y) == 3

    def test_prepare_training_data_empty(self):
        """Test training data preparation with empty events."""
        tool = CalibrationTool()
        codebase = CodebaseMetrics(
            total_files=0,
            total_lines=0,
            package_structure={},
            dependency_graph={},
            ast_complexity={},
            git_activity={},
        )

        X, y = tool._prepare_training_data([], codebase)

        assert X.shape[0] == 0
        assert y.shape[0] == 0

    def test_extract_features(self, sample_events, sample_codebase):
        """Test feature extraction for a single event."""
        tool = CalibrationTool()
        event = sample_events[0]

        features = tool._extract_features(event, sample_codebase)

        assert features is not None
        assert len(features) == 7
        assert all(np.isfinite(features))

    def test_extract_features_missing_file(self, sample_codebase):
        """Test feature extraction for event with missing file."""
        tool = CalibrationTool()
        event = ThrashEvent(
            file_path="nonexistent/file.py",
            line_number=10,
            timestamp=1000.0,
            severity=0.5,
            frequency=1,
            latency_impact=10.0,
        )

        # Should still return features (with defaults for missing data)
        features = tool._extract_features(event, sample_codebase)

        assert features is not None
        assert len(features) == 7

    def test_predict_thrash_probability(self):
        """Test thrash probability prediction."""
        tool = CalibrationTool()

        # Sample feature matrix
        X = np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 1.0, 5.0, 2.0],
                [0.2, 0.3, 0.4, 0.5, 2.0, 3.0, 1.0],
                [0.3, 0.4, 0.5, 0.6, 1.5, 4.0, 3.0],
            ]
        )

        alpha = 1.2
        weights = DistanceWeights(tok=1.0, ast=2.0, mod=3.0, git=4.0)

        predictions = tool._predict_thrash_probability(X, alpha, weights)

        assert len(predictions) == 3
        assert all(0 <= p <= 1 for p in predictions)  # Normalized to [0, 1]
        assert all(np.isfinite(predictions))

    def test_predict_thrash_probability_empty(self):
        """Test thrash probability prediction with empty input."""
        tool = CalibrationTool()

        X = np.array([]).reshape(0, 7)
        alpha = 1.2
        weights = DistanceWeights()

        predictions = tool._predict_thrash_probability(X, alpha, weights)

        assert len(predictions) == 0

    def test_grid_search_alpha(self, sample_events, sample_codebase):
        """Test alpha parameter grid search."""
        tool = CalibrationTool()

        X, y = tool._prepare_training_data(sample_events, sample_codebase)

        # Mock to avoid actual computation
        with patch.object(tool, "_predict_thrash_probability") as mock_predict:
            # Return predictable values for correlation calculation
            mock_predict.side_effect = [
                np.array([0.1, 0.2, 0.3]),  # First alpha
                np.array([0.2, 0.3, 0.4]),  # Second alpha
                np.array([0.3, 0.4, 0.5]),  # Third alpha
            ]

            optimal_alpha, best_score = tool._grid_search_alpha(X, y, (1.0, 1.2), 3)

            assert 1.0 <= optimal_alpha <= 1.2
            assert isinstance(best_score, float)

    def test_optimize_weights(self, sample_events, sample_codebase):
        """Test weight optimization."""
        tool = CalibrationTool()

        X, y = tool._prepare_training_data(sample_events, sample_codebase)
        alpha = 1.2

        # Test with limited iterations to speed up test
        optimal_weights, score, iterations, converged = tool._optimize_weights(
            X, y, alpha, max_iterations=5, tolerance=1e-3
        )

        assert isinstance(optimal_weights, DistanceWeights)
        assert optimal_weights.tok > 0
        assert optimal_weights.ast > 0
        assert optimal_weights.mod > 0
        assert optimal_weights.git > 0
        assert isinstance(score, float)
        assert isinstance(iterations, int)
        assert isinstance(converged, bool)

    def test_cross_validate(self, sample_events, sample_codebase):
        """Test cross-validation."""
        tool = CalibrationTool()

        X, y = tool._prepare_training_data(sample_events, sample_codebase)
        alpha = 1.2
        weights = DistanceWeights()

        # Test with small k for limited data
        score = tool._cross_validate(X, y, alpha, weights, k_folds=2)

        assert isinstance(score, float)
        assert -1 <= score <= 1  # Correlation should be in [-1, 1]

    def test_cross_validate_small_dataset(self):
        """Test cross-validation with very small dataset."""
        tool = CalibrationTool()

        # Very small dataset (less than k_folds)
        X = np.array([[0.1, 0.2, 0.3, 0.4, 1.0, 5.0, 2.0]])
        y = np.array([0.5])
        alpha = 1.2
        weights = DistanceWeights()

        # Should fall back to leave-one-out
        score = tool._cross_validate(X, y, alpha, weights, k_folds=5)

        assert isinstance(score, float)

    def test_leave_one_out_validate(self):
        """Test leave-one-out validation."""
        tool = CalibrationTool()

        X = np.array([[0.1, 0.2, 0.3, 0.4, 1.0, 5.0, 2.0], [0.2, 0.3, 0.4, 0.5, 2.0, 3.0, 1.0]])
        y = np.array([0.5, 0.7])
        alpha = 1.2
        weights = DistanceWeights()

        score = tool._leave_one_out_validate(X, y, alpha, weights)

        assert isinstance(score, float)

    def test_fit_parameters_insufficient_data(self):
        """Test parameter fitting with insufficient data."""
        tool = CalibrationTool()

        with pytest.raises(ValueError, match="Need at least one historical thrash event"):
            tool.fit_parameters([], CodebaseMetrics(0, 0, {}, {}, {}, {}))

    def test_fit_parameters_success(self, sample_events, sample_codebase):
        """Test successful parameter fitting."""
        tool = CalibrationTool()

        # Use small parameters to speed up test
        result = tool.fit_parameters(
            sample_events,
            sample_codebase,
            alpha_range=(1.1, 1.3),
            alpha_steps=3,
            max_iterations=5,
        )

        assert isinstance(result.optimal_alpha, float)
        assert 1.1 <= result.optimal_alpha <= 1.3
        assert isinstance(result.optimal_weights, DistanceWeights)
        assert isinstance(result.correlation_score, float)
        assert isinstance(result.validation_score, float)
        assert isinstance(result.iterations, int)
        assert isinstance(result.converged, bool)

    def test_validate_predictions(self, sample_events, sample_codebase):
        """Test prediction validation."""
        tool = CalibrationTool()

        metrics = tool.validate_predictions(sample_events, sample_codebase)

        assert "pearson_correlation" in metrics
        assert "pearson_p_value" in metrics
        assert "spearman_correlation" in metrics
        assert "rmse" in metrics
        assert "r_squared" in metrics
        assert "sample_size" in metrics

        assert metrics["sample_size"] == len(sample_events)

    def test_validate_predictions_empty(self):
        """Test prediction validation with empty test set."""
        tool = CalibrationTool()
        codebase = CodebaseMetrics(0, 0, {}, {}, {}, {})

        metrics = tool.validate_predictions([], codebase)

        assert "error" in metrics
        assert metrics["error"] == "Empty test set"

    def test_suggest_config_updates(self):
        """Test configuration update suggestions."""
        tool = CalibrationTool()

        # Create calibration result with different parameters
        calibration_result = CalibrationResult(
            optimal_alpha=1.4,
            optimal_weights=DistanceWeights(tok=2.0, ast=3.0, mod=4.0, git=5.0),
            correlation_score=0.85,
            validation_score=0.80,
            iterations=50,
            converged=True,
        )

        suggestions = tool.suggest_config_updates(calibration_result)

        assert "summary" in suggestions
        assert "alpha_change" in suggestions
        assert "weight_changes" in suggestions
        assert "expected_improvement" in suggestions
        assert "confidence" in suggestions

        # Should suggest alpha change (1.4 vs default 1.2)
        assert suggestions["alpha_change"]["to"] == 1.4

        # Should suggest weight changes
        assert "tok" in suggestions["weight_changes"]
        assert "ast" in suggestions["weight_changes"]

        assert suggestions["confidence"] == "high"  # converged=True

    def test_suggest_config_updates_no_changes(self):
        """Test suggestions when no changes are needed."""
        tool = CalibrationTool()
        current_config = PolymerConfig()

        # Calibration result matches current config
        calibration_result = CalibrationResult(
            optimal_alpha=current_config.alpha,
            optimal_weights=current_config.weights,
            correlation_score=0.85,
            validation_score=0.80,
            iterations=50,
            converged=True,
        )

        suggestions = tool.suggest_config_updates(calibration_result, current_config)

        assert not suggestions["alpha_change"]  # No alpha change
        assert not suggestions["weight_changes"]  # No weight changes
        assert "already well-tuned" in suggestions["summary"]

    def test_explain_alpha_change(self):
        """Test alpha change explanations."""
        tool = CalibrationTool()

        # Large positive change
        explanation = tool._explain_alpha_change(0.2)
        assert "Increase contact decay" in explanation

        # Large negative change
        explanation = tool._explain_alpha_change(-0.2)
        assert "Decrease contact decay" in explanation

        # Small change
        explanation = tool._explain_alpha_change(0.05)
        assert "Minor adjustment" in explanation

    def test_explain_weight_change(self):
        """Test weight change explanations."""
        tool = CalibrationTool()

        # Large positive change
        explanation = tool._explain_weight_change("tok", 0.8)
        assert "Increase token-level distance importance significantly" in explanation

        # Small positive change
        explanation = tool._explain_weight_change("ast", 0.2)
        assert "Increase AST-level distance importance" in explanation

        # Large negative change
        explanation = tool._explain_weight_change("mod", -0.8)
        assert "Decrease module-level distance importance significantly" in explanation

        # Small change
        explanation = tool._explain_weight_change("git", 0.05)
        assert "Minor adjustment" in explanation


class TestSampleDataGeneration:
    """Test sample data generation functions."""

    def test_create_sample_thrash_events(self):
        """Test sample thrash events generation."""
        file_paths = ["file1.py", "file2.py", "file3.py"]
        events = create_sample_thrash_events(file_paths, event_count=10, random_seed=42)

        assert len(events) == 10

        for event in events:
            assert isinstance(event, ThrashEvent)
            assert event.file_path in file_paths
            assert 1 <= event.line_number <= 500
            assert event.timestamp >= 0
            assert 0 <= event.severity <= 1
            assert event.frequency >= 1
            assert event.latency_impact >= 0
            assert isinstance(event.related_files, list)

    def test_create_sample_thrash_events_reproducible(self):
        """Test that sample events are reproducible with same seed."""
        file_paths = ["file1.py", "file2.py"]

        events1 = create_sample_thrash_events(file_paths, event_count=5, random_seed=42)
        events2 = create_sample_thrash_events(file_paths, event_count=5, random_seed=42)

        # Should be identical with same seed
        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2):
            assert e1.file_path == e2.file_path
            assert e1.line_number == e2.line_number
            assert e1.severity == e2.severity

    def test_create_sample_codebase_metrics(self):
        """Test sample codebase metrics generation."""
        file_paths = ["pkg1/file1.py", "pkg1/file2.py", "pkg2/file3.py"]
        metrics = create_sample_codebase_metrics(file_paths)

        assert isinstance(metrics, CodebaseMetrics)
        assert metrics.total_files == len(file_paths)
        assert metrics.total_lines > 0
        assert len(metrics.package_structure) > 0
        assert len(metrics.dependency_graph) == len(file_paths)
        assert len(metrics.ast_complexity) == len(file_paths)
        assert len(metrics.git_activity) == len(file_paths)

        # Check that all files are represented
        for file_path in file_paths:
            assert file_path in metrics.dependency_graph
            assert file_path in metrics.ast_complexity
            assert file_path in metrics.git_activity

    def test_create_sample_codebase_metrics_packages(self):
        """Test package structure extraction in sample metrics."""
        file_paths = [
            "genomevault/api/handlers.py",
            "genomevault/core/engine.py",
            "utils/helpers.py",
        ]
        metrics = create_sample_codebase_metrics(file_paths)

        # Should extract packages from file paths
        assert "genomevault" in metrics.package_structure
        assert "utils" in metrics.package_structure
        assert metrics.package_structure["genomevault"] >= 2  # At least 2 files
        assert metrics.package_structure["utils"] >= 1  # At least 1 file


class TestCalibrationEdgeCases:
    """Test edge cases and error conditions in calibration."""

    def test_calibration_with_single_event(self):
        """Test calibration with only one event."""
        tool = CalibrationTool()

        event = ThrashEvent("test.py", 10, 1000.0, 0.5, 1, 10.0)
        codebase = create_sample_codebase_metrics(["test.py"])

        # Should handle single event gracefully
        result = tool.fit_parameters(
            [event], codebase, alpha_range=(1.1, 1.3), alpha_steps=3, max_iterations=5
        )

        assert isinstance(result.optimal_alpha, float)

    def test_calibration_with_identical_events(self):
        """Test calibration with identical severity events."""
        tool = CalibrationTool()

        # Create identical events (same severity, frequency, latency)
        events = [ThrashEvent("file1.py", 10, float(i), 0.5, 2, 20.0) for i in range(5)]

        codebase = create_sample_codebase_metrics(["file1.py"])

        # Should handle identical targets
        result = tool.fit_parameters(
            events, codebase, alpha_range=(1.1, 1.3), alpha_steps=3, max_iterations=5
        )

        assert isinstance(result, CalibrationResult)

    def test_feature_extraction_edge_cases(self):
        """Test feature extraction with edge case inputs."""
        tool = CalibrationTool()

        # Empty codebase
        empty_codebase = CodebaseMetrics(0, 0, {}, {}, {}, {})

        event = ThrashEvent("test.py", 1, 1000.0, 0.5, 1, 10.0)

        # Should handle empty codebase
        features = tool._extract_features(event, empty_codebase)

        assert features is not None
        assert len(features) == 7
        assert all(np.isfinite(features))

    def test_distance_calculations(self):
        """Test individual distance calculation methods."""
        tool = CalibrationTool()
        event = ThrashEvent("test/file.py", 100, 1000.0, 0.5, 1, 10.0)

        codebase = CodebaseMetrics(
            total_files=1,
            total_lines=1000,
            package_structure={"test": 1},
            dependency_graph={"test/file.py": []},
            ast_complexity={"test/file.py": 3.0},
            git_activity={"test/file.py": 0.5},
        )

        # Test individual distance calculations
        tok_dist = tool._calculate_token_distance(event, codebase)
        ast_dist = tool._calculate_ast_distance(event, codebase)
        mod_dist = tool._calculate_module_distance(event, codebase)
        git_dist = tool._calculate_git_distance(event, codebase)

        assert isinstance(tok_dist, float)
        assert isinstance(ast_dist, float)
        assert isinstance(mod_dist, float)
        assert isinstance(git_dist, float)

        assert tok_dist > 0
        assert ast_dist > 0
        assert mod_dist > 0
        assert git_dist > 0


if __name__ == "__main__":
    pytest.main([__file__])
