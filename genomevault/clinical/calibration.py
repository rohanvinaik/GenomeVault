from genomevault.observability.logging import configure_logging

logger = configure_logging()
# genomevault/clinical/calibration.py
"""Clinical calibration suite for error budget management and compliance."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score


class UseCaseCategory(Enum):
    """Clinical use case categories with error tolerances."""

    SCREENING = ("screening", 0.05, "Population health screening")
    DIAGNOSTIC = ("diagnostic", 0.01, "Clinical diagnosis")
    RESEARCH = ("research", 0.02, "Research applications")
    REGULATORY = ("regulatory", 0.005, "Regulatory submissions")
    PRECISION_MEDICINE = ("precision_medicine", 0.008, "Personalized treatment")

    def __init__(self, key: str, error_budget: float, description: str):
        self.key = key
        self.error_budget = error_budget
        self.description = description


@dataclass
class CalibrationMetrics:
    """Metrics from calibration process."""

    use_case: UseCaseCategory
    achieved_error: float
    compression_ratio: float
    sensitivity: float
    specificity: float
    auc_score: float
    precision_at_threshold: float
    recall_at_threshold: float
    optimal_threshold: float
    confidence_interval: tuple[float, float]
    sample_size: int
    calibration_timestamp: float


@dataclass
class ParetoPoint:
    """Point on Pareto frontier."""

    compression_ratio: float
    error_rate: float
    settings: dict[str, Any]
    metrics: CalibrationMetrics


class CalibrationDataset(Protocol):
    """Protocol for calibration datasets."""

    def get_samples(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Get n samples with labels."""
        ...

    def get_validation_set(self) -> tuple[np.ndarray, np.ndarray]:
        """Get validation dataset."""
        ...


class ClinicalCalibrationSuite:
    """Comprehensive calibration suite for clinical compliance."""

    def __init__(self, model: Any, output_dir: Path | None = None):  # HybridKANHD model
        self.model = model
        self.output_dir = output_dir or Path("calibration_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Calibration results cache
        self.calibration_results: dict[str, CalibrationMetrics] = {}
        self.pareto_frontiers: dict[str, list[ParetoPoint]] = {}

    def calibrate_for_use_case(
        self,
        use_case: UseCaseCategory,
        dataset: CalibrationDataset,
        n_iterations: int = 100,
        confidence_level: float = 0.95,
    ) -> CalibrationMetrics:
        """Calibrate model for specific clinical use case."""
        logger.info(
            f"Calibrating for {use_case.description} (error budget: {use_case.error_budget})"
        )

        # Get calibration data
        X_cal, y_cal = dataset.get_samples(n=1000)
        X_val, y_val = dataset.get_validation_set()

        # Parameter search space
        param_space = {
            "hd_resolution": [10000, 15000, 20000],
            "noise_scale": np.logspace(-3, -1, 10),
            "kan_layers": [3, 4, 5],
            "spline_knots": [8, 10, 12, 16],
        }

        best_metrics = None
        best_error = float("inf")

        # Grid search with early stopping
        for resolution in param_space["hd_resolution"]:
            for noise in param_space["noise_scale"]:
                for layers in param_space["kan_layers"]:
                    for knots in param_space["spline_knots"]:
                        # Configure model
                        self._configure_model(resolution, noise, layers, knots)

                        # Evaluate
                        metrics = self._evaluate_configuration(X_cal, y_cal, X_val, y_val, use_case)

                        # Check if meets error budget
                        if metrics.achieved_error <= use_case.error_budget:
                            if metrics.achieved_error < best_error:
                                best_error = metrics.achieved_error
                                best_metrics = metrics

                        # Early stopping if we're well within budget
                        if metrics.achieved_error < use_case.error_budget * 0.8:
                            break

        if not best_metrics:
            raise ValueError(f"Could not achieve error budget for {use_case.description}")

        # Bootstrap confidence intervals
        ci_lower, ci_upper = self._bootstrap_confidence_interval(
            X_val, y_val, best_metrics, confidence_level
        )
        best_metrics.confidence_interval = (ci_lower, ci_upper)

        # Store results
        self.calibration_results[use_case.key] = best_metrics
        self._save_calibration_results(use_case, best_metrics)

        return best_metrics

    def compute_pareto_frontier(
        self, dataset: CalibrationDataset, use_cases: list[UseCaseCategory]
    ) -> dict[str, list[ParetoPoint]]:
        """Compute Pareto frontier for compression vs accuracy tradeoff."""
        X_test, y_test = dataset.get_validation_set()

        frontiers = {}

        for use_case in use_cases:
            points = []

            # Sample parameter space
            for compression_target in np.logspace(1, 2.7, 20):  # 10x to 500x
                # Find best error for this compression
                best_error = float("inf")
                best_settings = None
                best_metrics = None

                for _ in range(10):  # Random search iterations
                    settings = self._sample_random_settings(compression_target)

                    # Configure and evaluate
                    self._configure_model(**settings)
                    metrics = self._evaluate_configuration(
                        X_test[:100],
                        y_test[:100],
                        X_test,
                        y_test,
                        use_case,  # Quick eval
                    )

                    if metrics.achieved_error < best_error:
                        best_error = metrics.achieved_error
                        best_settings = settings
                        best_metrics = metrics

                if best_settings:
                    points.append(
                        ParetoPoint(
                            compression_ratio=compression_target,
                            error_rate=best_error,
                            settings=best_settings,
                            metrics=best_metrics,
                        )
                    )

            # Filter to Pareto optimal points
            pareto_points = self._filter_pareto_optimal(points)
            frontiers[use_case.key] = pareto_points

        self.pareto_frontiers = frontiers
        return frontiers

    def validate_compliance(
        self, use_case: UseCaseCategory, test_dataset: CalibrationDataset
    ) -> dict[str, Any]:
        """Validate model meets compliance requirements."""
        if use_case.key not in self.calibration_results:
            raise ValueError(f"Model not calibrated for {use_case.description}")

        calibration = self.calibration_results[use_case.key]
        X_test, y_test = test_dataset.get_validation_set()

        # Re-evaluate on test set
        current_metrics = self._evaluate_configuration(
            X_test[:100], y_test[:100], X_test, y_test, use_case
        )

        # Compliance checks
        checks = {
            "error_budget_met": current_metrics.achieved_error <= use_case.error_budget,
            "performance_stable": abs(current_metrics.achieved_error - calibration.achieved_error)
            < 0.01,
            "confidence_interval_valid": calibration.confidence_interval[0]
            <= current_metrics.achieved_error
            <= calibration.confidence_interval[1],
            "minimum_auc_met": current_metrics.auc_score >= 0.85,
            "sample_size_adequate": calibration.sample_size >= 1000,
        }

        compliance_score = sum(checks.values()) / len(checks)

        return {
            "use_case": use_case.key,
            "compliant": all(checks.values()),
            "compliance_score": compliance_score,
            "checks": checks,
            "current_metrics": current_metrics,
            "calibration_metrics": calibration,
            "recommendations": self._generate_compliance_recommendations(checks, current_metrics),
        }

    def generate_calibration_report(self) -> dict[str, Any]:
        """Generate comprehensive calibration report."""
        report = {
            "calibration_summary": {},
            "pareto_analysis": {},
            "compliance_matrix": {},
            "recommendations": [],
        }

        # Calibration summary
        for use_case_key, metrics in self.calibration_results.items():
            report["calibration_summary"][use_case_key] = {
                "achieved_error": metrics.achieved_error,
                "compression_ratio": metrics.compression_ratio,
                "auc_score": metrics.auc_score,
                "confidence_interval": metrics.confidence_interval,
                "meets_requirements": metrics.achieved_error <= metrics.use_case.error_budget,
            }

        # Pareto analysis
        for use_case_key, frontier in self.pareto_frontiers.items():
            if frontier:
                report["pareto_analysis"][use_case_key] = {
                    "optimal_points": len(frontier),
                    "compression_range": (
                        min(p.compression_ratio for p in frontier),
                        max(p.compression_ratio for p in frontier),
                    ),
                    "error_range": (
                        min(p.error_rate for p in frontier),
                        max(p.error_rate for p in frontier),
                    ),
                }

        # Compliance matrix
        use_cases = [uc for uc in UseCaseCategory]
        compliance_matrix = {}

        for uc1 in use_cases:
            compliance_matrix[uc1.key] = {}
            for uc2 in use_cases:
                # Check if calibration for uc1 meets requirements for uc2
                if uc1.key in self.calibration_results:
                    metrics = self.calibration_results[uc1.key]
                    meets = metrics.achieved_error <= uc2.error_budget
                    compliance_matrix[uc1.key][uc2.key] = meets

        report["compliance_matrix"] = compliance_matrix

        # Generate recommendations
        report["recommendations"] = self._generate_global_recommendations()

        # Save report
        report_path = self.output_dir / "calibration_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report

    def _configure_model(
        self, resolution: int, noise_scale: float, n_layers: int, n_knots: int
    ) -> None:
        """Configure model with given parameters."""
        # This would configure the actual HybridKANHD model
        # Placeholder implementation
        pass

    def _evaluate_configuration(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        use_case: UseCaseCategory,
    ) -> CalibrationMetrics:
        """Evaluate model configuration."""
        # Encode data
        encoded_results = []
        for x in X_test:
            result = self.model.encode_genomic_data(
                x,
                resolution=15000,
                use_case=use_case.key,  # From configuration
            )
            encoded_results.append(result)

        # Compute predictions (placeholder - would use actual predictor)
        predictions = np.random.rand(len(y_test))  # Replace with actual predictions

        # Compute metrics
        auc = roc_auc_score(y_test, predictions)
        precision, recall, thresholds = precision_recall_curve(y_test, predictions)

        # Find optimal threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        # Binary predictions at threshold
        y_pred = (predictions >= optimal_threshold).astype(int)

        # Error rate
        error_rate = np.mean(y_pred != y_test)

        # Compression ratio (from encoded results)
        compression_ratio = np.mean(
            [r["compression_metrics"].compression_ratio for r in encoded_results]
        )

        return CalibrationMetrics(
            use_case=use_case,
            achieved_error=error_rate,
            compression_ratio=compression_ratio,
            sensitivity=recall[optimal_idx],
            specificity=1 - (np.sum((y_test == 0) & (y_pred == 1)) / np.sum(y_test == 0)),
            auc_score=auc,
            precision_at_threshold=precision[optimal_idx],
            recall_at_threshold=recall[optimal_idx],
            optimal_threshold=optimal_threshold,
            confidence_interval=(0.0, 0.0),  # Will be set by bootstrap
            sample_size=len(y_test),
            calibration_timestamp=time.time(),
        )

    def _bootstrap_confidence_interval(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metrics: CalibrationMetrics,
        confidence_level: float,
        n_bootstrap: int = 1000,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval."""
        errors = []
        n_samples = len(y)

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Evaluate on bootstrap sample
            boot_metrics = self._evaluate_configuration(
                X_boot[:100], y_boot[:100], X_boot, y_boot, metrics.use_case
            )
            errors.append(boot_metrics.achieved_error)

        # Compute percentiles
        alpha = 1 - confidence_level
        lower = np.percentile(errors, 100 * alpha / 2)
        upper = np.percentile(errors, 100 * (1 - alpha / 2))

        return lower, upper

    def _sample_random_settings(self, target_compression: float) -> dict[str, Any]:
        """Sample random model settings targeting compression ratio."""
        # Heuristic: higher resolution → lower compression
        # More layers/knots → better accuracy but lower compression

        if target_compression > 100:
            resolution = 10000
            n_layers = 3
            n_knots = 8
        elif target_compression > 50:
            resolution = 15000
            n_layers = 4
            n_knots = 10
        else:
            resolution = 20000
            n_layers = 5
            n_knots = 16

        # Add some randomness
        resolution = int(resolution * np.random.uniform(0.8, 1.2))
        noise_scale = np.random.uniform(0.001, 0.1)

        return {
            "resolution": resolution,
            "noise_scale": noise_scale,
            "n_layers": n_layers,
            "n_knots": n_knots,
        }

    def _filter_pareto_optimal(self, points: list[ParetoPoint]) -> list[ParetoPoint]:
        """Filter points to Pareto optimal set."""
        pareto_points = []

        for candidate in points:
            is_dominated = False

            for other in points:
                if other == candidate:
                    continue

                # Check if other dominates candidate
                if (
                    other.compression_ratio >= candidate.compression_ratio
                    and other.error_rate <= candidate.error_rate
                    and (
                        other.compression_ratio > candidate.compression_ratio
                        or other.error_rate < candidate.error_rate
                    )
                ):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_points.append(candidate)

        return sorted(pareto_points, key=lambda p: p.compression_ratio)

    def _save_calibration_results(
        self, use_case: UseCaseCategory, metrics: CalibrationMetrics
    ) -> None:
        """Save calibration results to disk."""
        results_file = self.output_dir / f"calibration_{use_case.key}.json"

        data = {
            "use_case": use_case.key,
            "error_budget": use_case.error_budget,
            "achieved_error": metrics.achieved_error,
            "compression_ratio": metrics.compression_ratio,
            "metrics": {
                "sensitivity": metrics.sensitivity,
                "specificity": metrics.specificity,
                "auc_score": metrics.auc_score,
                "precision": metrics.precision_at_threshold,
                "recall": metrics.recall_at_threshold,
                "optimal_threshold": metrics.optimal_threshold,
            },
            "confidence_interval": metrics.confidence_interval,
            "sample_size": metrics.sample_size,
            "timestamp": metrics.calibration_timestamp,
        }

        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)

    def _generate_compliance_recommendations(
        self, checks: dict[str, bool], metrics: CalibrationMetrics
    ) -> list[str]:
        """Generate recommendations based on compliance checks."""
        recommendations = []

        if not checks["error_budget_met"]:
            recommendations.append(
                f"Error rate ({metrics.achieved_error:.3f}) exceeds budget. "
                "Consider: reducing compression, increasing model capacity, or "
                "improving feature engineering."
            )

        if not checks["performance_stable"]:
            recommendations.append(
                "Performance has drifted from calibration. "
                "Recommend: recalibration with current data distribution."
            )

        if not checks["minimum_auc_met"]:
            recommendations.append(
                f"AUC score ({metrics.auc_score:.3f}) below minimum threshold. "
                "Consider: model architecture improvements or additional training data."
            )

        if not checks["sample_size_adequate"]:
            recommendations.append(
                "Calibration sample size insufficient for statistical validity. "
                "Increase to at least 1000 samples."
            )

        return recommendations

    def _generate_global_recommendations(self) -> list[str]:
        """Generate overall recommendations from all calibrations."""
        recommendations = []

        # Check coverage
        calibrated_use_cases = set(self.calibration_results.keys())
        all_use_cases = {uc.key for uc in UseCaseCategory}
        missing = all_use_cases - calibrated_use_cases

        if missing:
            recommendations.append(
                f"Missing calibrations for: {', '.join(missing)}. "
                "Complete calibration for all use cases before deployment."
            )

        # Check Pareto optimality
        if self.pareto_frontiers:
            max_compression = max(
                max(p.compression_ratio for p in frontier)
                for frontier in self.pareto_frontiers.values()
                if frontier
            )
            if max_compression < 100:
                recommendations.append(
                    "Maximum compression below 100x. Consider exploring "
                    "more aggressive compression strategies."
                )

        # Cross-use-case compatibility
        if len(self.calibration_results) >= 2:
            errors = [m.achieved_error for m in self.calibration_results.values()]
            if max(errors) / min(errors) > 10:
                recommendations.append(
                    "Large variation in error rates across use cases. "
                    "Consider separate model variants for different applications."
                )

        return recommendations
