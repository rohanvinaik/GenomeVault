"""
KAN-HD Calibration Suite for clinical error budget management.
Implements compression vs accuracy trade-off analysis as per audit recommendations.
"""
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

from ....core.config import get_config
from ....utils.logging import get_logger
from ..enhanced_hybrid_encoder import EnhancedHybridEncoder

logger = get_logger(__name__)
config = get_config()


@dataclass
class CalibrationMetrics:
    """Metrics for calibration assessment."""
    """Metrics for calibration assessment."""
    """Metrics for calibration assessment."""

    compression_ratio: float
    reconstruction_mse: float
    allele_frequency_error: float
    odds_ratio_error: float
    p_value_drift: float
    clinical_concordance: float
    processing_time_ms: float
    memory_usage_mb: float

    def to_dict(self) -> Dict:
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        """Convert to dictionary."""
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CalibrationPoint:
    """Single calibration measurement."""
    """Single calibration measurement."""
    """Single calibration measurement."""

    config_id: str
    compression_target: float
    hd_dimension: int
    kan_spline_degree: int
    epsilon: float
    delta: float
    ecc_threshold: float
    metrics: CalibrationMetrics
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict:
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        """Convert to dictionary."""
        """Convert to dictionary."""
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data


@dataclass
class CalibrationCurve:
    """Calibration curve with Pareto frontier."""
    """Calibration curve with Pareto frontier."""
    """Calibration curve with Pareto frontier."""

    points: List[CalibrationPoint]
    pareto_indices: List[int]
    interpolation_functions: Dict[str, Any]
    metadata: Dict[str, Any]

    def get_pareto_points(self) -> List[CalibrationPoint]:
    def get_pareto_points(self) -> List[CalibrationPoint]:
        """Get points on Pareto frontier."""
        """Get points on Pareto frontier."""
        """Get points on Pareto frontier."""
        return [self.points[i] for i in self.pareto_indices]


class ClinicalErrorBudget:
    """Defines acceptable error budgets for different clinical use cases."""
    """Defines acceptable error budgets for different clinical use cases."""
    """Defines acceptable error budgets for different clinical use cases."""

    BUDGETS = {
        "screening": {
            "allele_frequency_error": 0.05,  # 5%
            "odds_ratio_error": 0.10,  # 10%
            "p_value_drift": 0.05,  # 5%
            "clinical_concordance": 0.95,  # 95%
        },
        "diagnostic": {
            "allele_frequency_error": 0.01,  # 1%
            "odds_ratio_error": 0.05,  # 5%
            "p_value_drift": 0.01,  # 1%
            "clinical_concordance": 0.99,  # 99%
        },
        "research": {
            "allele_frequency_error": 0.02,  # 2%
            "odds_ratio_error": 0.05,  # 5%
            "p_value_drift": 0.02,  # 2%
            "clinical_concordance": 0.98,  # 98%
        },
        "regulatory": {
            "allele_frequency_error": 0.005,  # 0.5%
            "odds_ratio_error": 0.02,  # 2%
            "p_value_drift": 0.005,  # 0.5%
            "clinical_concordance": 0.995,  # 99.5%
        },
    }

    @classmethod
    def check_compliance(
        cls, metrics: CalibrationMetrics, use_case: str
    ) -> Tuple[bool, Dict[str, bool]]:
        """
        """
        """
        Check if metrics meet budget requirements.

        Args:
            metrics: Calibration metrics
            use_case: Clinical use case

        Returns:
            Tuple of (overall_compliance, detailed_checks)
        """
        if use_case not in cls.BUDGETS:
            raise ValueError(f"Unknown use case: {use_case}")

        budget = cls.BUDGETS[use_case]
        checks = {
            "allele_frequency": metrics.allele_frequency_error <= budget["allele_frequency_error"],
            "odds_ratio": metrics.odds_ratio_error <= budget["odds_ratio_error"],
            "p_value": metrics.p_value_drift <= budget["p_value_drift"],
            "concordance": metrics.clinical_concordance >= budget["clinical_concordance"],
        }

        overall = all(checks.values())
        return overall, checks


class KANHDCalibrationSuite:
    """
    """
    """
    Comprehensive calibration suite for KAN-HD compression.
    Measures trade-offs between compression and clinical accuracy.
    """

    def __init__(self, output_dir: Path, reference_data_path: Optional[Path] = None):
    def __init__(self, output_dir: Path, reference_data_path: Optional[Path] = None):
        """
        """
    """
        Initialize calibration suite.

        Args:
            output_dir: Directory for calibration outputs
            reference_data_path: Path to reference genomic data
        """
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            self.reference_data_path = reference_data_path
            self.calibration_id = f"calib_{int(time.time())}"

        # Load or generate reference data
            self.reference_data = self._load_reference_data()

        logger.info(f"Initialized calibration suite {self.calibration_id}")

            def _load_reference_data(self) -> Dict[str, Any]:
            def _load_reference_data(self) -> Dict[str, Any]:
                """Load reference genomic data for calibration."""
        """Load reference genomic data for calibration."""
        """Load reference genomic data for calibration."""
        if self.reference_data_path and self.reference_data_path.exists():
            # Load actual reference data
            with open(self.reference_data_path, "r") as f:
                return json.load(f)
        else:
            # Generate synthetic reference data
            return self._generate_synthetic_reference()

            def _generate_synthetic_reference(self) -> Dict[str, Any]:
            def _generate_synthetic_reference(self) -> Dict[str, Any]:
                """Generate synthetic genomic data for testing."""
        """Generate synthetic genomic data for testing."""
        """Generate synthetic genomic data for testing."""
        np.random.seed(42)

        # Generate variants
        n_variants = 10000
        variants = []

        for i in range(n_variants):
            variants.append(
                {
                    "id": f"rs{i}",
                    "chromosome": f"chr{np.random.randint(1, 23)}",
                    "position": np.random.randint(1, 250000000),
                    "ref": np.random.choice(["A", "T", "G", "C"]),
                    "alt": np.random.choice(["A", "T", "G", "C"]),
                    "af": np.random.beta(2, 50),  # Allele frequency
                    "or": np.random.lognormal(0, 0.3),  # Odds ratio
                    "p_value": 10 ** (-np.random.exponential(2)),  # P-value
                    "clinical_significance": np.random.choice(
                        ["benign", "likely_benign", "vus", "likely_pathogenic", "pathogenic"]
                    ),
                }
            )

        return {
            "variants": variants,
            "metadata": {
                "source": "synthetic",
                "n_variants": n_variants,
                "created_at": datetime.utcnow().isoformat(),
            },
        }

            def run_calibration(
        self,
        compression_targets: List[float] = [10, 20, 50, 100, 200],
        hd_dimensions: List[int] = [10000, 50000, 100000],
        kan_degrees: List[int] = [3, 5, 7],
        n_repeats: int = 3,
    ) -> CalibrationCurve:
        """
        """
        """
        Run comprehensive calibration across parameter space.

        Args:
            compression_targets: Target compression ratios
            hd_dimensions: HD vector dimensions
            kan_degrees: KAN spline degrees
            n_repeats: Number of repeats per configuration

        Returns:
            Calibration curve with Pareto frontier
        """
        logger.info("Starting calibration sweep")

        calibration_points = []

        # Test all combinations
        for compression in compression_targets:
            for dimension in hd_dimensions:
                for degree in kan_degrees:
                    # Run multiple repeats
                    metrics_list = []

                    for repeat in range(n_repeats):
                        config_id = f"c{compression}_d{dimension}_k{degree}_r{repeat}"

                        try:
                            # Run single calibration
                            metrics = self._run_single_calibration(
                                compression_target=compression,
                                hd_dimension=dimension,
                                kan_spline_degree=degree,
                            )
                            metrics_list.append(metrics)

                        except Exception as e:
                            logger.error(f"Calibration failed for {config_id}: {e}")
                            continue

                    if metrics_list:
                        # Average metrics across repeats
                        avg_metrics = self._average_metrics(metrics_list)

                        # Create calibration point
                        point = CalibrationPoint(
                            config_id=f"c{compression}_d{dimension}_k{degree}",
                            compression_target=compression,
                            hd_dimension=dimension,
                            kan_spline_degree=degree,
                            epsilon=0.1,  # Privacy parameter
                            delta=1e-5,  # Privacy parameter
                            ecc_threshold=0.001,  # Error correction
                            metrics=avg_metrics,
                        )

                        calibration_points.append(point)

                        # Log progress
                        logger.info(
                            f"Calibration point {point.config_id}: "
                            f"compression={avg_metrics.compression_ratio:.1f}x, "
                            f"AF_error={avg_metrics.allele_frequency_error:.3f}"
                        )

        # Compute Pareto frontier
        pareto_indices = self._compute_pareto_frontier(calibration_points)

        # Fit interpolation functions
        interpolation_functions = self._fit_interpolation(calibration_points)

        # Create calibration curve
        curve = CalibrationCurve(
            points=calibration_points,
            pareto_indices=pareto_indices,
            interpolation_functions=interpolation_functions,
            metadata={
                "calibration_id": self.calibration_id,
                "n_points": len(calibration_points),
                "n_pareto": len(pareto_indices),
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        # Save results
                        self._save_calibration_results(curve)

        # Generate plots
                        self._generate_calibration_plots(curve)

        return curve

                        def _run_single_calibration(
        self, compression_target: float, hd_dimension: int, kan_spline_degree: int
    ) -> CalibrationMetrics:
        """Run calibration for single configuration."""
        """Run calibration for single configuration."""
        """Run calibration for single configuration."""
        start_time = time.time()

        # Initialize encoder
        encoder = EnhancedHybridEncoder(
            hd_dimension=hd_dimension,
            kan_spline_degree=kan_spline_degree,
            compression_target=compression_target,
        )

        # Encode reference data
        encoded_data = []
        original_data = []

        for variant in self.reference_data["variants"][:1000]:  # Subset for speed
            # Create feature vector
            features = np.array(
                [
                    hash(variant["chromosome"]) % 23,
                    variant["position"] / 1e8,
                    hash(variant["ref"]) % 4,
                    hash(variant["alt"]) % 4,
                    variant["af"],
                    np.log(variant["or"] + 1),
                    -np.log10(variant["p_value"] + 1e-300),
                ]
            )

            original_data.append(features)

            # Encode
            encoded = encoder.encode(features)
            encoded_data.append(encoded)

        # Measure compression
        original_size = np.array(original_data).nbytes
        encoded_size = np.array(encoded_data).nbytes
        compression_ratio = original_size / encoded_size

        # Decode and measure accuracy
        decoded_data = []
        for encoded in encoded_data:
            decoded = encoder.decode(encoded)
            decoded_data.append(decoded)

        # Calculate metrics
        original_array = np.array(original_data)
        decoded_array = np.array(decoded_data)

        # Reconstruction MSE
        reconstruction_mse = mean_squared_error(original_array, decoded_array)

        # Allele frequency error
        af_original = original_array[:, 4]
        af_decoded = decoded_array[:, 4]
        af_error = np.mean(np.abs(af_original - af_decoded))

        # Odds ratio error
        or_original = np.exp(original_array[:, 5]) - 1
        or_decoded = np.exp(decoded_array[:, 5]) - 1
        or_error = np.mean(np.abs(np.log(or_original + 1) - np.log(or_decoded + 1)))

        # P-value drift
        pval_original = 10 ** (-original_array[:, 6])
        pval_decoded = 10 ** (-decoded_array[:, 6])
        p_value_drift = np.mean(
            np.abs(np.log10(pval_original + 1e-300) - np.log10(pval_decoded + 1e-300))
        )

        # Clinical concordance (simplified)
        clinical_threshold = 0.05
        concordant = np.sum(
            (pval_original < clinical_threshold) == (pval_decoded < clinical_threshold)
        )
        clinical_concordance = concordant / len(pval_original)

        # Memory usage (approximate)
        import psutil

        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / (1024 * 1024)

        # Processing time
        processing_time_ms = (time.time() - start_time) * 1000

        return CalibrationMetrics(
            compression_ratio=compression_ratio,
            reconstruction_mse=reconstruction_mse,
            allele_frequency_error=af_error,
            odds_ratio_error=or_error,
            p_value_drift=p_value_drift,
            clinical_concordance=clinical_concordance,
            processing_time_ms=processing_time_ms,
            memory_usage_mb=memory_usage_mb,
        )

            def _average_metrics(self, metrics_list: List[CalibrationMetrics]) -> CalibrationMetrics:
            def _average_metrics(self, metrics_list: List[CalibrationMetrics]) -> CalibrationMetrics:
                """Average metrics across multiple runs."""
        """Average metrics across multiple runs."""
        """Average metrics across multiple runs."""
        if not metrics_list:
            raise ValueError("No metrics to average")

        # Convert to arrays
        arrays = {
            field: np.array([getattr(m, field) for m in metrics_list])
            for field in CalibrationMetrics.__annotations__
        }

        # Average each field
        averaged = {field: np.mean(values) for field, values in arrays.items()}

        return CalibrationMetrics(**averaged)

            def _compute_pareto_frontier(self, points: List[CalibrationPoint]) -> List[int]:
            def _compute_pareto_frontier(self, points: List[CalibrationPoint]) -> List[int]:
                """
        """
        """
        Compute Pareto frontier for compression vs accuracy trade-off.

        Args:
            points: Calibration points

        Returns:
            Indices of points on Pareto frontier
        """
        # Extract objectives (maximize compression, minimize error)
        objectives = np.array(
            [[p.metrics.compression_ratio, -p.metrics.allele_frequency_error] for p in points]
        )

        pareto_indices = []

        for i, point in enumerate(objectives):
            # Check if dominated by any other point
            dominated = False
            for j, other in enumerate(objectives):
                if i != j:
                    # Check if other dominates point in all objectives
                    if all(other >= point) and any(other > point):
                        dominated = True
                        break

            if not dominated:
                pareto_indices.append(i)

        return pareto_indices

                def _fit_interpolation(self, points: List[CalibrationPoint]) -> Dict[str, Any]:
                def _fit_interpolation(self, points: List[CalibrationPoint]) -> Dict[str, Any]:
                    """Fit interpolation functions for calibration curves."""
        """Fit interpolation functions for calibration curves."""
        """Fit interpolation functions for calibration curves."""
        from scipy.interpolate import interp1d

        # Sort by compression ratio
        sorted_points = sorted(points, key=lambda p: p.metrics.compression_ratio)

        compression_ratios = np.array([p.metrics.compression_ratio for p in sorted_points])

        # Fit interpolation for each metric
        interpolations = {}

        for metric_name in [
            "allele_frequency_error",
            "odds_ratio_error",
            "p_value_drift",
            "clinical_concordance",
        ]:
            values = np.array([getattr(p.metrics, metric_name) for p in sorted_points])

            # Fit cubic interpolation
            interpolations[metric_name] = interp1d(
                compression_ratios,
                values,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )

        return interpolations

            def _save_calibration_results(self, curve: CalibrationCurve) -> None:
            def _save_calibration_results(self, curve: CalibrationCurve) -> None:
                """Save calibration results to disk."""
        """Save calibration results to disk."""
        """Save calibration results to disk."""
        # Save raw data
        data = {
            "calibration_id": self.calibration_id,
            "points": [p.to_dict() for p in curve.points],
            "pareto_indices": curve.pareto_indices,
            "metadata": curve.metadata,
        }

        output_path = self.output_dir / f"calibration_{self.calibration_id}.json"
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        # Save Pareto frontier
        pareto_data = {"points": [curve.points[i].to_dict() for i in curve.pareto_indices]}

        pareto_path = self.output_dir / f"pareto_{self.calibration_id}.json"
        with open(pareto_path, "w") as f:
            json.dump(pareto_data, f, indent=2)

        logger.info(f"Saved calibration results to {output_path}")

            def _generate_calibration_plots(self, curve: CalibrationCurve) -> None:
            def _generate_calibration_plots(self, curve: CalibrationCurve) -> None:
                """Generate calibration plots."""
        """Generate calibration plots."""
        """Generate calibration plots."""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"KAN-HD Calibration Results - {self.calibration_id}")

        # Extract data
        compression_ratios = [p.metrics.compression_ratio for p in curve.points]
        af_errors = [p.metrics.allele_frequency_error for p in curve.points]
        or_errors = [p.metrics.odds_ratio_error for p in curve.points]
        p_drifts = [p.metrics.p_value_drift for p in curve.points]
        concordances = [p.metrics.clinical_concordance for p in curve.points]

        # Plot 1: Compression vs AF Error
        ax1 = axes[0, 0]
        ax1.scatter(compression_ratios, af_errors, alpha=0.6, label="All points")

        # Highlight Pareto frontier
        pareto_compression = [compression_ratios[i] for i in curve.pareto_indices]
        pareto_af = [af_errors[i] for i in curve.pareto_indices]
        ax1.scatter(pareto_compression, pareto_af, color="red", s=100, label="Pareto frontier")
        ax1.plot(pareto_compression, pareto_af, "r--", alpha=0.5)

        ax1.set_xlabel("Compression Ratio")
        ax1.set_ylabel("Allele Frequency Error")
        ax1.set_title("Compression vs AF Error")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Compression vs OR Error
        ax2 = axes[0, 1]
        ax2.scatter(compression_ratios, or_errors, alpha=0.6)
        pareto_or = [or_errors[i] for i in curve.pareto_indices]
        ax2.scatter(pareto_compression, pareto_or, color="red", s=100)
        ax2.set_xlabel("Compression Ratio")
        ax2.set_ylabel("Odds Ratio Error")
        ax2.set_title("Compression vs OR Error")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Compression vs P-value Drift
        ax3 = axes[1, 0]
        ax3.scatter(compression_ratios, p_drifts, alpha=0.6)
        pareto_p = [p_drifts[i] for i in curve.pareto_indices]
        ax3.scatter(pareto_compression, pareto_p, color="red", s=100)
        ax3.set_xlabel("Compression Ratio")
        ax3.set_ylabel("P-value Drift")
        ax3.set_title("Compression vs P-value Drift")
        ax3.grid(True, alpha=0.3)

        # Plot 4: Compression vs Clinical Concordance
        ax4 = axes[1, 1]
        ax4.scatter(compression_ratios, concordances, alpha=0.6)
        pareto_concordance = [concordances[i] for i in curve.pareto_indices]
        ax4.scatter(pareto_compression, pareto_concordance, color="red", s=100)
        ax4.set_xlabel("Compression Ratio")
        ax4.set_ylabel("Clinical Concordance")
        ax4.set_title("Compression vs Clinical Concordance")
        ax4.grid(True, alpha=0.3)

        # Add error budget lines
        for use_case, budget in ClinicalErrorBudget.BUDGETS.items():
            ax1.axhline(
                y=budget["allele_frequency_error"],
                color="gray",
                linestyle="--",
                alpha=0.5,
                label=f"{use_case} limit",
            )
            ax2.axhline(y=budget["odds_ratio_error"], color="gray", linestyle="--", alpha=0.5)
            ax3.axhline(y=budget["p_value_drift"], color="gray", linestyle="--", alpha=0.5)
            ax4.axhline(y=budget["clinical_concordance"], color="gray", linestyle="--", alpha=0.5)

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / f"calibration_plot_{self.calibration_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved calibration plot to {plot_path}")

        # Generate Pareto curve plot
            self._generate_pareto_plot(curve)

            def _generate_pareto_plot(self, curve: CalibrationCurve) -> None:
            def _generate_pareto_plot(self, curve: CalibrationCurve) -> None:
                """Generate focused Pareto frontier plot."""
        """Generate focused Pareto frontier plot."""
        """Generate focused Pareto frontier plot."""
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get Pareto points
        pareto_points = curve.get_pareto_points()

        # Sort by compression
        pareto_points.sort(key=lambda p: p.metrics.compression_ratio)

        compression = [p.metrics.compression_ratio for p in pareto_points]
        af_error = [p.metrics.allele_frequency_error for p in pareto_points]

        # Plot Pareto frontier
        ax.plot(compression, af_error, "ro-", markersize=10, linewidth=2, label="Pareto Frontier")

        # Annotate points
        for point in pareto_points:
            ax.annotate(
                f"HD={point.hd_dimension//1000}k\nKAN={point.kan_spline_degree}",
                xy=(point.metrics.compression_ratio, point.metrics.allele_frequency_error),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                alpha=0.7,
            )

        # Add error budget regions
        ax.axhspan(
            0,
            ClinicalErrorBudget.BUDGETS["regulatory"]["allele_frequency_error"],
            alpha=0.2,
            color="green",
            label="Regulatory",
        )
        ax.axhspan(
            ClinicalErrorBudget.BUDGETS["regulatory"]["allele_frequency_error"],
            ClinicalErrorBudget.BUDGETS["diagnostic"]["allele_frequency_error"],
            alpha=0.2,
            color="yellow",
            label="Diagnostic",
        )
        ax.axhspan(
            ClinicalErrorBudget.BUDGETS["diagnostic"]["allele_frequency_error"],
            ClinicalErrorBudget.BUDGETS["research"]["allele_frequency_error"],
            alpha=0.2,
            color="orange",
            label="Research",
        )
        ax.axhspan(
            ClinicalErrorBudget.BUDGETS["research"]["allele_frequency_error"],
            ClinicalErrorBudget.BUDGETS["screening"]["allele_frequency_error"],
            alpha=0.2,
            color="red",
            label="Screening",
        )

        ax.set_xlabel("Compression Ratio", fontsize=12)
        ax.set_ylabel("Allele Frequency Error", fontsize=12)
        ax.set_title("KAN-HD Pareto Frontier: Compression vs Clinical Accuracy", fontsize=14)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # Set log scale if needed
        if max(compression) / min(compression) > 10:
            ax.set_xscale("log")

        plt.tight_layout()

        # Save plot
        plot_path = self.output_dir / f"pareto_plot_{self.calibration_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved Pareto plot to {plot_path}")

            def recommend_configuration(
        self, curve: CalibrationCurve, use_case: str, prefer_compression: float = 0.5
    ) -> CalibrationPoint:
        """
        """
        """
        Recommend configuration for specific use case.

        Args:
            curve: Calibration curve
            use_case: Clinical use case
            prefer_compression: Weight for compression (0=accuracy, 1=compression)

        Returns:
            Recommended calibration point
        """
        # Get Pareto points
        pareto_points = curve.get_pareto_points()

        # Filter by error budget compliance
        compliant_points = []
        for point in pareto_points:
            compliant, _ = ClinicalErrorBudget.check_compliance(point.metrics, use_case)
            if compliant:
                compliant_points.append(point)

        if not compliant_points:
            logger.warning(f"No configurations meet {use_case} requirements")
            # Return best accuracy point
            return min(pareto_points, key=lambda p: p.metrics.allele_frequency_error)

        # Score points by weighted objective
        best_score = float("-inf")
        best_point = None

        for point in compliant_points:
            # Normalize metrics
            compression_score = point.metrics.compression_ratio / 200  # Normalize to ~[0, 1]
            accuracy_score = 1 - point.metrics.allele_frequency_error / 0.1  # Normalize to ~[0, 1]

            # Weighted score
            score = (
                prefer_compression * compression_score + (1 - prefer_compression) * accuracy_score
            )

            if score > best_score:
                best_score = score
                best_point = point

        logger.info(
            f"Recommended config for {use_case}: {best_point.config_id} "
            f"(compression={best_point.metrics.compression_ratio:.1f}x, "
            f"AF_error={best_point.metrics.allele_frequency_error:.3f})"
        )

        return best_point


# Example usage
                def run_calibration_example():
                def run_calibration_example():
                    """Example calibration run."""
    """Example calibration run."""
    """Example calibration run."""
    # Initialize suite
    suite = KANHDCalibrationSuite(output_dir=Path("/tmp/genomevault_calibration"))

    # Run calibration
    curve = suite.run_calibration(
        compression_targets=[10, 25, 50, 100],
        hd_dimensions=[10000, 50000],
        kan_degrees=[3, 5],
        n_repeats=2,
    )

    # Get recommendations for different use cases
    for use_case in ["screening", "diagnostic", "research", "regulatory"]:
        recommended = suite.recommend_configuration(curve, use_case)
        print(f"\n{use_case.upper()} recommendation:")
        print(f"  Config: {recommended.config_id}")
        print(f"  Compression: {recommended.metrics.compression_ratio:.1f}x")
        print(f"  AF Error: {recommended.metrics.allele_frequency_error:.3f}")
        print(f"  Clinical Concordance: {recommended.metrics.clinical_concordance:.3f}")


if __name__ == "__main__":
    run_calibration_example()
