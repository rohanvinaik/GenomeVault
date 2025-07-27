"""
Scientific Interpretability Module

Implements scientific discovery capabilities from KAN-HD insights,
extracting human-understandable patterns and formulas from learned functions.
"""
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    import sympy as sp
    from scipy import optimize, stats
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .kan_layer import KANLayer, LinearKAN


class BiologicalFunction(Enum):
    """Types of biological functions that can be discovered"""
    """Types of biological functions that can be discovered"""
    """Types of biological functions that can be discovered"""

    EXPONENTIAL_DECAY = "exponential_decay"  # Drug metabolism, radioactive decay
    SIGMOIDAL = "sigmoidal"  # Dose-response curves
    POWER_LAW = "power_law"  # Allometric scaling
    PERIODIC = "periodic"  # Circadian rhythms
    LOGARITHMIC = "logarithmic"  # pH, decibel scales
    LINEAR = "linear"  # Simple proportional relationships
    THRESHOLD = "threshold"  # Activation thresholds
    OSCILLATORY = "oscillatory"  # Gene expression oscillations


@dataclass
class DiscoveredFunction:
    """Represents a discovered biological function"""
    """Represents a discovered biological function"""
    """Represents a discovered biological function"""

    function_type: BiologicalFunction
    symbolic_expression: str
    parameters: Dict[str, float]
    confidence: float
    biological_interpretation: str
    input_variable: str
    output_variable: str
    r_squared: float


@dataclass
class PatternAnalysis:
    """Analysis of patterns in KAN functions"""
    """Analysis of patterns in KAN functions"""
    """Analysis of patterns in KAN functions"""

    monotonicity: str  # 'increasing', 'decreasing', 'mixed'
    concavity: str  # 'concave_up', 'concave_down', 'mixed'
    critical_points: List[float]
    inflection_points: List[float]
    asymptotes: List[float]
    dominant_frequency: Optional[float] = None


class KANFunctionAnalyzer(nn.Module):
    """
    """
    """
    Analyzer for extracting interpretable patterns from KAN layer functions

    Implements the insight that KANs can "easily interact with human users"
    and help rediscover known formulas in science.
    """

    def __init__(self) -> None:
        """TODO: Add docstring for __init__"""
        """TODO: Add docstring for __init__"""
            """TODO: Add docstring for __init__"""
    super().__init__()

        # Template functions for pattern matching
        self.function_templates = self._initialize_function_templates()

        # Biological interpretations
        self.biological_meanings = {
            BiologicalFunction.EXPONENTIAL_DECAY: "Represents decay processes like drug clearance, protein degradation, or radioactive decay",
            BiologicalFunction.SIGMOIDAL: "Represents dose-response relationships, enzyme kinetics, or logistic growth",
            BiologicalFunction.POWER_LAW: "Represents scaling relationships like metabolic rate vs body mass",
            BiologicalFunction.PERIODIC: "Represents circadian rhythms, seasonal patterns, or cell cycle dynamics",
            BiologicalFunction.LOGARITHMIC: "Represents pH relationships, sensory perception (Weber-Fechner law), or information content",
            BiologicalFunction.LINEAR: "Represents simple proportional relationships between biological variables",
            BiologicalFunction.THRESHOLD: "Represents activation thresholds in neurons, gene expression switches",
            BiologicalFunction.OSCILLATORY: "Represents damped oscillations in biological systems, population dynamics",
        }

        def _initialize_function_templates(self) -> Dict[BiologicalFunction, Callable]:
            """TODO: Add docstring for _initialize_function_templates"""
        """TODO: Add docstring for _initialize_function_templates"""
            """TODO: Add docstring for _initialize_function_templates"""
    """Initialize function templates for pattern matching"""
        if not SCIPY_AVAILABLE:
            # Return simplified templates that don't require scipy
            return {
                BiologicalFunction.LINEAR: lambda x, a, b: a * x + b,
                BiologicalFunction.EXPONENTIAL_DECAY: lambda x, a, b: a * np.exp(-b * x),
                BiologicalFunction.POWER_LAW: lambda x, a, b: a * np.power(np.abs(x) + 1e-8, b),
            }

        return {
            BiologicalFunction.EXPONENTIAL_DECAY: lambda x, a, b: a * np.exp(-b * x),
            BiologicalFunction.SIGMOIDAL: lambda x, a, b, c: a / (1 + np.exp(-b * (x - c))),
            BiologicalFunction.POWER_LAW: lambda x, a, b: a * np.power(np.abs(x) + 1e-8, b),
            BiologicalFunction.PERIODIC: lambda x, a, b, c, d: a * np.sin(b * x + c) + d,
            BiologicalFunction.LOGARITHMIC: lambda x, a, b: a * np.log(np.abs(b * x) + 1e-8),
            BiologicalFunction.LINEAR: lambda x, a, b: a * x + b,
            BiologicalFunction.THRESHOLD: lambda x, a, b, c: a * (x > b).astype(float) + c,
            BiologicalFunction.OSCILLATORY: lambda x, a, b, c, d, e: a
            * np.exp(-b * x)
            * np.sin(c * x + d)
            + e,
        }

            def analyze_kan_layer(
        self,
        kan_layer: Union[LinearKAN, KANLayer],
        input_range: Tuple[float, float] = (-2.0, 2.0),
        num_points: int = 1000,
    ) -> Dict[str, Any]:
        """
        """
        """
        Comprehensive analysis of a KAN layer

        Args:
            kan_layer: KAN layer to analyze
            input_range: Range of input values to analyze
            num_points: Number of points for function evaluation

        Returns:
            Dictionary containing all discovered patterns and functions
        """
        analysis_results = {
            "layer_type": type(kan_layer).__name__,
            "discovered_functions": {},
            "pattern_analyses": {},
            "biological_insights": [],
            "symbolic_expressions": [],
            "interpretability_score": 0.0,
        }

        # Generate input points
        x_values = torch.linspace(input_range[0], input_range[1], num_points)

        # Analyze each univariate function in the layer
        if isinstance(kan_layer, LinearKAN):
            functions_analyzed = self._analyze_linear_kan(kan_layer, x_values)
        elif isinstance(kan_layer, KANLayer):
            functions_analyzed = self._analyze_full_kan(kan_layer, x_values)
        else:
            raise ValueError(f"Unsupported layer type: {type(kan_layer)}")

        analysis_results.update(functions_analyzed)

        # Compute overall interpretability score
        analysis_results["interpretability_score"] = self._compute_interpretability_score(
            analysis_results
        )

        return analysis_results

            def _analyze_linear_kan(self, linear_kan: LinearKAN, x_values: torch.Tensor) -> Dict[str, Any]:
                """TODO: Add docstring for _analyze_linear_kan"""
        """TODO: Add docstring for _analyze_linear_kan"""
            """TODO: Add docstring for _analyze_linear_kan"""
    """Analyze Linear KAN layer functions"""
        results = {
            "discovered_functions": {},
            "pattern_analyses": {},
            "biological_insights": [],
            "symbolic_expressions": [],
        }

        # For LinearKAN, analyze the piecewise linear functions
        if hasattr(linear_kan, "values") and hasattr(linear_kan, "breakpoints"):
            num_functions = linear_kan.values.shape[0] * linear_kan.values.shape[1]

            for i in range(min(num_functions, 20)):  # Analyze up to 20 functions
                out_idx = i // linear_kan.values.shape[1]
                in_idx = i % linear_kan.values.shape[1]

                # Extract piecewise linear function
                breakpoints = linear_kan.breakpoints[out_idx, in_idx, :].detach().numpy()
                values = linear_kan.values[out_idx, in_idx, :].detach().numpy()

                # Evaluate function
                y_values = self._evaluate_piecewise_linear(x_values.numpy(), breakpoints, values)

                # Discover function type
                discovered = self._discover_function_type(x_values.numpy(), y_values)

                if discovered:
                    function_key = f"function_{out_idx}_{in_idx}"
                    results["discovered_functions"][function_key] = discovered

                    # Pattern analysis
                    pattern = self._analyze_function_pattern(x_values.numpy(), y_values)
                    results["pattern_analyses"][function_key] = pattern

                    # Generate biological insight
                    insight = self._generate_biological_insight(discovered, pattern)
                    if insight:
                        results["biological_insights"].append(insight)

                    # Generate symbolic expression
                    symbolic = self._generate_symbolic_expression(discovered)
                    if symbolic:
                        results["symbolic_expressions"].append(symbolic)

        return results

                        def _analyze_full_kan(self, kan_layer: KANLayer, x_values: torch.Tensor) -> Dict[str, Any]:
                            """TODO: Add docstring for _analyze_full_kan"""
        """TODO: Add docstring for _analyze_full_kan"""
            """TODO: Add docstring for _analyze_full_kan"""
    """Analyze full KAN layer with spline functions"""
        results = {
            "discovered_functions": {},
            "pattern_analyses": {},
            "biological_insights": [],
            "symbolic_expressions": [],
        }

        # Analyze spline functions
        if hasattr(kan_layer, "splines"):
            for i, spline_row in enumerate(kan_layer.splines[:5]):  # Analyze first 5 outputs
                for j, spline in enumerate(spline_row[:10]):  # Analyze first 10 inputs
                    # Evaluate spline function
                    with torch.no_grad():
                        y_values = spline(x_values).numpy()

                    # Discover function type
                    discovered = self._discover_function_type(x_values.numpy(), y_values)

                    if discovered:
                        function_key = f"spline_{i}_{j}"
                        results["discovered_functions"][function_key] = discovered

                        # Pattern analysis
                        pattern = self._analyze_function_pattern(x_values.numpy(), y_values)
                        results["pattern_analyses"][function_key] = pattern

                        # Generate insights
                        insight = self._generate_biological_insight(discovered, pattern)
                        if insight:
                            results["biological_insights"].append(insight)

                        symbolic = self._generate_symbolic_expression(discovered)
                        if symbolic:
                            results["symbolic_expressions"].append(symbolic)

        return results

                            def _evaluate_piecewise_linear(
        self, x: np.ndarray, breakpoints: np.ndarray, values: np.ndarray
    ) -> np.ndarray:
        """TODO: Add docstring for _evaluate_piecewise_linear"""
        """TODO: Add docstring for _evaluate_piecewise_linear"""
            """TODO: Add docstring for _evaluate_piecewise_linear"""
    """Evaluate piecewise linear function"""
        y = np.zeros_like(x)

        for i in range(len(x)):
            # Find which segment x[i] belongs to
            segment_idx = np.searchsorted(breakpoints[1:], x[i])
            segment_idx = min(segment_idx, len(values) - 2)

            # Linear interpolation within segment
            if segment_idx < len(breakpoints) - 1:
                x1, x2 = breakpoints[segment_idx], breakpoints[segment_idx + 1]
                y1, y2 = values[segment_idx], values[segment_idx + 1]

                if x2 != x1:
                    t = (x[i] - x1) / (x2 - x1)
                    y[i] = y1 * (1 - t) + y2 * t
                else:
                    y[i] = y1
            else:
                y[i] = values[-1]

        return y

                def _discover_function_type(self, x: np.ndarray, y: np.ndarray) -> Optional[DiscoveredFunction]:
                    """TODO: Add docstring for _discover_function_type"""
        """TODO: Add docstring for _discover_function_type"""
            """TODO: Add docstring for _discover_function_type"""
    """Discover the type of function by fitting templates"""
        best_fit = None
        best_r_squared = -np.inf

        # Clean data
        valid_mask = np.isfinite(x) & np.isfinite(y)
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]

        if len(x_clean) < 10:  # Need minimum points
            return None

        # Try fitting each template function
        for func_type, template in self.function_templates.items():
            try:
                fit_result = self._fit_function_template(x_clean, y_clean, template, func_type)

                if fit_result and fit_result.r_squared > best_r_squared:
                    best_fit = fit_result
                    best_r_squared = fit_result.r_squared

            except Exception:
                # Skip functions that can't be fitted
                continue

        # Only return if fit is reasonably good
        if best_fit and best_fit.r_squared > 0.7:
            return best_fit

        return None

            def _fit_function_template(
        self, x: np.ndarray, y: np.ndarray, template: Callable, func_type: BiologicalFunction
    ) -> Optional[DiscoveredFunction]:
        """TODO: Add docstring for _fit_function_template"""
        """TODO: Add docstring for _fit_function_template"""
            """TODO: Add docstring for _fit_function_template"""
    """Fit a specific function template to data"""
        try:
            if not SCIPY_AVAILABLE:
                # Simplified fitting for basic functions
                return self._simple_fit(x, y, template, func_type)

            # Use scipy for more sophisticated fitting
            return self._scipy_fit(x, y, template, func_type)

        except Exception:
            return None

            def _simple_fit(
        self, x: np.ndarray, y: np.ndarray, template: Callable, func_type: BiologicalFunction
    ) -> Optional[DiscoveredFunction]:
        """TODO: Add docstring for _simple_fit"""
        """TODO: Add docstring for _simple_fit"""
            """TODO: Add docstring for _simple_fit"""
    """Simple fitting without scipy"""

        if func_type == BiologicalFunction.LINEAR:
            # Simple linear regression
            A = np.vstack([x, np.ones(len(x))]).T
            params, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a, b = params

            y_pred = template(x, a, b)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            symbolic_expr = f"y = {a:.4f}*x + {b:.4f}"

            return DiscoveredFunction(
                function_type=func_type,
                symbolic_expression=symbolic_expr,
                parameters={"a": a, "b": b},
                confidence=min(r_squared, 1.0),
                biological_interpretation=self.biological_meanings.get(func_type, "Unknown"),
                input_variable="x",
                output_variable="y",
                r_squared=r_squared,
            )

        return None

            def _scipy_fit(
        self, x: np.ndarray, y: np.ndarray, template: Callable, func_type: BiologicalFunction
    ) -> Optional[DiscoveredFunction]:
        """TODO: Add docstring for _scipy_fit"""
        """TODO: Add docstring for _scipy_fit"""
            """TODO: Add docstring for _scipy_fit"""
    """Sophisticated fitting using scipy"""
        # Determine number of parameters
        import inspect

        sig = inspect.signature(template)
        num_params = len(sig.parameters) - 1  # Subtract x parameter

        # Initial parameter guesses
        p0 = self._get_initial_params(func_type, x, y, num_params)

        # Fit function
        popt, pcov = optimize.curve_fit(template, x, y, p0=p0, maxfev=10000)

        # Calculate R-squared
        y_pred = template(x, *popt)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Create parameter dictionary
        param_names = list(inspect.signature(template).parameters.keys())[1:]  # Skip 'x'
        parameters = dict(zip(param_names, popt))

        # Generate symbolic expression
        symbolic_expr = self._create_symbolic_expression(func_type, parameters)

        return DiscoveredFunction(
            function_type=func_type,
            symbolic_expression=symbolic_expr,
            parameters=parameters,
            confidence=min(r_squared, 1.0),
            biological_interpretation=self.biological_meanings.get(
                func_type, "Unknown biological function"
            ),
            input_variable="x",
            output_variable="y",
            r_squared=r_squared,
        )

        def _get_initial_params(
        self, func_type: BiologicalFunction, x: np.ndarray, y: np.ndarray, num_params: int
    ) -> List[float]:
        """TODO: Add docstring for _get_initial_params"""
        """TODO: Add docstring for _get_initial_params"""
            """TODO: Add docstring for _get_initial_params"""
    """Get initial parameter guesses based on function type"""
        if func_type == BiologicalFunction.LINEAR:
            return [1.0, 0.0]
        elif func_type == BiologicalFunction.EXPONENTIAL_DECAY:
            return [np.max(y), 1.0]
        elif func_type == BiologicalFunction.SIGMOIDAL:
            return [np.max(y) - np.min(y), 1.0, np.mean(x)]
        elif func_type == BiologicalFunction.POWER_LAW:
            return [1.0, 1.0]
        elif func_type == BiologicalFunction.PERIODIC:
            return [np.std(y), 2 * np.pi / np.ptp(x), 0.0, np.mean(y)]
        elif func_type == BiologicalFunction.LOGARITHMIC:
            return [1.0, 1.0]
        elif func_type == BiologicalFunction.THRESHOLD:
            return [np.max(y) - np.min(y), np.mean(x), np.min(y)]
        elif func_type == BiologicalFunction.OSCILLATORY:
            return [np.std(y), 0.1, 2 * np.pi / np.ptp(x), 0.0, np.mean(y)]
        else:
            return [1.0] * num_params

            def _create_symbolic_expression(
        self, func_type: BiologicalFunction, parameters: Dict[str, float]
    ) -> str:
        """TODO: Add docstring for _create_symbolic_expression"""
        """TODO: Add docstring for _create_symbolic_expression"""
            """TODO: Add docstring for _create_symbolic_expression"""
    """Create symbolic expression for the discovered function"""

        # Format parameters with reasonable precision
        formatted_params = {k: f"{v:.4f}" for k, v in parameters.items()}

        if func_type == BiologicalFunction.LINEAR:
            return f"y = {formatted_params['a']}*x + {formatted_params['b']}"
        elif func_type == BiologicalFunction.EXPONENTIAL_DECAY:
            return f"y = {formatted_params['a']}*exp(-{formatted_params['b']}*x)"
        elif func_type == BiologicalFunction.SIGMOIDAL:
            return f"y = {formatted_params['a']}/(1 + exp(-{formatted_params['b']}*(x - {formatted_params['c']})))"
        elif func_type == BiologicalFunction.POWER_LAW:
            return f"y = {formatted_params['a']}*x^{formatted_params['b']}"
        elif func_type == BiologicalFunction.PERIODIC:
            return f"y = {formatted_params['a']}*sin({formatted_params['b']}*x + {formatted_params['c']}) + {formatted_params['d']}"
        elif func_type == BiologicalFunction.LOGARITHMIC:
            return f"y = {formatted_params['a']}*log({formatted_params['b']}*x)"
        elif func_type == BiologicalFunction.THRESHOLD:
            return f"y = {formatted_params['a']}*H(x - {formatted_params['b']}) + {formatted_params['c']}"
        elif func_type == BiologicalFunction.OSCILLATORY:
            return f"y = {formatted_params['a']}*exp(-{formatted_params['b']}*x)*sin({formatted_params['c']}*x + {formatted_params['d']}) + {formatted_params['e']}"
        else:
            return "Unknown function type"

            def _analyze_function_pattern(  # noqa: C901
        self, x: np.ndarray, y: np.ndarray
    ) -> PatternAnalysis:  # noqa: C901
        """Analyze mathematical patterns in the function"""

        # Calculate derivatives numerically
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        dy_dx = np.gradient(y, dx)
        d2y_dx2 = np.gradient(dy_dx, dx)

        # Monotonicity analysis
        if np.all(dy_dx >= -1e-6):
            monotonicity = "increasing"
        elif np.all(dy_dx <= 1e-6):
            monotonicity = "decreasing"
        else:
            monotonicity = "mixed"

        # Concavity analysis
        if np.all(d2y_dx2 >= -1e-6):
            concavity = "concave_up"
        elif np.all(d2y_dx2 <= 1e-6):
            concavity = "concave_down"
        else:
            concavity = "mixed"

        # Find critical points (where derivative ≈ 0)
        critical_points = []
        for i in range(1, len(dy_dx) - 1):
            if abs(dy_dx[i]) < 1e-3 and dy_dx[i - 1] * dy_dx[i + 1] <= 0:
                critical_points.append(x[i])

        # Find inflection points (where second derivative ≈ 0)
        inflection_points = []
        for i in range(1, len(d2y_dx2) - 1):
            if abs(d2y_dx2[i]) < 1e-3 and d2y_dx2[i - 1] * d2y_dx2[i + 1] <= 0:
                inflection_points.append(x[i])

        # Detect asymptotes (simplified)
        asymptotes = []
        if len(y) > 10:
            # Horizontal asymptotes
            left_limit = np.mean(y[:5])
            right_limit = np.mean(y[-5:])
            if abs(right_limit - left_limit) < 0.1 * np.std(y):
                asymptotes.append(right_limit)

        # Dominant frequency analysis (for periodic functions)
        dominant_frequency = None
        try:
            fft = np.fft.fft(y - np.mean(y))
            freqs = np.fft.fftfreq(len(y), dx)
            power = np.abs(fft) ** 2

            # Find dominant frequency (excluding DC component)
            non_zero_mask = freqs != 0
            if np.any(non_zero_mask):
                dominant_idx = np.argmax(power[non_zero_mask])
                dominant_frequency = abs(freqs[non_zero_mask][dominant_idx])
        except:
            pass

        return PatternAnalysis(
            monotonicity=monotonicity,
            concavity=concavity,
            critical_points=critical_points[:5],  # Limit to first 5
            inflection_points=inflection_points[:5],
            asymptotes=asymptotes[:3],
            dominant_frequency=dominant_frequency,
        )

            def _generate_biological_insight(  # noqa: C901
        self, discovered: DiscoveredFunction, pattern: PatternAnalysis
    ) -> Optional[str]:
        """TODO: Add docstring for _generate_biological_insight"""
        """TODO: Add docstring for _generate_biological_insight"""
            """TODO: Add docstring for _generate_biological_insight"""
    """Generate biological insight from discovered function and pattern"""

        insights = []

        # Function-specific insights
        if discovered.function_type == BiologicalFunction.EXPONENTIAL_DECAY:
            if "b" in discovered.parameters:
                half_life = np.log(2) / discovered.parameters["b"]
                insights.append(f"Exponential decay with half-life of {half_life:.2f} units")

        elif discovered.function_type == BiologicalFunction.SIGMOIDAL:
            if "c" in discovered.parameters and "b" in discovered.parameters:
                ec50 = discovered.parameters["c"]
                hill_coeff = discovered.parameters["b"]
                insights.append(
                    f"Sigmoidal response with EC50 = {ec50:.2f} and Hill coefficient = {hill_coeff:.2f}"
                )

        elif discovered.function_type == BiologicalFunction.POWER_LAW:
            if "b" in discovered.parameters:
                exponent = discovered.parameters["b"]
                if abs(exponent - 0.75) < 0.1:
                    insights.append(
                        "Power law with exponent ≈ 3/4, consistent with metabolic scaling laws"
                    )
                elif abs(exponent - 2 / 3) < 0.1:
                    insights.append(
                        "Power law with exponent ≈ 2/3, consistent with surface area scaling"
                    )

        elif discovered.function_type == BiologicalFunction.PERIODIC:
            if "b" in discovered.parameters:
                period = 2 * np.pi / discovered.parameters["b"]
                if 23 < period < 25:
                    insights.append(
                        f"Periodic function with ~24h period, suggesting circadian regulation"
                    )
                elif 28 < period < 32:
                    insights.append(
                        f"Periodic function with ~30 day period, suggesting monthly biological cycle"
                    )

        # Pattern-specific insights
        if pattern.monotonicity == "increasing":
            insights.append("Monotonically increasing relationship suggests positive regulation")
        elif pattern.monotonicity == "decreasing":
            insights.append("Monotonically decreasing relationship suggests negative regulation")

        if len(pattern.critical_points) > 0:
            insights.append(
                f"Function has {len(pattern.critical_points)} critical points, suggesting complex regulation"
            )

        if pattern.dominant_frequency and 0.04 < pattern.dominant_frequency < 0.05:
            insights.append("Dominant frequency suggests circadian (24h) periodicity")

        return "; ".join(insights) if insights else None

            def _generate_symbolic_expression(self, discovered: DiscoveredFunction) -> Optional[str]:
                """TODO: Add docstring for _generate_symbolic_expression"""
        """TODO: Add docstring for _generate_symbolic_expression"""
            """TODO: Add docstring for _generate_symbolic_expression"""
    """Generate clean symbolic expression"""
        return discovered.symbolic_expression

                def _compute_interpretability_score(self, analysis_results: Dict[str, Any]) -> float:
                    """TODO: Add docstring for _compute_interpretability_score"""
        """TODO: Add docstring for _compute_interpretability_score"""
            """TODO: Add docstring for _compute_interpretability_score"""
    """Compute overall interpretability score"""

        num_functions = len(analysis_results["discovered_functions"])
        if num_functions == 0:
            return 0.0

        # Average confidence of discovered functions
        confidences = [
            func.confidence for func in analysis_results["discovered_functions"].values()
        ]
        avg_confidence = np.mean(confidences) if confidences else 0.0

        # Bonus for biological insights
        insight_bonus = min(len(analysis_results["biological_insights"]) * 0.1, 0.5)

        # Bonus for symbolic expressions
        symbolic_bonus = min(len(analysis_results["symbolic_expressions"]) * 0.05, 0.3)

        # Penalty for too many complex functions (harder to interpret)
        complexity_penalty = max(0, (num_functions - 10) * 0.02)

        score = avg_confidence + insight_bonus + symbolic_bonus - complexity_penalty

        return max(0.0, min(score, 1.0))

            def generate_interpretability_report(self, analysis_results: Dict[str, Any]) -> str:
                """TODO: Add docstring for generate_interpretability_report"""
        """TODO: Add docstring for generate_interpretability_report"""
            """TODO: Add docstring for generate_interpretability_report"""
    """Generate human-readable interpretability report"""

        report = []
        report.append("=== KAN Function Interpretability Report ===\n")

        # Summary
        num_functions = len(analysis_results["discovered_functions"])
        score = analysis_results["interpretability_score"]

        report.append(
            f"Analyzed {num_functions} functions with interpretability score: {score:.3f}"
        )
        report.append(f"Layer type: {analysis_results['layer_type']}\n")

        # Discovered functions
        if analysis_results["discovered_functions"]:
            report.append("--- Discovered Biological Functions ---")

            for func_name, func in analysis_results["discovered_functions"].items():
                report.append(f"\n{func_name}:")
                report.append(f"  Type: {func.function_type.value}")
                report.append(f"  Expression: {func.symbolic_expression}")
                report.append(f"  Confidence: {func.confidence:.3f}")
                report.append(f"  R²: {func.r_squared:.3f}")
                report.append(f"  Interpretation: {func.biological_interpretation}")

        # Biological insights
        if analysis_results["biological_insights"]:
            report.append("\n--- Biological Insights ---")
            for i, insight in enumerate(analysis_results["biological_insights"], 1):
                report.append(f"{i}. {insight}")

        # Symbolic expressions
        if analysis_results["symbolic_expressions"]:
            report.append("\n--- Symbolic Expressions ---")
            for i, expr in enumerate(analysis_results["symbolic_expressions"], 1):
                report.append(f"{i}. {expr}")

        report.append("\n" + "=" * 50)

        return "\n".join(report)


# Integration with existing KAN-HD system
class InterpretableKANHybridEncoder(nn.Module):
    """
    """
    """
    KAN-HD Hybrid Encoder with built-in interpretability analysis

    Extends the existing hybrid encoder with scientific interpretability features.
    """

    def __init__(self, base_dim: int = 10000, compressed_dim: int = 100) -> None:
        """TODO: Add docstring for __init__"""
        """TODO: Add docstring for __init__"""
            """TODO: Add docstring for __init__"""
    super().__init__()

        # Import the enhanced hybrid encoder
        from .enhanced_hybrid_encoder import EnhancedKANHybridEncoder

        self.hybrid_encoder = EnhancedKANHybridEncoder(
            base_dim=base_dim, compressed_dim=compressed_dim, enable_interpretability=True
        )
        self.function_analyzer = KANFunctionAnalyzer()

        # Store analysis results
        self.interpretability_cache = {}

        def forward(self, *args, **kwargs) -> None:
            """TODO: Add docstring for forward"""
        """TODO: Add docstring for forward"""
            """TODO: Add docstring for forward"""
    """Forward pass with optional interpretability analysis"""
        return self.hybrid_encoder(*args, **kwargs)

            def encode_genomic_data(self, *args, **kwargs) -> None:
                """TODO: Add docstring for encode_genomic_data"""
        """TODO: Add docstring for encode_genomic_data"""
            """TODO: Add docstring for encode_genomic_data"""
    """Delegate to hybrid encoder"""
        return self.hybrid_encoder.encode_genomic_data(*args, **kwargs)

                def encode_multimodal_data(self, *args, **kwargs) -> None:
                    """TODO: Add docstring for encode_multimodal_data"""
        """TODO: Add docstring for encode_multimodal_data"""
            """TODO: Add docstring for encode_multimodal_data"""
    """Delegate to hybrid encoder"""
        return self.hybrid_encoder.encode_multimodal_data(*args, **kwargs)

                    def analyze_interpretability(  # noqa: C901
        self, layer_name: Optional[str] = None
    ) -> Dict[str, Any]:  # noqa: C901
        """
        Analyze interpretability of KAN layers in the hybrid encoder

        Args:
            layer_name: Specific layer to analyze, or None for all layers

        Returns:
            Interpretability analysis results
        """
        results = {}

        # Analyze KAN compressor layers
        if hasattr(self.hybrid_encoder, "adaptive_compressor"):
            compressor = self.hybrid_encoder.adaptive_compressor

            if hasattr(compressor, "compressors"):
                for comp_name, comp in compressor.compressors.items():
                    if hasattr(comp, "encoder"):
                        for i, layer in enumerate(comp.encoder):
                            if isinstance(layer, (LinearKAN, KANLayer)):
                                layer_key = f"adaptive_{comp_name}_encoder_layer_{i}"
                                if layer_name is None or layer_name == layer_key:
                                    analysis = self.function_analyzer.analyze_kan_layer(layer)
                                    results[layer_key] = analysis

        # Analyze domain projections
        if hasattr(self.hybrid_encoder, "domain_projections"):
            for proj_name, projection in self.hybrid_encoder.domain_projections.items():
                if isinstance(projection, nn.Sequential):
                    for i, layer in enumerate(projection):
                        if isinstance(layer, (LinearKAN, KANLayer)):
                            layer_key = f"projection_{proj_name}_layer_{i}"
                            if layer_name is None or layer_name == layer_key:
                                analysis = self.function_analyzer.analyze_kan_layer(layer)
                                results[layer_key] = analysis

        # Analyze hierarchical encoder
        if hasattr(self.hybrid_encoder, "hierarchical_encoder"):
            hier_encoder = self.hybrid_encoder.hierarchical_encoder

            # Analyze resolution generators
            if hasattr(hier_encoder, "resolution_generators"):
                for res_name, res_gen in hier_encoder.resolution_generators.items():
                    if isinstance(res_gen, (LinearKAN, KANLayer)):
                        layer_key = f"resolution_{res_name}"
                        if layer_name is None or layer_name == layer_key:
                            analysis = self.function_analyzer.analyze_kan_layer(res_gen)
                            results[layer_key] = analysis

        # Cache results
                            self.interpretability_cache.update(results)

        return results

                            def generate_scientific_report(self) -> str:
                                """TODO: Add docstring for generate_scientific_report"""
        """TODO: Add docstring for generate_scientific_report"""
            """TODO: Add docstring for generate_scientific_report"""
    """Generate comprehensive scientific interpretability report"""

        # Ensure we have analysis results
        if not self.interpretability_cache:
            self.analyze_interpretability()

        report = []
        report.append("=== GenomeVault KAN-HD Scientific Interpretability Report ===\n")

        # Overall summary
        total_functions = sum(
            len(analysis["discovered_functions"])
            for analysis in self.interpretability_cache.values()
        )

        avg_interpretability = (
            np.mean(
                [
                    analysis["interpretability_score"]
                    for analysis in self.interpretability_cache.values()
                ]
            )
            if self.interpretability_cache
            else 0.0
        )

        report.append(f"Total analyzable functions discovered: {total_functions}")
        report.append(f"Average interpretability score: {avg_interpretability:.3f}")
        report.append(f"Layers analyzed: {len(self.interpretability_cache)}\n")

        # Per-layer analysis
        for layer_name, analysis in self.interpretability_cache.items():
            report.append(f"--- {layer_name.upper()} ---")
            layer_report = self.function_analyzer.generate_interpretability_report(analysis)
            report.append(layer_report)
            report.append("")

        return "\n".join(report)

            def export_discovered_functions(self, filepath: str) -> None:
                """TODO: Add docstring for export_discovered_functions"""
        """TODO: Add docstring for export_discovered_functions"""
            """TODO: Add docstring for export_discovered_functions"""
    """Export discovered functions to JSON for further analysis"""

        # Ensure we have analysis results
        if not self.interpretability_cache:
            self.analyze_interpretability()

        # Convert to serializable format
        export_data = {}

        for layer_name, analysis in self.interpretability_cache.items():
            layer_data = {
                "interpretability_score": analysis["interpretability_score"],
                "discovered_functions": {},
                "biological_insights": analysis["biological_insights"],
                "symbolic_expressions": analysis["symbolic_expressions"],
            }

            # Convert discovered functions
            for func_name, func in analysis["discovered_functions"].items():
                layer_data["discovered_functions"][func_name] = {
                    "function_type": func.function_type.value,
                    "symbolic_expression": func.symbolic_expression,
                    "parameters": func.parameters,
                    "confidence": func.confidence,
                    "biological_interpretation": func.biological_interpretation,
                    "r_squared": func.r_squared,
                }

            export_data[layer_name] = layer_data

        # Save to file
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"Discovered functions exported to {filepath}")
