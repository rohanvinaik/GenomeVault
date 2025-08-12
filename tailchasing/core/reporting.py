"""
Chromatin-inspired visualizations for performance analysis.

This module implements Hi-C style contact matrices and polymer physics metrics
for visualizing and analyzing performance bottlenecks in a biologically-inspired way.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class TAD:
    """Topologically Associating Domain representation."""

    start: int
    end: int
    name: str
    activity_level: float = 0.0

    def size(self) -> int:
        """Get TAD size."""
        return self.end - self.start

    def contains(self, position: int) -> bool:
        """Check if position is within this TAD."""
        return self.start <= position <= self.end


@dataclass
class ThrashCluster:
    """Represents a cluster of performance thrashing events."""

    positions: List[int]
    risk_score: float
    frequency: int
    avg_latency: float

    def center(self) -> float:
        """Get cluster center position."""
        return sum(self.positions) / len(self.positions) if self.positions else 0.0


class HiCHeatmapGenerator:
    """
    Generates Hi-C style contact heatmaps for performance analysis.

    Maps performance events to a contact matrix where:
    - Axes represent code positions/functions/modules
    - Contact intensity represents interaction frequency/severity
    - TAD boundaries represent logical separation boundaries
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self.unicode_blocks = [
            " ",  # 0-12.5%
            "▁",  # 12.5-25%
            "▂",  # 25-37.5%
            "▃",  # 37.5-50%
            "▄",  # 50-62.5%
            "▅",  # 62.5-75%
            "▆",  # 75-87.5%
            "▇",  # 87.5-100%
            "█",  # 100%
        ]

    def generate_contact_heatmap(
        self,
        contact_matrix: np.ndarray,
        tads: Optional[List[TAD]] = None,
        title: str = "Performance Contact Matrix",
    ) -> str:
        """
        Generate ASCII/Unicode art heatmap of contact matrix.

        Args:
            contact_matrix: 2D array representing interaction frequencies
            tads: Optional list of TAD boundaries
            title: Title for the heatmap

        Returns:
            Formatted string representation of the heatmap
        """
        if contact_matrix.size == 0:
            return "Empty contact matrix"

        # Normalize matrix for visualization
        normalized = self._normalize_matrix(contact_matrix)

        # Create ASCII representation
        ascii_heatmap = self._matrix_to_ascii(normalized)

        # Add TAD boundaries if provided
        if tads:
            ascii_heatmap = self._add_tad_boundaries(ascii_heatmap, tads, normalized.shape)

        # Create rich panel
        heatmap_text = Text()
        for line in ascii_heatmap:
            heatmap_text.append(line + "\n")

        panel = Panel(
            heatmap_text,
            title=title,
            subtitle=f"Matrix size: {contact_matrix.shape[0]}×{contact_matrix.shape[1]}",
        )

        # Capture panel output as string
        with self.console.capture() as capture:
            self.console.print(panel)

        return capture.get()

    def highlight_thrash_clusters(
        self, matrix: np.ndarray, risk_scores: Dict[Tuple[int, int], float]
    ) -> str:
        """
        Mark high-risk regions in the contact matrix.

        Args:
            matrix: Contact matrix
            risk_scores: Dictionary mapping matrix positions to risk scores

        Returns:
            Formatted heatmap with highlighted risk regions
        """
        if matrix.size == 0:
            return "Empty matrix"

        normalized = self._normalize_matrix(matrix)
        ascii_matrix = self._matrix_to_ascii(normalized)

        # Overlay risk indicators
        risk_overlay = self._create_risk_overlay(ascii_matrix, risk_scores, matrix.shape)

        # Create table for better formatting
        table = Table(title="Thrash Risk Analysis", show_header=False, box=None)

        for i, line in enumerate(risk_overlay):
            if i < len(ascii_matrix):
                # Color code based on risk level
                risk_line = self._colorize_risk_line(line, i, risk_scores, matrix.shape)
                table.add_row(risk_line)

        with self.console.capture() as capture:
            self.console.print(table)

        return capture.get()

    def show_tad_boundaries(self, matrix: np.ndarray, tad_map: Dict[str, TAD]) -> str:
        """
        Visual TAD delimitation on contact matrix.

        Args:
            matrix: Contact matrix
            tad_map: Dictionary mapping TAD names to TAD objects

        Returns:
            Formatted heatmap with TAD boundaries
        """
        if matrix.size == 0:
            return "Empty matrix"

        normalized = self._normalize_matrix(matrix)
        ascii_matrix = self._matrix_to_ascii(normalized)

        # Draw TAD boundaries
        tad_bounded = self._draw_tad_boundaries(ascii_matrix, list(tad_map.values()), matrix.shape)

        # Create legend
        legend_table = Table(title="TAD Legend", show_header=True)
        legend_table.add_column("TAD", style="cyan")
        legend_table.add_column("Range", style="green")
        legend_table.add_column("Size", style="yellow")
        legend_table.add_column("Activity", style="red")

        for name, tad in tad_map.items():
            legend_table.add_row(
                name,
                f"{tad.start}-{tad.end}",
                str(tad.size()),
                f"{tad.activity_level:.2f}",
            )

        # Combine heatmap and legend
        heatmap_panel = Panel("\n".join(tad_bounded), title="TAD Boundary Analysis")

        with self.console.capture() as capture:
            self.console.print(heatmap_panel)
            self.console.print(legend_table)

        return capture.get()

    def _normalize_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Normalize matrix values to 0-1 range."""
        if matrix.max() == matrix.min():
            return np.zeros_like(matrix, dtype=float)
        return (matrix - matrix.min()) / (matrix.max() - matrix.min())

    def _matrix_to_ascii(self, normalized_matrix: np.ndarray) -> List[str]:
        """Convert normalized matrix to ASCII using Unicode blocks."""
        ascii_lines = []
        height, width = normalized_matrix.shape

        for i in range(height):
            line = ""
            for j in range(width):
                value = normalized_matrix[i, j]
                block_index = min(
                    int(value * len(self.unicode_blocks)), len(self.unicode_blocks) - 1
                )
                line += self.unicode_blocks[block_index]
            ascii_lines.append(line)

        return ascii_lines

    def _add_tad_boundaries(
        self, ascii_lines: List[str], tads: List[TAD], shape: Tuple[int, int]
    ) -> List[str]:
        """Add TAD boundary markers to ASCII representation."""
        modified_lines = ascii_lines.copy()
        height, width = shape

        for tad in tads:
            # Scale TAD positions to matrix coordinates
            start_pos = int((tad.start / 100) * width)  # Assuming positions are 0-100 scale
            end_pos = int((tad.end / 100) * width)

            # Draw vertical boundaries
            for i in range(height):
                if start_pos < len(modified_lines[i]):
                    line = list(modified_lines[i])
                    line[start_pos] = "|"
                    modified_lines[i] = "".join(line)
                if end_pos < len(modified_lines[i]):
                    line = list(modified_lines[i])
                    line[end_pos] = "|"
                    modified_lines[i] = "".join(line)

        return modified_lines

    def _create_risk_overlay(
        self,
        ascii_matrix: List[str],
        risk_scores: Dict[Tuple[int, int], float],
        shape: Tuple[int, int],
    ) -> List[str]:
        """Create risk overlay on ASCII matrix."""
        overlay = ascii_matrix.copy()
        height, width = shape

        for (i, j), risk in risk_scores.items():
            if 0 <= i < height and 0 <= j < width and i < len(overlay):
                if risk > 0.7:  # High risk
                    if j < len(overlay[i]):
                        line = list(overlay[i])
                        line[j] = "⚠"
                        overlay[i] = "".join(line)

        return overlay

    def _colorize_risk_line(
        self,
        line: str,
        row_idx: int,
        risk_scores: Dict[Tuple[int, int], float],
        shape: Tuple[int, int],
    ) -> Text:
        """Apply color coding to risk line based on risk scores."""
        text = Text()

        for col_idx, char in enumerate(line):
            risk = risk_scores.get((row_idx, col_idx), 0.0)

            if risk > 0.8:
                style = "bold red"
            elif risk > 0.6:
                style = "red"
            elif risk > 0.4:
                style = "yellow"
            elif risk > 0.2:
                style = "blue"
            else:
                style = "dim"

            text.append(char, style=style)

        return text

    def _draw_tad_boundaries(
        self, ascii_matrix: List[str], tads: List[TAD], shape: Tuple[int, int]
    ) -> List[str]:
        """Draw TAD boundaries on ASCII matrix."""
        bounded = ascii_matrix.copy()
        height, width = shape

        for tad in tads:
            # Scale positions
            start = int((tad.start / 100) * width)
            end = int((tad.end / 100) * width)

            # Draw boundaries
            for i in range(height):
                if i < len(bounded):
                    line = list(bounded[i])
                    if start < len(line):
                        line[start] = "┃"
                    if end < len(line):
                        line[end] = "┃"
                    bounded[i] = "".join(line)

        return bounded


class PolymerMetricsReport:
    """
    Polymer physics-inspired metrics for performance analysis.

    Treats code execution as polymer dynamics where:
    - Functions are monomers
    - Call chains are polymer segments
    - TADs represent functional domains
    - Contact probabilities represent interaction likelihoods
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def calculate_polymer_distances(
        self, tads: List[TAD], interactions: List[Tuple[int, int, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate average polymer distance within and across TADs.

        Args:
            tads: List of TAD objects
            interactions: List of (pos1, pos2, strength) tuples

        Returns:
            Dictionary with intra-TAD and inter-TAD distance metrics
        """
        metrics = {
            "intra_tad_distances": {},
            "inter_tad_distances": {},
            "global_metrics": {},
        }

        # Calculate intra-TAD distances
        for tad in tads:
            intra_distances = []
            for pos1, pos2, strength in interactions:
                if tad.contains(pos1) and tad.contains(pos2):
                    distance = abs(pos1 - pos2)
                    intra_distances.append(distance * strength)

            if intra_distances:
                metrics["intra_tad_distances"][tad.name] = {
                    "mean": np.mean(intra_distances),
                    "std": np.std(intra_distances),
                    "median": np.median(intra_distances),
                    "count": len(intra_distances),
                }

        # Calculate inter-TAD distances
        for i, tad1 in enumerate(tads):
            for j, tad2 in enumerate(tads):
                if i < j:  # Avoid duplicates
                    inter_distances = []
                    for pos1, pos2, strength in interactions:
                        if (tad1.contains(pos1) and tad2.contains(pos2)) or (
                            tad2.contains(pos1) and tad1.contains(pos2)
                        ):
                            distance = abs(pos1 - pos2)
                            inter_distances.append(distance * strength)

                    if inter_distances:
                        pair_name = f"{tad1.name}-{tad2.name}"
                        metrics["inter_tad_distances"][pair_name] = {
                            "mean": np.mean(inter_distances),
                            "std": np.std(inter_distances),
                            "median": np.median(inter_distances),
                            "count": len(inter_distances),
                        }

        # Global metrics
        all_distances = [abs(pos1 - pos2) * strength for pos1, pos2, strength in interactions]
        if all_distances:
            metrics["global_metrics"] = {
                "overall_mean_distance": np.mean(all_distances),
                "overall_std_distance": np.std(all_distances),
                "total_interactions": len(interactions),
                "polymer_compactness": self._calculate_compactness(all_distances),
            }

        return metrics

    def calculate_contact_probabilities(
        self, interactions: List[Tuple[int, int, float]], max_distance: int = 1000
    ) -> Dict[str, Union[List[float], Dict[str, float]]]:
        """
        Calculate contact probability distributions.

        Args:
            interactions: List of interaction data
            max_distance: Maximum distance to consider

        Returns:
            Contact probability distribution data
        """
        distance_bins = np.linspace(0, max_distance, 50)
        probabilities = np.zeros(len(distance_bins) - 1)

        for pos1, pos2, strength in interactions:
            distance = abs(pos1 - pos2)
            if distance <= max_distance:
                bin_idx = np.digitize(distance, distance_bins) - 1
                if 0 <= bin_idx < len(probabilities):
                    probabilities[bin_idx] += strength

        # Normalize
        if probabilities.sum() > 0:
            probabilities = probabilities / probabilities.sum()

        return {
            "distance_bins": distance_bins[:-1].tolist(),
            "probabilities": probabilities.tolist(),
            "statistics": {
                "mean_contact_distance": (
                    np.average(distance_bins[:-1], weights=probabilities)
                    if probabilities.sum() > 0
                    else 0
                ),
                "contact_decay_rate": self._calculate_decay_rate(distance_bins[:-1], probabilities),
                "short_range_fraction": probabilities[:10].sum(),  # First 10 bins
                "long_range_fraction": probabilities[-10:].sum(),  # Last 10 bins
            },
        }

    def predict_thrash_reduction(
        self, fix_strategies: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Predict thrash reduction from each fix strategy.

        Args:
            fix_strategies: List of fix strategy descriptions

        Returns:
            Predicted reduction metrics for each strategy
        """
        predictions = {}

        for strategy in fix_strategies:
            strategy_name = strategy.get("name", "Unknown")
            impact_score = strategy.get("impact_score", 0.5)
            complexity = strategy.get("complexity", 0.5)
            confidence = strategy.get("confidence", 0.5)

            # Simple prediction model based on strategy parameters
            reduction_estimate = impact_score * confidence * (1 - complexity * 0.3)
            risk_factor = complexity * (1 - confidence)

            predictions[strategy_name] = {
                "estimated_reduction": reduction_estimate,
                "implementation_risk": risk_factor,
                "roi_score": reduction_estimate / (complexity + 0.1),  # Avoid division by zero
                "recommended_priority": self._calculate_priority(
                    reduction_estimate, risk_factor, complexity
                ),
            }

        return predictions

    def visualize_replication_timing(self, timeline_data: List[Dict[str, Any]]) -> str:
        """
        Generate replication timing schedule visualization.

        Args:
            timeline_data: List of timing events

        Returns:
            Formatted timeline visualization
        """
        if not timeline_data:
            return "No timeline data available"

        # Sort by timestamp
        sorted_data = sorted(timeline_data, key=lambda x: x.get("timestamp", 0))

        table = Table(title="Replication Timing Schedule", show_header=True)
        table.add_column("Time", style="cyan")
        table.add_column("Event", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Impact", style="red")
        table.add_column("Status", style="blue")

        for event in sorted_data:
            timestamp = event.get("timestamp", 0)
            name = event.get("name", "Unknown")
            duration = event.get("duration", 0)
            impact = event.get("impact", 0.0)
            status = event.get("status", "pending")

            # Format impact with visual indicator
            impact_str = f"{impact:.2f} {'🔥' if impact > 0.7 else '⚡' if impact > 0.4 else '💚'}"

            table.add_row(f"{timestamp:.2f}ms", name, f"{duration:.2f}ms", impact_str, status)

        with self.console.capture() as capture:
            self.console.print(table)

        return capture.get()

    def generate_comprehensive_report(
        self,
        tads: List[TAD],
        interactions: List[Tuple[int, int, float]],
        fix_strategies: List[Dict[str, Any]],
        timeline_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive polymer metrics report.

        Args:
            tads: TAD definitions
            interactions: Interaction data
            fix_strategies: Fix strategy data
            timeline_data: Timeline event data

        Returns:
            Complete metrics report dictionary
        """
        report = {
            "polymer_distances": self.calculate_polymer_distances(tads, interactions),
            "contact_probabilities": self.calculate_contact_probabilities(interactions),
            "thrash_predictions": self.predict_thrash_reduction(fix_strategies),
            "timeline_analysis": self._analyze_timeline(timeline_data),
            "summary_metrics": {},
        }

        # Calculate summary metrics
        polymer_data = report["polymer_distances"]
        if polymer_data.get("global_metrics"):
            report["summary_metrics"] = {
                "overall_health_score": self._calculate_health_score(
                    polymer_data, report["thrash_predictions"]
                ),
                "optimization_potential": self._calculate_optimization_potential(
                    report["thrash_predictions"]
                ),
                "stability_index": self._calculate_stability_index(timeline_data),
            }

        return report

    def _calculate_compactness(self, distances: List[float]) -> float:
        """Calculate polymer compactness metric."""
        if not distances:
            return 0.0

        mean_dist = np.mean(distances)
        max_dist = max(distances)

        return 1.0 - (mean_dist / max_dist) if max_dist > 0 else 1.0

    def _calculate_decay_rate(self, distances: np.ndarray, probabilities: np.ndarray) -> float:
        """Calculate contact probability decay rate."""
        if len(distances) < 2 or probabilities.sum() == 0:
            return 0.0

        # Fit exponential decay: P(d) = exp(-d/λ)
        log_probs = np.log(probabilities + 1e-10)  # Avoid log(0)
        valid_idx = np.isfinite(log_probs)

        if valid_idx.sum() < 2:
            return 0.0

        # Simple linear fit to log probabilities
        x = distances[valid_idx]
        y = log_probs[valid_idx]

        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            return -slope if slope < 0 else 0.0

        return 0.0

    def _calculate_priority(self, reduction: float, risk: float, complexity: float) -> str:
        """Calculate priority rating for fix strategy."""
        score = reduction * 2 - risk - complexity * 0.5

        if score > 1.0:
            return "High"
        elif score > 0.5:
            return "Medium"
        else:
            return "Low"

    def _analyze_timeline(self, timeline_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze timeline data for patterns."""
        if not timeline_data:
            return {}

        durations = [event.get("duration", 0) for event in timeline_data]
        impacts = [event.get("impact", 0) for event in timeline_data]

        return {
            "total_duration": sum(durations),
            "average_duration": np.mean(durations),
            "max_impact_event": max(impacts) if impacts else 0,
            "total_events": len(timeline_data),
            "high_impact_events": sum(1 for impact in impacts if impact > 0.7),
        }

    def _calculate_health_score(
        self, polymer_data: Dict[str, Any], predictions: Dict[str, Dict[str, float]]
    ) -> float:
        """Calculate overall system health score."""
        base_score = 1.0

        # Penalize based on global metrics
        global_metrics = polymer_data.get("global_metrics", {})
        compactness = global_metrics.get("polymer_compactness", 1.0)

        health_score = base_score * compactness

        # Factor in prediction confidence
        if predictions:
            avg_reduction = np.mean([pred["estimated_reduction"] for pred in predictions.values()])
            health_score *= (
                1.0 - avg_reduction * 0.5
            )  # Higher predicted reduction = lower current health

        return max(0.0, min(1.0, health_score))

    def _calculate_optimization_potential(self, predictions: Dict[str, Dict[str, float]]) -> float:
        """Calculate optimization potential score."""
        if not predictions:
            return 0.0

        roi_scores = [pred["roi_score"] for pred in predictions.values()]
        return np.mean(roi_scores) if roi_scores else 0.0

    def _calculate_stability_index(self, timeline_data: List[Dict[str, Any]]) -> float:
        """Calculate system stability index."""
        if not timeline_data:
            return 1.0

        impacts = [event.get("impact", 0) for event in timeline_data]
        durations = [event.get("duration", 0) for event in timeline_data]

        if not impacts or not durations:
            return 1.0

        # Lower variance in impacts and durations = higher stability
        impact_stability = 1.0 - (np.std(impacts) / (np.mean(impacts) + 0.1))
        duration_stability = 1.0 - (np.std(durations) / (np.mean(durations) + 0.1))

        return max(0.0, min(1.0, (impact_stability + duration_stability) / 2))


def integrate_chromatin_analysis(
    existing_report: Dict[str, Any],
    contact_matrix: np.ndarray,
    tads: List[TAD],
    interactions: List[Tuple[int, int, float]],
    fix_strategies: List[Dict[str, Any]],
    timeline_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Integrate chromatin analysis into existing JSON reporting.

    Args:
        existing_report: Existing report dictionary
        contact_matrix: Performance contact matrix
        tads: TAD definitions
        interactions: Interaction data
        fix_strategies: Fix strategies
        timeline_data: Timeline events

    Returns:
        Enhanced report with chromatin analysis section
    """
    # Generate chromatin-inspired metrics
    heatmap_gen = HiCHeatmapGenerator()
    polymer_reporter = PolymerMetricsReport()

    # Create comprehensive analysis
    chromatin_analysis = {
        "contact_matrix_summary": {
            "dimensions": contact_matrix.shape,
            "total_contacts": int(np.sum(contact_matrix)),
            "max_contact_strength": float(np.max(contact_matrix)),
            "mean_contact_strength": float(np.mean(contact_matrix)),
        },
        "tad_analysis": {
            "total_tads": len(tads),
            "tad_details": [
                {
                    "name": tad.name,
                    "start": tad.start,
                    "end": tad.end,
                    "size": tad.size(),
                    "activity_level": tad.activity_level,
                }
                for tad in tads
            ],
            "average_tad_size": np.mean([tad.size() for tad in tads]) if tads else 0,
        },
        "polymer_metrics": polymer_reporter.generate_comprehensive_report(
            tads, interactions, fix_strategies, timeline_data
        ),
        "visualization_data": {
            "heatmap_ascii": heatmap_gen.generate_contact_heatmap(contact_matrix, tads),
            "contact_probabilities": polymer_reporter.calculate_contact_probabilities(interactions),
        },
    }

    # Add polymer-based risk scoring
    risk_scores = {}
    for i in range(contact_matrix.shape[0]):
        for j in range(contact_matrix.shape[1]):
            if contact_matrix[i, j] > 0:
                # Risk based on contact strength and distance
                distance = abs(i - j) + 1
                strength = contact_matrix[i, j]
                risk = strength / distance  # Higher strength, shorter distance = higher risk
                risk_scores[(i, j)] = float(risk)

    chromatin_analysis["risk_analysis"] = {
        "total_risk_positions": len(risk_scores),
        "high_risk_positions": len([r for r in risk_scores.values() if r > 0.7]),
        "average_risk_score": np.mean(list(risk_scores.values())) if risk_scores else 0.0,
        "risk_distribution": {
            "low": len([r for r in risk_scores.values() if r <= 0.3]),
            "medium": len([r for r in risk_scores.values() if 0.3 < r <= 0.7]),
            "high": len([r for r in risk_scores.values() if r > 0.7]),
        },
    }

    # Integrate into existing report
    enhanced_report = existing_report.copy()
    enhanced_report["chromatin_analysis"] = chromatin_analysis

    return enhanced_report


def generate_comparative_matrices(
    before_matrix: np.ndarray,
    after_matrix: np.ndarray,
    tads: List[TAD],
    strategy_name: str = "optimization",
) -> Dict[str, Any]:
    """
    Generate comparative before/after contact matrices.

    Args:
        before_matrix: Contact matrix before optimization
        after_matrix: Contact matrix after optimization
        tads: TAD definitions
        strategy_name: Name of the optimization strategy

    Returns:
        Comparative analysis results
    """
    heatmap_gen = HiCHeatmapGenerator()

    # Generate difference matrix
    diff_matrix = after_matrix - before_matrix

    # Calculate improvement metrics
    total_before = np.sum(before_matrix)
    total_after = np.sum(after_matrix)
    reduction_ratio = (total_before - total_after) / total_before if total_before > 0 else 0

    comparative_analysis = {
        "strategy_name": strategy_name,
        "metrics": {
            "total_contacts_before": int(total_before),
            "total_contacts_after": int(total_after),
            "absolute_reduction": int(total_before - total_after),
            "reduction_percentage": float(reduction_ratio * 100),
            "improvement_score": float(max(0, reduction_ratio)),
        },
        "visualizations": {
            "before_heatmap": heatmap_gen.generate_contact_heatmap(
                before_matrix, tads, f"Before {strategy_name}"
            ),
            "after_heatmap": heatmap_gen.generate_contact_heatmap(
                after_matrix, tads, f"After {strategy_name}"
            ),
            "difference_heatmap": heatmap_gen.generate_contact_heatmap(
                np.abs(diff_matrix), tads, f"Change from {strategy_name}"
            ),
        },
        "tad_specific_improvements": {},
    }

    # Calculate TAD-specific improvements
    for tad in tads:
        # Scale TAD coordinates to matrix indices
        start_idx = int((tad.start / 100) * before_matrix.shape[0])
        end_idx = int((tad.end / 100) * before_matrix.shape[0])

        if start_idx < before_matrix.shape[0] and end_idx <= before_matrix.shape[0]:
            tad_before = before_matrix[start_idx:end_idx, start_idx:end_idx]
            tad_after = after_matrix[start_idx:end_idx, start_idx:end_idx]

            tad_reduction = (
                (np.sum(tad_before) - np.sum(tad_after)) / np.sum(tad_before)
                if np.sum(tad_before) > 0
                else 0
            )

            comparative_analysis["tad_specific_improvements"][tad.name] = {
                "reduction_percentage": float(tad_reduction * 100),
                "contacts_before": int(np.sum(tad_before)),
                "contacts_after": int(np.sum(tad_after)),
            }

    return comparative_analysis
