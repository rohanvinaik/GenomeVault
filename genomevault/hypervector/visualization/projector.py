"""
Topographical Projection and Semantic Drift Detection for Model Training

This module provides visualization tools for understanding model evolution
during training using dimensionality reduction techniques.
"""

import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)

# Suppress UMAP warnings in production
warnings.filterwarnings("ignore", category=UserWarning)


class ModelEvolutionVisualizer:
    """Visualize model semantic evolution during training"""

    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = output_dir
        self.projections = {}
        self.drift_history = []

    def visualize_semantic_space(
        self,
        hypervectors: list[np.ndarray],
        labels: list[str],
        title: str = "Model Semantic Evolution",
        save_path: str | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Visualize high-dimensional model representations in 2D space.

        Args:
            hypervectors: List of high-dimensional model representations
            labels: Labels for each hypervector (e.g., epoch numbers)
            title: Title for the visualization
            save_path: Path to save the figure

        Returns:
            Dictionary with 'tsne' and 'umap' projections
        """
        logger.info("Projecting %slen(hypervectors) hypervectors to 2D space")

        # Convert to numpy array
        X = np.array(hypervectors)

        # T-SNE for local structure preservation
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, len(hypervectors) - 1),
            random_state=42,
            n_iter=1000,
        )
        embeddings_tsne = tsne.fit_transform(X)

        # UMAP for global structure preservation
        reducer = umap.UMAP(
            n_neighbors=min(15, len(hypervectors) - 1), min_dist=0.1, random_state=42
        )
        embeddings_umap = reducer.fit_transform(X)

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Color map for progression
        colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))

        # T-SNE plot
        ax1.scatter(
            embeddings_tsne[:, 0],
            embeddings_tsne[:, 1],
            c=colors,
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
        )

        # Add trajectory lines
        for i in range(1, len(embeddings_tsne)):
            ax1.plot(
                [embeddings_tsne[i - 1, 0], embeddings_tsne[i, 0]],
                [embeddings_tsne[i - 1, 1], embeddings_tsne[i, 1]],
                "k-",
                alpha=0.3,
                linewidth=1,
            )

        # Label key points
        for i, label in enumerate(labels):
            if i % max(1, len(labels) // 10) == 0 or i == len(labels) - 1:
                ax1.annotate(
                    label,
                    (embeddings_tsne[i, 0], embeddings_tsne[i, 1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        ax1.set_title("t-SNE: Semantic Projection of Training Dynamics", fontsize=14)
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")
        ax1.grid(True, alpha=0.3)

        # UMAP plot
        ax2.scatter(
            embeddings_umap[:, 0],
            embeddings_umap[:, 1],
            c=colors,
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidth=1,
        )

        # Add trajectory lines
        for i in range(1, len(embeddings_umap)):
            ax2.plot(
                [embeddings_umap[i - 1, 0], embeddings_umap[i, 0]],
                [embeddings_umap[i - 1, 1], embeddings_umap[i, 1]],
                "k-",
                alpha=0.3,
                linewidth=1,
            )

        # Label key points
        for i, label in enumerate(labels):
            if i % max(1, len(labels) // 10) == 0 or i == len(labels) - 1:
                ax2.annotate(
                    label,
                    (embeddings_umap[i, 0], embeddings_umap[i, 1]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        ax2.set_title("UMAP: Global Structure of Model Evolution", fontsize=14)
        ax2.set_xlabel("UMAP 1")
        ax2.set_ylabel("UMAP 2")
        ax2.grid(True, alpha=0.3)

        # Add colorbar
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(cmap="viridis"),
            ax=[ax1, ax2],
            orientation="horizontal",
            pad=0.1,
            aspect=40,
        )
        cbar.set_label("Training Progress", fontsize=12)

        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

        # Store projections
        self.projections = {"tsne": embeddings_tsne, "umap": embeddings_umap}

        return self.projections

    def detect_semantic_drift(
        self,
        snapshot_vectors: list[np.ndarray],
        threshold: float = 0.15,
        window_size: int = 5,
    ) -> tuple[list[float], list[int]]:
        """
        Detect semantic drift in model evolution.

        Args:
            snapshot_vectors: List of model representation vectors
            threshold: Drift threshold for alerting
            window_size: Window for computing rolling average drift

        Returns:
            Tuple of (drift_scores, anomaly_indices)
        """
        logger.info("Detecting semantic drift in model evolution")

        drift_scores = []
        anomaly_indices = []

        # Compute pairwise drift
        for i in range(1, len(snapshot_vectors)):
            # Cosine similarity between consecutive snapshots
            similarity = cosine_similarity(
                snapshot_vectors[i - 1].reshape(1, -1),
                snapshot_vectors[i].reshape(1, -1),
            )[0, 0]

            drift = 1 - similarity
            drift_scores.append(drift)

            # Check for anomalous drift
            if drift > threshold:
                anomaly_indices.append(i)
                logger.warning(
                    "⚠️  Semantic drift detected at epoch %si * 50: %sdrift:.3f "
                    "(threshold: %sthreshold)"
                )

        # Compute rolling average drift for trend analysis
        if len(drift_scores) >= window_size:
            rolling_drift = np.convolve(
                drift_scores, np.ones(window_size) / window_size, mode="valid"
            )

            # Check for sustained high drift
            sustained_drift_indices = np.where(rolling_drift > threshold * 0.8)[0]
            if len(sustained_drift_indices) > 0:
                logger.warning(
                    "Sustained semantic drift detected in epochs "
                    "%ssustained_drift_indices[0] * 50 to %ssustained_drift_indices[-1] * 50"
                )

        self.drift_history = drift_scores
        return drift_scores, anomaly_indices

    def plot_drift_analysis(
        self,
        drift_scores: list[float],
        anomaly_indices: list[int],
        labels: list[str] | None = None,
        save_path: str | None = None,
    ) -> None:
        """
        Plot semantic drift analysis over training.

        Args:
            drift_scores: List of drift scores
            anomaly_indices: Indices where anomalies detected
            labels: Optional epoch labels
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        epochs = range(1, len(drift_scores) + 1)

        # Drift score plot
        ax1.plot(epochs, drift_scores, "b-", linewidth=2, label="Semantic Drift")
        ax1.axhline(y=0.15, color="r", linestyle="--", label="Alert Threshold")
        ax1.fill_between(epochs, 0, drift_scores, alpha=0.3)

        # Mark anomalies
        if anomaly_indices:
            anomaly_scores = [drift_scores[i - 1] for i in anomaly_indices]
            ax1.scatter(
                anomaly_indices,
                anomaly_scores,
                color="red",
                s=100,
                marker="x",
                linewidth=3,
                label="Anomalies",
            )

        ax1.set_ylabel("Semantic Drift Score", fontsize=12)
        ax1.set_title("Model Semantic Drift During Training", fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Cumulative drift plot
        cumulative_drift = np.cumsum(drift_scores)
        ax2.plot(epochs, cumulative_drift, "g-", linewidth=2)
        ax2.fill_between(epochs, 0, cumulative_drift, alpha=0.3, color="green")

        ax2.set_xlabel("Training Epoch", fontsize=12)
        ax2.set_ylabel("Cumulative Drift", fontsize=12)
        ax2.set_title("Cumulative Semantic Drift", fontsize=14)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def analyze_trajectory_smoothness(
        self, projections: np.ndarray, projection_type: str = "umap"
    ) -> dict[str, float]:
        """
        Analyze the smoothness of model evolution trajectory.

        Args:
            projections: 2D projections of model states
            projection_type: Type of projection ('tsne' or 'umap')

        Returns:
            Dictionary of smoothness metrics
        """
        if len(projections) < 3:
            return {"smoothness": 1.0, "curvature": 0.0}

        # Compute path length
        distances = []
        for i in range(1, len(projections)):
            dist = np.linalg.norm(projections[i] - projections[i - 1])
            distances.append(dist)

        total_path_length = sum(distances)

        # Compute direct distance
        direct_distance = np.linalg.norm(projections[-1] - projections[0])

        # Smoothness metric (1 = straight line, higher = more curved)
        smoothness = total_path_length / (direct_distance + 1e-8)

        # Compute curvature at each point
        curvatures = []
        for i in range(1, len(projections) - 1):
            v1 = projections[i] - projections[i - 1]
            v2 = projections[i + 1] - projections[i]

            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            curvatures.append(angle)

        avg_curvature = np.mean(curvatures) if curvatures else 0.0
        max_curvature = np.max(curvatures) if curvatures else 0.0

        metrics = {
            "smoothness": smoothness,
            "avg_curvature": avg_curvature,
            "max_curvature": max_curvature,
            "total_path_length": total_path_length,
            "direct_distance": direct_distance,
            "efficiency": direct_distance / (total_path_length + 1e-8),
        }

        logger.info(
            "%sprojection_type trajectory analysis: "
            "smoothness=%ssmoothness:.3f, efficiency=%smetrics['efficiency']:.3f"
        )

        return metrics

    def detect_training_phases(
        self, snapshot_vectors: list[np.ndarray], n_phases: int = 3
    ) -> list[tuple[int, int]]:
        """
        Detect distinct phases in model training based on semantic changes.

        Args:
            snapshot_vectors: List of model representation vectors
            n_phases: Expected number of training phases

        Returns:
            List of (start_idx, end_idx) tuples for each phase
        """
        if len(snapshot_vectors) < n_phases:
            return [(0, len(snapshot_vectors) - 1)]

        # Compute drift scores
        drift_scores, _ = self.detect_semantic_drift(snapshot_vectors, threshold=float("inf"))

        # Find phase boundaries using change point detection
        # Simple approach: find points with largest drift changes
        if len(drift_scores) < 2:
            return [(0, len(snapshot_vectors) - 1)]

        # Compute second-order differences
        drift_changes = np.abs(np.diff(drift_scores))

        # Find top n_phases-1 change points
        change_points = np.argsort(drift_changes)[-1 - (n_phases - 1) :]
        change_points = sorted(change_points) + [len(snapshot_vectors) - 1]

        # Create phase boundaries
        phases = []
        start_idx = 0
        for end_idx in change_points:
            phases.append((start_idx, end_idx + 1))
            start_idx = end_idx + 1

        logger.info("Detected %slen(phases) training phases: %sphases")

        return phases

    def create_phase_visualization(
        self,
        hypervectors: list[np.ndarray],
        phases: list[tuple[int, int]],
        labels: list[str],
        save_path: str | None = None,
    ) -> None:
        """
        Visualize training phases with different colors.

        Args:
            hypervectors: List of model representations
            phases: List of (start, end) phase boundaries
            labels: Epoch labels
            save_path: Path to save figure
        """
        # Get UMAP projection
        X = np.array(hypervectors)
        reducer = umap.UMAP(n_neighbors=min(15, len(X) - 1), min_dist=0.1, random_state=42)
        embeddings = reducer.fit_transform(X)

        plt.figure(figsize=(10, 8))

        # Color each phase differently
        phase_colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))

        for idx, (start, end) in enumerate(phases):
            phase_embeddings = embeddings[start:end]
            # phase_labels = labels[start:end]  # Keep for potential future use

            # Plot phase points
            plt.scatter(
                phase_embeddings[:, 0],
                phase_embeddings[:, 1],
                c=[phase_colors[idx]] * len(phase_embeddings),
                s=100,
                alpha=0.7,
                edgecolors="black",
                linewidth=1,
                label=f"Phase {idx + 1}",
            )

            # Connect points within phase
            for i in range(1, len(phase_embeddings)):
                plt.plot(
                    [phase_embeddings[i - 1, 0], phase_embeddings[i, 0]],
                    [phase_embeddings[i - 1, 1], phase_embeddings[i, 1]],
                    color=phase_colors[idx],
                    alpha=0.5,
                    linewidth=2,
                )

        plt.title("Training Phase Analysis", fontsize=16)
        plt.xlabel("UMAP 1", fontsize=12)
        plt.ylabel("UMAP 2", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()


def create_semantic_debugging_report(
    visualizer: ModelEvolutionVisualizer,
    snapshot_vectors: list[np.ndarray],
    labels: list[str],
    output_path: str = "./semantic_debug_report.png",
) -> dict[str, Any]:
    """
    Create comprehensive semantic debugging report.

    Args:
        visualizer: ModelEvolutionVisualizer instance
        snapshot_vectors: List of model hypervectors
        labels: Epoch labels
        output_path: Path to save report

    Returns:
        Dictionary with analysis results
    """
    # Create figure with subplots
    plt.figure(figsize=(20, 12))

    # 1. Projection visualization
    ax1 = plt.subplot(2, 3, 1)
    X = np.array(snapshot_vectors)
    reducer = umap.UMAP(n_neighbors=min(15, len(X) - 1), random_state=42)
    embeddings = reducer.fit_transform(X)

    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    ax1.scatter(embeddings[:, 0], embeddings[:, 1], c=colors, s=100, alpha=0.7)
    for i in range(1, len(embeddings)):
        ax1.plot(
            [embeddings[i - 1, 0], embeddings[i, 0]],
            [embeddings[i - 1, 1], embeddings[i, 1]],
            "k-",
            alpha=0.3,
        )
    ax1.set_title("Semantic Evolution (UMAP)")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")

    # 2. Drift analysis
    ax2 = plt.subplot(2, 3, 2)
    drift_scores, anomalies = visualizer.detect_semantic_drift(snapshot_vectors)
    epochs = range(1, len(drift_scores) + 1)
    ax2.plot(epochs, drift_scores, "b-", linewidth=2)
    ax2.axhline(y=0.15, color="r", linestyle="--", alpha=0.5)
    ax2.fill_between(epochs, 0, drift_scores, alpha=0.3)
    ax2.set_title("Semantic Drift Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Drift")

    # 3. Phase detection
    ax3 = plt.subplot(2, 3, 3)
    phases = visualizer.detect_training_phases(snapshot_vectors)
    phase_colors = plt.cm.Set3(np.linspace(0, 1, len(phases)))

    for idx, (start, end) in enumerate(phases):
        ax3.axvspan(start, end, alpha=0.3, color=phase_colors[idx], label=f"Phase {idx + 1}")
    ax3.plot(range(len(snapshot_vectors)), [0] * len(snapshot_vectors), "k.", markersize=10)
    ax3.set_title("Training Phases")
    ax3.set_xlabel("Snapshot Index")
    ax3.legend()

    # 4. Trajectory smoothness
    ax4 = plt.subplot(2, 3, 4)
    metrics = visualizer.analyze_trajectory_smoothness(embeddings, "umap")
    metric_names = list(metrics.keys())[:4]
    metric_values = [metrics[k] for k in metric_names]
    ax4.bar(metric_names, metric_values)
    ax4.set_title("Trajectory Metrics")
    ax4.set_ylabel("Value")
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

    # 5. Distance matrix heatmap
    ax5 = plt.subplot(2, 3, 5)
    similarity_matrix = cosine_similarity(X)
    sns.heatmap(similarity_matrix, cmap="viridis", ax=ax5, cbar=True)
    ax5.set_title("Cosine Similarity Matrix")
    ax5.set_xlabel("Snapshot")
    ax5.set_ylabel("Snapshot")

    # 6. Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    summary_text = f"""
    Semantic Analysis Summary
    ========================
    Total Snapshots: {len(snapshot_vectors)}
    Anomalies Detected: {len(anomalies)}
    Number of Phases: {len(phases)}
    Avg Drift: {np.mean(drift_scores):.3f}
    Max Drift: {np.max(drift_scores):.3f}
    Trajectory Smoothness: {metrics["smoothness"]:.3f}
    Path Efficiency: {metrics["efficiency"]:.3f}
    """
    ax6.text(
        0.1,
        0.5,
        summary_text,
        fontsize=12,
        family="monospace",
        verticalalignment="center",
    )

    plt.suptitle("Model Semantic Debugging Report", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

    # Return analysis results
    return {
        "drift_scores": drift_scores,
        "anomaly_indices": anomalies,
        "phases": phases,
        "trajectory_metrics": metrics,
        "embeddings": embeddings,
    }
