"""
Federated Model Lineage Tracking for Distributed Training

This module tracks model evolution across federated learning rounds,
creating a verifiable DAG (Directed Acyclic Graph) of model updates.
"""

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx

from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class NodeRole(Enum):
    """Role of a federated learning participant"""

    AGGREGATOR = "aggregator"
    CLIENT = "client"
    VALIDATOR = "validator"
    ORCHESTRATOR = "orchestrator"


class UpdateType(Enum):
    """Type of model update in federated learning"""

    LOCAL_TRAINING = "local_training"
    AGGREGATION = "aggregation"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"
    ROLLBACK = "rollback"


@dataclass
class ModelVersion:
    """Represents a specific version of the model"""

    version_id: str
    parent_version: str | None
    model_hash: str
    timestamp: int
    round_number: int
    contributor_id: str
    update_type: UpdateType
    metrics: dict[str, float]
    metadata: dict[str, Any]


@dataclass
class LineageEdge:
    """Edge in the model lineage graph"""

    from_version: str
    to_version: str
    edge_type: str  # 'update', 'aggregate', 'validate'
    weight: float  # Contribution weight for aggregation
    metadata: dict[str, Any]


class FederatedModelLineage:
    """
    Tracks model lineage in federated learning systems.

    Creates an immutable DAG showing:
    1. Local model updates from clients
    2. Aggregation operations
    3. Validation checkpoints
    4. Fork/merge history
    """

    def __init__(self, federation_id: str, initial_model_hash: str):
        """
        Initialize federated model lineage tracker.

        Args:
            federation_id: Unique ID for this federation
            initial_model_hash: Hash of the initial model
        """
        self.federation_id = federation_id
        self.lineage_graph = nx.DiGraph()
        self.versions: dict[str, ModelVersion] = {}
        self.current_round = 0
        self.active_branches: set[str] = set()

        # Create initial version
        initial_version = ModelVersion(
            version_id="v0",
            parent_version=None,
            model_hash=initial_model_hash,
            timestamp=int(time.time()),
            round_number=0,
            contributor_id="system",
            update_type=UpdateType.CHECKPOINT,
            metrics={},
            metadata={"initial": True},
        )

        self._add_version(initial_version)
        self.active_branches.add("v0")

        logger.info(f"Initialized federated lineage for {federation_id}")

    def record_local_update(
        self,
        client_id: str,
        parent_version: str,
        new_model_hash: str,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Record a local model update from a client.

        Args:
            client_id: ID of the client
            parent_version: Version this update is based on
            new_model_hash: Hash of updated model
            metrics: Training metrics (loss, accuracy, etc.)
            metadata: Additional metadata

        Returns:
            New version ID
        """
        if parent_version not in self.versions:
            raise ValueError(f"Parent version {parent_version} not found")

        # Generate version ID
        version_data = f"{parent_version}{client_id}{new_model_hash}{time.time()}"
        version_id = f"v{hashlib.sha256(version_data.encode()).hexdigest()[:8]}"

        # Create version
        version = ModelVersion(
            version_id=version_id,
            parent_version=parent_version,
            model_hash=new_model_hash,
            timestamp=int(time.time()),
            round_number=self.current_round,
            contributor_id=client_id,
            update_type=UpdateType.LOCAL_TRAINING,
            metrics=metrics,
            metadata=metadata or {},
        )

        self._add_version(version)

        # Add edge
        edge = LineageEdge(
            from_version=parent_version,
            to_version=version_id,
            edge_type="update",
            weight=1.0,
            metadata={"client_id": client_id},
        )
        self._add_edge(edge)

        self.active_branches.add(version_id)

        logger.info(
            f"Recorded local update {version_id} from {client_id} "
            f"(parent: {parent_version}, round: {self.current_round})"
        )

        return version_id

    def record_aggregation(
        self,
        aggregator_id: str,
        input_versions: list[tuple[str, float]],
        aggregated_model_hash: str,
        aggregation_method: str,
        metrics: dict[str, float],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Record model aggregation operation.

        Args:
            aggregator_id: ID of the aggregator
            input_versions: List of (version_id, weight) tuples
            aggregated_model_hash: Hash of aggregated model
            aggregation_method: Method used (FedAvg, FedProx, etc.)
            metrics: Aggregation metrics
            metadata: Additional metadata

        Returns:
            Aggregated version ID
        """
        # Validate input versions
        for version_id, weight in input_versions:
            if version_id not in self.versions:
                raise ValueError(f"Input version {version_id} not found")

        # Generate aggregated version ID
        input_str = "".join(f"{v}{w}" for v, w in sorted(input_versions))
        version_data = f"agg{input_str}{aggregated_model_hash}{time.time()}"
        version_id = f"v{hashlib.sha256(version_data.encode()).hexdigest()[:8]}"

        # Create aggregated version
        version = ModelVersion(
            version_id=version_id,
            parent_version=None,  # Multiple parents
            model_hash=aggregated_model_hash,
            timestamp=int(time.time()),
            round_number=self.current_round,
            contributor_id=aggregator_id,
            update_type=UpdateType.AGGREGATION,
            metrics=metrics,
            metadata={
                "aggregation_method": aggregation_method,
                "input_count": len(input_versions),
                **(metadata or {}),
            },
        )

        self._add_version(version)

        # Add edges from all inputs
        for input_version, weight in input_versions:
            edge = LineageEdge(
                from_version=input_version,
                to_version=version_id,
                edge_type="aggregate",
                weight=weight,
                metadata={"aggregator_id": aggregator_id, "method": aggregation_method},
            )
            self._add_edge(edge)

            # Remove input from active branches
            self.active_branches.discard(input_version)

        self.active_branches.add(version_id)

        logger.info(
            f"Recorded aggregation {version_id} from {len(input_versions)} inputs "
            f"using {aggregation_method} (round: {self.current_round})"
        )

        return version_id

    def record_validation(
        self,
        validator_id: str,
        version_id: str,
        validation_metrics: dict[str, float],
        validation_passed: bool,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Record model validation checkpoint.

        Args:
            validator_id: ID of the validator
            version_id: Version being validated
            validation_metrics: Validation metrics
            validation_passed: Whether validation passed
            metadata: Additional metadata

        Returns:
            Validation version ID
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        # Generate validation version ID
        val_data = f"val{version_id}{validator_id}{time.time()}"
        val_version_id = f"v{hashlib.sha256(val_data.encode()).hexdigest()[:8]}"

        # Create validation version
        version = ModelVersion(
            version_id=val_version_id,
            parent_version=version_id,
            model_hash=self.versions[version_id].model_hash,  # Same model
            timestamp=int(time.time()),
            round_number=self.current_round,
            contributor_id=validator_id,
            update_type=UpdateType.VALIDATION,
            metrics=validation_metrics,
            metadata={
                "validation_passed": validation_passed,
                "validated_version": version_id,
                **(metadata or {}),
            },
        )

        self._add_version(version)

        # Add validation edge
        edge = LineageEdge(
            from_version=version_id,
            to_version=val_version_id,
            edge_type="validate",
            weight=1.0,
            metadata={"validator_id": validator_id, "passed": validation_passed},
        )
        self._add_edge(edge)

        if validation_passed:
            # Update active branch
            self.active_branches.discard(version_id)
            self.active_branches.add(val_version_id)

        logger.info(
            f"Recorded validation {val_version_id} for {version_id} "
            f"(passed: {validation_passed})"
        )

        return val_version_id

    def advance_round(self) -> int:
        """Advance to next federated learning round"""
        self.current_round += 1
        logger.info(f"Advanced to round {self.current_round}")
        return self.current_round

    def get_lineage_path(self, version_id: str) -> list[str]:
        """
        Get the lineage path from initial version to specified version.

        Args:
            version_id: Target version

        Returns:
            List of version IDs in lineage path
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        # Find shortest path from initial version
        try:
            path = nx.shortest_path(self.lineage_graph, "v0", version_id)
            return path
        except nx.NetworkXNoPath:
            # Handle multiple parent case (aggregation)
            # Find all ancestors
            ancestors = nx.ancestors(self.lineage_graph, version_id)
            ancestors.add(version_id)

            # Return topologically sorted ancestors
            subgraph = self.lineage_graph.subgraph(ancestors)
            return list(nx.topological_sort(subgraph))

    def get_version_provenance(self, version_id: str) -> dict[str, Any]:
        """
        Get complete provenance information for a model version.

        Args:
            version_id: Version to query

        Returns:
            Provenance information
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")

        version = self.versions[version_id]

        # Get ancestors
        ancestors = list(nx.ancestors(self.lineage_graph, version_id))

        # Get immediate parents
        parents = list(self.lineage_graph.predecessors(version_id))

        # Get contribution weights if aggregated
        contributions = {}
        if version.update_type == UpdateType.AGGREGATION:
            for parent in parents:
                edge_data = self.lineage_graph[parent][version_id]
                contributions[parent] = edge_data.get("weight", 1.0)

        provenance = {
            "version_id": version_id,
            "version_info": asdict(version),
            "lineage_depth": len(self.get_lineage_path(version_id)),
            "total_ancestors": len(ancestors),
            "immediate_parents": parents,
            "contributions": contributions,
            "is_validated": any(
                self.versions[v].update_type == UpdateType.VALIDATION
                for v in nx.descendants(self.lineage_graph, version_id)
            ),
            "downstream_versions": list(nx.descendants(self.lineage_graph, version_id)),
        }

        return provenance

    def detect_forks(self) -> list[tuple[str, list[str]]]:
        """
        Detect fork points in the lineage graph.

        Returns:
            List of (fork_point, branches) tuples
        """
        forks = []

        for node in self.lineage_graph.nodes():
            successors = list(self.lineage_graph.successors(node))

            # Fork if multiple successors that aren't aggregation
            non_agg_successors = [
                s for s in successors if self.versions[s].update_type != UpdateType.AGGREGATION
            ]

            if len(non_agg_successors) > 1:
                forks.append((node, non_agg_successors))

        return forks

    def visualize_lineage(
        self,
        highlight_versions: list[str] | None = None,
        save_path: str | None = None,
    ):
        """
        Visualize the model lineage graph.

        Args:
            highlight_versions: Versions to highlight
            save_path: Path to save visualization
        """
        plt.figure(figsize=(16, 12))

        # Create layout
        pos = nx.spring_layout(self.lineage_graph, k=2, iterations=50)

        # Color nodes by type
        node_colors = []
        node_sizes = []

        for node in self.lineage_graph.nodes():
            version = self.versions[node]

            if version.update_type == UpdateType.LOCAL_TRAINING:
                node_colors.append("lightblue")
                node_sizes.append(800)
            elif version.update_type == UpdateType.AGGREGATION:
                node_colors.append("lightgreen")
                node_sizes.append(1200)
            elif version.update_type == UpdateType.VALIDATION:
                node_colors.append("gold")
                node_sizes.append(1000)
            elif version.update_type == UpdateType.CHECKPOINT:
                node_colors.append("lightcoral")
                node_sizes.append(1500)
            else:
                node_colors.append("lightgray")
                node_sizes.append(600)

        # Draw nodes
        nx.draw_networkx_nodes(
            self.lineage_graph,
            pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
        )

        # Highlight specified versions
        if highlight_versions:
            highlight_nodes = [n for n in self.lineage_graph.nodes() if n in highlight_versions]
            nx.draw_networkx_nodes(
                self.lineage_graph,
                pos,
                nodelist=highlight_nodes,
                node_color="red",
                node_size=1500,
                alpha=0.5,
            )

        # Draw edges with different styles
        edge_styles = {
            "update": {"edge_color": "blue", "style": "solid", "width": 2},
            "aggregate": {"edge_color": "green", "style": "dashed", "width": 3},
            "validate": {"edge_color": "orange", "style": "dotted", "width": 2},
        }

        for edge_type, style in edge_styles.items():
            edges = [
                (u, v)
                for u, v, d in self.lineage_graph.edges(data=True)
                if d.get("edge_type") == edge_type
            ]

            if edges:
                nx.draw_networkx_edges(
                    self.lineage_graph,
                    pos,
                    edgelist=edges,
                    edge_color=style["edge_color"],
                    style=style["style"],
                    width=style["width"],
                    alpha=0.7,
                    arrows=True,
                    arrowsize=20,
                )

        # Draw labels
        labels = {}
        for node in self.lineage_graph.nodes():
            version = self.versions[node]
            labels[node] = f"{node[:4]}...\nR{version.round_number}"

        nx.draw_networkx_labels(self.lineage_graph, pos, labels, font_size=8, font_weight="bold")

        # Add legend
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="lightblue", label="Local Training"),
            Patch(facecolor="lightgreen", label="Aggregation"),
            Patch(facecolor="gold", label="Validation"),
            Patch(facecolor="lightcoral", label="Checkpoint"),
        ]
        plt.legend(handles=legend_elements, loc="upper right", fontsize=10)

        plt.title(f"Federated Model Lineage - {self.federation_id}", fontsize=16)
        plt.axis("off")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def export_lineage_dag(self) -> dict[str, Any]:
        """
        Export the lineage DAG in a serializable format.

        Returns:
            DAG representation
        """
        nodes = []
        edges = []

        # Export nodes
        for version_id, version in self.versions.items():
            nodes.append(
                {
                    "id": version_id,
                    "data": asdict(version),
                    "round": version.round_number,
                    "type": version.update_type.value,
                }
            )

        # Export edges
        for u, v, data in self.lineage_graph.edges(data=True):
            edges.append(
                {
                    "from": u,
                    "to": v,
                    "type": data.get("edge_type", "unknown"),
                    "weight": data.get("weight", 1.0),
                    "metadata": data.get("metadata", {}),
                }
            )

        return {
            "federation_id": self.federation_id,
            "current_round": self.current_round,
            "total_versions": len(self.versions),
            "active_branches": list(self.active_branches),
            "nodes": nodes,
            "edges": edges,
            "forks": self.detect_forks(),
            "export_time": int(time.time()),
        }

    def _add_version(self, version: ModelVersion):
        """Add a version to the lineage"""
        self.versions[version.version_id] = version
        self.lineage_graph.add_node(version.version_id, **asdict(version))

    def _add_edge(self, edge: LineageEdge):
        """Add an edge to the lineage graph"""
        self.lineage_graph.add_edge(
            edge.from_version,
            edge.to_version,
            edge_type=edge.edge_type,
            weight=edge.weight,
            metadata=edge.metadata,
        )

    def compute_lineage_hash(self) -> str:
        """
        Compute a hash of the entire lineage graph.

        Returns:
            Lineage hash
        """
        # Create deterministic representation
        nodes = sorted(self.versions.keys())
        edges = sorted(
            [
                (u, v, data["edge_type"], data["weight"])
                for u, v, data in self.lineage_graph.edges(data=True)
            ]
        )

        lineage_data = {
            "federation_id": self.federation_id,
            "nodes": nodes,
            "edges": edges,
            "rounds": self.current_round,
        }

        lineage_str = json.dumps(lineage_data, sort_keys=True)
        return hashlib.sha256(lineage_str.encode()).hexdigest()
