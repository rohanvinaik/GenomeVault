"""
Reference data manager for PIR system.
Handles pangenome graphs, annotations, and population-specific data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import gzip
import json
import time

from genomevault.utils.logging import get_logger, logger

logger = get_logger(__name__)


class ReferenceDataType(Enum):
    """Types of reference data."""

    PANGENOME_GRAPH = "pangenome_graph"
    VARIANT_ANNOTATIONS = "variant_annotations"
    POPULATION_FREQUENCIES = "population_frequencies"
    GENE_ANNOTATIONS = "gene_annotations"
    REGULATORY_ELEMENTS = "regulatory_elements"
    CLINICAL_VARIANTS = "clinical_variants"


@dataclass
class GenomicRegion:
    """Genomic region specification."""

    chromosome: str
    start: int
    end: int

    def __str__(self):
        """Return string representation."""
        return "{self.chromosome}:{self.start}-{self.end}"

    def overlaps(self, other: "GenomicRegion") -> bool:
        """Check if regions overlap."""
        return (
            self.chromosome == other.chromosome
            and self.start < other.end
            and self.end > other.start
        )


@dataclass
class PangenomeNode:
    """Node in pangenome graph."""

    node_id: int
    sequence: str
    chromosome: str
    position: int
    populations: set[str] = field(default_factory=set)
    frequency: float = 0.0

    def to_bytes(self) -> bytes:
        """Convert to bytes for PIR storage."""
        data = {
            "id": self.node_id,
            "seq": self.sequence,
            "chr": self.chromosome,
            "pos": self.position,
            "pop": list(self.populations),
            "freq": self.frequency,
        }
        return json.dumps(data).encode()


@dataclass
class PangenomeEdge:
    """Edge in pangenome graph."""

    source_id: int
    target_id: int
    support: int  # Number of genomes with this edge
    populations: set[str] = field(default_factory=set)


@dataclass
class VariantAnnotation:
    """Annotation for a genetic variant."""

    chromosome: str
    position: int
    ref_allele: str
    alt_allele: str
    gene_impact: str | None = None
    protein_change: str | None = None
    conservation_score: float | None = None
    pathogenicity_score: float | None = None
    population_frequencies: dict[str, float] = field(default_factory=dict)
    clinical_significance: str | None = None

    def to_bytes(self) -> bytes:
        """Convert to bytes for PIR storage."""
        data = {
            "chr": self.chromosome,
            "pos": self.position,
            "ref": self.ref_allele,
            "alt": self.alt_allele,
            "impact": self.gene_impact,
            "protein": self.protein_change,
            "cons": self.conservation_score,
            "path": self.pathogenicity_score,
            "freq": self.population_frequencies,
            "clin": self.clinical_significance,
        }
        return json.dumps(data).encode()


class ReferenceDataManager:
    """
    Manages reference data for PIR queries.
    Handles pangenome graphs, annotations, and population data.
    """

    def __init__(self, data_directory: Path):
        """
        Initialize reference data manager.

        Args:
            data_directory: Directory for reference data storage
        """
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)

        # Pangenome graph
        self.nodes: dict[int, PangenomeNode] = {}
        self.edges: list[PangenomeEdge] = []
        self.node_index: dict[tuple[str, int], list[int]] = {}  # (chr, pos) -> node_ids

        # Annotations
        self.variant_annotations: dict[str, VariantAnnotation] = {}
        self.gene_annotations: dict[str, dict] = {}

        # Indexing structures
        self.region_index: dict[str, list[int]] = {}  # chr -> sorted positions
        self.population_nodes: dict[str, set[int]] = {}  # population -> node_ids

        # Metadata
        self.metadata = {
            "version": "1.0",
            "created": time.time(),
            "populations": set(),
            "total_nodes": 0,
            "total_edges": 0,
            "total_variants": 0,
        }

        # Load existing data
        self._load_reference_data()

        logger.info(f"ReferenceDataManager initialized with {len(self.nodes)} nodes")

    def _load_reference_data(self) -> None:
        """Load reference data from disk."""
        # Load pangenome graph
        graph_path = self.data_directory / "pangenome_graph.json.gz"
        if graph_path.exists():
            with gzip.open(graph_path, "rt") as f:
                graph_data = json.load(f)
                self._load_graph_from_dict(graph_data)

        # Load annotations
        annotations_path = self.data_directory / "variant_annotations.json.gz"
        if annotations_path.exists():
            with gzip.open(annotations_path, "rt") as f:
                annotations_data = json.load(f)
                self._load_annotations_from_dict(annotations_data)

    def _load_graph_from_dict(self, data: dict) -> None:
        """Load pangenome graph from dictionary."""
        # Load nodes
        for node_data in data.get("nodes", []):
            node = PangenomeNode(
                node_id=node_data["id"],
                sequence=node_data["seq"],
                chromosome=node_data["chr"],
                position=node_data["pos"],
                populations=set(node_data.get("pop", [])),
                frequency=node_data.get("freq", 0.0),
            )
            self.add_node(node)

        # Load edges
        for edge_data in data.get("edges", []):
            edge = PangenomeEdge(
                source_id=edge_data["source"],
                target_id=edge_data["target"],
                support=edge_data.get("support", 1),
                populations=set(edge_data.get("pop", [])),
            )
            self.add_edge(edge)

        # Update metadata
        if "metadata" in data:
            self.metadata.update(data["metadata"])

    def _load_annotations_from_dict(self, data: dict) -> None:
        """Load variant annotations from dictionary."""
        for var_key, ann_data in data.items():
            annotation = VariantAnnotation(
                chromosome=ann_data["chr"],
                position=ann_data["pos"],
                ref_allele=ann_data["ref"],
                alt_allele=ann_data["alt"],
                gene_impact=ann_data.get("impact"),
                protein_change=ann_data.get("protein"),
                conservation_score=ann_data.get("cons"),
                pathogenicity_score=ann_data.get("path"),
                population_frequencies=ann_data.get("freq", {}),
                clinical_significance=ann_data.get("clin"),
            )
            self.variant_annotations[var_key] = annotation

    def add_node(self, node: PangenomeNode) -> None:
        """Add node to pangenome graph."""
        self.nodes[node.node_id] = node

        # Update index
        key = (node.chromosome, node.position)
        if key not in self.node_index:
            self.node_index[key] = []
        self.node_index[key].append(node.node_id)

        # Update population index
        for pop in node.populations:
            if pop not in self.population_nodes:
                self.population_nodes[pop] = set()
            self.population_nodes[pop].add(node.node_id)
            self.metadata["populations"].add(pop)

        # Update region index
        if node.chromosome not in self.region_index:
            self.region_index[node.chromosome] = []
        self.region_index[node.chromosome].append(node.position)

        self.metadata["total_nodes"] += 1

    def add_edge(self, edge: PangenomeEdge) -> None:
        """Add edge to pangenome graph."""
        self.edges.append(edge)

        # Update population tracking
        for pop in edge.populations:
            self.metadata["populations"].add(pop)

        self.metadata["total_edges"] += 1

    def add_variant_annotation(self, annotation: VariantAnnotation) -> None:
        """Add variant annotation."""
        key = "{annotation.chromosome}:{annotation.position}:{annotation.ref_allele}:{annotation.alt_allele}"
        self.variant_annotations[key] = annotation
        self.metadata["total_variants"] += 1

    def get_nodes_in_region(
        self, region: GenomicRegion, population: str | None = None
    ) -> list[PangenomeNode]:
        """
        Get nodes in a genomic region.

        Args:
            region: Genomic region to query
            population: Optional population filter

        Returns:
            List of nodes in the region
        """
        nodes = []

        # Get positions in region
        if region.chromosome in self.region_index:
            positions = self.region_index[region.chromosome]

            # Binary search for start position
            import bisect

            start_idx = bisect.bisect_left(positions, region.start)
            end_idx = bisect.bisect_right(positions, region.end)

            # Get nodes at these positions
            for pos in positions[start_idx:end_idx]:
                key = (region.chromosome, pos)
                if key in self.node_index:
                    for node_id in self.node_index[key]:
                        node = self.nodes[node_id]

                        # Apply population filter if specified
                        if population is None or population in node.populations:
                            nodes.append(node)

        return nodes

    def get_variant_annotation(
        self, chromosome: str, position: int, ref_allele: str, alt_allele: str
    ) -> VariantAnnotation | None:
        """Get annotation for a specific variant."""
        key = "{chromosome}:{position}:{ref_allele}:{alt_allele}"
        return self.variant_annotations.get(key)

    def prepare_for_pir(self, data_type: ReferenceDataType) -> list[bytes]:
        """
        Prepare reference data for PIR storage.

        Args:
            data_type: Type of reference data to prepare

        Returns:
            List of serialized data items
        """
        items = []

        if data_type == ReferenceDataType.PANGENOME_GRAPH:
            # Serialize nodes
            for node in self.nodes.values():
                items.append(node.to_bytes())

        elif data_type == ReferenceDataType.VARIANT_ANNOTATIONS:
            # Serialize annotations
            for annotation in self.variant_annotations.values():
                items.append(annotation.to_bytes())

        elif data_type == ReferenceDataType.POPULATION_FREQUENCIES:
            # Aggregate population frequencies
            pop_data = self._aggregate_population_data()
            for item in pop_data:
                items.append(json.dumps(item).encode())

        logger.info(f"Prepared {len(items)} items of type {data_type.value} for PIR")
        return items

    def _aggregate_population_data(self) -> list[dict]:
        """Aggregate population-specific data."""
        pop_data = []

        for population in self.metadata["populations"]:
            # Get nodes for this population
            pop_nodes = self.population_nodes.get(population, set())

            # Calculate statistics
            data = {
                "population": population,
                "node_count": len(pop_nodes),
                "unique_variants": 0,
                "avg_frequency": 0.0,
            }

            # Count unique variants
            unique_positions = set()
            total_freq = 0.0

            for node_id in pop_nodes:
                node = self.nodes[node_id]
                unique_positions.add((node.chromosome, node.position))
                total_freq += node.frequency

            data["unique_variants"] = len(unique_positions)
            data["avg_frequency"] = total_freq / len(pop_nodes) if pop_nodes else 0.0

            pop_data.append(data)

        return pop_data

    def create_pir_index(self, region: GenomicRegion) -> dict[str, Any]:
        """
        Create PIR index for a genomic region.

        Args:
            region: Region to index

        Returns:
            Index structure for PIR queries
        """
        nodes = self.get_nodes_in_region(region)

        # Create position-based index
        index = {
            "region": str(region),
            "node_count": len(nodes),
            "positions": {},
            "populations": {},
        }

        # Index by position
        for i, node in enumerate(nodes):
            pos_key = "{node.chromosome}:{node.position}"
            if pos_key not in index["positions"]:
                index["positions"][pos_key] = []
            index["positions"][pos_key].append(i)

            # Index by population
            for pop in node.populations:
                if pop not in index["populations"]:
                    index["populations"][pop] = []
                index["populations"][pop].append(i)

        return index

    def save_reference_data(self) -> None:
        """Save reference data to disk."""
        # Save pangenome graph
        graph_data = {
            "nodes": [
                {
                    "id": node.node_id,
                    "seq": node.sequence,
                    "chr": node.chromosome,
                    "pos": node.position,
                    "pop": list(node.populations),
                    "freq": node.frequency,
                }
                for node in self.nodes.values()
            ],
            "edges": [
                {
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "support": edge.support,
                    "pop": list(edge.populations),
                }
                for edge in self.edges
            ],
            "metadata": self.metadata,
        }

        graph_path = self.data_directory / "pangenome_graph.json.gz"
        with gzip.open(graph_path, "wt") as f:
            json.dump(graph_data, f)

        # Save annotations
        annotations_data = {
            key: {
                "chr": ann.chromosome,
                "pos": ann.position,
                "ref": ann.ref_allele,
                "alt": ann.alt_allele,
                "impact": ann.gene_impact,
                "protein": ann.protein_change,
                "cons": ann.conservation_score,
                "path": ann.pathogenicity_score,
                "freq": ann.population_frequencies,
                "clin": ann.clinical_significance,
            }
            for key, ann in self.variant_annotations.items()
        }

        annotations_path = self.data_directory / "variant_annotations.json.gz"
        with gzip.open(annotations_path, "wt") as f:
            json.dump(annotations_data, f)

        logger.info("Reference data saved")

    def get_statistics(self) -> dict[str, Any]:
        """Get reference data statistics."""
        # Calculate size metrics
        total_sequence_length = sum(len(node.sequence) for node in self.nodes.values())

        # Population diversity
        pop_stats = {}
        for pop, node_ids in self.population_nodes.items():
            pop_stats[pop] = {
                "nodes": len(node_ids),
                "percentage": (len(node_ids) / len(self.nodes) * 100 if self.nodes else 0),
            }

        # Chromosome distribution
        chr_stats = {}
        for node in self.nodes.values():
            if node.chromosome not in chr_stats:
                chr_stats[node.chromosome] = 0
            chr_stats[node.chromosome] += 1

        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_variants": len(self.variant_annotations),
            "total_sequence_length": total_sequence_length,
            "populations": len(self.metadata["populations"]),
            "population_stats": pop_stats,
            "chromosome_distribution": chr_stats,
            "metadata": self.metadata,
        }


# Example usage
if __name__ == "__main__":
    import tempfile

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = ReferenceDataManager(Path(temp_dir))

        # Add some test nodes
        for i in range(100):
            node = PangenomeNode(
                node_id=i,
                sequence="ACGT" * 10,
                chromosome="chr1",
                position=1000000 + i * 100,
                populations={"EUR", "AFR"} if i % 2 == 0 else {"EAS", "SAS"},
                frequency=0.1 + (i % 10) * 0.05,
            )
            manager.add_node(node)

        # Add some edges
        for i in range(99):
            edge = PangenomeEdge(
                source_id=i, target_id=i + 1, support=50, populations={"EUR", "AFR"}
            )
            manager.add_edge(edge)

        # Add some annotations
        for i in range(50):
            annotation = VariantAnnotation(
                chromosome="chr1",
                position=1000000 + i * 200,
                ref_allele="A",
                alt_allele="G",
                gene_impact="MODERATE",
                conservation_score=0.8,
                pathogenicity_score=0.3,
                population_frequencies={"EUR": 0.1, "AFR": 0.05},
            )
            manager.add_variant_annotation(annotation)

        # Query a region
        region = GenomicRegion("chr1", 1000000, 1010000)
        nodes = manager.get_nodes_in_region(region)
        logger.info(f"Found {len(nodes)} nodes in {region}")

        # Prepare for PIR
        pir_data = manager.prepare_for_pir(ReferenceDataType.PANGENOME_GRAPH)
        logger.info(f"Prepared {len(pir_data)} items for PIR")

        # Get statistics
        stats = manager.get_statistics()
        logger.info(f"\nStatistics: {json.dumps(stats}, indent=2)")

        # Save data
        manager.save_reference_data()
