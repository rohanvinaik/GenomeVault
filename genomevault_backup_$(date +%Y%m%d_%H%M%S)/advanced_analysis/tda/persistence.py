"""
Topological Data Analysis for genomic structure analysis

"""

from __future__ import annotations

from dataclasses import dataclass
from scipy.spatial import distance_matrix
import networkx as nx

import numpy as np

MAX_HOMOLOGY_DIMENSION = 2
PERSISTENCE_THRESHOLD = 0.1


@dataclass
class PersistencePair:
    """Represents a birth-death pair in persistent homology"""

    birth: float
    death: float
    dimension: int
    representatives: list[int] | None = None

    @property
    def persistence(self) -> float:
        """Lifetime of the topological feature"""
        return self.death - self.birth if self.death != float("in") else float("inf")

    @property
    def midpoint(self) -> float:
        """Midpoint of the persistence interval"""
        if self.death == float("inf"):
            return self.birth
        return (self.birth + self.death) / 2


@dataclass
class PersistenceDiagram:
    """Collection of persistence pairs"""

    pairs: list[PersistencePair]
    dimension: int

    def filter_by_persistence(self, threshold: float) -> "PersistenceDiagram":
        """Filter pairs by persistence threshold"""
        filtered_pairs = [p for p in self.pairs if p.persistence >= threshold]
        return PersistenceDiagram(filtered_pairs, self.dimension)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array format"""
        if not self.pairs:
            return np.array([])
        return np.array([[p.birth, p.death] for p in self.pairs])


class TopologicalAnalyzer:
    """
    Performs topological data analysis on genomic hypervector representations
    """

    def __init__(self, max_dimension: int = MAX_HOMOLOGY_DIMENSION):
        """Initialize instance.

        Args:
            max_dimension: Dimension value.
        """
        self.max_dimension = max_dimension

    def compute_persistence_diagram(
        self, data: np.ndarray, max_scale: float = None
    ) -> dict[int, PersistenceDiagram]:
        """
        Compute persistence diagrams for genomic data

        Args:
            data: Point cloud data (n_samples, n_features)
            max_scale: Maximum scale for filtration

        Returns:
            Dictionary mapping dimension to persistence diagram
        """
        # Compute distance matrix
        distances = distance_matrix(data, data)

        if max_scale is None:
            max_scale = np.max(distances)

        # Build Vietoris-Rips filtration
        filtration = self._build_vietoris_rips(distances, max_scale)

        # Compute persistent homology
        persistence_diagrams = {}
        for dim in range(self.max_dimension + 1):
            diagram = self._compute_homology_persistence(filtration, dim)
            persistence_diagrams[dim] = diagram

        return persistence_diagrams

    def _build_vietoris_rips(
        self, distances: np.ndarray, max_scale: float
    ) -> list[tuple[float, list[int]]]:
        """Build Vietoris-Rips filtration from distance matrix"""
        n = distances.shape[0]
        filtration = []

        # Add vertices (0-simplices)
        for i in range(n):
            filtration.append((0.0, [i]))

        # Add edges (1-simplices)
        for i in range(n):
            for j in range(i + 1, n):
                if distances[i, j] <= max_scale:
                    filtration.append((distances[i, j], [i, j]))

        # Add triangles (2-simplices) if max_dimension >= 2
        if self.max_dimension >= 2:
            for i in range(n):
                for j in range(i + 1, n):
                    for k in range(j + 1, n):
                        # Check if all edges exist
                        max_edge = max(distances[i, j], distances[i, k], distances[j, k])
                        if max_edge <= max_scale:
                            filtration.append((max_edge, [i, j, k]))

        # Sort by filtration value
        filtration.sort(key=lambda x: x[0])

        return filtration

    def _compute_homology_persistence(
        self, filtration: list[tuple[float, list[int]]], dimension: int
    ) -> PersistenceDiagram:
        """Compute persistent homology in given dimension"""
        pairs = []

        if dimension == 0:
            # 0-dimensional persistence (connected components)
            pairs = self._compute_0d_persistence(filtration)
        elif dimension == 1:
            # 1-dimensional persistence (loops)
            pairs = self._compute_1d_persistence(filtration)
        elif dimension == 2:
            # 2-dimensional persistence (voids)
            pairs = self._compute_2d_persistence(filtration)

        return PersistenceDiagram(pairs, dimension)

    def _compute_0d_persistence(
        self, filtration: list[tuple[float, list[int]]]
    ) -> list[PersistencePair]:
        """Compute 0-dimensional persistence (connected components)"""
        pairs = []
        union_find = UnionFind()
        components_birth = {}

        for value, simplex in filtration:
            if len(simplex) == 1:
                # Vertex birth
                vertex = simplex[0]
                union_find.add(vertex)
                components_birth[vertex] = value
            elif len(simplex) == 2:
                # Edge connects components
                u, v = simplex
                root_u = union_find.find(u)
                root_v = union_find.find(v)

                if root_u != root_v:
                    # Components merge - one dies
                    union_find.union(u, v)

                    # The younger component dies
                    birth_u = components_birth[root_u]
                    birth_v = components_birth[root_v]

                    if birth_u < birth_v:
                        pairs.append(PersistencePair(birth_v, value, 0, [root_v]))
                        components_birth[union_find.find(u)] = birth_u
                    else:
                        pairs.append(PersistencePair(birth_u, value, 0, [root_u]))
                        components_birth[union_find.find(u)] = birth_v

        # Add infinite persistence for remaining components
        remaining_roots = set()
        for vertex in union_find.parent.keys():
            remaining_roots.add(union_find.find(vertex))

        for root in remaining_roots:
            pairs.append(PersistencePair(components_birth[root], float("inf"), 0, [root]))

        return pairs

    def _compute_1d_persistence(
        self, filtration: list[tuple[float, list[int]]]
    ) -> list[PersistencePair]:
        """Compute 1-dimensional persistence (loops)"""
        # Simplified implementation - would use boundary matrices in practice
        pairs = []

        # Build graph from edges
        edges = [(s[0], s[1], val) for val, s in filtration if len(s) == 2]
        triangles = [(s[0], s[1], s[2], val) for val, s in filtration if len(s) == 3]

        # Track when loops form and die
        G = nx.Graph()
        loop_births = {}

        for u, v, val in edges:
            if G.has_node(u) and G.has_node(v):
                # Check if edge creates a cycle
                if not nx.has_path(G, u, v):
                    G.add_edge(u, v, weight=val)
                else:
                    # Cycle formed - record birth
                    cycle = nx.shortest_path(G, u, v)
                    cycle_key = tuple(sorted(cycle + [u]))
                    if cycle_key not in loop_births:
                        loop_births[cycle_key] = val
                    G.add_edge(u, v, weight=val)
            else:
                G.add_edge(u, v, weight=val)

        # Triangles kill loops
        for i, j, k, val in triangles:
            # Check if triangle fills a loop
            for cycle_key, birth in list(loop_births.items()):
                if {i, j, k}.issubset(set(cycle_key)):
                    pairs.append(PersistencePair(birth, val, 1))
                    del loop_births[cycle_key]
                    break

        # Remaining loops have infinite persistence
        for cycle_key, birth in loop_births.items():
            pairs.append(PersistencePair(birth, float("inf"), 1))

        return pairs

    def _compute_2d_persistence(
        self, filtration: list[tuple[float, list[int]]]
    ) -> list[PersistencePair]:
        """
        Compute 2-dimensional persistence (voids) for genomic structural analysis.

        Args:
            filtration: Sorted list of simplices with their filtration values

        Returns:
            List of 2D persistence pairs representing topological voids
        """
        pairs = []

        # Extract tetrahedra (3-simplices) and triangles (2-simplices)
        triangles = [(val, s) for val, s in filtration if len(s) == 3]
        tetrahedra = [(val, s) for val, s in filtration if len(s) == 4]

        # Track 2D voids (cavities)
        void_births = {}

        # Triangles can create 2D voids
        for val, triangle in triangles:
            triangle_key = tuple(sorted(triangle))

            # Check if triangle is boundary of a void
            # In a simplified model, isolated triangles create voids
            neighboring_triangles = self._find_neighboring_triangles(triangle, triangles)

            if len(neighboring_triangles) < 3:  # Not fully surrounded
                void_births[triangle_key] = val

        # Tetrahedra can fill voids
        for val, tetrahedron in tetrahedra:
            tetrahedron_faces = self._get_tetrahedron_faces(tetrahedron)

            # Check if tetrahedron fills any existing void
            for face in tetrahedron_faces:
                face_key = tuple(sorted(face))
                if face_key in void_births:
                    birth_time = void_births[face_key]
                    pairs.append(PersistencePair(birth_time, val, 2, list(face)))
                    del void_births[face_key]
                    break

        # Remaining voids have infinite persistence
        for triangle_key, birth_time in void_births.items():
            pairs.append(PersistencePair(birth_time, float("inf"), 2, list(triangle_key)))

        return pairs

    def _find_neighboring_triangles(
        self, triangle: list[int], all_triangles: list[tuple[float, list[int]]]
    ) -> list[list[int]]:
        """Find triangles that share an edge with the given triangle"""
        neighbors = []
        triangle_set = set(triangle)

        for _, other_triangle in all_triangles:
            other_set = set(other_triangle)
            if len(triangle_set.intersection(other_set)) >= 2:  # Share at least an edge
                neighbors.append(other_triangle)

        return neighbors

    def _get_tetrahedron_faces(self, tetrahedron: list[int]) -> list[list[int]]:
        """Get the four triangular faces of a tetrahedron"""
        faces = []
        for i in range(4):
            face = [tetrahedron[j] for j in range(4) if j != i]
            faces.append(face)
        return faces

    def compute_bottleneck_distance(
        self, diagram1: PersistenceDiagram, diagram2: PersistenceDiagram
    ) -> float:
        """Compute bottleneck distance between two persistence diagrams"""
        arr1 = diagram1.to_array()
        arr2 = diagram2.to_array()

        if len(arr1) == 0 and len(arr2) == 0:
            return 0.0

        # Simple approximation - would use Hungarian algorithm in practice
        if len(arr1) == 0:
            return np.max([p.persistence for p in diagram2.pairs])
        if len(arr2) == 0:
            return np.max([p.persistence for p in diagram1.pairs])

        # Compute pairwise distances
        n1, n2 = len(arr1), len(arr2)
        cost_matrix = np.zeros((max(n1, n2), max(n1, n2)))

        for i in range(n1):
            for j in range(n2):
                cost_matrix[i, j] = max(abs(arr1[i, 0] - arr2[j, 0]), abs(arr1[i, 1] - arr2[j, 1]))

        # Add diagonal (deletion cost)
        for i in range(n1):
            for j in range(n2, max(n1, n2)):
                cost_matrix[i, j] = arr1[i, 1] - arr1[i, 0]

        for i in range(n1, max(n1, n2)):
            for j in range(n2):
                cost_matrix[i, j] = arr2[j, 1] - arr2[j, 0]

        return np.min(np.max(cost_matrix, axis=1))

    def compute_persistence_landscape(
        self, diagram: PersistenceDiagram, resolution: int = 100
    ) -> np.ndarray:
        """Compute persistence landscape from diagram"""
        if not diagram.pairs:
            return np.zeros((1, resolution))

        # Get range
        finite_pairs = [p for p in diagram.pairs if p.death != float("inf")]
        if not finite_pairs:
            return np.zeros((1, resolution))

        min_val = min(p.birth for p in finite_pairs)
        max_val = max(p.death for p in finite_pairs)

        # Create grid
        t = np.linspace(min_val, max_val, resolution)

        # Compute landscape functions
        landscapes = []
        for p in finite_pairs:
            landscape = np.zeros(resolution)
            for i, ti in enumerate(t):
                if p.birth <= ti <= p.death:
                    landscape[i] = min(ti - p.birth, p.death - ti)
            landscapes.append(landscape)

        # Stack and sort
        if landscapes:
            landscapes = np.vstack(landscapes)
            landscapes = -np.sort(-landscapes, axis=0)  # Sort descending
        else:
            landscapes = np.zeros((1, resolution))

        return landscapes


class UnionFind:
    """Union-Find data structure for connected components"""

    def __init__(self):
        """Initialize instance."""
        self.parent = {}
        self.rank = {}

    def add(self, x):
        """Add element to structure"""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        """Find root of element with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union two sets by rank"""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


class StructuralSignatureAnalyzer:
    """
    Analyzes topological signatures of DNA structures
    """

    def __init__(self):
        """Initialize instance."""
        self.analyzer = TopologicalAnalyzer()

    def compute_dna_structural_signature(
        self, contact_matrix: np.ndarray, threshold: float = 0.1
    ) -> dict[str, any]:
        """
        Compute topological signature from DNA contact matrix

        Args:
            contact_matrix: Hi-C or similar contact frequency matrix
            threshold: Contact threshold for building graph

        Returns:
            Structural signature including persistence diagrams
        """
        # Convert contact matrix to distance matrix
        distance_matrix = 1.0 / (contact_matrix + 1e-6)
        np.fill_diagonal(distance_matrix, 0)

        # Compute persistence diagrams
        persistence_diagrams = self.analyzer.compute_persistence_diagram(distance_matrix)

        # Extract topological features
        features = {
            "num_loops": len(
                [p for p in persistence_diagrams[1].pairs if p.persistence > PERSISTENCE_THRESHOLD]
            ),
            "num_domains": len(
                [p for p in persistence_diagrams[0].pairs if p.persistence > PERSISTENCE_THRESHOLD]
            ),
            "max_loop_persistence": max(
                [p.persistence for p in persistence_diagrams[1].pairs], default=0
            ),
            "total_persistence": sum(
                p.persistence
                for d in persistence_diagrams.values()
                for p in d.pairs
                if p.death != float("inf")
            ),
        }

        return {
            "persistence_diagrams": persistence_diagrams,
            "topological_features": features,
            "distance_matrix": distance_matrix,
        }

    def compare_structural_signatures(self, sig1: dict, sig2: dict) -> float:
        """Compare two structural signatures"""
        # Compare persistence diagrams
        distances = []
        for dim in range(self.analyzer.max_dimension + 1):
            d1 = sig1["persistence_diagrams"][dim]
            d2 = sig2["persistence_diagrams"][dim]
            dist = self.analyzer.compute_bottleneck_distance(d1, d2)
            distances.append(dist)

        # Weighted average of distances
        weights = [1.0, 2.0, 1.0]  # Weight loops more heavily
        weighted_distance = sum(w * d for w, d in zip(weights, distances)) / sum(weights)

        return weighted_distance
