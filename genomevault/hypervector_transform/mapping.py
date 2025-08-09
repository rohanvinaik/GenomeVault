"""
Similarity-preserving mappings for hypervector transformations

This module implements various mappings that preserve biological relationships
while transforming data into the hyperdimensional space.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from genomevault.core.constants import OmicsType
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MappingConfig:
    """Configuration for similarity-preserving mappings"""

    preserve_distances: bool = True
    preserve_angles: bool = True
    preserve_neighborhoods: bool = True
    neighborhood_size: int = 10
    learning_rate: float = 0.01
    num_iterations: int = 1000
    regularization: float = 0.001


class SimilarityPreservingMapper:
    """
    Implements mappings that preserve similarity relationships

    This is crucial for maintaining biological relationships when
    transforming data into hyperdimensional space.
    """

    def __init__(self, input_dim: int, output_dim: int, config: MappingConfig | None = None):
        """
        Initialize the mapper

        Args:
            input_dim: Input dimension
            output_dim: Output dimension (hypervector dimension)
            config: Mapping configuration
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config or MappingConfig()

        # Initialize mapping matrix
        self.mapping_matrix = None
        self._initialize_mapping()

        logger.info("Initialized mapper: %sinput_dimD -> %soutput_dimD")

    def _initialize_mapping(self):
        """Initialize the mapping matrix"""
        if self.output_dim >= self.input_dim:
            # Random orthogonal mapping for expansion
            q, _ = torch.qr(torch.randn(self.output_dim, self.input_dim))
            self.mapping_matrix = q
        else:
            # PCA-like mapping for reduction
            self.mapping_matrix = torch.randn(self.output_dim, self.input_dim)
            self.mapping_matrix /= torch.norm(self.mapping_matrix, dim=1, keepdim=True)

    def fit(self, data: torch.Tensor, similarity_matrix: torch.Tensor | None = None):
        """
        Fit the mapping to preserve similarities in the data

        Args:
            data: Input data matrix (n_samples x input_dim)
            similarity_matrix: Pre-computed similarity matrix (optional)
        """
        if similarity_matrix is None:
            # Compute pairwise similarities
            similarity_matrix = self._compute_similarities(data)

        # Optimize mapping to preserve similarities
        self._optimize_mapping(data, similarity_matrix)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Transform data using the learned mapping

        Args:
            data: Input data (n_samples x input_dim or input_dim)

        Returns:
            Transformed data in hyperdimensional space
        """
        if data.dim() == 1:
            # Single vector
            return torch.matmul(self.mapping_matrix, data)
        else:
            # Batch of vectors
            return torch.matmul(data, self.mapping_matrix.T)

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Fit and transform in one step"""
        self.fit(data)
        return self.transform(data)

    def _compute_similarities(self, data: torch.Tensor) -> torch.Tensor:
        """Compute pairwise similarity matrix"""
        # Normalize data
        normalized = data / torch.norm(data, dim=1, keepdim=True)

        # Compute cosine similarities
        similarities = torch.matmul(normalized, normalized.T)

        return similarities

    def _optimize_mapping(self, data: torch.Tensor, target_similarities: torch.Tensor):
        """Optimize mapping matrix to preserve similarities"""
        optimizer = torch.optim.Adam([self.mapping_matrix], lr=self.config.learning_rate)

        for iteration in range(self.config.num_iterations):
            # Transform data
            transformed = self.transform(data)

            # Compute similarities in transformed space
            transformed_similarities = self._compute_similarities(transformed)

            # Compute loss
            loss = 0

            if self.config.preserve_distances:
                # Distance preservation loss
                distance_loss = torch.mean((transformed_similarities - target_similarities) ** 2)
                loss += distance_loss

            if self.config.preserve_angles:
                # Angle preservation loss
                angle_loss = self._angle_preservation_loss(data, transformed)
                loss += angle_loss

            if self.config.preserve_neighborhoods:
                # Neighborhood preservation loss
                neighborhood_loss = self._neighborhood_preservation_loss(
                    target_similarities, transformed_similarities
                )
                loss += neighborhood_loss

            # Regularization
            loss += self.config.regularization * torch.norm(self.mapping_matrix)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Orthogonalize if expanding dimensions
            if self.output_dim >= self.input_dim and iteration % 10 == 0:
                self._orthogonalize_mapping()

            if iteration % 100 == 0:
                logger.debug("Iteration %siteration, Loss: %sloss.item():.4f")

    def _angle_preservation_loss(
        self, original: torch.Tensor, transformed: torch.Tensor
    ) -> torch.Tensor:
        """Compute angle preservation loss"""
        # Sample random triplets
        n = original.shape[0]
        num_triplets = min(n * 10, 1000)

        loss = 0
        for _ in range(num_triplets):
            # Random triplet
            indices = torch.randperm(n)[:3]
            o1, o2, o3 = original[indices]
            t1, t2, t3 = transformed[indices]

            # Original angles
            angle_o = self._compute_angle(o1 - o2, o3 - o2)
            angle_t = self._compute_angle(t1 - t2, t3 - t2)

            # Angle difference
            loss += (angle_o - angle_t) ** 2

        return loss / num_triplets

    def _compute_angle(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        """Compute angle between two vectors"""
        cos_angle = torch.dot(v1, v2) / (torch.norm(v1) * torch.norm(v2) + 1e-8)
        return torch.acos(torch.clamp(cos_angle, -1, 1))

    def _neighborhood_preservation_loss(
        self, original_sim: torch.Tensor, transformed_sim: torch.Tensor
    ) -> torch.Tensor:
        """Compute neighborhood preservation loss"""
        n = original_sim.shape[0]
        k = self.config.neighborhood_size

        # Get k-nearest neighbors in original space
        _, original_neighbors = torch.topk(original_sim, k=k + 1, dim=1)

        # Get k-nearest neighbors in transformed space
        _, transformed_neighbors = torch.topk(transformed_sim, k=k + 1, dim=1)

        # Compute overlap
        loss = 0
        for i in range(n):
            # Convert to sets (excluding self)
            orig_set = set(original_neighbors[i, 1:].tolist())
            trans_set = set(transformed_neighbors[i, 1:].tolist())

            # Jaccard distance
            intersection = len(orig_set & trans_set)
            union = len(orig_set | trans_set)
            jaccard = intersection / union if union > 0 else 0

            loss += 1 - jaccard

        return loss / n

    def _orthogonalize_mapping(self):
        """Orthogonalize the mapping matrix"""
        q, r = torch.qr(self.mapping_matrix.T)
        self.mapping_matrix = q.T


class BiologicalSimilarityMapper(SimilarityPreservingMapper):
    """
    Specialized mapper for biological data that preserves domain-specific similarities
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        omics_type: OmicsType,
        config: MappingConfig | None = None,
    ):
        """
        Initialize biological similarity mapper

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            omics_type: Type of omics data
            config: Mapping configuration
        """
        super().__init__(input_dim, output_dim, config)
        self.omics_type = omics_type
        self.similarity_functions = self._get_similarity_functions()

    def _get_similarity_functions(self) -> dict[str, Callable]:
        """Get omics-specific similarity functions"""
        if self.omics_type == OmicsType.GENOMIC:
            return {
                "variant": self._variant_similarity,
                "haplotype": self._haplotype_similarity,
                "structural": self._structural_similarity,
            }
        elif self.omics_type == OmicsType.TRANSCRIPTOMIC:
            return {
                "expression": self._expression_similarity,
                "coexpression": self._coexpression_similarity,
                "pathway": self._pathway_similarity,
            }
        elif self.omics_type == OmicsType.EPIGENOMIC:
            return {
                "methylation": self._methylation_similarity,
                "chromatin": self._chromatin_similarity,
            }
        else:
            return {"default": self._default_similarity}

    def compute_biological_similarity(
        self,
        data1: torch.Tensor,
        data2: torch.Tensor,
        similarity_type: str = "default",
    ) -> float:
        """
        Compute biological similarity between two data points

        Args:
            data1: First data point
            data2: Second data point
            similarity_type: Type of similarity to compute

        Returns:
            Similarity score
        """
        sim_func = self.similarity_functions.get(similarity_type, self._default_similarity)
        return sim_func(data1, data2)

    def _variant_similarity(self, v1: torch.Tensor, v2: torch.Tensor) -> float:
        """Compute similarity between variant profiles"""
        # Consider shared variants and allele frequencies
        shared = torch.min(v1, v2).sum()
        total = torch.max(v1, v2).sum()
        return (shared / total).item() if total > 0 else 0

    def _haplotype_similarity(self, h1: torch.Tensor, h2: torch.Tensor) -> float:
        """Compute haplotype similarity"""
        # Hamming distance normalized
        matches = (h1 == h2).float().mean()
        return matches.item()

    def _structural_similarity(self, s1: torch.Tensor, s2: torch.Tensor) -> float:
        """Compute structural variant similarity"""
        # Overlap coefficient for structural variants
        overlap = torch.min(torch.abs(s1), torch.abs(s2)).sum()
        min_sum = min(torch.abs(s1).sum(), torch.abs(s2).sum())
        return (overlap / min_sum).item() if min_sum > 0 else 0

    def _expression_similarity(self, e1: torch.Tensor, e2: torch.Tensor) -> float:
        """Compute gene expression similarity"""
        # Pearson correlation
        e1_centered = e1 - e1.mean()
        e2_centered = e2 - e2.mean()

        numerator = (e1_centered * e2_centered).sum()
        denominator = torch.sqrt((e1_centered**2).sum() * (e2_centered**2).sum())

        return (numerator / denominator).item() if denominator > 0 else 0

    def _coexpression_similarity(self, c1: torch.Tensor, c2: torch.Tensor) -> float:
        """Compute co-expression pattern similarity"""
        # Use rank correlation for robustness
        rank1 = torch.argsort(torch.argsort(c1)).float()
        rank2 = torch.argsort(torch.argsort(c2)).float()

        return self._expression_similarity(rank1, rank2)

    def _pathway_similarity(self, p1: torch.Tensor, p2: torch.Tensor) -> float:
        """Compute pathway activity similarity"""
        # Jaccard index for pathway membership
        active1 = p1 > 0.5
        active2 = p2 > 0.5

        intersection = (active1 & active2).sum().float()
        union = (active1 | active2).sum().float()

        return (intersection / union).item() if union > 0 else 0

    def _methylation_similarity(self, m1: torch.Tensor, m2: torch.Tensor) -> float:
        """Compute methylation pattern similarity"""
        # Beta-value difference
        diff = torch.abs(m1 - m2)
        similarity = 1 - diff.mean()
        return similarity.item()

    def _chromatin_similarity(self, c1: torch.Tensor, c2: torch.Tensor) -> float:
        """Compute chromatin accessibility similarity"""
        # Peak overlap
        threshold = 0.1
        peaks1 = c1 > threshold
        peaks2 = c2 > threshold

        overlap = (peaks1 & peaks2).sum().float()
        total = (peaks1 | peaks2).sum().float()

        return (overlap / total).item() if total > 0 else 0

    def _default_similarity(self, d1: torch.Tensor, d2: torch.Tensor) -> float:
        """Default similarity using cosine similarity"""
        return torch.nn.functional.cosine_similarity(d1.view(1, -1), d2.view(1, -1)).item()


class ManifoldPreservingMapper:
    """
    Mapper that preserves manifold structure of biological data
    """

    def __init__(self, input_dim: int, output_dim: int, n_neighbors: int = 15):
        """
        Initialize manifold-preserving mapper

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            n_neighbors: Number of neighbors for manifold approximation
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neighbors = n_neighbors
        self.embedding = None

        logger.info("Initialized ManifoldPreservingMapper: %sinput_dimD -> %soutput_dimD")

    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """
        Fit and transform data preserving manifold structure

        Args:
            data: Input data (n_samples x input_dim)

        Returns:
            Transformed data preserving manifold
        """
        n_samples = data.shape[0]

        # Compute k-nearest neighbor graph
        distances = torch.cdist(data, data)
        _, neighbors = torch.topk(distances, k=self.n_neighbors + 1, largest=False, dim=1)

        # Initialize random embedding
        self.embedding = torch.randn(n_samples, self.output_dim)
        self.embedding /= torch.norm(self.embedding, dim=1, keepdim=True)

        # Optimize to preserve local structure
        self._optimize_embedding(data, neighbors)

        return self.embedding

    def _optimize_embedding(
        self,
        data: torch.Tensor,
        neighbors: torch.Tensor,
        n_iterations: int = 500,
    ):
        """Optimize embedding to preserve local structure"""
        optimizer = torch.optim.Adam([self.embedding], lr=0.01)

        for iteration in range(n_iterations):
            loss = 0

            # For each point
            for i in range(data.shape[0]):
                # Get neighbors
                neighbor_indices = neighbors[i, 1:]  # Exclude self

                # Preserve distances to neighbors
                for j in neighbor_indices:
                    # Original distance
                    orig_dist = torch.norm(data[i] - data[j])

                    # Embedded distance
                    embed_dist = torch.norm(self.embedding[i] - self.embedding[j])

                    # Distance preservation loss
                    loss += (orig_dist - embed_dist) ** 2

            # Normalize
            loss /= data.shape[0] * self.n_neighbors

            # Add repulsion term to prevent collapse
            repulsion = 0
            for i in range(data.shape[0]):
                for j in range(i + 1, data.shape[0]):
                    if j not in neighbors[i]:
                        # Repel non-neighbors
                        dist = torch.norm(self.embedding[i] - self.embedding[j])
                        repulsion -= torch.log(dist + 1e-8)

            loss += 0.01 * repulsion / (data.shape[0] ** 2)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % 100 == 0:
                logger.debug("Manifold optimization iteration %siteration, Loss: %sloss.item():.4f")


# Convenience functions
def create_biological_mapper(
    input_dim: int, output_dim: int, omics_type: OmicsType
) -> BiologicalSimilarityMapper:
    """Create a biological similarity mapper"""
    return BiologicalSimilarityMapper(input_dim, output_dim, omics_type)


def preserve_similarities(data: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Transform data while preserving similarities"""
    mapper = SimilarityPreservingMapper(data.shape[1], target_dim)
    return mapper.fit_transform(data)
