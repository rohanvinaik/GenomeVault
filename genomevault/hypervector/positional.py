"""
Positional encoding for single-nucleotide accuracy
Implements sparse, memory-efficient position vectors for SNP-level granularity
"""

import hashlib

import numpy as np
import torch

from genomevault.core.exceptions import HypervectorError
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


class PositionalEncoder:
    """
    Memory-efficient positional encoding for genomic positions
    Uses sparse representations and hash-based seeds for 10M+ positions
    """

    def __init__(self, dimension: int = 100000, sparsity: float = 0.01, cache_size: int = 10000):
        """
        Initialize positional encoder

        Args:
            dimension: Hypervector dimension
            sparsity: Sparsity level for position vectors (default 1%)
            cache_size: Number of position vectors to cache
        """
        self.dimension = dimension
        self.sparsity = sparsity
        self.cache_size = cache_size
        self._cache: dict[int, torch.Tensor] = {}

        # Pre-compute constants for efficiency
        self.nnz = int(dimension * sparsity)  # Number of non-zero elements
        logger.info(f"Initialized PositionalEncoder: {dimension}D, {self.nnz} non-zeros per vector")

    def make_position_vector(self, position: int, seed: int | None = None) -> torch.Tensor:
        """
        Generate orthogonal position vector using hash-based seed

        Args:
            position: Genomic position
            seed: Additional seed for determinism

        Returns:
            Sparse position hypervector
        """
        # Check cache first
        cache_key = (position, seed)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Generate deterministic seed from position
        if seed is None:
            seed = self._position_to_seed(position)
        else:
            seed = self._position_to_seed(position) ^ seed

        # Create sparse vector
        vec = self._create_sparse_vector(seed)

        # Cache if space available
        if len(self._cache) < self.cache_size:
            self._cache[cache_key] = vec

        return vec

    def make_position_vectors_batch(
        self, positions: list[int], seed: int | None = None
    ) -> torch.Tensor:
        """
        Generate batch of position vectors efficiently

        Args:
            positions: List of genomic positions
            seed: Additional seed for determinism

        Returns:
            Tensor of shape (len(positions), dimension)
        """
        vectors = []
        for pos in positions:
            vec = self.make_position_vector(pos, seed)
            vectors.append(vec)

        return torch.stack(vectors)

    def encode_snp_positions(
        self, chromosome: str, positions: list[int], panel_name: str = "default"
    ) -> torch.Tensor:
        """
        Encode a set of SNP positions into a single panel hypervector

        Args:
            chromosome: Chromosome name
            positions: List of SNP positions
            panel_name: Name of the SNP panel

        Returns:
            Combined panel hypervector
        """
        try:
            # Generate chromosome-specific seed
            chr_seed = self._chromosome_to_seed(chromosome)

            # Encode each position
            position_vectors = []
            for pos in positions:
                # Combine chromosome and panel seeds
                combined_seed = chr_seed ^ hash(panel_name)
                vec = self.make_position_vector(pos, combined_seed)
                position_vectors.append(vec)

            # Bundle positions using XOR (preserves orthogonality)
            if position_vectors:
                # Stack and XOR for bundling
                stacked = torch.stack(position_vectors)
                # Use sign to binarize, then XOR
                binary = (stacked > 0).float()
                xor_sum = binary.sum(dim=0) % 2
                panel_hv = xor_sum * 2 - 1  # Convert back to {-1, 1}

                return panel_hv
            else:
                return torch.zeros(self.dimension)

        except Exception as e:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            raise HypervectorError(f"Failed to encode SNP positions: {e!s}")
            raise

    def _position_to_seed(self, position: int) -> int:
        """Convert genomic position to deterministic seed"""
        # Use SHA256 for good distribution
        pos_bytes = str(position).encode()
        hash_bytes = hashlib.sha256(pos_bytes).digest()
        # Use first 4 bytes as seed
        return int.from_bytes(hash_bytes[:4], byteorder="big")

    def _chromosome_to_seed(self, chromosome: str) -> int:
        """Convert chromosome to deterministic seed"""
        # Normalize chromosome name
        chr_norm = chromosome.lower().replace("chr", "")

        # Map to numeric value
        chr_map = {
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "6": 6,
            "7": 7,
            "8": 8,
            "9": 9,
            "10": 10,
            "11": 11,
            "12": 12,
            "13": 13,
            "14": 14,
            "15": 15,
            "16": 16,
            "17": 17,
            "18": 18,
            "19": 19,
            "20": 20,
            "21": 21,
            "22": 22,
            "x": 23,
            "y": 24,
            "m": 25,
            "mt": 25,
        }

        chr_num = chr_map.get(chr_norm, 26)
        return chr_num * 1000000  # Space out chromosome seeds

    def _create_sparse_vector(self, seed: int) -> torch.Tensor:
        """Create sparse hypervector from seed"""
        # Set random state
        rng = np.random.RandomState(seed)

        # Create sparse vector
        vec = torch.zeros(self.dimension)

        # Random positions
        indices = rng.choice(self.dimension, size=self.nnz, replace=False)

        # Random values from {-1, +1}
        values = rng.choice([-1.0, 1.0], size=self.nnz)

        vec[indices] = torch.tensor(values)

        return vec

    def get_memory_usage(self) -> dict[str, float]:
        """Get memory usage statistics"""
        cache_size_mb = sum(vec.element_size() * vec.numel() for vec in self._cache.values()) / (
            1024**2
        )

        return {
            "cache_entries": len(self._cache),
            "cache_size_mb": cache_size_mb,
            "max_cache_size": self.cache_size,
            "dimension": self.dimension,
            "sparsity": self.sparsity,
            "nnz_per_vector": self.nnz,
        }

    def clear_cache(self):
        """Clear the position vector cache"""
        self._cache.clear()
        logger.info("Cleared position vector cache")


class SNPPanel:
    """
    Manages SNP panels for variant encoding
    """

    def __init__(self, positional_encoder: PositionalEncoder):
        """
        Initialize SNP panel manager

        Args:
            positional_encoder: PositionalEncoder instance
        """
        self.encoder = positional_encoder
        self.panels: dict[str, dict] = {}

        # Initialize default panels
        self._init_default_panels()

    def _init_default_panels(self):
        """Initialize default SNP panels"""
        # Common SNPs panel (example positions)
        self.panels["common"] = {
            "name": "Common SNPs",
            "size": 1000000,
            "description": "Common variants (MAF > 5%)",
            "positions": {},  # Would be loaded from file
        }

        # Clinical panel
        self.panels["clinical"] = {
            "name": "ClinVar/dbSNP",
            "size": 10000000,
            "description": "Clinical and dbSNP variants",
            "positions": {},  # Would be loaded from file
        }

        logger.info(f"Initialized {len(self.panels)} default SNP panels")

    def load_panel_from_file(self, panel_name: str, file_path: str, file_type: str = "bed"):
        """
        Load SNP panel from BED/VCF file

        Args:
            panel_name: Name for the panel
            file_path: Path to BED/VCF file
            file_type: File type ('bed' or 'vcf')
        """
        positions_by_chr = {}

        try:
            if file_type == "bed":
                # Parse BED file
                with open(file_path) as f:
                    for line in f:
                        if line.startswith("#"):
                            continue
                        parts = line.strip().split("\t")
                        if len(parts) >= 3:
                            chrom = parts[0]
                            start = int(parts[1])
                            end = int(parts[2])

                            if chrom not in positions_by_chr:
                                positions_by_chr[chrom] = []

                            # Add all positions in range
                            positions_by_chr[chrom].extend(range(start, end + 1))

            elif file_type == "vcf":
                # Parse VCF file
                with open(file_path) as f:
                    for line in f:
                        if line.startswith("#"):
                            continue
                        parts = line.strip().split("\t")
                        if len(parts) >= 2:
                            chrom = parts[0]
                            pos = int(parts[1])

                            if chrom not in positions_by_chr:
                                positions_by_chr[chrom] = []

                            positions_by_chr[chrom].append(pos)

            # Count total positions
            total_positions = sum(len(positions) for positions in positions_by_chr.values())

            # Store panel
            self.panels[panel_name] = {
                "name": panel_name,
                "size": total_positions,
                "description": f"Custom panel from {file_path}",
                "positions": positions_by_chr,
                "file_path": file_path,
            }

            logger.info(f"Loaded panel '{panel_name}' with {total_positions} positions")

        except Exception as e:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            raise HypervectorError(f"Failed to load panel from {file_path}: {e!s}")
            raise

    def encode_with_panel(
        self, panel_name: str, chromosome: str, observed_bases: dict[int, str]
    ) -> torch.Tensor:
        """
        Encode observed bases using specified SNP panel

        Args:
            panel_name: Name of the SNP panel to use
            chromosome: Chromosome name
            observed_bases: Dict mapping position -> observed base

        Returns:
            Panel-encoded hypervector
        """
        if panel_name not in self.panels:
            raise ValueError(f"Unknown panel: {panel_name}")

        panel = self.panels[panel_name]

        # Get panel positions for this chromosome
        if chromosome not in panel.get("positions", {}):
            logger.warning(f"No positions for {chromosome} in panel {panel_name}")
            return torch.zeros(self.encoder.dimension)

        panel_positions = panel["positions"][chromosome]

        # Encode observed positions that are in the panel
        encoded_positions = []
        for pos in panel_positions:
            if pos in observed_bases:
                # Encode this position with its base
                pos_vec = self.encoder.make_position_vector(pos)
                base_vec = self._encode_base(observed_bases[pos])

                # Bind position and base
                bound_vec = self._bind_vectors(pos_vec, base_vec)
                encoded_positions.append(bound_vec)

        # Bundle all encoded positions
        if encoded_positions:
            return self._bundle_vectors(encoded_positions)
        else:
            return torch.zeros(self.encoder.dimension)

    def _encode_base(self, base: str) -> torch.Tensor:
        """Encode nucleotide base"""
        base_seeds = {"A": 1, "T": 2, "G": 3, "C": 4, "N": 5}
        seed = base_seeds.get(base.upper(), 5)
        return self.encoder._create_sparse_vector(seed * 1000)

    def _bind_vectors(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """Bind two hypervectors using circular convolution"""
        # For sparse vectors, use element-wise multiplication
        return vec1 * vec2

    def _bundle_vectors(self, vectors: list[torch.Tensor]) -> torch.Tensor:
        """Bundle multiple vectors using XOR-like operation"""
        stacked = torch.stack(vectors)
        # Majority vote for each dimension
        bundled = torch.sign(stacked.sum(dim=0))
        # Handle zeros
        bundled[bundled == 0] = 1
        return bundled

    def get_panel_info(self, panel_name: str) -> dict:
        """Get information about a panel"""
        if panel_name not in self.panels:
            raise ValueError(f"Unknown panel: {panel_name}")

        panel = self.panels[panel_name]
        return {
            "name": panel["name"],
            "size": panel["size"],
            "description": panel["description"],
            "chromosomes": list(panel.get("positions", {}).keys()),
            "file_path": panel.get("file_path", "built-in"),
        }

    def list_panels(self) -> list[str]:
        """List available panel names"""
        return list(self.panels.keys())
