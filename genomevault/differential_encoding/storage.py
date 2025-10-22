"""
Storage and Serialization for Differential Encoding

This module provides efficient storage and serialization for encoded genomes,
including hypervectors, metadata, and verification information.

Section 7.2 of the specification.
"""

import gzip
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import hashlib

import numpy as np

from genomevault.differential_encoding.metadata import DifferentialEncodingMetadata

logger = logging.getLogger(__name__)


@dataclass
class EncodedGenome:
    """
    Complete encoded genome representation.

    Contains all hypervectors, metadata, and verification information needed
    to store and retrieve a differentially encoded genome.

    The encoded genome can be serialized to compressed JSON for efficient storage,
    achieving significant compression ratios compared to raw VCF files.

    Attributes:
        genome_id: Unique identifier for the genome
        assembly: Reference assembly (e.g., "GRCh38")
        bundled_hypervector: Bundled genome-level hypervector (10,000D)
        chunk_hypervectors: Individual chunk hypervectors
        metadata: Metadata for each chunk
        statistics: Encoding statistics
        master_seed: Master seed used for encoding (for reproducibility)
        encoding_hash: Hash of the encoding for verification
        created_at: Timestamp when encoding was created
        version: Encoding version for compatibility

    Example:
        >>> result = pipeline.encode_experimental_genome(genome, AnalysisType.SLIDING_WINDOW)
        >>> encoded = EncodedGenome.from_encoding_result(
        ...     genome_id=genome.genome_id,
        ...     assembly=genome.assembly,
        ...     result=result,
        ...     master_seed=b"seed" * 8,
        ... )
        >>> encoded.save("patient_001.enc.gz")
        >>> loaded = EncodedGenome.load("patient_001.enc.gz")
        >>> print(f"Storage size: {loaded.storage_size_kb():.2f} KB")
    """

    genome_id: str
    assembly: str
    bundled_hypervector: np.ndarray
    chunk_hypervectors: List[np.ndarray]
    metadata: List[DifferentialEncodingMetadata]
    statistics: Dict[str, Any]
    master_seed: bytes
    encoding_hash: str
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

    def __post_init__(self):
        """Validate data after initialization."""
        if len(self.chunk_hypervectors) != len(self.metadata):
            raise ValueError(
                f"Mismatch: {len(self.chunk_hypervectors)} hypervectors but "
                f"{len(self.metadata)} metadata entries"
            )

        # Ensure bundled hypervector is numpy array
        if not isinstance(self.bundled_hypervector, np.ndarray):
            self.bundled_hypervector = np.array(self.bundled_hypervector, dtype=np.float32)

        # Ensure chunk hypervectors are numpy arrays
        self.chunk_hypervectors = [
            np.array(hv, dtype=np.float32) if not isinstance(hv, np.ndarray) else hv
            for hv in self.chunk_hypervectors
        ]

    @classmethod
    def from_encoding_result(
        cls,
        genome_id: str,
        assembly: str,
        result: Any,  # EncodingResult
        master_seed: bytes,
    ) -> "EncodedGenome":
        """
        Create EncodedGenome from an EncodingResult.

        Args:
            genome_id: Genome identifier
            assembly: Reference assembly
            result: EncodingResult from pipeline
            master_seed: Master seed used for encoding

        Returns:
            EncodedGenome instance

        Raises:
            ValueError: If bundled hypervector is not available
        """
        if result.bundled_hypervector is None:
            raise ValueError("Encoding result must have bundled hypervector")

        # Compute encoding hash
        encoding_hash = cls._compute_encoding_hash(
            bundled_hv=result.bundled_hypervector,
            chunk_hvs=result.hypervectors,
            master_seed=master_seed,
        )

        return cls(
            genome_id=genome_id,
            assembly=assembly,
            bundled_hypervector=result.bundled_hypervector,
            chunk_hypervectors=result.hypervectors,
            metadata=result.metadata,
            statistics=result.statistics,
            master_seed=master_seed,
            encoding_hash=encoding_hash,
        )

    def save(self, filepath: Path | str, compress: bool = True) -> int:
        """
        Save encoded genome to file.

        Serializes to compressed JSON format with hex-encoded numpy arrays.
        Achieves significant compression compared to raw VCF files.

        Args:
            filepath: Path to save file (will add .gz if compress=True)
            compress: Whether to use gzip compression (default: True)

        Returns:
            Number of bytes written

        Example:
            >>> encoded.save("patient_001.enc.gz")
            45678
            >>> encoded.save("patient_001.enc.json", compress=False)
            123456
        """
        filepath = Path(filepath)

        # Convert to serializable format
        data = self._to_serializable()

        # Serialize to JSON
        json_str = json.dumps(data, indent=2)
        json_bytes = json_str.encode('utf-8')

        # Write to file (compressed or uncompressed)
        if compress:
            if not str(filepath).endswith('.gz'):
                filepath = Path(str(filepath) + '.gz')

            with gzip.open(filepath, 'wb') as f:
                f.write(json_bytes)
        else:
            with open(filepath, 'wb') as f:
                f.write(json_bytes)

        file_size = filepath.stat().st_size

        logger.info(
            f"Saved encoded genome {self.genome_id} to {filepath} "
            f"({file_size:,} bytes, compress={compress})"
        )

        return file_size

    @classmethod
    def load(cls, filepath: Path | str) -> "EncodedGenome":
        """
        Load encoded genome from file.

        Deserializes from compressed JSON, reconstructs numpy arrays,
        and validates integrity.

        Args:
            filepath: Path to encoded genome file

        Returns:
            EncodedGenome instance

        Raises:
            ValueError: If file is corrupted or validation fails

        Example:
            >>> encoded = EncodedGenome.load("patient_001.enc.gz")
            >>> print(f"Loaded {len(encoded.chunk_hypervectors)} chunks")
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read file (auto-detect compression)
        try:
            with gzip.open(filepath, 'rb') as f:
                json_bytes = f.read()
        except gzip.BadGzipFile:
            # Not compressed, read as regular file
            with open(filepath, 'rb') as f:
                json_bytes = f.read()

        # Deserialize JSON
        try:
            data = json.loads(json_bytes.decode('utf-8'))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file: {e}")

        # Reconstruct EncodedGenome
        encoded_genome = cls._from_serializable(data)

        # Validate integrity
        encoded_genome._validate_integrity()

        logger.info(
            f"Loaded encoded genome {encoded_genome.genome_id} from {filepath} "
            f"({len(encoded_genome.chunk_hypervectors)} chunks)"
        )

        return encoded_genome

    def storage_size_kb(self) -> float:
        """
        Calculate total storage size in KB.

        Returns:
            Storage size in kilobytes (uncompressed JSON)
        """
        data = self._to_serializable()
        json_str = json.dumps(data)
        json_bytes = json_str.encode('utf-8')
        return len(json_bytes) / 1024

    def compression_ratio(self, vcf_size_kb: float) -> float:
        """
        Calculate compression ratio vs. original VCF.

        Args:
            vcf_size_kb: Size of original VCF file in KB

        Returns:
            Compression ratio (vcf_size / encoded_size)

        Example:
            >>> vcf_size = 1500  # KB
            >>> ratio = encoded.compression_ratio(vcf_size)
            >>> print(f"Compression: {ratio:.1f}x")
        """
        encoded_size = self.storage_size_kb()
        if encoded_size == 0:
            return float('inf')
        return vcf_size_kb / encoded_size

    def verify(self) -> bool:
        """
        Verify encoding integrity.

        Recomputes encoding hash and compares with stored value.

        Returns:
            True if verification passes, False otherwise
        """
        try:
            self._validate_integrity()
            return True
        except ValueError:
            return False

    def _to_serializable(self) -> Dict[str, Any]:
        """
        Convert to JSON-serializable format.

        Returns:
            Dictionary with all data in serializable format
        """
        return {
            "genome_id": self.genome_id,
            "assembly": self.assembly,
            "bundled_hypervector": self._encode_array(self.bundled_hypervector),
            "chunk_hypervectors": [
                self._encode_array(hv) for hv in self.chunk_hypervectors
            ],
            "metadata": [
                meta.to_dict() for meta in self.metadata
            ],
            "statistics": self._serialize_statistics(self.statistics),
            "master_seed": self.master_seed.hex(),
            "encoding_hash": self.encoding_hash,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def _from_serializable(cls, data: Dict[str, Any]) -> "EncodedGenome":
        """
        Reconstruct from serialized format.

        Args:
            data: Serialized data dictionary

        Returns:
            EncodedGenome instance
        """
        return cls(
            genome_id=data["genome_id"],
            assembly=data["assembly"],
            bundled_hypervector=cls._decode_array(data["bundled_hypervector"]),
            chunk_hypervectors=[
                cls._decode_array(hv_data) for hv_data in data["chunk_hypervectors"]
            ],
            metadata=[
                DifferentialEncodingMetadata.from_dict(meta_data)
                for meta_data in data["metadata"]
            ],
            statistics=cls._deserialize_statistics(data["statistics"]),
            master_seed=bytes.fromhex(data["master_seed"]),
            encoding_hash=data["encoding_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            version=data["version"],
        )

    def _validate_integrity(self):
        """
        Validate encoding integrity.

        Raises:
            ValueError: If validation fails
        """
        # Recompute hash
        computed_hash = self._compute_encoding_hash(
            bundled_hv=self.bundled_hypervector,
            chunk_hvs=self.chunk_hypervectors,
            master_seed=self.master_seed,
        )

        if computed_hash != self.encoding_hash:
            raise ValueError(
                f"Encoding hash mismatch: expected {self.encoding_hash}, "
                f"got {computed_hash}"
            )

        # Validate hypervector dimensions
        bundled_dim = len(self.bundled_hypervector)
        for i, hv in enumerate(self.chunk_hypervectors):
            if len(hv) != bundled_dim:
                raise ValueError(
                    f"Chunk {i} has dimension {len(hv)}, expected {bundled_dim}"
                )

        # Validate hypervector norms
        bundled_norm = np.linalg.norm(self.bundled_hypervector)
        if not np.isclose(bundled_norm, 1.0, atol=1e-5):
            raise ValueError(
                f"Bundled hypervector not normalized: norm={bundled_norm}"
            )

        for i, hv in enumerate(self.chunk_hypervectors):
            norm = np.linalg.norm(hv)
            if not np.isclose(norm, 1.0, atol=1e-5):
                raise ValueError(
                    f"Chunk {i} hypervector not normalized: norm={norm}"
                )

    @staticmethod
    def _compute_encoding_hash(
        bundled_hv: np.ndarray,
        chunk_hvs: List[np.ndarray],
        master_seed: bytes,
    ) -> str:
        """
        Compute cryptographic hash of encoding.

        Args:
            bundled_hv: Bundled hypervector
            chunk_hvs: Chunk hypervectors
            master_seed: Master seed

        Returns:
            Hex-encoded hash string
        """
        hasher = hashlib.sha256()

        # Hash bundled hypervector
        hasher.update(bundled_hv.tobytes())

        # Hash each chunk hypervector
        for hv in chunk_hvs:
            hasher.update(hv.tobytes())

        # Hash master seed
        hasher.update(master_seed)

        return hasher.hexdigest()

    @staticmethod
    def _encode_array(arr: np.ndarray) -> Dict[str, Any]:
        """
        Encode numpy array to JSON-serializable format.

        Args:
            arr: Numpy array

        Returns:
            Dictionary with shape, dtype, and hex-encoded data
        """
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "data": arr.tobytes().hex(),
        }

    @staticmethod
    def _decode_array(data: Dict[str, Any]) -> np.ndarray:
        """
        Decode numpy array from JSON format.

        Args:
            data: Dictionary with shape, dtype, and hex-encoded data

        Returns:
            Reconstructed numpy array
        """
        arr_bytes = bytes.fromhex(data["data"])
        arr = np.frombuffer(arr_bytes, dtype=data["dtype"])
        return arr.reshape(data["shape"])

    @staticmethod
    def _serialize_statistics(stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize statistics to JSON-compatible format.

        Args:
            stats: Statistics dictionary

        Returns:
            Serialized statistics
        """
        serialized = {}
        for key, value in stats.items():
            if isinstance(value, (int, float, str, bool, type(None))):
                serialized[key] = value
            elif isinstance(value, list):
                serialized[key] = value
            elif isinstance(value, (np.integer, np.floating)):
                serialized[key] = float(value)
            elif isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            else:
                serialized[key] = str(value)
        return serialized

    @staticmethod
    def _deserialize_statistics(stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize statistics from JSON format.

        Args:
            stats: Serialized statistics

        Returns:
            Statistics dictionary
        """
        # For now, just return as-is since we serialized to compatible types
        return stats

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EncodedGenome("
            f"id={self.genome_id}, "
            f"assembly={self.assembly}, "
            f"chunks={len(self.chunk_hypervectors)}, "
            f"dimension={len(self.bundled_hypervector)}, "
            f"size={self.storage_size_kb():.1f}KB)"
        )

    def summary(self) -> Dict[str, Any]:
        """
        Get summary information about the encoded genome.

        Returns:
            Dictionary with summary statistics
        """
        return {
            "genome_id": self.genome_id,
            "assembly": self.assembly,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "hypervector_dimension": len(self.bundled_hypervector),
            "total_chunks": len(self.chunk_hypervectors),
            "total_differences": self.statistics.get("total_differences", 0),
            "new_mutations": self.statistics.get("new_mutations", 0),
            "missing_variants": self.statistics.get("missing_variants", 0),
            "genotype_differences": self.statistics.get("genotype_differences", 0),
            "chromosomes": self.statistics.get("chromosomes", []),
            "storage_size_kb": self.storage_size_kb(),
            "encoding_hash": self.encoding_hash[:16] + "...",  # Truncated for display
        }
