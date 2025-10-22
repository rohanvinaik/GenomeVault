"""
Differential Encoding Metadata for GenomeVault.

This module implements metadata structures for differential encoding results,
capturing all necessary information for cryptographic verification, reference
tracking, and difference statistics.

Section 5.2: Differential Encoding Metadata
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from genomevault.differential_encoding.chunking import AnalysisType
from genomevault.differential_encoding.differences import DifferenceType
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


# JSON Schema for metadata validation
METADATA_SCHEMA = {
    "type": "object",
    "required": [
        "chunk_id",
        "chromosome",
        "start_position",
        "end_position",
        "reference_genome_id",
        "reference_seed",
        "reference_hash",
        "chunking_strategy",
        "chunking_seed",
        "analysis_type",
        "difference_counts",
        "cryptographic_binding",
        "created_timestamp",
    ],
    "properties": {
        "chunk_id": {"type": "string", "pattern": "^[0-9a-f]+$"},
        "chromosome": {"type": "string"},
        "start_position": {"type": "integer", "minimum": 0},
        "end_position": {"type": "integer", "minimum": 0},
        "reference_genome_id": {"type": "string"},
        "reference_seed": {"type": "string", "pattern": "^[0-9a-f]+$"},
        "reference_hash": {"type": "string", "pattern": "^[0-9a-f]+$"},
        "chunking_strategy": {"type": "string"},
        "chunking_seed": {"type": "string", "pattern": "^[0-9a-f]+$"},
        "analysis_type": {"type": "string"},
        "difference_counts": {
            "type": "object",
            "properties": {
                "new_mutations": {"type": "integer", "minimum": 0},
                "missing_variants": {"type": "integer", "minimum": 0},
                "genotype_differences": {"type": "integer", "minimum": 0},
                "total": {"type": "integer", "minimum": 0},
            },
            "required": ["new_mutations", "missing_variants", "genotype_differences", "total"],
        },
        "cryptographic_binding": {"type": "string", "pattern": "^[0-9a-f]+$"},
        "created_timestamp": {"type": "string"},
        "feature_associations": {
            "type": "array",
            "items": {"type": "string"},
        },
        "metadata": {"type": "object"},
    },
}


@dataclass
class DifferentialEncodingMetadata:
    """
    Metadata for differential encoding results.

    This class captures all essential information about a differentially encoded
    genome chunk, including:
    - Chunk identification and location
    - Reference genome selection
    - Chunking strategy and parameters
    - Differential statistics
    - Cryptographic verification binding
    - Temporal and feature associations

    All cryptographic fields (chunk_id, seeds, hashes, bindings) are stored as
    bytes internally but serialized as hex strings for JSON compatibility.

    Attributes:
        # Chunk Information
        chunk_id: Unique identifier for this chunk (HMAC-SHA256)
        chromosome: Chromosome identifier
        start_position: Start position in chromosome (0-based)
        end_position: End position in chromosome (exclusive)

        # Reference Selection
        reference_genome_id: ID of selected reference genome
        reference_seed: Cryptographic seed used for reference selection
        reference_hash: Hash of reference genome section

        # Chunking Information
        chunking_strategy: Name of chunking strategy used
        chunking_seed: Seed used for chunk generation
        analysis_type: Type of analysis (AnalysisType enum value)

        # Differential Statistics
        difference_counts: Counts of each difference type

        # Cryptographic Binding
        cryptographic_binding: HMAC binding of chunk to reference

        # Temporal Information
        created_timestamp: When this metadata was created

        # Optional Associations
        feature_associations: List of genomic features in this chunk
        metadata: Additional key-value metadata

    Example:
        >>> metadata = DifferentialEncodingMetadata(
        ...     chunk_id=b"\\x01" * 32,
        ...     chromosome="chr1",
        ...     start_position=100000,
        ...     end_position=200000,
        ...     reference_genome_id="GRCh38_ref_001",
        ...     reference_seed=b"\\x02" * 32,
        ...     reference_hash=b"\\x03" * 32,
        ...     chunking_strategy="sliding_window",
        ...     chunking_seed=b"\\x04" * 32,
        ...     analysis_type="sliding_window",
        ...     difference_counts={
        ...         "new_mutations": 5,
        ...         "missing_variants": 3,
        ...         "genotype_differences": 2,
        ...         "total": 10
        ...     },
        ...     cryptographic_binding=b"\\x05" * 32
        ... )
        >>> data = metadata.to_dict()
        >>> restored = DifferentialEncodingMetadata.from_dict(data)
        >>> assert metadata == restored
    """

    # Chunk Information
    chunk_id: bytes
    chromosome: str
    start_position: int
    end_position: int

    # Reference Selection
    reference_genome_id: str
    reference_seed: bytes
    reference_hash: bytes

    # Chunking Information
    chunking_strategy: str
    chunking_seed: bytes
    analysis_type: str

    # Differential Statistics
    difference_counts: Dict[str, int]

    # Cryptographic Binding
    cryptographic_binding: bytes

    # Temporal Information
    created_timestamp: datetime = field(default_factory=datetime.utcnow)

    # Optional Associations
    feature_associations: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate metadata on creation."""
        self.validate()

    def validate(self) -> None:
        """
        Validate metadata integrity and consistency.

        Checks:
        - Position ranges are valid
        - All cryptographic fields have correct length
        - Difference counts are non-negative and consistent
        - Analysis type is valid

        Raises:
            ValueError: If validation fails
        """
        # Validate positions
        if self.start_position < 0:
            raise ValueError(f"start_position must be non-negative, got {self.start_position}")
        if self.end_position <= self.start_position:
            raise ValueError(
                f"end_position ({self.end_position}) must be > start_position ({self.start_position})"
            )

        # Validate cryptographic field lengths (SHA-256 = 32 bytes)
        crypto_fields = {
            "chunk_id": self.chunk_id,
            "reference_seed": self.reference_seed,
            "reference_hash": self.reference_hash,
            "chunking_seed": self.chunking_seed,
            "cryptographic_binding": self.cryptographic_binding,
        }

        for name, value in crypto_fields.items():
            if not isinstance(value, bytes):
                raise ValueError(f"{name} must be bytes, got {type(value)}")
            if len(value) != 32:
                raise ValueError(f"{name} must be 32 bytes (SHA-256), got {len(value)}")

        # Validate difference counts
        required_keys = {"new_mutations", "missing_variants", "genotype_differences", "total"}
        if not all(key in self.difference_counts for key in required_keys):
            raise ValueError(
                f"difference_counts must contain {required_keys}, "
                f"got {set(self.difference_counts.keys())}"
            )

        for key, value in self.difference_counts.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"difference_counts[{key}] must be non-negative int, got {value}")

        # Validate total matches sum
        expected_total = (
            self.difference_counts["new_mutations"]
            + self.difference_counts["missing_variants"]
            + self.difference_counts["genotype_differences"]
        )
        if self.difference_counts["total"] != expected_total:
            raise ValueError(
                f"difference_counts['total'] ({self.difference_counts['total']}) "
                f"must equal sum of components ({expected_total})"
            )

        # Validate analysis type
        valid_types = {at.value for at in AnalysisType}
        if self.analysis_type not in valid_types:
            raise ValueError(
                f"analysis_type must be one of {valid_types}, got {self.analysis_type}"
            )

        # Validate chromosome format
        if not self.chromosome:
            raise ValueError("chromosome cannot be empty")

        # Validate reference genome ID
        if not self.reference_genome_id:
            raise ValueError("reference_genome_id cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize metadata to dictionary with hex-encoded cryptographic fields.

        All bytes fields are converted to hex strings for JSON compatibility.
        Timestamps are converted to ISO format strings.

        Returns:
            Dictionary representation suitable for JSON serialization

        Example:
            >>> metadata = DifferentialEncodingMetadata(...)
            >>> data = metadata.to_dict()
            >>> json.dumps(data)  # Can be serialized to JSON
        """
        return {
            # Chunk information
            "chunk_id": self.chunk_id.hex(),
            "chromosome": self.chromosome,
            "start_position": self.start_position,
            "end_position": self.end_position,
            # Reference selection
            "reference_genome_id": self.reference_genome_id,
            "reference_seed": self.reference_seed.hex(),
            "reference_hash": self.reference_hash.hex(),
            # Chunking information
            "chunking_strategy": self.chunking_strategy,
            "chunking_seed": self.chunking_seed.hex(),
            "analysis_type": self.analysis_type,
            # Differential statistics
            "difference_counts": self.difference_counts.copy(),
            # Cryptographic binding
            "cryptographic_binding": self.cryptographic_binding.hex(),
            # Temporal information
            "created_timestamp": self.created_timestamp.isoformat(),
            # Optional associations
            "feature_associations": (
                self.feature_associations.copy() if self.feature_associations else None
            ),
            "metadata": self.metadata.copy() if self.metadata else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DifferentialEncodingMetadata:
        """
        Deserialize metadata from dictionary.

        Converts hex-encoded strings back to bytes and ISO timestamps back to datetime.

        Args:
            data: Dictionary representation (from to_dict())

        Returns:
            DifferentialEncodingMetadata instance

        Raises:
            ValueError: If data is invalid or missing required fields

        Example:
            >>> data = metadata.to_dict()
            >>> restored = DifferentialEncodingMetadata.from_dict(data)
        """
        try:
            return cls(
                # Chunk information
                chunk_id=bytes.fromhex(data["chunk_id"]),
                chromosome=data["chromosome"],
                start_position=data["start_position"],
                end_position=data["end_position"],
                # Reference selection
                reference_genome_id=data["reference_genome_id"],
                reference_seed=bytes.fromhex(data["reference_seed"]),
                reference_hash=bytes.fromhex(data["reference_hash"]),
                # Chunking information
                chunking_strategy=data["chunking_strategy"],
                chunking_seed=bytes.fromhex(data["chunking_seed"]),
                analysis_type=data["analysis_type"],
                # Differential statistics
                difference_counts=data["difference_counts"].copy(),
                # Cryptographic binding
                cryptographic_binding=bytes.fromhex(data["cryptographic_binding"]),
                # Temporal information
                created_timestamp=datetime.fromisoformat(data["created_timestamp"]),
                # Optional associations
                feature_associations=(
                    data["feature_associations"].copy()
                    if data.get("feature_associations")
                    else None
                ),
                metadata=data["metadata"].copy() if data.get("metadata") else None,
            )
        except KeyError as e:
            raise ValueError(f"Missing required field in metadata: {e}")
        except ValueError as e:
            raise ValueError(f"Invalid metadata format: {e}")

    def to_json(self, indent: Optional[int] = 2) -> str:
        """
        Serialize metadata to JSON string.

        Args:
            indent: JSON indentation level (None for compact)

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> DifferentialEncodingMetadata:
        """
        Deserialize metadata from JSON string.

        Args:
            json_str: JSON string representation

        Returns:
            DifferentialEncodingMetadata instance

        Raises:
            ValueError: If JSON is invalid
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

    def verify_binding(self, chunk_data: bytes, reference_data: bytes) -> bool:
        """
        Verify cryptographic binding between chunk and reference.

        The binding is computed as:
            HMAC-SHA256(chunking_seed, chunk_hash || reference_hash)

        Args:
            chunk_data: Raw chunk data
            reference_data: Raw reference data

        Returns:
            True if binding is valid, False otherwise

        Example:
            >>> is_valid = metadata.verify_binding(chunk_data, reference_data)
            >>> if not is_valid:
            ...     print("WARNING: Cryptographic binding verification failed!")
        """
        # Compute chunk hash
        chunk_hash = hashlib.sha256(chunk_data).digest()

        # Compute reference hash
        reference_hash = hashlib.sha256(reference_data).digest()

        # Compute expected binding
        expected_binding = hmac.new(
            self.chunking_seed, chunk_hash + reference_hash, hashlib.sha256
        ).digest()

        # Constant-time comparison
        return hmac.compare_digest(expected_binding, self.cryptographic_binding)

    @staticmethod
    def compute_binding(chunk_data: bytes, reference_data: bytes, seed: bytes) -> bytes:
        """
        Compute cryptographic binding between chunk and reference.

        Args:
            chunk_data: Raw chunk data
            reference_data: Raw reference data
            seed: Cryptographic seed for HMAC

        Returns:
            32-byte HMAC-SHA256 binding

        Example:
            >>> binding = DifferentialEncodingMetadata.compute_binding(
            ...     chunk_data, reference_data, seed
            ... )
        """
        chunk_hash = hashlib.sha256(chunk_data).digest()
        reference_hash = hashlib.sha256(reference_data).digest()
        return hmac.new(seed, chunk_hash + reference_hash, hashlib.sha256).digest()

    def get_region_string(self) -> str:
        """
        Get genomic region as string (chr:start-end).

        Returns:
            Region string in format "chr1:100000-200000"
        """
        return f"{self.chromosome}:{self.start_position}-{self.end_position}"

    def get_region_size(self) -> int:
        """
        Get size of genomic region in base pairs.

        Returns:
            Region size in bp
        """
        return self.end_position - self.start_position

    def __str__(self) -> str:
        """String representation of metadata."""
        return (
            f"DifferentialEncodingMetadata("
            f"region={self.get_region_string()}, "
            f"reference={self.reference_genome_id}, "
            f"analysis={self.analysis_type}, "
            f"differences={self.difference_counts['total']})"
        )

    def __repr__(self) -> str:
        """Detailed representation of metadata."""
        return (
            f"DifferentialEncodingMetadata("
            f"chunk_id={self.chunk_id.hex()[:16]}..., "
            f"chromosome={self.chromosome}, "
            f"start_position={self.start_position}, "
            f"end_position={self.end_position}, "
            f"reference_genome_id={self.reference_genome_id}, "
            f"analysis_type={self.analysis_type}, "
            f"difference_counts={self.difference_counts})"
        )


def validate_metadata_schema(data: Dict[str, Any]) -> bool:
    """
    Validate metadata dictionary against JSON schema.

    Args:
        data: Metadata dictionary to validate

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    # Simple schema validation (for full validation, use jsonschema library)
    required_fields = METADATA_SCHEMA["required"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Validate difference_counts structure
    diff_counts = data.get("difference_counts", {})
    required_diff_keys = METADATA_SCHEMA["properties"]["difference_counts"]["required"]
    for key in required_diff_keys:
        if key not in diff_counts:
            raise ValueError(f"Missing required difference count: {key}")

    return True


def create_metadata_from_chunk(
    chunk_id: bytes,
    chromosome: str,
    start_position: int,
    end_position: int,
    reference_genome_id: str,
    reference_seed: bytes,
    reference_hash: bytes,
    chunking_strategy: str,
    chunking_seed: bytes,
    analysis_type: AnalysisType,
    new_mutations: int,
    missing_variants: int,
    genotype_differences: int,
    chunk_data: bytes,
    reference_data: bytes,
    feature_associations: Optional[List[str]] = None,
    additional_metadata: Optional[Dict[str, Any]] = None,
) -> DifferentialEncodingMetadata:
    """
    Factory function to create metadata from chunk components.

    This convenience function computes the cryptographic binding and creates
    a complete metadata object.

    Args:
        chunk_id: Unique chunk identifier
        chromosome: Chromosome identifier
        start_position: Start position
        end_position: End position
        reference_genome_id: Reference genome ID
        reference_seed: Reference selection seed
        reference_hash: Reference genome hash
        chunking_strategy: Chunking strategy name
        chunking_seed: Chunking seed
        analysis_type: Analysis type
        new_mutations: Count of new mutations
        missing_variants: Count of missing variants
        genotype_differences: Count of genotype differences
        chunk_data: Raw chunk data for binding
        reference_data: Raw reference data for binding
        feature_associations: Optional feature associations
        additional_metadata: Optional additional metadata

    Returns:
        DifferentialEncodingMetadata instance

    Example:
        >>> metadata = create_metadata_from_chunk(
        ...     chunk_id=chunk_id,
        ...     chromosome="chr1",
        ...     start_position=100000,
        ...     end_position=200000,
        ...     reference_genome_id="GRCh38_001",
        ...     reference_seed=ref_seed,
        ...     reference_hash=ref_hash,
        ...     chunking_strategy="sliding_window",
        ...     chunking_seed=chunk_seed,
        ...     analysis_type=AnalysisType.SLIDING_WINDOW,
        ...     new_mutations=5,
        ...     missing_variants=3,
        ...     genotype_differences=2,
        ...     chunk_data=chunk_bytes,
        ...     reference_data=ref_bytes
        ... )
    """
    # Compute cryptographic binding
    binding = DifferentialEncodingMetadata.compute_binding(
        chunk_data, reference_data, chunking_seed
    )

    # Compute total differences
    total = new_mutations + missing_variants + genotype_differences

    return DifferentialEncodingMetadata(
        chunk_id=chunk_id,
        chromosome=chromosome,
        start_position=start_position,
        end_position=end_position,
        reference_genome_id=reference_genome_id,
        reference_seed=reference_seed,
        reference_hash=reference_hash,
        chunking_strategy=chunking_strategy,
        chunking_seed=chunking_seed,
        analysis_type=analysis_type.value if isinstance(analysis_type, AnalysisType) else analysis_type,
        difference_counts={
            "new_mutations": new_mutations,
            "missing_variants": missing_variants,
            "genotype_differences": genotype_differences,
            "total": total,
        },
        cryptographic_binding=binding,
        feature_associations=feature_associations,
        metadata=additional_metadata,
    )
