"""
Hypervector search index with efficient bit-packing for Hamming distance.

This module provides indexing and search capabilities for hypervectors,
with optimized bit-packing for Hamming distance computation and support
for multiple distance metrics.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import struct

from numpy.typing import NDArray
import numpy as np

from genomevault.api.types import IndexManifest
from genomevault.hypervector.types import (
    VectorBool,
)
from genomevault.hypervector.operations.hamming_lut import (
    generate_popcount_lut,
    hamming_distance_batch_cpu,
)


def pack_binary_vectors(vectors: List[VectorBool]) -> NDArray[np.uint64]:
    """
    Pack binary vectors into uint64 chunks for efficient Hamming distance.

    Args:
        vectors: List of binary vectors (boolean or 0/1 arrays)

    Returns:
        Packed uint64 array of shape (n_vectors, n_chunks)
    """
    if not vectors:
        raise ValueError("Cannot pack empty vector list")

    n_vectors = len(vectors)
    dimension = len(vectors[0])

    # Ensure all vectors have same dimension
    for i, vec in enumerate(vectors):
        if len(vec) != dimension:
            raise ValueError(f"Vector {i} has dimension {len(vec)}, expected {dimension}")

    # Calculate number of 64-bit chunks needed
    n_chunks = (dimension + 63) // 64
    packed_dim = n_chunks * 64

    # Create padded binary matrix
    binary_matrix = np.zeros((n_vectors, packed_dim), dtype=np.uint8)
    for i, vec in enumerate(vectors):
        # Convert to binary if needed
        if vec.dtype == bool:
            binary_vec = vec.astype(np.uint8)
        elif vec.dtype in [np.float32, np.float64]:
            binary_vec = (vec > 0).astype(np.uint8)
        else:
            binary_vec = vec.astype(np.uint8)

        binary_matrix[i, :dimension] = binary_vec

    # Pack into uint64 chunks
    packed = np.zeros((n_vectors, n_chunks), dtype=np.uint64)
    for i in range(n_vectors):
        for chunk_idx in range(n_chunks):
            start = chunk_idx * 64
            end = min(start + 64, packed_dim)

            # Pack 64 bits into single uint64
            chunk_val = 0
            for bit_idx in range(start, end):
                if bit_idx < dimension and binary_matrix[i, bit_idx]:
                    chunk_val |= 1 << (bit_idx - start)

            packed[i, chunk_idx] = chunk_val

    return packed


def unpack_binary_vectors(packed: NDArray[np.uint64], dimension: int) -> List[VectorBool]:
    """
    Unpack uint64 chunks back to binary vectors.

    Args:
        packed: Packed uint64 array of shape (n_vectors, n_chunks)
        dimension: Original dimension of vectors

    Returns:
        List of binary vectors
    """
    n_vectors, n_chunks = packed.shape
    vectors = []

    for i in range(n_vectors):
        vec = np.zeros(dimension, dtype=np.uint8)
        for chunk_idx in range(n_chunks):
            chunk_val = packed[i, chunk_idx]
            start = chunk_idx * 64

            for bit_idx in range(64):
                global_idx = start + bit_idx
                if global_idx < dimension:
                    vec[global_idx] = (int(chunk_val) >> bit_idx) & 1

        vectors.append(vec)

    return vectors


def build(vectors: List[np.ndarray], ids: List[str], path: Path, metric: str = "hamming") -> None:
    """
    Build a search index from vectors.

    Args:
        vectors: List of hypervectors
        ids: List of unique identifiers for each vector
        path: Directory path to save index files
        metric: Distance metric ('hamming', 'cosine', 'euclidean')
    """
    if len(vectors) != len(ids):
        raise ValueError(f"Mismatch: {len(vectors)} vectors vs {len(ids)} ids")

    if not vectors:
        raise ValueError("Cannot build index from empty vectors")

    # Validate metric
    valid_metrics = ["hamming", "cosine", "euclidean"]
    if metric not in valid_metrics:
        raise ValueError(f"Invalid metric: {metric}. Must be one of {valid_metrics}")

    # Create output directory
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Get vector properties
    dimension = len(vectors[0])
    dtype = vectors[0].dtype

    # Validate all vectors have same dimension
    for i, vec in enumerate(vectors):
        if len(vec) != dimension:
            raise ValueError(f"Vector {i} has dimension {len(vec)}, expected {dimension}")

    # Handle different metrics
    if metric == "hamming":
        # Pack binary vectors for efficient storage
        packed_vectors = pack_binary_vectors(vectors)

        # Save packed vectors
        index_file = path / "index.bin"
        with open(index_file, "wb") as f:
            # Write header: n_vectors, n_chunks, dimension
            n_vectors, n_chunks = packed_vectors.shape
            header = struct.pack("III", n_vectors, n_chunks, dimension)
            f.write(header)

            # Write packed data
            packed_vectors.tofile(f)
    else:
        # For cosine/euclidean, store as float32
        float_vectors = np.array([v.astype(np.float32) for v in vectors])

        # Normalize for cosine similarity
        if metric == "cosine":
            norms = np.linalg.norm(float_vectors, axis=1, keepdims=True)
            float_vectors = float_vectors / (norms + 1e-8)

        # Save float vectors
        index_file = path / "index.bin"
        with open(index_file, "wb") as f:
            # Write header: n_vectors, dimension
            n_vectors = len(float_vectors)
            header = struct.pack("II", n_vectors, dimension)
            f.write(header)

            # Write float data
            float_vectors.tofile(f)

    # Save metadata
    manifest: IndexManifest = {
        "ids": ids,
        "dimension": dimension,
        "metric": metric,  # type: ignore
        "n_vectors": len(vectors),
        "dtype": str(dtype),
        "version": "1.0",
        "created_at": datetime.utcnow().isoformat(),
        "metadata": None,
    }

    manifest_file = path / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Index built: {len(vectors)} vectors, dimension {dimension}, metric {metric}")
    print(f"Saved to: {path}")


def search(
    query: np.ndarray, path: Path, k: int = 5, metric: Optional[str] = None
) -> List[Dict[str, Union[str, float]]]:
    """
    Search for k nearest neighbors in the index.

    Args:
        query: Query hypervector
        path: Directory path containing index files
        k: Number of nearest neighbors to return
        metric: Distance metric (if None, uses metric from index)

    Returns:
        List of dicts with 'id' and 'distance' for top-k results
    """
    path = Path(path)

    # Load manifest
    manifest_file = path / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Index manifest not found: {manifest_file}")

    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    # Use metric from manifest if not specified
    if metric is None:
        metric = manifest["metric"]
    elif metric != manifest["metric"]:
        raise ValueError(f"Metric mismatch: requested {metric} but index uses {manifest['metric']}")

    # Validate query dimension
    if len(query) != manifest["dimension"]:
        raise ValueError(
            f"Query dimension {len(query)} doesn't match index dimension {manifest['dimension']}"
        )

    # Load index
    index_file = path / "index.bin"
    if not index_file.exists():
        raise FileNotFoundError(f"Index file not found: {index_file}")

    # Compute distances based on metric
    if metric == "hamming":
        # Load packed vectors
        with open(index_file, "rb") as f:
            # Read header
            header = f.read(12)
            n_vectors, n_chunks, dimension = struct.unpack("III", header)

            # Read packed data
            packed_vectors = np.fromfile(f, dtype=np.uint64).reshape(n_vectors, n_chunks)

        # Pack query vector
        query_packed = pack_binary_vectors([query])[0]

        # Compute Hamming distances using LUT
        lut = generate_popcount_lut()

        # Reshape for batch computation
        query_batch = query_packed.reshape(1, -1)
        distances_matrix = hamming_distance_batch_cpu(query_batch, packed_vectors, lut)
        distances = distances_matrix[0]

    elif metric == "cosine":
        # Load float vectors
        with open(index_file, "rb") as f:
            # Read header
            header = f.read(8)
            n_vectors, dimension = struct.unpack("II", header)

            # Read float data
            vectors = np.fromfile(f, dtype=np.float32).reshape(n_vectors, dimension)

        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-8)

        # Compute cosine similarities (convert to distances)
        similarities = np.dot(vectors, query_norm)
        distances = 1 - similarities  # Convert similarity to distance

    elif metric == "euclidean":
        # Load float vectors
        with open(index_file, "rb") as f:
            # Read header
            header = f.read(8)
            n_vectors, dimension = struct.unpack("II", header)

            # Read float data
            vectors = np.fromfile(f, dtype=np.float32).reshape(n_vectors, dimension)

        # Compute Euclidean distances
        diff = vectors - query.astype(np.float32)
        distances = np.linalg.norm(diff, axis=1)

    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Get top-k indices
    k = min(k, len(distances))
    top_k_indices = np.argpartition(distances, k - 1)[:k]
    top_k_indices = top_k_indices[np.argsort(distances[top_k_indices])]

    # Build results
    results = []
    for idx in top_k_indices:
        results.append({"id": manifest["ids"][idx], "distance": float(distances[idx])})

    return results


def load_index_metadata(path: Path) -> IndexManifest:
    """
    Load index metadata without loading vectors.

    Args:
        path: Directory path containing index files

    Returns:
        Dictionary containing index metadata
    """
    path = Path(path)
    manifest_file = path / "manifest.json"

    if not manifest_file.exists():
        raise FileNotFoundError(f"Index manifest not found: {manifest_file}")

    with open(manifest_file, "r") as f:
        return json.load(f)


def add_vectors(vectors: List[np.ndarray], ids: List[str], path: Path) -> None:
    """
    Add new vectors to an existing index.

    Args:
        vectors: List of new hypervectors to add
        ids: List of unique identifiers for new vectors
        path: Directory path containing index files
    """
    if len(vectors) != len(ids):
        raise ValueError(f"Mismatch: {len(vectors)} vectors vs {len(ids)} ids")

    path = Path(path)

    # Load existing manifest
    manifest = load_index_metadata(path)

    # Validate dimensions
    for i, vec in enumerate(vectors):
        if len(vec) != manifest["dimension"]:
            raise ValueError(
                f"Vector {i} has dimension {len(vec)}, expected {manifest['dimension']}"
            )

    # Load existing index
    index_file = path / "index.bin"

    if manifest["metric"] == "hamming":
        # Load existing packed vectors
        with open(index_file, "rb") as f:
            header = f.read(12)
            n_vectors, n_chunks, dimension = struct.unpack("III", header)
            existing_packed = np.fromfile(f, dtype=np.uint64).reshape(n_vectors, n_chunks)

        # Pack new vectors
        new_packed = pack_binary_vectors(vectors)

        # Combine
        combined_packed = np.vstack([existing_packed, new_packed])

        # Write updated index
        with open(index_file, "wb") as f:
            new_n_vectors = len(combined_packed)
            header = struct.pack("III", new_n_vectors, n_chunks, dimension)
            f.write(header)
            combined_packed.tofile(f)
    else:
        # Load existing float vectors
        with open(index_file, "rb") as f:
            header = f.read(8)
            n_vectors, dimension = struct.unpack("II", header)
            existing_vectors = np.fromfile(f, dtype=np.float32).reshape(n_vectors, dimension)

        # Process new vectors
        new_vectors = np.array([v.astype(np.float32) for v in vectors])

        if manifest["metric"] == "cosine":
            norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
            new_vectors = new_vectors / (norms + 1e-8)

        # Combine
        combined_vectors = np.vstack([existing_vectors, new_vectors])

        # Write updated index
        with open(index_file, "wb") as f:
            new_n_vectors = len(combined_vectors)
            header = struct.pack("II", new_n_vectors, dimension)
            f.write(header)
            combined_vectors.tofile(f)

    # Update manifest
    manifest["ids"].extend(ids)
    manifest["n_vectors"] = len(manifest["ids"])

    manifest_file = path / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Added {len(vectors)} vectors to index. Total: {manifest['n_vectors']}")


def batch_hamming_distance(
    queries: NDArray[np.uint64], targets: NDArray[np.uint64]
) -> NDArray[np.uint32]:
    """
    Compute batch Hamming distances between packed binary vectors.

    Args:
        queries: Packed query vectors (n_queries, n_chunks) as uint64
        targets: Packed target vectors (n_targets, n_chunks) as uint64

    Returns:
        Distance matrix (n_queries, n_targets) as uint32
    """
    if queries.ndim != 2 or targets.ndim != 2:
        raise ValueError("Both queries and targets must be 2D arrays")

    if queries.shape[1] != targets.shape[1]:
        raise ValueError(f"Chunk dimension mismatch: {queries.shape[1]} vs {targets.shape[1]}")

    lut = generate_popcount_lut()
    distances = hamming_distance_batch_cpu(queries, targets, lut)
    return distances.astype(np.uint32)


def compute_packed_hamming(vec1: NDArray[np.uint64], vec2: NDArray[np.uint64]) -> np.uint32:
    """
    Compute Hamming distance between two packed uint64 vectors.

    Args:
        vec1: First packed vector (n_chunks,) as uint64
        vec2: Second packed vector (n_chunks,) as uint64

    Returns:
        Hamming distance as uint32
    """
    if vec1.shape != vec2.shape:
        raise ValueError(f"Shape mismatch: {vec1.shape} vs {vec2.shape}")

    lut = generate_popcount_lut()
    distance = np.uint32(0)

    for i in range(len(vec1)):
        xor_val = vec1[i] ^ vec2[i]
        distance += (
            lut[(xor_val >> 0) & 0xFFFF]
            + lut[(xor_val >> 16) & 0xFFFF]
            + lut[(xor_val >> 32) & 0xFFFF]
            + lut[(xor_val >> 48) & 0xFFFF]
        )

    return distance
