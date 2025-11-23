"""Hyperdimensional Computing CLI commands."""

import typer
from typing import Optional
from pathlib import Path
import json
import numpy as np

app = typer.Typer(help="Hyperdimensional Computing operations")

@app.command()
def encode(
    input_file: Path = typer.Option(..., "--input", "-i", help="Input data file (JSON)"),
    dimension: int = typer.Option(10000, "--dim", "-d", help="Hypervector dimension"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    omics_type: str = typer.Option("genomic", "--type", "-t", help="Omics type"),
):
    """Encode data to hypervector."""
    from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig
    from genomevault.core.constants import OmicsType
    
    typer.echo(f"Encoding {input_file} to {dimension}D hypervector")
    
    # Load input data
    with open(input_file) as f:
        data = json.load(f)
    
    # Convert to numpy array
    if isinstance(data, list):
        data_array = np.array(data, dtype=np.float32)
    else:
        typer.echo("Error: Input must be a JSON array of numbers")
        raise typer.Exit(1)
    
    # Create encoder
    config = HypervectorConfig(dimension=dimension)
    encoder = HypervectorEncoder(config=config)
    
    # Encode
    omics = OmicsType[omics_type.upper()]
    encoded = encoder.encode(data_array, omics)
    
    # Convert to list for JSON serialization
    if hasattr(encoded, 'detach'):
        encoded_list = encoded.detach().cpu().numpy().tolist()
    elif isinstance(encoded, np.ndarray):
        encoded_list = encoded.tolist()
    else:
        encoded_list = list(encoded)
    
    # Save or print result
    result = {
        "dimension": dimension,
        "vector": encoded_list,
        "type": omics_type,
        "sparsity": float(np.sum(np.array(encoded_list) == 0) / dimension)
    }
    
    if output:
        with open(output, 'w') as f:
            json.dump(result, f, indent=2)
        typer.echo(f"Encoded vector saved to: {output}")
        typer.echo(f"Sparsity: {result['sparsity']:.1%}")
    else:
        typer.echo(json.dumps(result, indent=2))

@app.command()
def compare(
    vector1: Path = typer.Option(..., "--v1", help="First vector file"),
    vector2: Path = typer.Option(..., "--v2", help="Second vector file"),
    metric: str = typer.Option("hamming", "--metric", "-m", help="Distance metric"),
):
    """Compare two hypervectors."""
    from genomevault.hypervector.similarity import hamming_distance
    
    typer.echo(f"Comparing vectors using {metric} distance")
    
    # Load vectors
    with open(vector1) as f:
        v1_data = json.load(f)
    with open(vector2) as f:
        v2_data = json.load(f)
    
    v1 = np.array(v1_data.get("vector", v1_data))
    v2 = np.array(v2_data.get("vector", v2_data))
    
    if len(v1) != len(v2):
        typer.echo(f"Error: Vector dimensions don't match ({len(v1)} vs {len(v2)})")
        raise typer.Exit(1)
    
    # Calculate distance
    if metric == "hamming":
        # Binarize vectors
        v1_bin = (v1 > 0).astype(np.uint8)
        v2_bin = (v2 > 0).astype(np.uint8)
        distance = np.sum(v1_bin != v2_bin)
        similarity = 1 - (distance / len(v1))
    elif metric == "cosine":
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
        distance = 1 - similarity
    else:
        typer.echo(f"Unknown metric: {metric}")
        raise typer.Exit(1)
    
    typer.echo(f"\nResults:")
    typer.echo(f"  Dimension: {len(v1)}")
    typer.echo(f"  Distance: {distance:.4f}")
    typer.echo(f"  Similarity: {similarity:.1%}")

@app.command()
def benchmark(
    dimensions: str = typer.Option("1000,5000,10000", "--dims", help="Comma-separated dimensions"),
    samples: int = typer.Option(100, "--samples", "-n", help="Number of samples"),
):
    """Benchmark HDC encoding performance."""
    import time
    from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig
    from genomevault.core.constants import OmicsType
    
    dims = [int(d) for d in dimensions.split(",")]
    typer.echo(f"Benchmarking HDC encoding for dimensions: {dims}")
    
    # Generate test data
    data = np.random.randn(samples, 100).astype(np.float32)
    
    results = []
    for dim in dims:
        config = HypervectorConfig(dimension=dim)
        encoder = HypervectorEncoder(config=config)
        
        # Warm up
        _ = encoder.encode(data[0], OmicsType.GENOMIC)
        
        # Benchmark
        times = []
        for i in range(min(10, samples)):
            start = time.perf_counter()
            _ = encoder.encode(data[i], OmicsType.GENOMIC)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        results.append((dim, avg_time))
        typer.echo(f"  {dim}D: {avg_time:.3f} ms")
    
    typer.echo("\nSummary:")
    for dim, time_ms in results:
        throughput = 1000 / time_ms
        typer.echo(f"  {dim}D: {time_ms:.3f} ms ({throughput:.1f} ops/sec)")