"""Private Information Retrieval CLI commands."""

import typer
from typing import List, Optional
from pathlib import Path
import json

app = typer.Typer(help="Private Information Retrieval operations")

@app.command()
def serve(
    data: Path = typer.Option(..., "--data", "-d", help="Database JSON file"),
    port: int = typer.Option(8001, "--port", "-p", help="Server port"),
):
    """Start a PIR server."""
    typer.echo(f"Starting PIR server on port {port}")
    
    # Load database
    with open(data) as f:
        records = json.load(f)
    
    from genomevault.pir.servers import PIRServer
    server = PIRServer(records)
    
    typer.echo(f"Server ready with {len(records)} records")
    typer.echo("Press Ctrl+C to stop")
    
    # In a real implementation, this would start an HTTP server
    # For now, just keep the process alive
    import time
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        typer.echo("\nShutting down server")

@app.command()
def query(
    servers: str = typer.Option(..., "--servers", "-s", help="Comma-separated server URLs"),
    index: int = typer.Option(..., "--index", "-i", help="Record index to retrieve"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Query PIR servers privately."""
    server_urls = servers.split(",")
    typer.echo(f"Querying {len(server_urls)} servers for index {index}")
    
    # In a real implementation, this would query the servers
    # For demo, return mock result
    result = f"Record at index {index}"
    
    if output:
        output.write_text(result)
        typer.echo(f"Result saved to: {output}")
    else:
        typer.echo(f"Result: {result}")

@app.command()
def benchmark(
    size: int = typer.Option(1000, "--size", "-n", help="Database size"),
    queries: int = typer.Option(100, "--queries", "-q", help="Number of queries"),
):
    """Benchmark PIR performance."""
    import time
    import numpy as np
    from genomevault.pir.servers import PIRServer
    
    typer.echo(f"Benchmarking PIR with {size} records and {queries} queries")
    
    # Create test database
    records = [f"record_{i}".encode() for i in range(size)]
    server = PIRServer(records)
    
    # Run queries
    times = []
    for _ in range(queries):
        idx = np.random.randint(0, size)
        mask = np.zeros(size, dtype=np.uint8)
        mask[idx] = 1
        
        start = time.perf_counter()
        result = server.answer(mask)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    # Report statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    typer.echo(f"\nResults:")
    typer.echo(f"  Average: {avg_time:.3f} ms")
    typer.echo(f"  Std Dev: {std_time:.3f} ms")
    typer.echo(f"  Min: {min_time:.3f} ms")
    typer.echo(f"  Max: {max_time:.3f} ms")
    typer.echo(f"  Throughput: {1000/avg_time:.1f} queries/sec")