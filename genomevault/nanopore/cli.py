"""
CLI commands for nanopore streaming analysis.

Provides command-line interface for processing nanopore data
with biological signal detection.
"""

import asyncio
import json
from pathlib import Path

import click
import numpy as np

from genomevault.hypervector.encoding import HypervectorEncoder
from genomevault.nanopore.biological_signals import BiologicalSignalDetector
from genomevault.nanopore.streaming import NanoporeStreamProcessor
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


@click.group()
def nanopore():
    """Nanopore streaming analysis commands."""
    pass


@nanopore.command()
@click.argument("fast5_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file for results (JSON format)",
)
@click.option(
    "--slice-size",
    default=50000,
    help="Events per slice",
)
@click.option(
    "--catalytic-mb",
    default=100,
    help="Catalytic memory size in MB",
)
@click.option(
    "--gpu/--no-gpu",
    default=True,
    help="Enable GPU acceleration",
)
@click.option(
    "--anomaly-threshold",
    default=3.0,
    help="Anomaly detection threshold (z-score)",
)
@click.option(
    "--export-track",
    type=click.Choice(["bedgraph", "bed", "none"]),
    default="none",
    help="Export biological signals as genome browser track",
)
def process(
    fast5_file: str,
    output: str | None,
    slice_size: int,
    catalytic_mb: int,
    gpu: bool,
    anomaly_threshold: float,
    export_track: str,
):
    """
    Process Fast5 file for biological signals.

    Example:
        genomevault nanopore process sample.fast5 -o results.json --export-track bedgraph
    """
    click.echo(f"Processing {fast5_file}...")

    # Run async processing
    asyncio.run(
        _process_async(
            fast5_file,
            output,
            slice_size,
            catalytic_mb,
            gpu,
            anomaly_threshold,
            export_track,
        )
    )


async def _process_async(
    fast5_file: str,
    output: str | None,
    slice_size: int,
    catalytic_mb: int,
    gpu: bool,
    anomaly_threshold: float,
    export_track: str,
):
    """Async implementation of processing."""
    # Initialize components
    encoder = HypervectorEncoder(dimension=10000)
    processor = NanoporeStreamProcessor(
        hv_encoder=encoder,
        catalytic_space_mb=catalytic_mb,
        clean_space_mb=1,
        enable_gpu=gpu,
    )

    detector = BiologicalSignalDetector(
        anomaly_threshold=anomaly_threshold,
    )

    # Collect results
    all_results = []
    all_signals = []

    async def collect_results(result):
        all_results.append(result)

        # Detect biological signals
        if result["anomalies"]:
            variance_array = np.array([a[1] for a in result["anomalies"]])
            positions = np.array([a[0] for a in result["anomalies"]])

            signals = detector.detect_signals(
                variance_array=variance_array,
                genomic_positions=positions,
            )

            all_signals.extend(signals)

            # Progress update
            if len(all_results) % 10 == 0:
                click.echo(
                    f"  Processed {len(all_results)} slices, found {len(all_signals)} signals"
                )

    # Process file
    stats = await processor.process_fast5(
        fast5_file,
        output_callback=collect_results,
        anomaly_threshold=anomaly_threshold,
    )

    # Summary
    click.echo("\nProcessing complete!")
    click.echo(f"  Total events: {stats.total_events:,}")
    click.echo(f"  Total slices: {stats.total_slices}")
    click.echo(f"  Processing time: {stats.processing_time:.1f}s")
    click.echo(f"  Events/second: {stats.total_events / stats.processing_time:,.0f}")
    click.echo(f"  Anomalies found: {len(stats.variance_peaks)}")
    click.echo(f"  Biological signals: {len(all_signals)}")

    # Signal summary by type
    signal_counts = {}
    for signal in all_signals:
        sig_type = signal.signal_type.value
        signal_counts[sig_type] = signal_counts.get(sig_type, 0) + 1

    if signal_counts:
        click.echo("\nSignals by type:")
        for sig_type, count in sorted(signal_counts.items()):
            click.echo(f"  {sig_type}: {count}")

    # Export results
    if output:
        results_data = {
            "fast5_file": fast5_file,
            "stats": {
                "total_events": stats.total_events,
                "total_reads": stats.total_reads,
                "total_slices": stats.total_slices,
                "processing_time": stats.processing_time,
            },
            "signals": [
                {
                    "type": sig.signal_type.value,
                    "position": sig.genomic_position,
                    "confidence": sig.confidence,
                    "variance_score": sig.variance_score,
                    "context": sig.context,
                    "metadata": sig.metadata,
                }
                for sig in all_signals
            ],
        }

        with open(output, "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2)

        click.echo(f"\nResults saved to {output}")

    # Export track
    if export_track != "none" and all_signals:
        track_file = Path(fast5_file).stem + f".{export_track}"
        track_data = detector.export_signal_track(all_signals, export_track)

        with open(track_file, "w", encoding="utf-8") as f:
            f.write(track_data)

        click.echo(f"Track exported to {track_file}")


@nanopore.command()
@click.argument("results_file", type=click.Path(exists=True))
@click.option(
    "--min-confidence",
    default=0.5,
    help="Minimum confidence threshold",
)
@click.option(
    "--signal-type",
    type=click.Choice(["5mC", "6mA", "8oxoG", "SV", "repeat", "all"]),
    default="all",
    help="Filter by signal type",
)
def analyze(
    results_file: str,
    min_confidence: float,
    signal_type: str,
):
    """
    Analyze processed nanopore results.

    Example:
        genomevault nanopore analyze results.json --signal-type 5mC
    """
    # Load results
    with open(results_file) as f:
        data = json.load(f)

    signals = data.get("signals", [])

    # Filter signals
    filtered = []
    for sig in signals:
        if sig["confidence"] < min_confidence:
            continue

        if signal_type != "all" and sig["type"] != signal_type:
            continue

        filtered.append(sig)

    click.echo(f"Found {len(filtered)} signals matching criteria")

    if filtered:
        # Show top signals
        click.echo("\nTop signals by confidence:")

        sorted_signals = sorted(filtered, key=lambda x: x["confidence"], reverse=True)

        for i, sig in enumerate(sorted_signals[:10]):
            click.echo(
                f"  {i + 1}. {sig['type']} at position {sig['position']} "
                f"(confidence: {sig['confidence']:.2f}, variance: {sig['variance_score']:.2f})"
            )

        # Position distribution
        positions = [sig["position"] for sig in filtered]
        if positions:
            click.echo(f"\nPosition range: {min(positions):,} - {max(positions):,}")

        # Context analysis
        contexts = [sig.get("context", "") for sig in filtered if sig.get("context")]
        if contexts and signal_type == "5mC":
            cpg_count = sum(1 for ctx in contexts if "CG" in ctx)
            click.echo(
                f"CpG contexts: {cpg_count}/{len(contexts)} ({cpg_count / len(contexts) * 100:.1f}%)"
            )


@nanopore.command()
@click.option(
    "--slice-size",
    default=50000,
    help="Events per slice for benchmark",
)
@click.option(
    "--n-events",
    default=1000000,
    help="Total events to simulate",
)
@click.option(
    "--gpu/--no-gpu",
    default=True,
    help="Test GPU acceleration",
)
def benchmark(slice_size: int, n_events: int, gpu: bool):
    """
    Benchmark nanopore processing performance.

    Example:
        genomevault nanopore benchmark --n-events 5000000 --gpu
    """
    click.echo(f"Benchmarking with {n_events:,} events...")

    asyncio.run(_benchmark_async(slice_size, n_events, gpu))


async def _benchmark_async(slice_size: int, n_events: int, gpu: bool):
    """Async benchmark implementation."""
    # Initialize
    encoder = HypervectorEncoder(dimension=10000)
    processor = NanoporeStreamProcessor(
        hv_encoder=encoder,
        catalytic_space_mb=100,
        clean_space_mb=1,
        enable_gpu=gpu,
    )

    # Generate synthetic data
    click.echo("Generating synthetic nanopore data...")

    events = np.random.randn(n_events, 2).astype(np.float32)
    events[:, 0] *= 20  # Current values
    events[:, 1] = np.abs(events[:, 1]) * 0.01  # Dwell times

    # Add some artificial anomalies
    anomaly_positions = np.random.choice(n_events, size=100, replace=False)
    events[anomaly_positions, 0] *= 2.5

    # Process
    start_time = asyncio.get_event_loop().time()

    total_processed = 0
    slice_times = []

    for i in range(0, n_events, slice_size):
        slice_start = asyncio.get_event_loop().time()

        batch = events[i : i + slice_size]

        if gpu and processor.gpu_kernel:
            hv, var = await processor.gpu_kernel.process_events_async(batch, i, encoder)
        else:
            hv, var = processor._cpu_process_batch(batch, i)

        slice_time = asyncio.get_event_loop().time() - slice_start
        slice_times.append(slice_time)

        total_processed += len(batch)

        if (i // slice_size) % 10 == 0:
            click.echo(f"  Processed {total_processed:,} events...")

    total_time = asyncio.get_event_loop().time() - start_time

    # Results
    click.echo("\nBenchmark results:")
    click.echo(f"  Processing mode: {'GPU' if gpu else 'CPU'}")
    click.echo(f"  Total events: {n_events:,}")
    click.echo(f"  Total time: {total_time:.2f}s")
    click.echo(f"  Throughput: {n_events / total_time:,.0f} events/s")
    click.echo(f"  Average slice time: {np.mean(slice_times) * 1000:.1f}ms")
    click.echo(f"  Slice time std: {np.std(slice_times) * 1000:.1f}ms")

    if gpu and processor.gpu_kernel:
        mem_stats = processor.gpu_kernel.get_memory_usage()
        click.echo("\nGPU memory usage:")
        click.echo(f"  Allocated: {mem_stats['total_mb']:.1f} MB")
        click.echo(f"  Peak used: {mem_stats['gpu_used_mb']:.1f} MB")


# Add to main CLI
def add_nanopore_commands(cli):
    """Add nanopore commands to main CLI."""
    cli.add_command(nanopore)


if __name__ == "__main__":
    nanopore()
