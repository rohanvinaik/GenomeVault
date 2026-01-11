"""
Unified Pipeline CLI Commands

Provides CLI interface for the GenomeVault unified production pipeline.

Usage:
    # Run full pipeline
    python -m genomevault.cli.main unified run \
        --fastq-r1 sample_R1.fq.gz \
        --fastq-r2 sample_R2.fq.gz \
        --guides data/guide_strands \
        --output pipeline_output

    # Show configuration
    python -m genomevault.cli.main unified config

    # Check component status
    python -m genomevault.cli.main unified status
"""

import typer
from typing import Optional
from typing_extensions import Annotated
from pathlib import Path
import json
import time

app = typer.Typer(
    name="unified",
    help="Unified GenomeVault production pipeline (7-layer architecture)",
    no_args_is_help=True,
)


@app.command()
def run(
    fastq_r1: Annotated[Path, typer.Option(
        "--fastq-r1", "-1",
        help="Path to R1 FASTQ file",
    )],
    fastq_r2: Annotated[Path, typer.Option(
        "--fastq-r2", "-2",
        help="Path to R2 FASTQ file",
    )],
    guides: Annotated[Path, typer.Option(
        "--guides", "-g",
        help="Path to guide strand directory",
    )] = Path("data/guide_strands"),
    output: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Output directory for pipeline results",
    )] = Path("pipeline_output"),
    threads: Annotated[int, typer.Option(
        "--threads", "-t",
        help="Number of threads for parallel operations",
    )] = 8,
    schema: Annotated[str, typer.Option(
        "--schema", "-s",
        help="Analysis schema for selective HDV",
    )] = "clinical_risk",
    no_zk: Annotated[bool, typer.Option(
        "--no-zk",
        help="Disable ZK proof generation",
    )] = False,
    no_pir: Annotated[bool, typer.Option(
        "--no-pir",
        help="Disable PIR support",
    )] = False,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable verbose output",
    )] = False,
):
    """
    Run the unified production pipeline.

    Executes Layers 3-6 of the 7-layer architecture:
    - Layer 3: Align experimental FASTQ to guide pool
    - Layer 4: Generate GDiff differential encoding
    - Layer 5: Create HDC encoding (full + selective)
    - Layer 6: Generate ZK proof

    Requires pre-computed Layer 1-2 outputs (consensus + guide strands).
    """
    from genomevault.pipelines.unified_pipeline import (
        UnifiedPipeline,
        PipelineConfig,
    )
    import logging

    # Configure logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(message)s"
    )

    # Validate inputs
    if not fastq_r1.exists():
        typer.echo(f"Error: FASTQ R1 not found: {fastq_r1}", err=True)
        raise typer.Exit(1)
    if not fastq_r2.exists():
        typer.echo(f"Error: FASTQ R2 not found: {fastq_r2}", err=True)
        raise typer.Exit(1)
    if not guides.exists():
        typer.echo(f"Error: Guide directory not found: {guides}", err=True)
        raise typer.Exit(1)

    typer.echo(f"\n{'='*60}")
    typer.echo("GenomeVault Unified Pipeline")
    typer.echo(f"{'='*60}")
    typer.echo(f"Input R1:   {fastq_r1}")
    typer.echo(f"Input R2:   {fastq_r2}")
    typer.echo(f"Guides:     {guides}")
    typer.echo(f"Output:     {output}")
    typer.echo(f"Threads:    {threads}")
    typer.echo(f"Schema:     {schema}")
    typer.echo(f"ZK Proofs:  {not no_zk}")
    typer.echo(f"PIR:        {not no_pir}")
    typer.echo(f"{'='*60}\n")

    # Create pipeline
    config = PipelineConfig(
        output_dir=output,
        guide_fasta_dir=guides,
        threads=threads,
        enable_zk=not no_zk,
        enable_pir=not no_pir,
    )

    pipeline = UnifiedPipeline(config)

    # Run pipeline
    typer.echo("Starting pipeline execution...\n")
    start_time = time.time()

    result = pipeline.run_experimental_pipeline(
        query_fastq_1=fastq_r1,
        query_fastq_2=fastq_r2,
        schema=schema,
    )

    elapsed = time.time() - start_time

    # Report results
    typer.echo(f"\n{'='*60}")
    if result.success:
        typer.echo("Pipeline completed successfully!")
        typer.echo(f"{'='*60}")
        typer.echo(f"\nOutputs:")
        typer.echo(f"  BAM:           {result.query_bam}")
        typer.echo(f"  GDiff:         {result.gdiff_path}")
        typer.echo(f"  HDV:           {result.hdv_path}")
        typer.echo(f"  Selective HDV: {result.selective_hdv_path}")
        typer.echo(f"\nMetrics:")
        typer.echo(f"  Variants:      {result.num_variants}")
        typer.echo(f"  HDV Size:      {result.hdv_size_bytes} bytes")
        typer.echo(f"  Compression:   {result.compression_ratio:.1f}x")
        typer.echo(f"\nTiming:")
        for layer, layer_time in result.layer_times.items():
            typer.echo(f"  {layer:15} {layer_time:.1f}s")
        typer.echo(f"  {'TOTAL':15} {result.total_time:.1f}s")

        if result.zk_proof:
            typer.echo(f"\nZK Proof:")
            typer.echo(f"  Circuit:  {result.zk_proof.circuit_name}")
            typer.echo(f"  Size:     {len(result.zk_proof.proof_data)} bytes")

    else:
        typer.echo("Pipeline FAILED!", err=True)
        typer.echo(f"{'='*60}", err=True)
        typer.echo(f"Error: {result.error_message}", err=True)
        raise typer.Exit(1)


@app.command()
def config(
    guides: Annotated[Path, typer.Option(
        "--guides", "-g",
        help="Path to guide strand directory",
    )] = Path("data/guide_strands"),
    output: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Output directory",
    )] = Path("pipeline_output"),
    save: Annotated[Optional[Path], typer.Option(
        "--save", "-s",
        help="Save configuration to JSON file",
    )] = None,
):
    """Show or save pipeline configuration."""
    from genomevault.pipelines.unified_pipeline import PipelineConfig

    config = PipelineConfig(
        output_dir=output,
        guide_fasta_dir=guides,
    )

    config_dict = {
        "output_dir": str(config.output_dir),
        "guide_fasta_dir": str(config.guide_fasta_dir),
        "hdc_dimension": config.hdc_dimension,
        "hdc_chunk_size": config.hdc_chunk_size,
        "hdc_num_banks": config.hdc_num_banks,
        "enable_zk": config.enable_zk,
        "zk_circuit": config.zk_circuit,
        "enable_pir": config.enable_pir,
        "pir_num_servers": config.pir_num_servers,
        "threads": config.threads,
        "k_anonymity": config.k_anonymity,
        "min_base_quality": config.min_base_quality,
        "min_mapping_quality": config.min_mapping_quality,
    }

    if save:
        with open(save, 'w') as f:
            json.dump(config_dict, f, indent=2)
        typer.echo(f"Configuration saved to: {save}")
    else:
        typer.echo("Pipeline Configuration:")
        typer.echo("-" * 40)
        for key, value in config_dict.items():
            typer.echo(f"  {key:25} {value}")


@app.command()
def status(
    guides: Annotated[Path, typer.Option(
        "--guides", "-g",
        help="Path to guide strand directory to check",
    )] = Path("data/guide_strands"),
):
    """Check component status and readiness."""
    typer.echo("\nComponent Status Check")
    typer.echo("=" * 50)

    # Check guide strands
    typer.echo("\nLayer 2 (Guide Strands):")
    if guides.exists():
        fastas = list(guides.glob("*.fa.gz")) + list(guides.glob("*.fasta.gz"))
        bams = list(guides.glob("*.bam"))
        typer.echo(f"  Directory:    {guides}")
        typer.echo(f"  FASTA files:  {len(fastas)}")
        typer.echo(f"  BAM files:    {len(bams)}")
        if len(fastas) >= 2:
            typer.echo(f"  Status:       OK (k={len(fastas)})")
        else:
            typer.echo(f"  Status:       INSUFFICIENT (need k>=2)")
    else:
        typer.echo(f"  Directory:    {guides} NOT FOUND")
        typer.echo(f"  Status:       MISSING")

    # Check HDC encoder
    typer.echo("\nLayer 5 (HDC Encoder):")
    try:
        from genomevault.hypervector_transform import AdaptiveEncoder
        typer.echo(f"  AdaptiveEncoder:  OK")
    except ImportError as e:
        typer.echo(f"  AdaptiveEncoder:  FAILED ({e})")

    # Check ZK Prover
    typer.echo("\nLayer 6 (ZK Proofs):")
    try:
        from genomevault.zk_proofs.prover import Prover, CIRCOM_AVAILABLE
        prover = Prover()
        status = "PRODUCTION" if prover.is_production_ready else "MOCK"
        typer.echo(f"  Prover:           OK ({status})")
        typer.echo(f"  Circom backend:   {'available' if CIRCOM_AVAILABLE else 'unavailable'}")
    except Exception as e:
        typer.echo(f"  Prover:           FAILED ({e})")

    # Check PIR
    typer.echo("\nLayer 7 (PIR):")
    try:
        from genomevault.pir.it_pir_protocol import PIRProtocol, PIRParameters
        params = PIRParameters(database_size=100, num_servers=2)
        pir = PIRProtocol(params)
        typer.echo(f"  PIRProtocol:      OK (2-server IT-PIR)")
    except Exception as e:
        typer.echo(f"  PIRProtocol:      FAILED ({e})")

    typer.echo("\n" + "=" * 50)


@app.command("layer")
def run_layer(
    layer: Annotated[int, typer.Argument(help="Layer number (1-7)")],
    input_path: Annotated[Path, typer.Option(
        "--input", "-i",
        help="Input file or directory",
    )],
    output_path: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Output file or directory",
    )],
    guides: Annotated[Path, typer.Option(
        "--guides", "-g",
        help="Guide strand directory (for layers 3-5)",
    )] = Path("data/guide_strands"),
):
    """
    Run a specific pipeline layer.

    Layers:
    - 1: Build Byzantine consensus
    - 2: Create guide strands
    - 3: Align experimental data
    - 4: Generate GDiff encoding
    - 5: Create HDC encoding
    - 6: Generate ZK proof
    - 7: PIR query
    """
    from genomevault.pipelines.unified_pipeline import UnifiedPipeline, PipelineConfig

    if layer < 1 or layer > 7:
        typer.echo(f"Error: Layer must be 1-7, got {layer}", err=True)
        raise typer.Exit(1)

    config = PipelineConfig(
        output_dir=output_path.parent if output_path.suffix else output_path,
        guide_fasta_dir=guides,
    )
    pipeline = UnifiedPipeline(config)

    layer_names = {
        1: "Byzantine Consensus",
        2: "Guide Strand Creation",
        3: "Experimental Alignment",
        4: "GDiff Encoding",
        5: "HDC Encoding",
        6: "ZK Proof Generation",
        7: "PIR Query",
    }

    typer.echo(f"\nRunning Layer {layer}: {layer_names[layer]}")
    typer.echo("=" * 50)

    start_time = time.time()

    try:
        if layer == 1:
            # Consensus building
            refs = list(input_path.glob("*.fa")) + list(input_path.glob("*.fa.gz"))
            result = pipeline.build_consensus(refs, output_path)
            typer.echo(f"Output: {result}")

        elif layer == 2:
            # Guide strand creation
            bams = list(input_path.glob("*.bam"))
            result = pipeline.create_guide_strands(bams, output_path)
            typer.echo(f"Created {len(result)} guide strands")

        elif layer == 3:
            # Experimental alignment
            # Expect input_path to contain R1.fq.gz and R2.fq.gz
            r1 = list(input_path.glob("*R1*.f*.gz"))[0]
            r2 = list(input_path.glob("*R2*.f*.gz"))[0]
            result = pipeline.align_experimental(r1, r2, output_bam=output_path)
            typer.echo(f"Output: {result}")

        elif layer == 4:
            # GDiff encoding
            gdiff, path = pipeline.encode_gdiff(input_path, output_path=output_path)
            typer.echo(f"Output: {path}")
            typer.echo(f"Variants: {len(gdiff.differential_variants) if gdiff.differential_variants else 0}")

        elif layer == 5:
            # HDC encoding
            result = pipeline.encode_hdc(input_path, output_path=output_path)
            typer.echo(f"Output: {result}")

        elif layer == 6:
            # ZK proof
            variant = {"chrom": "chr1", "pos": 12345, "ref": "A", "alt": "T"}
            proof = pipeline.generate_zk_proof(variant)
            if proof:
                with open(output_path, 'w') as f:
                    json.dump(proof.to_dict(), f, indent=2)
                typer.echo(f"Output: {output_path}")
            else:
                typer.echo("ZK proofs disabled")

        elif layer == 7:
            # PIR query
            import numpy as np
            # Create dummy database for testing
            db = np.random.randint(0, 256, (100, 64), dtype=np.uint8)
            result = pipeline.pir_query(db, 42)
            typer.echo(f"PIR query result: {result[:10]}...")

        elapsed = time.time() - start_time
        typer.echo(f"\nLayer {layer} completed in {elapsed:.1f}s")

    except Exception as e:
        typer.echo(f"\nLayer {layer} failed: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
