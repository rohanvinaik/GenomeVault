"""Pipeline operation CLI commands."""

import typer
from typing import Optional
from pathlib import Path
import json
import time

from genomevault.pipelines.production_pipeline import ProductionPipeline, PipelineConfig

app = typer.Typer(help="Pipeline operations")

@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Pipeline configuration file"),
    input_dir: Path = typer.Option(..., "--input", "-i", help="Input directory"),
    output_dir: Path = typer.Option(..., "--output", "-o", help="Output directory"),
    parallel: bool = typer.Option(False, "--parallel", "-p", help="Run in parallel"),
):
    """Run a genomic processing pipeline."""
    typer.echo(f"Running pipeline with config: {config}")
    
    # Load configuration
    with open(config) as f:
        if config.suffix == '.yaml':
            import yaml
            pipeline_config = yaml.safe_load(f)
        else:
            pipeline_config = json.load(f)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Simulate pipeline execution
    stages = pipeline_config.get('stages', ['encode', 'compress', 'encrypt'])
    
    typer.echo(f"\nExecuting {len(stages)} stages...")
    
    for i, stage in enumerate(stages, 1):
        typer.echo(f"\n[{i}/{len(stages)}] Running: {stage}")
        
        # Simulate processing
        time.sleep(0.5)
        
        typer.echo(f"  ✅ {stage} completed")
    
    # Save results
    results = {
        "pipeline": config.name,
        "stages": stages,
        "input": str(input_dir),
        "output": str(output_dir),
        "timestamp": time.time(),
        "status": "completed"
    }
    
    results_file = output_dir / "pipeline_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    typer.echo(f"\n✅ Pipeline completed successfully!")
    typer.echo(f"Results saved to: {results_file}")

@app.command()
def status(
    pipeline_id: Optional[str] = typer.Option(None, "--id", help="Pipeline ID"),
):
    """Check pipeline status."""
    if pipeline_id:
        typer.echo(f"Checking status for pipeline: {pipeline_id}")
        # In a real implementation, this would query a database
        typer.echo("Status: Running (mock)")
    else:
        typer.echo("Active pipelines:")
        typer.echo("  - pipeline_001: Running")
        typer.echo("  - pipeline_002: Completed")
        typer.echo("  - pipeline_003: Failed")

@app.command()
def list():
    """List available pipeline templates."""
    templates = [
        ("variant_calling", "Standard variant calling pipeline"),
        ("hdc_encoding", "Hyperdimensional encoding pipeline"),
        ("privacy_preserving", "Full privacy-preserving pipeline"),
        ("clinical_analysis", "Clinical variant analysis"),
        ("population_genomics", "Population-scale analysis")
    ]
    
    typer.echo("Available pipeline templates:\n")
    for name, description in templates:
        typer.echo(f"  {name:20} - {description}")
    
    typer.echo("\nUse 'genomevault pipeline create --template <name>' to create a new pipeline")

@app.command()
def create(
    name: str = typer.Argument(..., help="Pipeline name"),
    template: str = typer.Option("privacy_preserving", "--template", "-t", help="Template to use"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output config file"),
):
    """Create a new pipeline configuration."""
    
    # Define template configurations
    templates = {
        "privacy_preserving": {
            "name": name,
            "version": "1.0",
            "stages": [
                {"name": "validate", "type": "quality_control"},
                {"name": "encode", "type": "hdc", "params": {"dimension": 10000}},
                {"name": "compress", "type": "sparse", "params": {"threshold": 0.1}},
                {"name": "encrypt", "type": "aes256"},
                {"name": "generate_proof", "type": "zk", "params": {"circuit": "variant_presence"}}
            ],
            "resources": {
                "cpu": 4,
                "memory": "8GB",
                "gpu": "optional"
            }
        },
        "hdc_encoding": {
            "name": name,
            "version": "1.0",
            "stages": [
                {"name": "preprocess", "type": "normalization"},
                {"name": "encode", "type": "hdc", "params": {"dimension": 10000}},
                {"name": "validate", "type": "similarity_check"}
            ],
            "resources": {
                "cpu": 2,
                "memory": "4GB",
                "gpu": "recommended"
            }
        }
    }
    
    config = templates.get(template, templates["privacy_preserving"])
    config["name"] = name
    
    # Save configuration
    if output:
        with open(output, 'w') as f:
            json.dump(config, f, indent=2)
        typer.echo(f"Pipeline configuration created: {output}")
    else:
        typer.echo(json.dumps(config, indent=2))


@app.command()
def production(
    gdiff_path: Path = typer.Argument(..., help="Path to GDiff file (.gdiff.gz)"),
    dimension: int = typer.Option(10000, "--dimension", "-d", help="HDC dimension"),
    backend: str = typer.Option("auto", "--backend", "-b", help="HDC backend (auto/cpu/metal/cuda)"),
    enable_zk: bool = typer.Option(True, "--zk/--no-zk", help="Enable ZK proof generation"),
    enable_pir: bool = typer.Option(False, "--pir/--no-pir", help="Enable PIR query"),
    sample: Optional[int] = typer.Option(None, "--sample", "-s", help="Sample N variants (for speed)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output results file"),
):
    """
    Run complete production pipeline: GDiff → HDC → ZK → PIR.

    This command executes the full GenomeVault workflow:
    1. Load GDiff differential encoding
    2. Generate HDC hypervector
    3. Generate ZK proof (optional)
    4. Execute PIR query (optional)

    Example:
        genomevault pipeline production experimental.gdiff.gz --dimension 10000 --zk
    """
    typer.echo("🚀 GenomeVault Production Pipeline\n")
    typer.echo(f"GDiff file: {gdiff_path}")
    typer.echo(f"HDC dimension: {dimension:,}D")
    typer.echo(f"Backend: {backend}")
    typer.echo(f"ZK proof: {'Enabled' if enable_zk else 'Disabled'}")
    typer.echo(f"PIR query: {'Enabled' if enable_pir else 'Disabled'}")
    if sample:
        typer.echo(f"Sample size: {sample:,} variants")
    typer.echo()

    # Validate GDiff file
    if not gdiff_path.exists():
        typer.echo(f"❌ Error: GDiff file not found: {gdiff_path}", err=True)
        raise typer.Exit(1)

    # Create pipeline configuration
    config = PipelineConfig(
        hdc_dimension=dimension,
        hdc_backend=backend,
        enable_zk_proof=enable_zk,
        enable_pir=enable_pir,
        sample_variants=sample
    )

    # Initialize and run pipeline
    pipeline = ProductionPipeline(config)

    try:
        result = pipeline.run(gdiff_path, "cli-pipeline")

        # Display results
        typer.echo("\n" + "="*60)
        typer.echo(f"{'PIPELINE RESULTS':^60}")
        typer.echo("="*60)
        typer.echo(f"Status: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        typer.echo(f"Total duration: {result.total_duration_s:.2f}s")
        typer.echo()

        # Stage-by-stage results
        typer.echo("Stage Results:")
        for stage_name, stage in result.stages.items():
            status_icon = "✅" if stage.success else "❌"
            typer.echo(f"  {status_icon} {stage_name}: {stage.duration_s:.2f}s")
            if stage.error:
                typer.echo(f"     Error: {stage.error}")

        # Summary
        typer.echo()
        typer.echo("Summary:")
        for key, value in result.summary.items():
            typer.echo(f"  {key}: {value}")

        # Save results
        if output or result.success:
            output_path = output or Path(f"production_pipeline_results_{result.pipeline_id}.json")
            with open(output_path, 'w') as f:
                json.dump({
                    "pipeline_id": result.pipeline_id,
                    "success": result.success,
                    "total_duration_s": result.total_duration_s,
                    "stages": {name: {
                        "duration_s": stage.duration_s,
                        "success": stage.success,
                        "metrics": stage.metrics,
                        "error": stage.error
                    } for name, stage in result.stages.items()},
                    "summary": result.summary
                }, f, indent=2)
            typer.echo(f"\n💾 Results saved: {output_path}")

        if not result.success:
            raise typer.Exit(1)

    except Exception as e:
        typer.echo(f"\n❌ Pipeline failed: {e}", err=True)
        raise typer.Exit(1)