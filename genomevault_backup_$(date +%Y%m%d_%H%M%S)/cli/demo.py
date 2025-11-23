"""Demo and example CLI commands."""

import typer
from typing import Optional
from pathlib import Path
import json
import time
import numpy as np

app = typer.Typer(help="Demo and example operations")

@app.command("simple")
def simple(
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Run a simple demonstration of core features."""
    from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig
    from genomevault.core.constants import OmicsType
    from genomevault.pir.servers import PIRServer
    
    typer.echo("🚀 Running Simple GenomeVault Demo")
    typer.echo("=" * 50)
    
    # 1. HDC Encoding
    typer.echo("\n1. Hyperdimensional Computing (HDC) Encoding")
    config = HypervectorConfig(dimension=1000)
    encoder = HypervectorEncoder(config=config)
    
    # Generate sample genomic data
    sample_data = np.random.randn(10).astype(np.float32)
    
    start = time.perf_counter()
    encoded = encoder.encode(sample_data, OmicsType.GENOMIC)
    encode_time = (time.perf_counter() - start) * 1000
    
    if hasattr(encoded, 'detach'):
        encoded_array = encoded.detach().cpu().numpy()
    else:
        encoded_array = np.array(encoded)
    
    sparsity = np.sum(encoded_array == 0) / len(encoded_array)
    typer.echo(f"  ✅ Encoded 10 features to 1000D hypervector")
    typer.echo(f"  ⏱️  Time: {encode_time:.2f} ms")
    typer.echo(f"  📊 Sparsity: {sparsity:.1%}")
    
    # 2. PIR Demo
    typer.echo("\n2. Private Information Retrieval (PIR)")
    records = [f"variant_{i}".encode() for i in range(100)]
    server = PIRServer(records)
    
    # Query for index 42
    mask = np.zeros(100, dtype=np.uint8)
    mask[42] = 1
    
    start = time.perf_counter()
    result = server.answer(mask)
    pir_time = (time.perf_counter() - start) * 1000
    
    typer.echo(f"  ✅ Retrieved record privately from 100 records")
    typer.echo(f"  ⏱️  Time: {pir_time:.2f} ms")
    decoded_result = result.rstrip(b'\0').decode()
    typer.echo(f"  🔒 Result: {decoded_result}")
    
    # 3. Summary
    typer.echo("\n" + "=" * 50)
    typer.echo("📊 Demo Summary:")
    typer.echo(f"  HDC Encoding: {encode_time:.2f} ms")
    typer.echo(f"  PIR Query: {pir_time:.2f} ms")
    typer.echo(f"  Total: {encode_time + pir_time:.2f} ms")
    typer.echo("\n✅ Demo completed successfully!")
    
    # Save results if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results = {
            "demo": "simple",
            "timestamp": time.time(),
            "hdc": {
                "dimension": 1000,
                "sparsity": float(sparsity),
                "time_ms": encode_time
            },
            "pir": {
                "records": 100,
                "time_ms": pir_time
            }
        }
        
        output_file = output_dir / "demo_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        typer.echo(f"\nResults saved to: {output_file}")

@app.command("full")
def full(
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Run a full E2E demonstration."""
    typer.echo("🚀 Running Full E2E GenomeVault Demo")
    typer.echo("=" * 50)
    
    # Import E2E demo script
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        from scripts.run_e2e_demo import run_complete_e2e_demo
        
        # Run the demo
        results = run_complete_e2e_demo(output_dir or Path("/tmp/genomevault_demo"))
        
        typer.echo("\n✅ Full E2E demo completed successfully!")
        typer.echo(f"Results saved to: {output_dir or '/tmp/genomevault_demo'}")
        
    except ImportError:
        typer.echo("❌ E2E demo script not found")
        typer.echo("Running simplified version instead...")
        simple(output_dir)

@app.command("clinical")
def clinical(
    scenario: str = typer.Option("diabetes", "--scenario", "-s", help="Clinical scenario"),
):
    """Run clinical use case demonstrations."""
    typer.echo(f"🏥 Running Clinical Demo: {scenario}")
    typer.echo("=" * 50)
    
    if scenario == "diabetes":
        typer.echo("\nDiabetes Risk Assessment Demo")
        typer.echo("-" * 30)
        
        # Simulate diabetes risk calculation
        risk_variants = ["TCF7L2", "PPARG", "KCNJ11", "FTO", "SLC30A8"]
        risk_scores = np.random.uniform(0.8, 1.3, len(risk_variants))
        
        typer.echo("\nVariant Analysis:")
        for variant, score in zip(risk_variants, risk_scores):
            risk_level = "↑" if score > 1.0 else "↓"
            typer.echo(f"  {variant}: {score:.2f} {risk_level}")
        
        overall_risk = np.prod(risk_scores)
        typer.echo(f"\nOverall Risk Score: {overall_risk:.2f}")
        
        if overall_risk > 1.2:
            typer.echo("⚠️  Elevated risk - recommend lifestyle intervention")
        else:
            typer.echo("✅ Normal risk profile")
    
    elif scenario == "pharmacogenomics":
        typer.echo("\nPharmacogenomics Demo")
        typer.echo("-" * 30)
        
        drugs = {
            "Warfarin": "CYP2C9*2 - Reduced metabolism",
            "Clopidogrel": "CYP2C19*2 - Poor metabolizer",
            "Simvastatin": "SLCO1B1 - Normal function"
        }
        
        typer.echo("\nDrug Metabolism Profile:")
        for drug, status in drugs.items():
            typer.echo(f"  {drug}: {status}")
        
    else:
        typer.echo(f"Unknown scenario: {scenario}")
        typer.echo("Available: diabetes, pharmacogenomics")