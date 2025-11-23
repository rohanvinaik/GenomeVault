"""Zero-knowledge proof CLI commands."""

import typer
from typing import Optional
from pathlib import Path
import json

app = typer.Typer(help="Zero-knowledge proof operations")

@app.command()
def prove(
    circuit: str = typer.Argument(..., help="Circuit name"),
    public_input: Path = typer.Option(..., "--public", "-p", help="Public input JSON file"),
    private_input: Path = typer.Option(..., "--private", "-s", help="Private input JSON file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output proof file"),
):
    """Generate a zero-knowledge proof."""
    from genomevault.zk_proofs.prover import Prover
    
    typer.echo(f"Generating proof for circuit: {circuit}")
    
    # Load inputs
    with open(public_input) as f:
        public = json.load(f)
    with open(private_input) as f:
        private = json.load(f)
    
    # Generate proof
    prover = Prover(use_circom=True)
    proof = prover.generate_proof(circuit_name=circuit, public_inputs=public, private_inputs=private)
    
    # Save proof
    if output:
        with open(output, 'w') as f:
            json.dump(proof.to_dict() if hasattr(proof, 'to_dict') else {"proof": str(proof)}, f, indent=2)
        typer.echo(f"Proof saved to: {output}")
    else:
        typer.echo(json.dumps(proof.to_dict() if hasattr(proof, 'to_dict') else {"proof": str(proof)}, indent=2))

@app.command()
def verify(
    circuit: str = typer.Argument(..., help="Circuit name"),
    proof: Path = typer.Option(..., "--proof", "-p", help="Proof JSON file"),
    public_input: Path = typer.Option(..., "--public", help="Public input JSON file"),
):
    """Verify a zero-knowledge proof."""
    from genomevault.zk_proofs.verifier import Verifier
    
    typer.echo(f"Verifying proof for circuit: {circuit}")
    
    # Load inputs
    with open(proof) as f:
        proof_data = json.load(f)
    with open(public_input) as f:
        public = json.load(f)
    
    # Verify proof
    verifier = Verifier()
    is_valid = verifier.verify(circuit_name=circuit, proof=proof_data, public_inputs=public)
    
    if is_valid:
        typer.echo("✅ Proof is VALID")
    else:
        typer.echo("❌ Proof is INVALID")
        raise typer.Exit(1)