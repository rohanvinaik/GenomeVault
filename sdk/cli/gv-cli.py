#!/usr/bin/env python3
"""
GenomeVault CLI Tool

Command-line interface for interacting with the GenomeVault API.
Provides convenient commands for encoding, querying, and analysis operations.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.json import JSON

# Import SDK (assuming it's installed)
try:
    from genomevault_sdk import GenomeVaultClient, GenomicVariant
    from genomevault_sdk.exceptions import GenomeVaultAPIError
except ImportError:
    print("GenomeVault SDK not found. Please install with: pip install genomevault-sdk")
    sys.exit(1)


console = Console()


class Config:
    """Configuration management for the CLI."""

    def __init__(self):
        self.config_file = Path.home() / ".genomevault" / "config.yaml"
        self.config_file.parent.mkdir(exist_ok=True)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load config: {e}[/yellow]")
        return {}

    def save(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, "w") as f:
                yaml.dump(self._config, f)
        except Exception as e:
            console.print(f"[red]Error saving config: {e}[/red]")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value

    def get_client_config(self) -> Dict[str, Any]:
        """Get client configuration."""
        return {
            "base_url": self.get("base_url", "https://api.genomevault.io"),
            "api_key": self.get("api_key"),
            "timeout": self.get("timeout", 30.0),
        }


config = Config()


def create_client() -> GenomeVaultClient:
    """Create API client from configuration."""
    client_config = config.get_client_config()

    if not client_config["api_key"]:
        console.print("[red]API key not configured. Use 'gv config set-api-key <key>'[/red]")
        sys.exit(1)

    return GenomeVaultClient(**client_config)


def handle_api_error(error: GenomeVaultAPIError) -> None:
    """Handle API errors with user-friendly messages."""
    if hasattr(error, "status_code"):
        if error.status_code == 401:
            console.print("[red]Authentication failed. Check your API key.[/red]")
        elif error.status_code == 429:
            console.print("[yellow]Rate limit exceeded. Try again later.[/yellow]")
        elif error.status_code >= 500:
            console.print("[red]Server error. Please try again.[/red]")
        else:
            console.print(f"[red]API error: {error.message}[/red]")
    else:
        console.print(f"[red]Error: {error.message}[/red]")

    if hasattr(error, "request_id") and error.request_id:
        console.print(f"[dim]Request ID: {error.request_id}[/dim]")


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """GenomeVault CLI - Privacy-preserving genomic computing."""
    pass


@cli.group()
def config():
    """Configuration management."""
    pass


@config.command("show")
def config_show():
    """Show current configuration."""
    table = Table(title="GenomeVault Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    client_config = config.get_client_config()
    for key, value in client_config.items():
        if key == "api_key" and value:
            value = f"{value[:8]}..." if len(value) > 8 else value
        table.add_row(key, str(value) if value else "[dim]Not set[/dim]")

    console.print(table)


@config.command("set-api-key")
@click.argument("api_key")
def config_set_api_key(api_key: str):
    """Set API key for authentication."""
    config.set("api_key", api_key)
    config.save()
    console.print("[green]API key configured successfully.[/green]")


@config.command("set-base-url")
@click.argument("base_url")
def config_set_base_url(base_url: str):
    """Set base URL for API endpoint."""
    config.set("base_url", base_url)
    config.save()
    console.print(f"[green]Base URL set to: {base_url}[/green]")


@cli.command()
def health():
    """Check API health status."""
    try:
        client = create_client()

        with console.status("[bold green]Checking health..."):
            result = asyncio.run(client.health_check())

        # Create status table
        table = Table(title="GenomeVault API Health")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")

        # Overall status
        status_color = "green" if result.status == "healthy" else "red"
        table.add_row("Overall", f"[{status_color}]{result.status.title()}[/{status_color}]")
        table.add_row("Version", result.version)

        # Service status
        if result.services:
            for service, status in result.services.items():
                service_color = "green" if status == "healthy" else "red"
                table.add_row(
                    f"  {service}", f"[{service_color}]{status.title()}[/{service_color}]"
                )

        console.print(table)

    except GenomeVaultAPIError as e:
        handle_api_error(e)
        sys.exit(1)


@cli.group()
def encode():
    """Hypervector encoding operations."""
    pass


@encode.command("variants")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output file (default: stdout)")
@click.option("--dim", default=8192, help="Hypervector dimension")
@click.option("--binary", is_flag=True, help="Return binary vectors")
@click.option("--format", "output_format", default="json", type=click.Choice(["json", "csv"]))
def encode_variants(
    input_file: str, output: Optional[str], dim: int, binary: bool, output_format: str
):
    """Encode genomic variants from VCF or JSON file."""
    try:
        client = create_client()

        # Load variants from file
        with console.status("[bold green]Loading variants..."):
            variants = load_variants_from_file(input_file)

        if not variants:
            console.print("[red]No variants found in input file.[/red]")
            sys.exit(1)

        console.print(f"[green]Loaded {len(variants)} variants[/green]")

        # Encode variants
        with console.status("[bold green]Encoding variants..."):
            result = asyncio.run(client.encode_variants(variants, dim=dim, binary=binary))

        # Format output
        if output_format == "json":
            output_data = {
                "dim": result.dim,
                "binary": result.binary,
                "vector": result.vector,
                "privacy_level": getattr(result, "privacy_level", None),
                "compression_ratio": getattr(result, "compression_ratio", None),
            }
            output_text = json.dumps(output_data, indent=2)
        else:  # csv
            output_text = ",".join(map(str, result.vector))

        # Write output
        if output:
            with open(output, "w") as f:
                f.write(output_text)
            console.print(f"[green]Results written to {output}[/green]")
        else:
            console.print(output_text)

    except GenomeVaultAPIError as e:
        handle_api_error(e)
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@encode.command("numeric")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", help="Output file (default: stdout)")
@click.option("--dim", default=8192, help="Hypervector dimension")
@click.option("--binary", is_flag=True, help="Return binary vectors")
def encode_numeric(input_file: str, output: Optional[str], dim: int, binary: bool):
    """Encode numeric features from CSV or JSON file."""
    try:
        client = create_client()

        # Load numeric data
        with console.status("[bold green]Loading numeric data..."):
            numeric_data = load_numeric_from_file(input_file)

        console.print(f"[green]Loaded {len(numeric_data)} features[/green]")

        # Encode data
        with console.status("[bold green]Encoding features..."):
            result = asyncio.run(client.encode_numeric(numeric_data, dim=dim, binary=binary))

        # Format output
        output_data = {
            "dim": result.dim,
            "binary": result.binary,
            "vector": result.vector,
            "privacy_level": getattr(result, "privacy_level", None),
            "compression_ratio": getattr(result, "compression_ratio", None),
        }
        output_text = json.dumps(output_data, indent=2)

        # Write output
        if output:
            with open(output, "w") as f:
                f.write(output_text)
            console.print(f"[green]Results written to {output}[/green]")
        else:
            console.print(output_text)

    except GenomeVaultAPIError as e:
        handle_api_error(e)
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.group()
def pir():
    """Private Information Retrieval operations."""
    pass


@pir.command("query")
@click.argument("index", type=int)
@click.option("--output", "-o", help="Output file for retrieved data")
@click.option("--timeout", default=30, help="Query timeout in seconds")
def pir_query(index: int, output: Optional[str], timeout: int):
    """Execute PIR query for specified index."""
    try:
        client = create_client()

        with console.status(f"[bold green]Executing PIR query for index {index}..."):
            result = asyncio.run(client.pir_query(index, timeout_seconds=timeout))

        # Decode base64 data
        import base64

        decoded_data = base64.b64decode(result.item_base64)

        # Display results
        table = Table(title="PIR Query Results")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Index", str(result.index))
        table.add_row("Data Size", f"{len(decoded_data)} bytes")
        if hasattr(result, "query_time_ms") and result.query_time_ms:
            table.add_row("Query Time", f"{result.query_time_ms}ms")

        console.print(table)

        # Save data if output specified
        if output:
            with open(output, "wb") as f:
                f.write(decoded_data)
            console.print(f"[green]Data saved to {output}[/green]")
        else:
            # Try to decode as text
            try:
                text_data = decoded_data.decode("utf-8")
                console.print(f"\n[bold]Retrieved Data:[/bold]\n{text_data}")
            except UnicodeDecodeError:
                console.print(f"\n[bold]Retrieved Data (hex):[/bold]\n{decoded_data.hex()}")

    except GenomeVaultAPIError as e:
        handle_api_error(e)
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.group()
def proof():
    """Zero-knowledge proof operations."""
    pass


@proof.command("generate")
@click.argument("proof_type", type=click.Choice(["genomic", "clinical", "research"]))
@click.option("--public-inputs", required=True, help="JSON file with public inputs")
@click.option("--private-hash", required=True, help="SHA-256 hash of private inputs")
@click.option("--output", "-o", help="Output file for proof")
def proof_generate(proof_type: str, public_inputs: str, private_hash: str, output: Optional[str]):
    """Generate zero-knowledge proof."""
    try:
        client = create_client()

        # Load public inputs
        with open(public_inputs) as f:
            public_data = json.load(f)

        with console.status(f"[bold green]Generating {proof_type} proof..."):
            result = asyncio.run(
                client.generate_proof(
                    proof_type=proof_type,
                    public_inputs=public_data,
                    private_inputs_hash=private_hash,
                )
            )

        # Display results
        table = Table(title="Proof Generation Results")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Proof ID", result.proof_id)
        table.add_row("Proof Type", proof_type)
        if hasattr(result, "validity_period_hours") and result.validity_period_hours:
            table.add_row("Valid For", f"{result.validity_period_hours} hours")

        console.print(table)

        # Save proof data
        proof_data = {
            "proof_id": result.proof_id,
            "proof_data": result.proof_data,
            "verification_key": result.verification_key,
            "public_signals": result.public_signals,
        }

        if output:
            with open(output, "w") as f:
                json.dump(proof_data, f, indent=2)
            console.print(f"[green]Proof saved to {output}[/green]")
        else:
            console.print(JSON.from_data(proof_data))

    except GenomeVaultAPIError as e:
        handle_api_error(e)
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def load_variants_from_file(file_path: str) -> List[GenomicVariant]:
    """Load genomic variants from file (VCF or JSON)."""
    path = Path(file_path)

    if path.suffix.lower() == ".vcf":
        return load_variants_from_vcf(file_path)
    elif path.suffix.lower() == ".json":
        with open(file_path) as f:
            data = json.load(f)

        variants = []
        for item in data if isinstance(data, list) else [data]:
            variant = GenomicVariant(**item)
            variants.append(variant)
        return variants
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def load_variants_from_vcf(file_path: str) -> List[GenomicVariant]:
    """Load variants from VCF file."""
    try:
        import pysam
    except ImportError:
        raise ImportError("pysam required for VCF parsing. Install with: pip install pysam")

    variants = []
    with pysam.VariantFile(file_path) as vcf:
        for record in vcf.fetch():
            variant = GenomicVariant(
                chrom=record.chrom,
                pos=record.pos,
                ref=record.ref,
                alt=record.alts[0] if record.alts else ".",
                quality=record.qual,
            )
            variants.append(variant)

    return variants


def load_numeric_from_file(file_path: str) -> List[float]:
    """Load numeric data from file (CSV or JSON)."""
    path = Path(file_path)

    if path.suffix.lower() == ".csv":
        import csv

        with open(file_path) as f:
            reader = csv.reader(f)
            # Assume first row might be header, try to parse as numbers
            data = []
            for row in reader:
                try:
                    numbers = [float(x) for x in row]
                    data.extend(numbers)
                except ValueError:
                    # Skip header row
                    continue
        return data
    elif path.suffix.lower() == ".json":
        with open(file_path) as f:
            data = json.load(f)

        if isinstance(data, list):
            return [float(x) for x in data]
        else:
            raise ValueError("JSON file must contain an array of numbers")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


if __name__ == "__main__":
    cli()
