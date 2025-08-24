#!/usr/bin/env python3
"""
Demo runner for Docker Compose stack.
"""

import json
import time
import sys
from pathlib import Path
import requests
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer()
console = Console()

class DemoRunner:
    """Run demo sequence against API."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()
        
    def wait_for_api(self, timeout: int = 60):
        """Wait for API to be ready."""
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                resp = self.session.get(f"{self.api_url}/health")
                if resp.status_code == 200:
                    return True
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(2)
        
        return False
    
    def run_demo(self):
        """Run complete demo sequence."""
        
        console.print("\n[bold cyan]🧬 GenomeVault Demo Runner[/bold cyan]")
        console.print("=" * 40)
        
        # Wait for API
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Waiting for API...", total=None)
            
            if not self.wait_for_api():
                console.print("[red]❌ API not responding[/red]")
                return False
            
            progress.update(task, completed=True)
        
        console.print("[green]✅ API is ready[/green]\n")
        
        # 1. Test HDC Compression
        console.print("[bold]1. Testing HDC Compression[/bold]")
        
        test_variants = [
            {"chr": "1", "pos": 14370, "ref": "G", "alt": "A"},
            {"chr": "1", "pos": 17330, "ref": "T", "alt": "A"},
            {"chr": "2", "pos": 20000, "ref": "C", "alt": "T"}
        ]
        
        resp = self.session.post(
            f"{self.api_url}/api/v1/hdc/compress",
            json={"variants": test_variants}
        )
        
        if resp.status_code == 200:
            result = resp.json()
            console.print(f"   Input: {len(test_variants)} variants")
            console.print(f"   Compressed size: {result.get('size_bytes', 0)} bytes")
            console.print(f"   [green]✅ Compression successful[/green]\n")
        else:
            console.print(f"   [red]❌ Compression failed: {resp.status_code}[/red]\n")
        
        # 2. Test ZK Proof
        console.print("[bold]2. Testing Zero-Knowledge Proof[/bold]")
        
        proof_request = {
            "circuit": "variant_presence",
            "inputs": {
                "variants": test_variants,
                "query": test_variants[0]
            }
        }
        
        resp = self.session.post(
            f"{self.api_url}/api/v1/zk/prove",
            json=proof_request
        )
        
        if resp.status_code == 200:
            proof = resp.json()
            console.print(f"   Circuit: {proof.get('circuit')}")
            console.print(f"   Proof ID: {proof.get('proof_id', 'N/A')[:8]}...")
            
            # Verify proof
            verify_resp = self.session.post(
                f"{self.api_url}/api/v1/zk/verify",
                json={"proof_id": proof.get("proof_id")}
            )
            
            if verify_resp.status_code == 200:
                console.print(f"   [green]✅ Proof verified[/green]\n")
            else:
                console.print(f"   [yellow]⚠️  Verification failed[/yellow]\n")
        else:
            console.print(f"   [red]❌ Proof generation failed[/red]\n")
        
        # 3. Test PIR Query
        console.print("[bold]3. Testing Private Information Retrieval[/bold]")
        
        pir_query = {
            "database": "reference_genome",
            "query": {"gene": "BRCA1"}
        }
        
        resp = self.session.post(
            f"{self.api_url}/api/v1/pir/query",
            json=pir_query
        )
        
        if resp.status_code == 200:
            result = resp.json()
            console.print(f"   Query: BRCA1")
            console.print(f"   Result retrieved (privacy preserved)")
            console.print(f"   [green]✅ PIR successful[/green]\n")
        else:
            console.print(f"   [yellow]⚠️  PIR not available[/yellow]\n")
        
        # 4. Check metrics
        console.print("[bold]4. Performance Metrics[/bold]")
        
        resp = self.session.get(f"{self.api_url}/api/v1/metrics/summary")
        if resp.status_code == 200:
            metrics = resp.json()
            console.print(f"   Total operations: {metrics.get('total_operations', 0)}")
            console.print(f"   Average latency: {metrics.get('avg_latency_ms', 0):.2f}ms")
            console.print(f"   [green]✅ Metrics available[/green]\n")
        
        console.print("=" * 40)
        console.print("[bold green]✅ Demo Complete![/bold green]\n")
        console.print("Access points:")
        console.print(f"  API:        {self.api_url}")
        console.print(f"  API Docs:   {self.api_url}/docs")
        console.print(f"  Grafana:    http://localhost:3000 (admin/admin)")
        console.print(f"  Prometheus: http://localhost:9090")
        
        return True

@app.command()
def run(
    api_url: str = typer.Option(
        "http://localhost:8000",
        "--api-url",
        help="API URL to test against"
    )
):
    """Run demo sequence."""
    runner = DemoRunner(api_url)
    success = runner.run_demo()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    app()