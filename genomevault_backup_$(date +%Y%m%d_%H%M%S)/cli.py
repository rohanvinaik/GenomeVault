#!/usr/bin/env python3
"""
GenomeVault Command Line Interface

Main entry point for the genomevault CLI tool.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create main CLI app
app = typer.Typer(
    name="genomevault",
    help="Privacy-preserving genomics platform",
    add_completion=False,
    rich_markup_mode="rich"
)

console = Console()

# Import and add main app with all subcommands
try:
    from genomevault.cli.main import app as main_app
    # Merge the main app's commands and subcommands
    for command_name, command in main_app.registered_commands:
        app.registered_commands.append((command_name, command))
    for group_name, group in main_app.registered_groups:
        app.registered_groups.append((group_name, group))
except ImportError as e:
    logger.warning(f"Could not import main CLI app: {e}")

# Add additional CLI modules
try:
    from genomevault.cli import zk, pir, hdc, demo, pipeline
    app.add_typer(zk.app, name="zk", help="Zero-knowledge proof operations")
    app.add_typer(pir.app, name="pir", help="Private Information Retrieval operations")
    app.add_typer(hdc.app, name="hdc", help="Hyperdimensional Computing operations")
    app.add_typer(demo.app, name="demo", help="Demo and example operations")
    app.add_typer(pipeline.app, name="pipeline", help="Pipeline operations")
except ImportError as e:
    logger.warning(f"Could not import CLI submodules: {e}")

@app.command()
def version():
    """Show version information."""
    try:
        from genomevault import __version__
        version_str = __version__
    except ImportError:
        version_str = "0.1.0"
    
    panel = Panel(
        f"[bold green]GenomeVault[/bold green] v{version_str}\n"
        f"Privacy-preserving genomics platform",
        title="Version Info",
        expand=False
    )
    console.print(panel)

@app.command()
def status():
    """Check system status and dependencies."""
    console.print("[bold]Checking system status...[/bold]")
    
    status = {}
    
    # Check Python version
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    status["Python"] = {
        'available': sys.version_info >= (3, 10),
        'details': python_version
    }
    
    # Check key dependencies
    dependencies = {
        "NumPy": "numpy",
        "PyTorch": "torch",
        "SQLAlchemy": "sqlalchemy",
        "FastAPI": "fastapi",
        "Typer": "typer",
        "Rich": "rich"
    }
    
    for name, module in dependencies.items():
        try:
            __import__(module)
            status[name] = {'available': True, 'details': 'Installed'}
        except ImportError:
            status[name] = {'available': False, 'details': 'Not installed'}
    
    # Check for Circom
    import subprocess
    try:
        result = subprocess.run(["circom", "--version"], capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            status["Circom"] = {'available': True, 'details': 'Installed'}
        else:
            status["Circom"] = {'available': False, 'details': 'Not found'}
    except (subprocess.SubprocessError, FileNotFoundError):
        status["Circom"] = {'available': False, 'details': 'Not found'}
    
    # Check for GPU
    try:
        import torch
        if torch.cuda.is_available():
            status["GPU"] = {'available': True, 'details': f'CUDA {torch.cuda.get_device_name(0)}'}
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            status["GPU"] = {'available': True, 'details': 'Apple Metal'}
        else:
            status["GPU"] = {'available': False, 'details': 'No GPU detected'}
    except ImportError:
        status["GPU"] = {'available': False, 'details': 'PyTorch not installed'}
    
    table = Table(title="System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details")
    
    for component, info in status.items():
        status_icon = "✅" if info['available'] else "❌"
        table.add_row(
            component,
            status_icon,
            info.get('details', '')
        )
    
    console.print(table)

@app.command()
def init(
    project_dir: Optional[Path] = typer.Option(
        None,
        "--dir", "-d",
        help="Project directory (default: current)"
    )
):
    """Initialize a new GenomeVault project."""
    if project_dir is None:
        project_dir = Path.cwd()
    
    console.print(f"[bold]Initializing GenomeVault project in {project_dir}[/bold]")
    
    # Create project structure
    directories = [
        "data/raw",
        "data/processed",
        "data/encrypted",
        "results",
        "logs",
        "keys",
        "circuits",
        "configs"
    ]
    
    for dir_path in directories:
        full_path = project_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        console.print(f"  Created: {dir_path}/")
    
    # Create default config
    config_path = project_dir / "configs" / "genomevault.yaml"
    if not config_path.exists():
        config_content = """# GenomeVault Configuration
version: 1.0

# Privacy settings
privacy:
  differential_privacy:
    epsilon: 1.0
    delta: 1e-5
  
  encryption:
    algorithm: AES-256-GCM
    key_derivation: PBKDF2

# Compression settings  
compression:
  hdc:
    dimension: 10000
    sparsity: 0.1
  
  tiers:
    mini:
      max_variants: 100
      target_size_kb: 1
    standard:
      max_variants: 1000
      target_size_kb: 10
    full:
      max_variants: 10000
      target_size_kb: 100

# Zero-knowledge settings
zk:
  backend: circom  # 'mock' or 'circom'
  trusted_setup_path: ./circuits/trusted_setup
  
# PIR settings
pir:
  num_servers: 3
  redundancy: 2
  
# Performance settings
performance:
  max_workers: 4
  cache_size: 1000
  gpu_enabled: auto
"""
        config_path.write_text(config_content)
        console.print(f"  Created: configs/genomevault.yaml")
    
    # Create .gitignore
    gitignore_path = project_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_content = """# GenomeVault gitignore
# Sensitive data
data/raw/*
data/encrypted/*
keys/*
*.key
*.pem
*.p12

# Generated files
*.hdc
*.proof
*.witness
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/

# Logs
logs/*
*.log

# Dependencies
node_modules/
venv/
.venv/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Temporary
tmp/
temp/
*.tmp

# Database
*.db
*.sqlite
*.sqlite3

# Secrets - NEVER commit these
.env
.env.local
.zk_transcript_key
*_secret*
*_private*
"""
        gitignore_path.write_text(gitignore_content)
        console.print("  Created: .gitignore")
    
    console.print("\n[bold green]✅ Project initialized successfully![/bold green]")
    console.print("\nNext steps:")
    console.print("  1. genomevault status      # Check system dependencies")
    console.print("  2. genomevault demo run    # Run a simple demo")
    console.print("  3. genomevault --help      # See all commands")

def main():
    """Main CLI entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        logger.exception("Unhandled exception")
        sys.exit(1)

if __name__ == "__main__":
    main()