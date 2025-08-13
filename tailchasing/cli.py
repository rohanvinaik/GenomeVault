"""
Command-line interface for tailchasing polymer physics configuration and calibration.

This module provides CLI commands for calibrating polymer physics parameters,
managing configuration files, and running performance analysis.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .calibrate import (
    CalibrationTool,
    create_sample_codebase_metrics,
    create_sample_thrash_events,
)
from .config import ConfigManager, PolymerConfig, get_config_manager


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TailchasingCLI:
    """
    Command-line interface for tailchasing polymer physics tools.
    """

    def __init__(self):
        self.console = Console()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser."""
        parser = argparse.ArgumentParser(
            prog="tailchasing",
            description="Chromatin-inspired performance analysis with polymer physics",
        )

        parser.add_argument("--config", type=str, help="Path to configuration file")

        parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # Config command
        config_parser = subparsers.add_parser("config", help="Manage configuration")
        config_subparsers = config_parser.add_subparsers(dest="config_action")

        config_subparsers.add_parser("show", help="Show current configuration")
        config_subparsers.add_parser("validate", help="Validate configuration")

        init_parser = config_subparsers.add_parser("init", help="Initialize default configuration")
        init_parser.add_argument("--force", action="store_true", help="Overwrite existing config")

        # Calibrate command
        calibrate_parser = subparsers.add_parser(
            "calibrate", help="Calibrate polymer physics parameters"
        )

        calibrate_parser.add_argument(
            "--thrash-events",
            type=str,
            help="Path to JSON file with historical thrash events",
        )

        calibrate_parser.add_argument(
            "--codebase-metrics",
            type=str,
            help="Path to JSON file with codebase metrics",
        )

        calibrate_parser.add_argument(
            "--alpha-range",
            type=float,
            nargs=2,
            default=[1.0, 1.5],
            help="Range for alpha parameter search (default: 1.0 1.5)",
        )

        calibrate_parser.add_argument(
            "--alpha-steps",
            type=int,
            default=26,
            help="Number of alpha values to test (default: 26)",
        )

        calibrate_parser.add_argument(
            "--max-iterations",
            type=int,
            default=100,
            help="Maximum optimization iterations (default: 100)",
        )

        calibrate_parser.add_argument(
            "--output", type=str, help="Output path for calibration results"
        )

        calibrate_parser.add_argument(
            "--demo", action="store_true", help="Run calibration with sample data"
        )

        # Validate command
        validate_parser = subparsers.add_parser(
            "validate", help="Validate configuration against test data"
        )

        validate_parser.add_argument(
            "--test-events",
            type=str,
            required=True,
            help="Path to JSON file with test thrash events",
        )

        validate_parser.add_argument(
            "--codebase-metrics",
            type=str,
            required=True,
            help="Path to JSON file with codebase metrics",
        )

        validate_parser.add_argument(
            "--polymer-config",
            type=str,
            help="Path to custom polymer configuration to test",
        )

        # Analysis command
        analysis_parser = subparsers.add_parser("analyze", help="Run performance analysis")

        analysis_parser.add_argument(
            "--input", type=str, required=True, help="Path to performance data file"
        )

        analysis_parser.add_argument("--output", type=str, help="Output path for analysis results")

        analysis_parser.add_argument(
            "--format",
            choices=["json", "yaml", "text"],
            default="json",
            help="Output format (default: json)",
        )

        return parser

    def run(self, args: Optional[List[str]] = None) -> int:
        """Run the CLI with given arguments."""
        parser = self.create_parser()
        parsed_args = parser.parse_args(args)

        if parsed_args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        try:
            return self._execute_command(parsed_args)
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/red]")
            if parsed_args.verbose:
                self.console.print_exception()
            return 1

    def _execute_command(self, args: argparse.Namespace) -> int:
        """Execute the specified command."""
        if args.command == "config":
            return self._handle_config_command(args)
        elif args.command == "calibrate":
            return self._handle_calibrate_command(args)
        elif args.command == "validate":
            return self._handle_validate_command(args)
        elif args.command == "analyze":
            return self._handle_analyze_command(args)
        else:
            self.console.print("[red]No command specified. Use --help for usage.[/red]")
            return 1

    def _handle_config_command(self, args: argparse.Namespace) -> int:
        """Handle config subcommands."""
        config_manager = get_config_manager(args.config)

        if args.config_action == "show":
            return self._show_config(config_manager)
        elif args.config_action == "validate":
            return self._validate_config(config_manager)
        elif args.config_action == "init":
            return self._init_config(config_manager, args.force)
        else:
            self.console.print("[red]No config action specified.[/red]")
            return 1

    def _handle_calibrate_command(self, args: argparse.Namespace) -> int:
        """Handle calibrate command."""
        if args.demo:
            return self._calibrate_demo(args)

        # Load thrash events
        if not args.thrash_events:
            self.console.print("[red]--thrash-events is required (or use --demo)[/red]")
            return 1

        thrash_events_path = Path(args.thrash_events)
        if not thrash_events_path.exists():
            self.console.print(f"[red]Thrash events file not found: {thrash_events_path}[/red]")
            return 1

        # Load codebase metrics
        if not args.codebase_metrics:
            self.console.print("[red]--codebase-metrics is required (or use --demo)[/red]")
            return 1

        codebase_metrics_path = Path(args.codebase_metrics)
        if not codebase_metrics_path.exists():
            self.console.print(
                f"[red]Codebase metrics file not found: {codebase_metrics_path}[/red]"
            )
            return 1

        return self._run_calibration(args, thrash_events_path, codebase_metrics_path)

    def _handle_validate_command(self, args: argparse.Namespace) -> int:
        """Handle validate command."""
        # Implementation for validation
        self.console.print("[yellow]Validation command not yet implemented[/yellow]")
        return 0

    def _handle_analyze_command(self, args: argparse.Namespace) -> int:
        """Handle analyze command."""
        # Implementation for analysis
        self.console.print("[yellow]Analysis command not yet implemented[/yellow]")
        return 0

    def _show_config(self, config_manager: ConfigManager) -> int:
        """Show current configuration."""
        try:
            config = config_manager.config

            # Create configuration display table
            table = Table(title="Polymer Physics Configuration", show_header=True)
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Description", style="dim")

            table.add_row("alpha", f"{config.alpha:.3f}", "Contact decay exponent (1.0-1.5)")
            table.add_row("epsilon", f"{config.epsilon:.2e}", "Numerical stability parameter")
            table.add_row("kappa", f"{config.kappa:.3f}", "Contact strength scaling")
            table.add_row(
                "max_distance",
                str(config.max_distance),
                "Maximum distance for contacts",
            )
            table.add_row(
                "min_threshold",
                f"{config.min_contact_threshold:.3f}",
                "Minimum contact threshold",
            )

            # Distance weights subtable
            weights_table = Table(show_header=True, box=None)
            weights_table.add_column("Type", style="blue")
            weights_table.add_column("Weight", style="yellow")

            weights_table.add_row("tok", f"{config.weights.tok:.2f}")
            weights_table.add_row("ast", f"{config.weights.ast:.2f}")
            weights_table.add_row("mod", f"{config.weights.mod:.2f}")
            weights_table.add_row("git", f"{config.weights.git:.2f}")

            # TAD patterns
            tad_text = Text()
            for i, pattern in enumerate(config.tad_patterns):
                if i > 0:
                    tad_text.append(", ")
                tad_text.append(pattern, style="magenta")

            self.console.print(table)
            self.console.print("\n")
            self.console.print(Panel(weights_table, title="Distance Weights"))
            self.console.print("\n")
            self.console.print(Panel(tad_text, title="TAD Patterns"))

            # Show config file location
            config_path = config_manager.config_path or PolymerConfig.get_default_config_path()
            self.console.print(f"\n[dim]Configuration file: {config_path}[/dim]")

            return 0

        except Exception as e:
            self.console.print(f"[red]Failed to load configuration: {e}[/red]")
            return 1

    def _validate_config(self, config_manager: ConfigManager) -> int:
        """Validate current configuration."""
        try:
            config = config_manager.config
            issues = config_manager.validate_config(config)

            if not issues:
                self.console.print("[green]✓ Configuration is valid[/green]")
                return 0
            else:
                self.console.print("[yellow]Configuration issues found:[/yellow]")
                for issue in issues:
                    if issue.startswith("Warning:"):
                        self.console.print(f"[yellow]  ⚠ {issue}[/yellow]")
                    else:
                        self.console.print(f"[red]  ✗ {issue}[/red]")

                # Return 1 only for errors, not warnings
                has_errors = any(not issue.startswith("Warning:") for issue in issues)
                return 1 if has_errors else 0

        except Exception as e:
            self.console.print(f"[red]Failed to validate configuration: {e}[/red]")
            return 1

    def _init_config(self, config_manager: ConfigManager, force: bool = False) -> int:
        """Initialize default configuration file."""
        config_path = config_manager.config_path or PolymerConfig.get_default_config_path()

        if config_path.exists() and not force:
            self.console.print(f"[yellow]Configuration file already exists: {config_path}[/yellow]")
            self.console.print("[yellow]Use --force to overwrite[/yellow]")
            return 1

        try:
            default_config = PolymerConfig()
            config_manager.save_config(default_config, config_path)

            self.console.print(f"[green]✓ Created default configuration: {config_path}[/green]")
            return 0

        except Exception as e:
            self.console.print(f"[red]Failed to create configuration file: {e}[/red]")
            return 1

    def _calibrate_demo(self, args: argparse.Namespace) -> int:
        """Run calibration with demo data."""
        pass  # Debug print removed

        try:
            # Create sample data
            sample_files = [
                "genomevault/api/handlers.py",
                "genomevault/core/engine.py",
                "genomevault/db/models.py",
                "genomevault/ui/components.py",
                "genomevault/utils/helpers.py",
                "genomevault/services/auth.py",
                "genomevault/controllers/main.py",
            ]

            thrash_events = create_sample_thrash_events(sample_files, event_count=30)
            codebase_metrics = create_sample_codebase_metrics(sample_files)

            # Run calibration
            config_manager = get_config_manager(args.config)
            calibration_tool = CalibrationTool(config_manager.config)

            self.console.print(f"[dim]Calibrating with {len(thrash_events)} sample events...[/dim]")

            result = calibration_tool.fit_parameters(
                thrash_events,
                codebase_metrics,
                tuple(args.alpha_range),
                args.alpha_steps,
                args.max_iterations,
            )

            # Display results
            self._display_calibration_results(result)

            # Save results if requested
            if args.output:
                self._save_calibration_results(result, args.output)

            # Ask if user wants to update configuration
            update = input("\nUpdate configuration with these results? [y/N]: ").lower().strip()
            if update == "y":
                config_manager.update_from_calibration(result)
                self.console.print("[green]✓ Configuration updated[/green]")

            return 0

        except Exception as e:
            self.console.print(f"[red]Calibration failed: {e}[/red]")
            if args.verbose:
                self.console.print_exception()
            return 1

    def _run_calibration(
        self,
        args: argparse.Namespace,
        thrash_events_path: Path,
        codebase_metrics_path: Path,
    ) -> int:
        """Run calibration with provided data files."""
        # Implementation for real calibration with user data
        self.console.print("[yellow]Real data calibration not yet implemented[/yellow]")
        self.console.print("[dim]Use --demo to run with sample data[/dim]")
        return 0

    def _display_calibration_results(self, result) -> None:
        """Display calibration results."""
        # Results summary
        summary_table = Table(title="Calibration Results", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Optimal Alpha", f"{result.optimal_alpha:.4f}")
        summary_table.add_row("Correlation Score", f"{result.correlation_score:.4f}")
        summary_table.add_row("Validation Score", f"{result.validation_score:.4f}")
        summary_table.add_row("Iterations", str(result.iterations))
        summary_table.add_row("Converged", "Yes" if result.converged else "No")

        self.console.print(summary_table)

        # Optimal weights
        weights_table = Table(title="Optimal Distance Weights", show_header=True)
        weights_table.add_column("Type", style="blue")
        weights_table.add_column("Weight", style="yellow")
        weights_table.add_column("Change", style="magenta")

        # Show changes from default
        default_weights = PolymerConfig().weights

        weights_table.add_row(
            "tok",
            f"{result.optimal_weights.tok:.3f}",
            f"{result.optimal_weights.tok - default_weights.tok:+.3f}",
        )
        weights_table.add_row(
            "ast",
            f"{result.optimal_weights.ast:.3f}",
            f"{result.optimal_weights.ast - default_weights.ast:+.3f}",
        )
        weights_table.add_row(
            "mod",
            f"{result.optimal_weights.mod:.3f}",
            f"{result.optimal_weights.mod - default_weights.mod:+.3f}",
        )
        weights_table.add_row(
            "git",
            f"{result.optimal_weights.git:.3f}",
            f"{result.optimal_weights.git - default_weights.git:+.3f}",
        )

        self.console.print("\n")
        self.console.print(weights_table)

    def _save_calibration_results(self, result, output_path: str) -> None:
        """Save calibration results to file."""
        output_file = Path(output_path)

        with open(output_file, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        self.console.print(f"[dim]Results saved to: {output_file}[/dim]")


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    cli = TailchasingCLI()
    return cli.run(args)


if __name__ == "__main__":
    sys.exit(main())
