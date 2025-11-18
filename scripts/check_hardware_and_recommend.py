#!/usr/bin/env python3
"""
Hardware capability checker and optimization recommender for GenomeVault.

This script detects your system's hardware capabilities and recommends
the optimal optimization phases to deploy.

Usage:
    python3 scripts/check_hardware_and_recommend.py

    # Save configuration to file
    python3 scripts/check_hardware_and_recommend.py --save-config

    # Quiet mode (just show commands)
    python3 scripts/check_hardware_and_recommend.py --quiet
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.compute.hardware_detector import HardwareDetector
from genomevault.compute.optimization_selector import OptimizationSelector


def main():
    parser = argparse.ArgumentParser(
        description="Check hardware and recommend GenomeVault optimizations"
    )
    parser.add_argument(
        "--save-config",
        action="store_true",
        help="Save configuration to YAML file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode - only show deployment commands"
    )
    parser.add_argument(
        "--config-file",
        default="genomevault_auto_config.yaml",
        help="Output configuration file path (default: genomevault_auto_config.yaml)"
    )

    args = parser.parse_args()

    # Initialize detector and selector
    detector = HardwareDetector()
    selector = OptimizationSelector(detector)

    if args.quiet:
        # Quiet mode - just show commands
        selector.generate_optimal_config()
        selector.print_deployment_commands()
    else:
        # Full report
        print("\n" + "=" * 70)
        print("GenomeVault Hardware-Aware Optimization Recommender")
        print("=" * 70)
        print("\nDetecting your system's hardware capabilities...\n")

        # Hardware detection report
        detector.print_report()

        # Generate optimal configuration
        selector.generate_optimal_config()

        # Print configuration summary
        selector.print_summary()

        # Print deployment commands
        selector.print_deployment_commands()

    # Save configuration if requested
    if args.save_config:
        selector.write_config_file(args.config_file)
        print(f"\n✅ Configuration saved to: {args.config_file}\n")

    # Final recommendation
    if not args.quiet:
        print("\n" + "=" * 70)
        print("📌 Next Steps")
        print("=" * 70)
        print("\n1. Wait for ref1 to finish sorting (~10-15 min)")
        print("2. Run the Phase 1 command shown above")
        print("3. Expect 5.6 hours savings on k=13 pipeline")
        print("\n💡 Tip: Run with --save-config to generate a config file")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
