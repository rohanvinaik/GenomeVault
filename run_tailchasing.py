#!/usr/bin/env python3
"""Run TailChasingFixer analysis on GenomeVault."""

import os
import sys

# Add TailChasingFixer to path
sys.path.insert(0, "/Users/rohanvinaik/Desktop/TailChasingFixer")

from tailchasing.cli import main

if __name__ == "__main__":
    # Set up arguments to analyze current directory
    sys.argv = [
        "tailchasing",
        "analyze",
        ".",
        "--explain",
        "--output",
        "tailchasing_report.json",
    ]

    # Change to GenomeVault directory
    os.chdir("/Users/rohanvinaik/genomevault")

    # Run analysis
    main()
