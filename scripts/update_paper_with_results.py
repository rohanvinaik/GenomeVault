"""
Update Academic Paper with Experimental Results

This script reads JSON result files from benchmarks and populates
placeholders in the academic paper markdown file.

Usage:
    python scripts/update_paper_with_results.py
    python scripts/update_paper_with_results.py --results-dir custom_results/
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_results(results_dir: Path) -> Dict[str, Any]:
    """Load all JSON result files from results directory."""
    results = {}

    json_files = list(results_dir.glob('**/*.json'))
    logger.info(f"Found {len(json_files)} JSON result files")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Use filename without extension as key
                key = json_file.stem
                results[key] = data
                logger.debug(f"Loaded {json_file.name}")
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")

    return results


def extract_value(results: Dict[str, Any], placeholder: str) -> str:
    """
    Extract value for a placeholder from results.

    Placeholder format: {{SOURCE.PATH.TO.VALUE}}
    Example: {{hdc_benchmark.encoding_time_ms}}
    """
    # Remove {{ and }}
    key_path = placeholder.strip('{}').strip()

    # Split into source and path
    parts = key_path.split('.', 1)
    if len(parts) != 2:
        logger.warning(f"Invalid placeholder format: {placeholder}")
        return placeholder

    source, path = parts

    # Get source data
    if source not in results:
        logger.warning(f"Source not found: {source}")
        return placeholder

    data = results[source]

    # Navigate path
    try:
        for key in path.split('.'):
            if isinstance(data, dict):
                data = data[key]
            elif isinstance(data, list):
                data = data[int(key)]
            else:
                logger.warning(f"Cannot navigate path {path} in {source}")
                return placeholder

        # Format value
        if isinstance(data, float):
            # Format floats with appropriate precision
            if abs(data) < 0.01 or abs(data) > 10000:
                return f"{data:.2e}"
            elif abs(data) < 1:
                return f"{data:.4f}"
            else:
                return f"{data:.2f}"
        elif isinstance(data, int):
            return str(data)
        else:
            return str(data)

    except (KeyError, IndexError, ValueError) as e:
        logger.warning(f"Failed to extract {path} from {source}: {e}")
        return placeholder


def update_paper(
    paper_path: Path,
    results: Dict[str, Any],
    output_path: Path = None,
) -> None:
    """Update paper markdown file with results."""

    if output_path is None:
        output_path = paper_path

    # Read paper
    with open(paper_path, 'r') as f:
        content = f.read()

    # Find all placeholders
    placeholders = re.findall(r'\{\{[^}]+\}\}', content)
    logger.info(f"Found {len(placeholders)} placeholders")

    # Replace placeholders
    updates = 0
    for placeholder in set(placeholders):  # Use set to avoid duplicates
        value = extract_value(results, placeholder)
        if value != placeholder:
            content = content.replace(placeholder, value)
            updates += 1
            logger.info(f"Updated {placeholder} → {value}")

    # Write updated paper
    with open(output_path, 'w') as f:
        f.write(content)

    logger.info(f"Updated {updates} placeholders in {output_path}")

    # Check for remaining placeholders
    remaining = re.findall(r'\{\{[^}]+\}\}', content)
    if remaining:
        logger.warning(f"Warning: {len(remaining)} placeholders remain:")
        for p in set(remaining):
            logger.warning(f"  {p}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update academic paper with experimental results"
    )
    parser.add_argument(
        '--paper',
        type=Path,
        default=Path('docs/GenomeVault_Academic_Paper.md'),
        help='Path to paper markdown file'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('results/paper_experiments'),
        help='Directory containing JSON result files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path (default: overwrites input)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check inputs exist
    if not args.paper.exists():
        logger.error(f"Paper file not found: {args.paper}")
        return 1

    if not args.results_dir.exists():
        logger.error(f"Results directory not found: {args.results_dir}")
        return 1

    # Load results
    logger.info(f"Loading results from {args.results_dir}")
    results = load_results(args.results_dir)

    if not results:
        logger.error("No results loaded")
        return 1

    logger.info(f"Loaded results from {len(results)} files")

    # Update paper
    logger.info(f"Updating paper: {args.paper}")
    update_paper(args.paper, results, args.output)

    logger.info("Done!")
    return 0


if __name__ == '__main__':
    exit(main())
