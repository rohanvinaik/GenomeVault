from genomevault.observability.logging import configure_logging

logger = configure_logging()
#!/usr/bin/env python3
"""
Analyze cyclomatic complexity in the genomevault package.
This script helps with step 7 of the audit checklist.
"""

import subprocess
import sys
from pathlib import Path


def analyze_complexity(package_dir: Path):
    """Run radon to analyze cyclomatic complexity."""
    try:
        # Run radon cc command
        result = subprocess.run(
            ["radon", "cc", "-s", "-a", str(package_dir)],
            check=False,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.info(f"Error running radon: {result.stderr}")
            return

        # Save full output
        output_file = package_dir.parent / "radon_complexity_report.txt"
        with open(output_file, "w") as f:
            f.write(result.stdout)

        logger.info(f"Full complexity report saved to: {output_file}")

        # Parse and find high complexity functions
        high_complexity = []
        lines = result.stdout.split("\n")

        for line in lines:
            # Look for lines with complexity ratings
            if " - " in line and "(" in line and ")" in line:
                # Extract complexity score
                try:
                    parts = line.split("(")
                    if len(parts) > 1:
                        score_part = parts[1].split(")")[0]
                        if score_part.isdigit():
                            score = int(score_part)
                            if score >= 12:
                                high_complexity.append((score, line.strip()))
                except Exception:
                    from genomevault.observability.logging import configure_logging

                    logger = configure_logging()
                    logger.exception("Unhandled exception")
                    continue
                    raise

        # Sort by complexity score (descending)
        high_complexity.sort(reverse=True)

        logger.info("\nFunctions with cyclomatic complexity >= 12:")
        logger.info("=" * 60)
        for score, func_info in high_complexity[:10]:  # Top 10
            logger.info(f"CC={score}: {func_info}")

        logger.info(f"\nTotal functions with CC >= 12: {len(high_complexity)}")

        # Generate refactoring recommendations
        if high_complexity:
            recommendations_file = package_dir.parent / "complexity_refactoring_guide.md"
            with open(recommendations_file, "w") as f:
                f.write("# Cyclomatic Complexity Refactoring Guide\n\n")
                f.write(f"Found {len(high_complexity)} functions with CC >= 12\n\n")
                f.write("## Top 10 Functions to Refactor:\n\n")

                for i, (score, func_info) in enumerate(high_complexity[:10], 1):
                    f.write(f"{i}. **CC={score}**: `{func_info}`\n")

                f.write("\n## Refactoring Strategies:\n\n")
                f.write(
                    "1. **Extract Helper Functions**: Break down large functions into smaller, focused helpers\n"
                )
                f.write("2. **Early Returns**: Use guard clauses to reduce nesting\n")
                f.write("3. **Replace Conditionals with Polymorphism**: For type-based switching\n")
                f.write("4. **Use Dictionary Dispatch**: Replace long if/elif chains\n")
                f.write(
                    "5. **Extract Validation Logic**: Move input validation to separate functions\n"
                )

            logger.info(f"\nRefactoring guide saved to: {recommendations_file}")

    except FileNotFoundError:
        logger.info("Error: radon not installed. Please run: pip install radon")
        sys.exit(1)


def main():
    """Main function to analyze complexity."""
    genomevault_dir = Path("/Users/rohanvinaik/genomevault/genomevault")

    if not genomevault_dir.exists():
        logger.info(f"Error: Directory not found: {genomevault_dir}")
        sys.exit(1)

    logger.info("Analyzing cyclomatic complexity...")
    analyze_complexity(genomevault_dir)


if __name__ == "__main__":
    main()
