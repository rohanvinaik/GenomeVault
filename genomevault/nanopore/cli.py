"""Cli module."""
from __future__ import annotations

from pathlib import Path
import argparse
def count_fastq_reads(path: Path) -> int:
    """Count fastq reads.

    Args:
        path: File or directory path.

    Returns:
        Integer result.
    """
    # minimal heuristic: count lines starting with '@' that precede sequence lines
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("@"):
                n += 1
    return n


def main(argv=None) -> int:
    """Main.

    Args:
        argv: Argv.

    Returns:
        Integer result.
    """
    p = argparse.ArgumentParser()
    p.add_argument("fastq", type=Path)
    args = p.parse_args(argv)
    print(count_fastq_reads(args.fastq))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
