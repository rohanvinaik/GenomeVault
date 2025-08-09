from __future__ import annotations
import argparse
from pathlib import Path

def count_fastq_reads(path: Path) -> int:
    # minimal heuristic: count lines starting with '@' that precede sequence lines
    n = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("@"):
                n += 1
    return n

def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("fastq", type=Path)
    args = p.parse_args(argv)
    print(count_fastq_reads(args.fastq))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())