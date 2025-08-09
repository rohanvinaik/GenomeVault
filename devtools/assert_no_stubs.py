# devtools/assert_no_stubs.py
from __future__ import annotations
import re
import sys
from pathlib import Path

PATTERNS = [
    r"\braise\s+NotImplementedError\b",
    r"^\s*pass\s*$",
    r"\bTODO\b", r"\bFIXME\b", r"\bXXX\b", r"\bWIP\b",
    r"^\s*\.\.\.\s*$",
]
STUB_RE = re.compile("|".join(PATTERNS), re.MULTILINE)

CRITICAL_DIRS = ["genomevault", "tests"]  # tighten scope here
EXCLUDE = {".git", ".venv", "venv", "__pycache__", "build", "dist", "node_modules"}

def main() -> int:
    root = Path(__file__).resolve().parents[1]
    offenders = []
    for d in CRITICAL_DIRS:
        for p in (root / d).rglob("*.py"):
            if any(part in EXCLUDE for part in p.parts):
                continue
            text = p.read_text(encoding="utf-8", errors="ignore")
            if STUB_RE.search(text):
                offenders.append(p.relative_to(root))
    if offenders:
        print("Stub‑guard failed. Remove TODO/pass/NotImplemented from critical paths:", file=sys.stderr)
        for f in offenders:
            print(f" - {f}", file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())