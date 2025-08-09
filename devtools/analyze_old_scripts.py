#!/usr/bin/env python3
import re
import subprocess
from pathlib import Path

root = Path(".")
outdir = Path(".tidy")
outdir.mkdir(exist_ok=True)

# 1) Gather tracked files
git_files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()

# 2) Candidate patterns
candidates = []
exts = {".sh", ".bash", ".zsh", ".ksh", ".cmd", ".bat", ".ps1"}
likely_dirs = ("scripts/", "bin/", "tools/")


def has_shebang(p: Path) -> bool:
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            first = f.readline()
        return first.startswith("#!") and (
            "sh" in first or "bash" in first or "zsh" in first or "python" in first
        )
    except Exception:
        return False


for rel in git_files:
    p = Path(rel)
    if not p.is_file():
        continue
    if p.suffix.lower() in exts or rel.startswith(likely_dirs) or has_shebang(p):
        candidates.append(p)

# 3) Build a search corpus to detect references
search_files = [
    f
    for f in git_files
    if not any(
        x in f
        for x in (
            ".git/",
            ".venv/",
            "venv/",
            ".mypy_cache/",
            ".ruff_cache/",
            ".pytest_cache/",
        )
    )
]
corpus = ""
for f in search_files:
    try:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            corpus += fh.read() + "\n"
    except Exception:
        pass


def referenced(relpath: str) -> bool:
    name = Path(relpath).name
    if re.search(rf"\b{re.escape(relpath)}\b", corpus):
        return True
    if re.search(rf"\b{re.escape(name)}\b", corpus):
        return True
    return False


def last_commit_days(relpath: str) -> int:
    try:
        ts = subprocess.check_output(
            ["git", "log", "-1", "--format=%ct", "--", relpath], text=True
        ).strip()
        if not ts:
            return 99999
        import time

        return int((time.time() - int(ts)) // 86400)
    except Exception:
        return 99999


rows = []
for p in sorted(set(candidates)):
    rel = str(p)
    st = p.stat()
    execbit = bool(st.st_mode & 0o111)
    sheb = ""
    try:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
            sheb = first if first.startswith("#!") else ""
    except Exception:
        pass
    rows.append(
        {
            "path": rel,
            "days_since_commit": last_commit_days(rel),
            "is_executable": execbit,
            "has_shebang": bool(sheb),
            "shebang": sheb,
            "size_kb": round(st.st_size / 1024, 1),
            "referenced": referenced(rel),
        }
    )

# Write TSV + summary
tsv = outdir / "old_scripts_report.tsv"
with tsv.open("w", encoding="utf-8") as f:
    f.write("path\tdays_since_commit\tis_executable\thas_shebang\tshebang\tsize_kb\treferenced\n")
    for r in sorted(rows, key=lambda x: (-x["days_since_commit"], x["path"])):
        f.write(
            f'{r["path"]}\t{r["days_since_commit"]}\t{int(r["is_executable"])}\t{int(r["has_shebang"])}\t{r["shebang"]}\t{r["size_kb"]}\t{int(r["referenced"])}\n'
        )

# Also create a shortlist: old (>=120d) and unreferenced
short = [r for r in rows if r["days_since_commit"] >= 120 and not r["referenced"]]
with (outdir / "old_unreferenced_scripts.txt").open("w", encoding="utf-8") as f:
    for r in sorted(short, key=lambda x: (-x["days_since_commit"], x["path"])):
        f.write(r["path"] + "\n")

print("Wrote:", tsv, "and", outdir / "old_unreferenced_scripts.txt")
print(f"Found {len(candidates)} script candidates")
print(f"Found {len(short)} old unreferenced scripts (>=120 days)")
