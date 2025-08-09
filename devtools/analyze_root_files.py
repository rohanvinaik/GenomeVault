#!/usr/bin/env python3

import re
import subprocess
import time
from pathlib import Path
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


ROOT = Path(".")
OUT = Path(".tidy")
OUT.mkdir(exist_ok=True)
now = time.time()

# Top-level tracked files only (no slashes)
tracked = [
    p
    for p in subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    if "/" not in p and Path(p).is_file()
]

# Heuristic keep-list (never propose)
KEEP = {
    "README.md",
    "LICENSE",
    "LICENSE.txt",
    "pyproject.toml",
    "poetry.lock",
    "Pipfile",
    "Pipfile.lock",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-ci.txt",
    ".gitignore",
    ".editorconfig",
    ".pre-commit-config.yaml",
    ".pylintrc",
    "mypy.ini",
    "ruff.toml",
    ".ruff.toml",
    "Dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "Makefile",
    ".env.example",
}

# Suspicious name patterns (likely junk)
NAME_PAT = re.compile(
    r"""
  (?:^|[_\-.])(old|backup|bak|tmp|temp|final|final2|final\w*|copy|copy\d*|draft|junk|unused|archive|archived|dead)$
""",
    re.I | re.X,
)

# Suspicious extensions
BAD_EXT = {
    ".log",
    ".tmp",
    ".bak",
    ".old",
    ".orig",
    ".swp",
    ".zip",
    ".tar",
    ".tgz",
    ".gz",
    ".7z",
    ".rar",
    ".csv",
    ".tsv",
    ".xlsx",
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ipynb",
}


def last_commit_days(path: str) -> int:
    try:
        ts = subprocess.check_output(
            ["git", "log", "-1", "--format=%ct", "--", path], text=True
        ).strip()
        if not ts:
            return 99999
        return int((now - int(ts)) // 86400)
    except Exception:
        return 99999


# build corpus for reference search
def readfile(p):
    try:
        return Path(p).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


corpus = []
for p in subprocess.check_output(["git", "ls-files"], text=True).splitlines():
    if p.startswith(".git") or p.startswith("venv/") or p.startswith(".venv/"):
        continue
    if "/" in p:  # include all files for reference matching
        corpus.append(readfile(p))
corpus = "\n".join(corpus)


def referenced(fname: str) -> bool:
    # check exact filename reference anywhere
    if re.search(rf"\b{re.escape(fname)}\b", corpus):
        return True
    # also if used in GH Actions or Makefile directives
    return False


rows = []
for f in tracked:
    P = Path(f)
    size = P.stat().st_size
    days = last_commit_days(f)
    ext = P.suffix.lower()
    suspicious_name = bool(NAME_PAT.search(P.stem))
    refd = referenced(P.name)
    shebang = ""
    try:
        first = P.open("r", encoding="utf-8", errors="ignore").readline().strip()
        shebang = first if first.startswith("#!") else ""
    except Exception:
        pass

    # categorize
    reason = []
    if f in KEEP:
        cat = "KEEP"
        reason.append("whitelist")
    else:
        cat = "MAYBE"
        if days >= 120:
            reason.append(f"old:{days}d")
        if size > 1_000_000:
            reason.append(f"large:{size//1024}KB")
        if suspicious_name:
            reason.append("suspicious_name")
        if ext in BAD_EXT:
            reason.append(f"ext:{ext}")
        if not refd:
            reason.append("unreferenced")
        # Promote to CANDIDATE if at least 2 signals OR clearly junk ext
        signals = sum([days >= 120, size > 1_000_000, suspicious_name, (ext in BAD_EXT), not refd])
        if (ext in BAD_EXT) or (signals >= 2):
            cat = "CANDIDATE"

    rows.append(
        (
            cat,
            f,
            days,
            size,
            ext,
            bool(shebang),
            suspicious_name,
            refd,
            ", ".join(reason),
        )
    )

# write report
report = OUT / "top_root_report.tsv"
with report.open("w", encoding="utf-8") as w:
    w.write(
        "category\tpath\tdays_since_commit\tsize_kb\text\thas_shebang\tsuspicious_name\treferenced\treasons\n"
    )
    for cat, f, days, size, ext, sb, susp, ref, why in sorted(
        rows, key=lambda x: (x[0] != "CANDIDATE", -x[2], -x[3], x[1])
    ):
        w.write(
            f"{cat}\t{f}\t{days}\t{size//1024}\t{ext}\t{int(sb)}\t{int(susp)}\t{int(ref)}\t{why}\n"
        )

# propose candidates list
cands = [f for cat, f, _, _, _, _, _, _, _ in rows if cat == "CANDIDATE"]
(Path(".tidy") / "top_root_candidates.txt").write_text("\n".join(cands) + "\n", encoding="utf-8")

logger.debug("Wrote:", report, "and", OUT / "top_root_candidates.txt")
print(
    f"Found {len(tracked)} root files: {len([1 for r in rows if r[0]=='KEEP'])} KEEP, {len([1 for r in rows if r[0]=='MAYBE'])} MAYBE, {len(cands)} CANDIDATE"
)
