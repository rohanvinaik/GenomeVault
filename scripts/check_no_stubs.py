#!/usr/bin/env python3
import os
import sys

SENTINEL = "File temporarily stubbed due to syntax error"

bad = []
for dp, dn, fns in os.walk("."):
    for fn in fns:
        if not fn.endswith((".py", ".ts", ".sol", ".md", ".json", ".yml", ".yaml", "Dockerfile")):
            continue
        p = os.path.join(dp, fn)
        try:
            with open(p, "r", errors="ignore") as f:
                s = f.read()
            if SENTINEL in s:
                bad.append(p)
        except Exception:
            pass

if bad:
    print("Stub sentinel found in files:")
    for b in bad:
        print(" -", b)
    sys.exit(1)
print("No stub sentinels found.")
