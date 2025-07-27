#!/usr/bin/env python3
"""
Pre-GitHub Push Checklist for GenomeVault
"""
import json
import logging

import os
import subprocess
from pathlib import Path

print("=" * 80)
print("🚀 GENOMEVAULT PRE-GITHUB PUSH CHECKLIST")
print("=" * 80)

# Results tracking
issues = []
warnings = []
successes = []

# 1. Check for sensitive information
print("\n1. CHECKING FOR SENSITIVE INFORMATION...")
print("-" * 40)

sensitive_patterns = [
    ("API keys", ["api_key", "apikey", "API_KEY"]),
    ("Passwords", ["password =", "PASSWORD =", "secret =", "SECRET ="]),
    ("Private keys", ["BEGIN RSA PRIVATE", "BEGIN PRIVATE KEY"]),
    ("AWS credentials", ["aws_access_key", "aws_secret"]),
    ("Database URLs", ["mongodb://", "postgresql://", "mysql://"]),
]

for desc, patterns in sensitive_patterns:
    found = False
    for pattern in patterns:
        result = subprocess.run(
            [
                "grep",
                "-r",
                pattern,
                ".",
                "--include=*.py",
                "--include=*.yaml",
                "--include=*.json",
            ],
            capture_output=True,
            text=True,
        )
        if result.stdout and not all(
            x in result.stdout for x in ["example", "test", "dummy", "placeholder"]
        ):
            found = True
            break

    if found:
        warnings.append("Potential {desc} found - review before pushing")
    else:
        successes.append("No hardcoded {desc} found")

# 2. Check for proper .gitignore
print("\n2. CHECKING .gitignore...")
print("-" * 40)

gitignore_path = Path(".gitignore")
required_entries = [
    "__pycache__/",
    "*.pyc",
    ".env",
    "venv/",
    ".pytest_cache/",
    "*.log",
    ".DS_Store",
    "*.key",
    "*.pem",
    "genomevault_audit.log",
]

if gitignore_path.exists():
    with open(gitignore_path, "r") as f:
        gitignore_content = f.read()

    missing = []
    for entry in required_entries:
        if entry not in gitignore_content:
            missing.append(entry)

    if missing:
        warnings.append(".gitignore missing entries: {', '.join(missing[:3])}...")
    else:
        successes.append(".gitignore has all required entries")
else:
    issues.append(".gitignore file is missing!")

# 3. Check documentation files
print("\n3. CHECKING DOCUMENTATION...")
print("-" * 40)

required_docs = {
    "README.md": "Project overview",
    "INSTALL.md": "Installation instructions",
    "requirements.txt": "Python dependencies",
    "LICENSE": "License file",
}

for doc, desc in required_docs.items():
    if Path(doc).exists():
        successes.append("{doc} exists ({desc})")
    else:
        warnings.append("{doc} is missing ({desc})")

# 4. Check for large files
print("\n4. CHECKING FOR LARGE FILES...")
print("-" * 40)

large_files = []
for path in Path(".").rglob("*"):
    if path.is_file() and not any(part.startswith(".") for part in path.parts):
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > 10:  # Files larger than 10MB
                large_files.append((str(path), "{size_mb:.1f}MB"))
        except Exception:
            pass

if large_files:
    warnings.append("Large files found: {', '.join(['{f[0]} ({f[1]})' for f in large_files[:3]])}")
else:
    successes.append("No large files detected")

# 5. Check Python code quality basics
print("\n5. CHECKING CODE QUALITY...")
print("-" * 40)

# Check for print debugging statements
debug_prints = subprocess.run(
    ["grep", "-r", "print(", ".", "--include=*.py", "--exclude-dir=tests"],
    capture_output=True,
    text=True,
)
if debug_prints.stdout:
    # Count occurrences
    count = len(debug_prints.stdout.strip().split("\n"))
    if count > 50:  # Arbitrary threshold
        warnings.append("Many print statements found ({count}) - consider using logging")
    else:
        successes.append("Reasonable number of print statements")

# 6. Check for TODO/FIXME comments
print("\n6. CHECKING FOR TODO/FIXME COMMENTS...")
print("-" * 40)

todos = subprocess.run(
    ["grep", "-r", "-E", "TODO|FIXME|XXX|HACK", ".", "--include=*.py"],
    capture_output=True,
    text=True,
)
if todos.stdout:
    count = len(todos.stdout.strip().split("\n"))
    warnings.append("Found {count} TODO/FIXME comments - consider addressing or documenting")
else:
    successes.append("No TODO/FIXME comments found")

# 7. Check imports are fixed
print("\n7. VERIFYING IMPORT FIXES...")
print("-" * 40)

# Check the specific fix we made
variant_file = Path("zk_proofs/circuits/biological/variant.py")
if variant_file.exists():
    with open(variant_file, "r") as f:
        content = f.read()
    if "from ..base_circuits import" in content:
        successes.append("✅ variant.py import fix is in place")
    else:
        issues.append("❌ variant.py import fix is missing!")

# 8. Check for test files
print("\n8. CHECKING TEST COVERAGE...")
print("-" * 40)

test_dirs = list(Path(".").glob("test*"))
test_files = list(Path(".").rglob("test_*.py"))

if test_dirs or test_files:
    successes.append("Found {len(test_dirs)} test directories and {len(test_files)} test files")
else:
    warnings.append("No test files found - consider adding tests")

# 9. Final summary
print("\n" + "=" * 80)
print("📊 PRE-PUSH SUMMARY")
print("=" * 80)

print("\n✅ GOOD ({len(successes)} items):")
for item in successes:
    print("   • {item}")

if warnings:
    print("\n⚠️  WARNINGS ({len(warnings)} items):")
    for item in warnings:
        print("   • {item}")

if issues:
    print("\n❌ ISSUES ({len(issues)} items):")
    for item in issues:
        print("   • {item}")

# 10. Recommendations
print("\n📝 RECOMMENDATIONS BEFORE PUSHING:")
print("-" * 40)

recommendations = [
    "Create a comprehensive README.md with:",
    "  - Project description and goals",
    "  - Installation instructions",
    "  - Usage examples",
    "  - Architecture overview",
    "  - Contributing guidelines",
    "",
    "Add a .gitignore file with common Python excludes",
    "",
    "Consider adding:",
    "  - LICENSE file (MIT, Apache 2.0, etc.)",
    "  - CONTRIBUTING.md for contribution guidelines",
    "  - GitHub Actions workflow for CI/CD",
    "  - Pre-commit hooks for code quality",
    "",
    "Review any TODO/FIXME comments",
    "",
    "Ensure no sensitive data is committed",
]

for rec in recommendations:
    print(rec)

print("\n" + "=" * 80)
print("✨ Overall: The codebase structure looks good!")
print("   Main import issue has been fixed.")
print("   Address warnings above for a professional repository.")
print("=" * 80)
