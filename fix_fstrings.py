#!/usr/bin/env python3
"""
Fix f-string issues in GenomeVault
"""
import os
import re


def fix_fstring_issues(directory: str = "."):
    """Fix missing f prefix on f-strings"""
    fixed_files = []

    # Pattern to find logger.error or similar with {} formatting but no f prefix
    patterns = [
        (r'(logger\.\w+)\("([^"]*\{[^}]+\}[^"]*)"', r'\1(f"\2"'),
        (r"(logger\.\w+)\('([^']*\{[^}]+\}[^']*)'", r"\1(f'\2'"),
    ]

    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        if any(
            skip in root for skip in [".git", "__pycache__", ".pytest_cache", "TailChasingFixer"]
        ):
            continue

        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)

                try:
                    with open(filepath, "r") as f:
                        content = f.read()

                    original = content

                    # Apply all patterns
                    for pattern, replacement in patterns:
                        content = re.sub(pattern, replacement, content)

                    if content != original:
                        with open(filepath, "w") as f:
                            f.write(content)
                        fixed_files.append(filepath)
                        print(f"Fixed: {filepath}")

                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    return fixed_files


if __name__ == "__main__":
    print("Fixing f-string issues in GenomeVault...")
    fixed = fix_fstring_issues("genomevault")
    print(f"\nFixed {len(fixed)} files")
