#!/usr/bin/env python3
"""Final aggressive cleanup of debug prints."""

import re
import os


def clean_file(filepath):
    """Aggressively remove debug prints from a file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except:
        return 0

    new_lines = []
    removed = 0
    skip_file = False

    # Check if this is a test file or demo that should keep prints
    if any(x in filepath for x in ["demo", "example", "test_", "_test.py", "main.py"]):
        # Be more selective for test/demo files
        for line in lines:
            if re.match(r"^\s*print\s*\(\s*\)\s*$", line):  # Empty print()
                removed += 1
                continue
            elif re.search(r"print.*# Debug print removed", line):
                # Already processed
                new_lines.append(line)
            elif re.search(
                r"print\s*\([^)]*(\bDEBUG\b|\bTODO\b|\bFIXME\b|\bXXX\b)",
                line,
                re.IGNORECASE,
            ):
                removed += 1
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    new_lines.append(" " * indent + "pass  # Debug print removed\n")
            else:
                new_lines.append(line)
    else:
        # More aggressive for non-test files
        for line in lines:
            # Remove various debug patterns
            if any(
                [
                    re.match(r"^\s*print\s*\(\s*\)\s*$", line),  # Empty print()
                    re.search(
                        r"print\s*\([^)]*(\bDEBUG\b|\bTODO\b|\bFIXME\b)",
                        line,
                        re.IGNORECASE,
                    ),
                    re.search(
                        r"print\s*\([^) \
                    ]*(>>>|---|===|\.\.\.|Processing|Checking|Starting|Step \d+)",
                        line,
                    ),
                    re.match(r'^\s*print\s*\([\'"][=-]+[\'"]\)', line),  # Separator lines
                    re.match(r'^\s*print\s*\([\'"]\\n[\'"]\)', line),  # Just newlines
                ]
            ):
                removed += 1
                # Preserve indentation with pass if needed
                indent = len(line) - len(line.lstrip())
                if indent > 0 and not line.strip().startswith("#"):
                    new_lines.append(" " * indent + "pass  # Debug print removed\n")
            else:
                new_lines.append(line)

    if removed > 0:
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            return removed
        except:
            return 0

    return 0


def main():
    """Main cleanup function."""
    print("=" * 70)
    pass  # Debug print removed
    print("=" * 70)

    total_removed = 0
    files_processed = []

    # Process all Python files
    for root, dirs, files in os.walk("."):
        dirs[:] = [
            d for d in dirs if d not in {".venv", "venv", "__pycache__", ".git", "node_modules"}
        ]

        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                removed = clean_file(filepath)
                if removed > 0:
                    total_removed += removed
                    files_processed.append((filepath, removed))

    pass  # Debug print removed

    if files_processed:
        print("\nTop files cleaned:")
        for filepath, count in sorted(files_processed, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {os.path.basename(filepath)}: {count} prints removed")

    print("\n" + "=" * 70)
    pass  # Debug print removed
    print("=" * 70)


if __name__ == "__main__":
    main()
