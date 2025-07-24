#!/usr/bin/env python3
"""
Fix common linting issues in GenomeVault
"""
import os
import re
import subprocess
from typing import List, Set, Dict


def get_python_files(directory: str = ".") -> List[str]:
    """Get all Python files in the directory"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        if any(
            skip in root for skip in [".git", "__pycache__", ".pytest_cache", "TailChasingFixer"]
        ):
            continue
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def fix_missing_typing_imports(filepath: str) -> bool:
    """Add missing typing imports to a file"""
    with open(filepath, "r") as f:
        content = f.read()

    # Check if file uses typing constructs
    typing_patterns = [
        r"\bDict\[",
        r"\bList\[",
        r"\bOptional\[",
        r"\bUnion\[",
        r"\bTuple\[",
        r"\bAny\b",
        r"\bSet\[",
        r"\bCallable\[",
        r": Dict",
        r": List",
        r": Optional",
        r": Union",
        r": Tuple",
        r": Any",
        r": Set",
        r": Callable",
    ]

    uses_typing = any(re.search(pattern, content) for pattern in typing_patterns)

    if not uses_typing:
        return False

    # Check what's already imported
    existing_imports = set()
    import_match = re.search(r"from typing import (.+)", content)
    if import_match:
        existing_imports = {item.strip() for item in import_match.group(1).split(",")}

    # Find what needs to be imported
    needed_imports = set()

    type_mapping = {
        r"\bDict\[|: Dict": "Dict",
        r"\bList\[|: List": "List",
        r"\bOptional\[|: Optional": "Optional",
        r"\bUnion\[|: Union": "Union",
        r"\bTuple\[|: Tuple": "Tuple",
        r"\bAny\b|: Any": "Any",
        r"\bSet\[|: Set": "Set",
        r"\bCallable\[|: Callable": "Callable",
    }

    for pattern, type_name in type_mapping.items():
        if re.search(pattern, content) and type_name not in existing_imports:
            needed_imports.add(type_name)

    if not needed_imports:
        return False

    # Add or update import
    all_imports = existing_imports | needed_imports
    new_import = f"from typing import {', '.join(sorted(all_imports))}"

    if import_match:
        # Replace existing import
        content = content.replace(import_match.group(0), new_import)
    else:
        # Add new import after other imports or at the beginning
        lines = content.split("\n")
        insert_pos = 0

        # Find the position after existing imports
        for i, line in enumerate(lines):
            if line.startswith(("import ", "from ")) and not line.startswith("from __future__"):
                insert_pos = i + 1
            elif line and not line.startswith(("#", '"""', "'''")) and insert_pos > 0:
                break

        lines.insert(insert_pos, new_import)
        content = "\n".join(lines)

    with open(filepath, "w") as f:
        f.write(content)

    return True


def fix_unused_exception_variables(filepath: str) -> bool:
    """Fix unused exception variables"""
    with open(filepath, "r") as f:
        content = f.read()

    original = content

    # Pattern to find except clauses with unused 'e' variable
    pattern = r"except\s+(\w+)\s+as\s+e:\s*\n\s*(logger\.error|print|pass|continue)"

    def replace_except(match):
        exception_type = match.group(1)
        next_statement = match.group(2)
        return f"except {exception_type}:\n        {next_statement}"

    content = re.sub(pattern, replace_except, content)

    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        return True

    return False


def add_logger_import(filepath: str) -> bool:
    """Add logger import if logger is used but not imported"""
    with open(filepath, "r") as f:
        content = f.read()

    # Check if logger is used
    if not re.search(r"\blogger\b", content):
        return False

    # Check if logger is already imported
    if re.search(r"from genomevault\.utils\.logging import get_logger", content) or re.search(
        r"logger\s*=\s*", content
    ):
        return False

    # Add logger import and initialization
    lines = content.split("\n")

    # Find where to insert import
    import_pos = 0
    for i, line in enumerate(lines):
        if line.startswith(("import ", "from ")):
            import_pos = i + 1

    # Add import
    lines.insert(import_pos, "from genomevault.utils.logging import get_logger")

    # Find where to add logger initialization (after imports, before first function/class)
    init_pos = import_pos + 1
    for i in range(import_pos + 1, len(lines)):
        if lines[i] and not lines[i].startswith(("import ", "from ", "#", '"""', "'''")):
            init_pos = i
            break

    lines.insert(init_pos, "")
    lines.insert(init_pos + 1, "logger = get_logger(__name__)")
    lines.insert(init_pos + 2, "")

    content = "\n".join(lines)

    with open(filepath, "w") as f:
        f.write(content)

    return True


def main():
    """Main function to fix common issues"""
    print("Fixing common linting issues in GenomeVault...")

    # Get all Python files
    python_files = get_python_files("genomevault")

    typing_fixed = 0
    exception_fixed = 0
    logger_fixed = 0

    for filepath in python_files:
        changed = False

        # Fix typing imports
        if fix_missing_typing_imports(filepath):
            typing_fixed += 1
            changed = True

        # Fix unused exception variables
        if fix_unused_exception_variables(filepath):
            exception_fixed += 1
            changed = True

        # Add logger imports
        if add_logger_import(filepath):
            logger_fixed += 1
            changed = True

        if changed:
            print(f"Fixed: {filepath}")

    print(f"\nSummary:")
    print(f"  Files with typing imports fixed: {typing_fixed}")
    print(f"  Files with exception variables fixed: {exception_fixed}")
    print(f"  Files with logger imports added: {logger_fixed}")

    # Run Black to ensure formatting
    print("\nRunning Black formatter...")
    subprocess.run(["/Users/rohanvinaik/miniconda3/bin/black", "genomevault/"], check=False)

    # Run isort to fix imports
    print("\nRunning isort...")
    subprocess.run(["/Users/rohanvinaik/miniconda3/bin/isort", "genomevault/"], check=False)

    print("\nDone! Run check_linters.py to see remaining issues.")


if __name__ == "__main__":
    main()
