#!/usr/bin/env python3
"""
Enhance __init__.py files with proper imports and exports based on module contents.
"""

import os
import ast
from pathlib import Path
from typing import List, Dict


def find_exportable_items(filepath: str) -> Dict[str, List[str]]:
    """Find classes, functions, and constants that should be exported."""
    exports = {"classes": [], "functions": [], "constants": []}

    try:
        with open(filepath, "r") as f:
            tree = ast.parse(f.read())

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                if not node.name.startswith("_"):
                    exports["classes"].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_"):
                    exports["functions"].append(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if target.id.isupper() and not target.id.startswith("_"):
                            exports["constants"].append(target.id)
    except:
        pass

    return exports


def enhance_init_file(init_path: str) -> bool:
    """Enhance an __init__.py file with proper imports."""

    dir_path = Path(init_path).parent
    module_name = dir_path.name

    # Find Python files in the directory
    py_files = []
    for f in dir_path.glob("*.py"):
        if f.stem != "__init__" and not f.stem.startswith("_"):
            # Skip test files in non-test directories
            if "test" not in str(dir_path) and "test" in f.stem:
                continue
            py_files.append(f)

    if not py_files:
        return False

    # Collect exports from each file
    all_exports = {}
    for py_file in py_files:
        exports = find_exportable_items(str(py_file))
        if any(exports.values()):
            all_exports[py_file.stem] = exports

    if not all_exports:
        return False

    # Read current content
    with open(init_path, "r") as f:
        current_content = f.read()

    # Generate new content
    lines = []

    # Keep existing docstring if present
    if '"""' in current_content:
        docstring_end = current_content.find('"""', 3) + 3
        lines.append(current_content[:docstring_end])
    else:
        # Generate docstring based on module type
        if "test" in str(dir_path):
            lines.append(f'"""Test suite for {module_name} module."""')
        elif "example" in str(dir_path):
            lines.append(f'"""Examples for {module_name} module."""')
        elif "crypto" in str(dir_path):
            lines.append(f'"""Cryptographic implementations for {module_name}."""')
        elif "zk" in str(dir_path):
            lines.append(f'"""Zero-knowledge proof implementations for {module_name}."""')
        elif "hypervector" in str(dir_path) or "hv" in str(dir_path):
            lines.append(f'"""Hyperdimensional computing implementations for {module_name}."""')
        elif "api" in str(dir_path):
            lines.append(f'"""API implementations for {module_name}."""')
        else:
            lines.append(f'"""Module {module_name} implementation."""')

    lines.append("")

    # Add imports
    all_items = []
    for module, exports in all_exports.items():
        items = exports["classes"] + exports["functions"] + exports["constants"]
        if items:
            # Group imports
            if len(items) <= 3:
                lines.append(f"from .{module} import {', '.join(items)}")
            else:
                lines.append(f"from .{module} import (")
                for item in items[:-1]:
                    lines.append(f"    {item},")
                lines.append(f"    {items[-1]}")
                lines.append(")")
            all_items.extend(items)

    # Add __all__ export
    if all_items:
        lines.append("")
        lines.append("__all__ = [")
        for item in sorted(all_items):
            lines.append(f'    "{item}",')
        lines.append("]")

    # Write new content
    new_content = "\n".join(lines) + "\n"

    # Only write if different from current
    if new_content != current_content:
        with open(init_path, "w") as f:
            f.write(new_content)
        return True

    return False


def main():
    """Main function to enhance __init__.py files."""
    print("=" * 70)
    print("ENHANCING __init__.py FILES WITH PROPER IMPORTS")
    print("=" * 70)

    # Priority directories to enhance
    priority_dirs = [
        "genomevault/hypervector",
        "genomevault/crypto",
        "genomevault/zk_proofs",
        "genomevault/api/routers",
        "genomevault/federated",
        "genomevault/pir/client",
        "genomevault/pir/server",
        "genomevault/blockchain",
        "genomevault/clinical",
        "genomevault/utils",
    ]

    enhanced = []

    # Process priority directories
    for dir_path in priority_dirs:
        if os.path.exists(dir_path):
            init_path = os.path.join(dir_path, "__init__.py")
            if os.path.exists(init_path):
                print(f"\nProcessing {dir_path}...")
                if enhance_init_file(init_path):
                    enhanced.append(init_path)
                    print(f"  ✓ Enhanced {init_path}")
                else:
                    print("  - No changes needed")

    # Process other genomevault directories
    for root, dirs, files in os.walk("genomevault"):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git"}]

        if "__init__.py" in files:
            init_path = os.path.join(root, "__init__.py")
            if init_path not in enhanced and init_path not in [
                os.path.join(d, "__init__.py") for d in priority_dirs
            ]:
                if enhance_init_file(init_path):
                    enhanced.append(init_path)
                    print(f"Enhanced: {init_path}")

    # Summary
    print("\n" + "=" * 70)
    print(f"Total __init__.py files enhanced: {len(enhanced)}")

    if enhanced:
        print("\nEnhanced files:")
        for filepath in enhanced[:10]:
            print(f"  - {filepath}")
        if len(enhanced) > 10:
            print(f"  ... and {len(enhanced) - 10} more")

    print("\n✅ __init__.py enhancement complete!")


if __name__ == "__main__":
    main()
