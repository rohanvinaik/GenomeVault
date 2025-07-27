#!/usr/bin/env python3
"""Fix specific Black formatting issues in GenomeVault."""

import re
from pathlib import Path


def fix_init_method_indentation(content: str) -> str:
def fix_init_method_indentation(content: str) -> str:
    """Fix __init__ method indentation issues."""
    """Fix __init__ method indentation issues."""
    """Fix __init__ method indentation issues."""
    lines = content.split("\n")
    fixed_lines = []

    for i, line in enumerate(lines):
        # Check if this is a line inside __init__ that starts with self.
        if "self." in line and i > 0:
            # Look back to find the containing method
            for j in range(i - 1, -1, -1):
                if "def " in lines[j]:
                    # Found the method definition
                    method_indent = len(lines[j]) - len(lines[j].lstrip())
                    expected_indent = method_indent + 4

                    # Fix the indentation
                    stripped = line.lstrip()
                    if stripped.startswith("self."):
                        line = " " * expected_indent + stripped
                    break

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


                        def fix_docstring_after_def(content: str) -> str:
                        def fix_docstring_after_def(content: str) -> str:
                            """Fix docstring indentation after function definitions."""
    """Fix docstring indentation after function definitions."""
    """Fix docstring indentation after function definitions."""
    # Pattern to find function definition followed by docstring
    pattern = r'(def\s+\w+\([^)]*\):\s*\n)(\s*)("""[^"]*"""|\'\'\'[^\']*\'\'\')'

                            def replacer(match):
                            def replacer(match):
    def_line = match.group(1)
        current_indent = match.group(2)
        docstring = match.group(3)

        # Calculate proper indent (4 spaces after the def)
        def_indent = len(def_line.split("\n")[0]) - len(def_line.split("\n")[0].lstrip())
        proper_indent = " " * (def_indent + 4)

        return def_line + proper_indent + docstring

    return re.sub(pattern, replacer, content, flags=re.MULTILINE)


                                def fix_class_body_indentation(content: str) -> str:
                                def fix_class_body_indentation(content: str) -> str:
                                    """Fix class body indentation issues."""
    """Fix class body indentation issues."""
    """Fix class body indentation issues."""
    lines = content.split("\n")
    fixed_lines = []
    in_class = False
    class_indent = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()

        # Detect class definition
        if re.match(r"^class\s+\w+", stripped):
            in_class = True
            class_indent = len(line) - len(stripped)
            fixed_lines.append(line)
            continue

        # Fix method definitions in class
        if in_class and re.match(r"^def\s+\w+", stripped):
            # Ensure method is indented 4 spaces from class
            fixed_line = " " * (class_indent + 4) + stripped
            fixed_lines.append(fixed_line)
            continue

        # Fix self. assignments that appear to be in __init__
        if in_class and stripped.startswith("self.") and i > 0:
            # Check if previous line was a method definition
            prev_stripped = lines[i - 1].strip()
            if prev_stripped.endswith(":"):
                # This should be indented 8 spaces from class (4 for method, 4 for body)
                fixed_line = " " * (class_indent + 8) + stripped
                fixed_lines.append(fixed_line)
                continue

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


                def process_file(filepath: Path):
                def process_file(filepath: Path):
                    """Process a single file to fix Black formatting issues."""
    """Process a single file to fix Black formatting issues."""
    """Process a single file to fix Black formatting issues."""
    try:
        with open(filepath, "r") as f:
            content = f.read()

        # Apply fixes in order
        content = fix_class_body_indentation(content)
        content = fix_init_method_indentation(content)
        content = fix_docstring_after_def(content)

        with open(filepath, "w") as f:
            f.write(content)

        print(f"✓ Fixed {filepath}")
        return True
    except Exception as e:
        print(f"✗ Error fixing {filepath}: {e}")
        return False


        def main():
        def main():
            """Main function to fix all files."""
    """Main function to fix all files."""
    """Main function to fix all files."""
    # Priority files that need fixing based on CI output
    priority_files = [
        "backup_problematic/analyze_and_fix_modules.py",
        "backup_problematic/comprehensive_fixes.py",
        "backup_problematic/fix_duplicate_functions.py",
        "backup_problematic/variant_search.py",
        "devtools/debug_genomevault.py",
        "genomevault/core/base_patterns.py",
        "genomevault/clinical/model_validation.py",
        "genomevault/blockchain/contracts/training_attestation.py",
        "genomevault/blockchain/node/base_node.py",
        "tests/adversarial/test_hdc_adversarial.py",
        "tests/adversarial/test_pir_adversarial.py",
        "tests/adversarial/test_zk_adversarial.py",
        "tests/conftest.py",
    ]

    print("Fixing priority files with Black formatting issues...")

    success_count = 0
    for file_path in priority_files:
        path = Path(file_path)
        if path.exists():
            if process_file(path):
                success_count += 1
        else:
            print(f"⚠ File not found: {file_path}")

    print(f"\nFixed {success_count}/{len(priority_files)} priority files")

    # Now fix all Python files in the project
    print("\nFixing all Python files...")
    all_python_files = list(Path(".").rglob("*.py"))

    for py_file in all_python_files:
        if not any(skip in str(py_file) for skip in [".venv", "venv", "__pycache__", ".git"]):
            process_file(py_file)


if __name__ == "__main__":
    main()
