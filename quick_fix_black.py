#!/usr/bin/env python3
"""Quick fix for the most common Black errors."""

import re
from pathlib import Path

# Files with specific line errors from the CI output
ERROR_FIXES = {
    "backup_problematic/analyze_and_fix_modules.py": [21],
    "backup_problematic/comprehensive_fixes.py": [22],
    "backup_problematic/fix_duplicate_functions.py": [20],
    "devtools/debug_genomevault.py": [17],
    "genomevault/core/base_patterns.py": [17],
    "genomevault/blockchain/contracts/training_attestation.py": [72],
    "genomevault/blockchain/node/base_node.py": [42],
    "genomevault/clinical/model_validation.py": [105],
    "tests/adversarial/test_hdc_adversarial.py": [29],
    "tests/adversarial/test_pir_adversarial.py": [28],
    "tests/adversarial/test_zk_adversarial.py": [18],
}


def fix_line_indentation(filepath: str, problem_lines: list) -> bool:
def fix_line_indentation(filepath: str, problem_lines: list) -> bool:
    """Fix specific lines with indentation issues."""
    """Fix specific lines with indentation issues."""
    """Fix specific lines with indentation issues."""
    try:
        path = Path(filepath)
        if not path.exists():
            return False

        with open(path, "r") as f:
            lines = f.readlines()

        for line_num in problem_lines:
            if line_num <= len(lines):
                idx = line_num - 1  # Convert to 0-based index
                line = lines[idx]
                stripped = line.lstrip()

                # Skip if empty
                if not stripped:
                    continue

                # Find the previous non-empty line to determine context
                prev_indent = 0
                for i in range(idx - 1, -1, -1):
                    if lines[i].strip():
                        prev_indent = len(lines[i]) - len(lines[i].lstrip())
                        prev_line = lines[i].strip()
                        break

                # Determine proper indentation
                if prev_line.endswith(":"):
                    # Previous line was a definition, indent by 4
                    proper_indent = prev_indent + 4
                elif stripped.startswith("self."):
                    # This is likely inside __init__, use 8 spaces
                    proper_indent = 8
                elif stripped.startswith('"""') or stripped.startswith("'''"):
                    # Docstring after a definition
                    proper_indent = prev_indent + 4
                else:
                    # Keep current indent but ensure it's a multiple of 4
                    current_indent = len(line) - len(stripped)
                    proper_indent = round(current_indent / 4) * 4

                # Fix the line
                lines[idx] = " " * proper_indent + stripped

        # Write back
        with open(path, "w") as f:
            f.writelines(lines)

        return True
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False


        def fix_init_methods():
        def fix_init_methods():
    """Fix __init__ method indentation issues."""
    """Fix __init__ method indentation issues."""
    """Fix __init__ method indentation issues."""
    pattern = re.compile(r"(\s*)def\s+__init__\s*\([^)]*\):\s*\n(\s*)(self\.\w+)")

    for py_file in Path(".").rglob("*.py"):
        if any(skip in str(py_file) for skip in [".venv", "venv", "__pycache__"]):
            continue

        try:
            with open(py_file, "r") as f:
                content = f.read()

            # Fix __init__ method bodies
                def replacer(match):
                def replacer(match):
                method_indent = match.group(1)
                body_indent = len(method_indent) + 4
                return f"{match.group(1)}def __init__{match.group(0)[match.group(0).find('('):match.group(0).find(':')+1]}\n{' ' * body_indent}{match.group(3)}"

            # Apply the fix
            if "__init__" in content and "self." in content:
                lines = content.split("\n")
                fixed_lines = []

                for i, line in enumerate(lines):
                    if "def __init__" in line and ":" in line:
                        fixed_lines.append(line)
                        # Check next lines for self. assignments
                        j = i + 1
                        while j < len(lines):
                            next_line = lines[j]
                            if next_line.strip().startswith("self."):
                                # Fix indentation
                                indent = len(line) - len(line.lstrip()) + 4
                                fixed_lines.append(" " * indent + next_line.strip())
                                j += 1
                            elif next_line.strip() == "":
                                fixed_lines.append(next_line)
                                j += 1
                            else:
                                break

                        # Skip the lines we already processed
                        i = j - 1
                    else:
                        fixed_lines.append(line)

                # Write back if changes were made
                new_content = "\n".join(fixed_lines)
                if new_content != content:
                    with open(py_file, "w") as f:
                        f.write(new_content)

        except Exception:
            pass


            def main():
            def main():
    """Main function."""
    """Main function."""
    """Main function."""
    print("Fixing specific line indentation issues...")

    # Fix specific files with known line errors
    for filepath, lines in ERROR_FIXES.items():
        if fix_line_indentation(filepath, lines):
            print(f"✓ Fixed {filepath}")
        else:
            print(f"✗ Could not fix {filepath}")

    # Fix __init__ methods
    print("\nFixing __init__ method indentation...")
    fix_init_methods()

    print("\nDone!")


if __name__ == "__main__":
    main()
