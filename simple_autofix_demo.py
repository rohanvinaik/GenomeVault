#!/usr/bin/env python3
"""
Simple demonstration of automated code fixes for GenomeVault
This is a simplified version showing the core concepts
"""

import os
import re
import sys


def fix_bare_except(content):
    """Fix bare except clauses by converting to except Exception"""
    # Pattern to match bare except:
    pattern = r"(\s*)except\s*:\s*$"
    replacement = r"\1except Exception:  # TODO: narrow this bare \'except\' to specific exception(s)"

    fixed_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    return fixed_content


def fix_print_statements(content):
    """Convert print statements to logger.info"""
    # Check if logging is already imported
    has_logging_import = "import logging" in content
    has_logger = "logger = logging.getLogger" in content

    # Pattern to match print statements
    pattern = r"print\s*\((.*?)\)"
    replacement = r"logger.info(\1)"

    fixed_content = re.sub(pattern, replacement, content)

    # Add logging imports if needed and print was replaced
    if fixed_content != content:  # If we made changes
        if not has_logging_import:
            fixed_content = "import logging\n" + fixed_content
        if not has_logger:
            # Add logger after imports
            lines = fixed_content.split("\n")
            import_end = 0
            for i, line in enumerate(lines):
                if line.startswith("import ") or line.startswith("from "):
                    import_end = i + 1
            lines.insert(import_end, "logger = logging.getLogger(__name__)\n")
            fixed_content = "\n".join(lines)

    return fixed_content


def fix_star_imports(content):
    """Add TODO comments to star imports"""
    pattern = r"(from\s+\S+\s+import\s+\*)"
    replacement = r"\1  # TODO: replace star import with explicit names"

    fixed_content = re.sub(pattern, replacement, content)
    return fixed_content


def fix_unused_params(content):
    """Simple regex-based approach to rename obviously unused params"""
    # This is a simplified version - the real implementation uses AST
    # Pattern to match function definitions
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        if "def " in line and "(" in line and ")" in line:
            # Extract function signature
            match = re.match(r"(\s*def\s+\w+\s*\()([^)]+)(\).*)", line)
            if match:
                indent, params, rest = match.groups()
                # Split parameters
                param_list = [p.strip() for p in params.split(",")]
                # For demo: rename params that contain 'unused'
                new_params = []
                for param in param_list:
                    if "unused" in param and not param.startswith("_"):
                        new_params.append("_" + param)
                    else:
                        new_params.append(param)
                fixed_line = indent + ", ".join(new_params) + rest
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def process_file(filepath, apply=False):
    """Process a single Python file"""
    print(f"Processing: {filepath}")

    try:
        with open(filepath, encoding="utf-8") as f:
            original_content = f.read()
    except Exception as e:
        print(f"  Error reading file: {e}")
        return False

    # Apply fixes
    content = original_content
    content = fix_bare_except(content)
    content = fix_print_statements(content)
    content = fix_star_imports(content)
    content = fix_unused_params(content)

    # Check if changes were made
    if content != original_content:
        print("  ✓ Changes detected")
        if apply:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(content)
                print("  ✓ Changes applied")
            except Exception as e:
                print(f"  ✗ Error writing file: {e}")
                return False
        else:
            print("  (Dry run - no changes written)")
        return True
    else:
        print("  No changes needed")
        return False


def main():
    """Main function"""
    apply_changes = "--apply" in sys.argv

    print("GenomeVault Simple Auto-Fix Demo")
    print("================================")
    print(f"Mode: {'APPLY CHANGES' if apply_changes else 'DRY RUN'}")
    print()

    # For demo, just process the test file
    test_file = "test_autofix_example.py"
    if os.path.exists(test_file):
        changed = process_file(test_file, apply=apply_changes)

        if not apply_changes and changed:
            print("\nTo apply changes, run with --apply flag")
    else:
        print(f"Test file {test_file} not found")
        print("Creating example file...")

        example_content = '''#!/usr/bin/env python3
"""Example file with issues to fix"""

# Issue 1: Print statements
print("Starting application")

# Issue 2: Bare except
try:
    risky_operation()
except:
    print("Error occurred")

# Issue 3: Star import
from os import *

# Issue 4: Unused parameters
def process_data(self, data, unused_param):
    return data * 2

# Issue 5: Broad exception
try:
    another_operation()
except Exception:
    pass
'''

        with open(test_file, "w") as f:
            f.write(example_content)
        print(f"Created {test_file}")
        print("Run the script again to process it")


if __name__ == "__main__":
    main()
