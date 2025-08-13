#!/usr/bin/env python3
"""Remove debug print statements from Python files."""

import re
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


    # Skip if it's a comment
    if stripped.startswith('#'):
        return False

    # Common debug print patterns
    debug_patterns = [
        r'^\s*print\s*\(\s*[\'"] \
            (?:DEBUG|debug|Debug|TODO|FIXME|XXX|>>>|--- \
                |\*\*\*|===|\.\.\.|Processing|Checking|Starting|Found|Error|Warning|Info).*[\'"]',
        r'^\s*print\s*\(\s*f[\'"](?:DEBUG|debug|Debug|Processing|Checking|Starting|Found).*[\'"]',
        r'^\s*print\s*\(\s*[\'"]\\n.*[\'"]',  # Prints starting with newline
        r'^\s*print\s*\(\s*[\'"]={5,}.*[\'"]',  # Separator lines
        r'^\s*print\s*\(\s*[\'"][=-]{3,}.*[\'"]',  # Separator lines
        r'^\s*print\s*\(\s*\)',  # Empty prints
        r'^\s*print\s*\(\s*[\'"]\\t.*[\'"]',  # Tab-indented prints
        r'^\s*print\s*\(\s*[\'"]Step \d+:.*[\'"]',  # Step indicators
        r'^\s*print\s*\(\s*[\'"]Test.*[\'"]',  # Test messages
        r'^\s*print\s*\(\s*"[^"]*\s+\.\.\.\s*"',  # Messages ending with ...
    ]

    for pattern in debug_patterns:
        if re.match(pattern, stripped):
            return True

    # Check for multiline debug prints
    if 'print(' in stripped and any(keyword in stripped.lower() for keyword in
                                    ['debug', 'trace', 'verbose', 'todo', 'fixme',
                                     'xxx', 'temp', 'test', 'check']):
        return True

    return False

def should_keep_print(line):
    """Check if a print statement should be kept (part of actual output)."""
    stripped = line.strip()

    # Keep prints that are part of actual program output
    keep_patterns = [
        r'^\s*print\s*\(\s*[\'"]Usage:',
        r'^\s*print\s*\(\s*[\'"]Results?:',
        r'^\s*print\s*\(\s*[\'"]Output:',
        r'^\s*print\s*\(.*__name__.*__main__',
        r'^\s*print\s*\(\s*json\.',
        r'^\s*print\s*\(\s*.*\.to_json',
        r'^\s*print\s*\(\s*help\(',
    ]

    for pattern in keep_patterns:
        if re.match(pattern, stripped):
            return True

    # Keep prints in main blocks that show results
    if '__main__' in stripped:
        return True

    return False

def remove_debug_prints_from_file(filepath):
    """Remove debug print statements from a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
        return 0

    new_lines = []
    removed_count = 0
    in_multiline_print = False
    multiline_buffer = []

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Handle multiline prints
        if in_multiline_print:
            multiline_buffer.append(line)
            if ')' in line and not line.rstrip().endswith(','):
                # End of multiline print
                in_multiline_print = False
                full_print = ''.join(multiline_buffer)
                    if indent > 0 and i + 1 < len(lines) and not lines[i + 1].strip():
                        new_lines.append(' ' * indent + 'pass  # Debug print removed\n')
                else:
                    new_lines.extend(multiline_buffer)
                multiline_buffer = []
            i += 1
            continue

        # Check if this starts a multiline print
        if 'print(' in stripped and not stripped.endswith(')'):
            in_multiline_print = True
            multiline_buffer = [line]
            i += 1
            continue

        # Single line print
        if is_debug_print(line) and not should_keep_print(line):
            removed_count += 1
            # Add pass if needed for indentation
            indent = len(line) - len(line.lstrip())
            if indent > 0 and i + 1 < len(lines) and not lines[i + 1].strip():
                new_lines.append(' ' * indent + 'pass  # Debug print removed\n')
        else:
            new_lines.append(line)

        i += 1

    # Write back only if changes were made
    if removed_count > 0:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
            return removed_count
        except Exception as e:
            logger.error(f"Error writing {filepath}: {e}")
            return 0

    return 0

def process_files(file_list):
    """Process a list of files to remove debug prints."""
    total_removed = 0

    for filepath in file_list:
        if os.path.exists(filepath):
            removed = remove_debug_prints_from_file(filepath)
            if removed > 0:
                logger.info(f"✓ {filepath}: Removed {removed} debug prints")
                total_removed += removed
        else:
            logger.warning(f"✗ {filepath}: File not found")

    return total_removed

def find_all_python_files():
    """Find all Python files in the project."""
    python_files = []
    for root, dirs, files in os.walk('.'):
        # Skip virtual environments and cache directories
        dirs[:] = [d for d in dirs if d not in {'.venv', \
            'venv', \
            '__pycache__', \
            '.git', \
            'node_modules'}]

        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))

    return python_files

def main():
    """Main function to remove debug prints."""
    logger.info("=" * 70)
    logger.info("REMOVING DEBUG PRINT STATEMENTS")
    logger.info("=" * 70)

    # Priority files with most debug prints
    priority_files = [
        'devtools/enhanced_cleanup.py',
        'examples/hdc_pir_zk_integration_demo.py',
        'devtools/comprehensive_cleanup.py',
        'devtools/final_validation.py',
        'tests/test_advanced_implementations.py',
    ]

    logger.info("\nProcessing priority files...")
    priority_removed = process_files(priority_files)

    logger.info(f"\nRemoved {priority_removed} debug prints from priority files")

    # Process all other Python files
    logger.info("\nSearching for remaining debug prints...")
    all_files = find_all_python_files()

    # Exclude priority files already processed
    remaining_files = [f for f in all_files if f not in priority_files]

    remaining_removed = process_files(remaining_files)

    total_removed = priority_removed + remaining_removed

    logger.info("\n" + "=" * 70)
    logger.info(f"COMPLETE: Removed {total_removed} debug print statements")
    logger.info("=" * 70)

    return total_removed

if __name__ == "__main__":
    main()
