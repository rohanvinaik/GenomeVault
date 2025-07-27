#!/usr/bin/env python3
"""Final aggressive fix for all indentation issues."""

import os
from pathlib import Path

def fix_file_aggressively(filepath):
    """Fix all indentation issues in a file."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        fixed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Check if this is a function/method definition
            if stripped.startswith('def ') and stripped.endswith(':'):
                # Add the def line
                fixed_lines.append(line)
                i += 1
                
                # Check next lines for docstring
                while i < len(lines):
                    next_line = lines[i]
                    next_stripped = next_line.strip()
                    
                    # Skip empty lines
                    if not next_stripped:
                        fixed_lines.append(next_line)
                        i += 1
                        continue
                    
                    # Check if it's a docstring
                    if (next_stripped.startswith('"""') or next_stripped.startswith("'''")):
                        # Calculate proper indentation
                        def_indent = len(line) - len(line.lstrip())
                        docstring_indent = def_indent + 4
                        
                        # Add properly indented docstring
                        fixed_lines.append(' ' * docstring_indent + next_stripped + '\n')
                        i += 1
                        break
                    else:
                        # Not a docstring, continue normally
                        break
            
            # Fix standalone lines that start with """TODO
            elif stripped.startswith('"""TODO:') or stripped.startswith("'''TODO:"):
                # This is a misplaced docstring
                # Find appropriate indentation by looking at context
                proper_indent = 8  # Default to method body level
                
                # Look backwards for context
                for j in range(len(fixed_lines) - 1, -1, -1):
                    prev_line = fixed_lines[j].rstrip()
                    if prev_line and not prev_line.isspace():
                        if ':' in prev_line:
                            # Found a definition, indent from there
                            prev_indent = len(fixed_lines[j]) - len(fixed_lines[j].lstrip())
                            proper_indent = prev_indent + 4
                            break
                
                fixed_lines.append(' ' * proper_indent + stripped + '\n')
                i += 1
            
            # Fix lines that have wrong indentation for class/method bodies
            elif line.startswith('        """TODO:'):
                # This might be correct, keep it
                fixed_lines.append(line)
                i += 1
            elif line.lstrip().startswith('"""TODO:') and not line.startswith('    '):
                # Wrong indentation, fix it
                # Assume it should be at least 4 spaces
                indent = len(line) - len(line.lstrip())
                if indent == 0:
                    indent = 8  # Method body level
                elif indent % 4 != 0:
                    indent = ((indent // 4) + 1) * 4
                
                fixed_lines.append(' ' * indent + line.lstrip())
                i += 1
            else:
                # Keep the line as is
                fixed_lines.append(line)
                i += 1
        
        # Write back
        new_content = ''.join(fixed_lines)
        with open(filepath, 'w') as f:
            f.write(new_content)
        
        return True
    except Exception as e:
        return False

def main():
    """Main function."""
    print("Applying aggressive fixes to all Python files...")
    
    fixed_count = 0
    for py_file in Path('.').rglob('*.py'):
        if any(skip in str(py_file) for skip in ['.venv', 'venv', '__pycache__', '.git']):
            continue
        
        if fix_file_aggressively(py_file):
            fixed_count += 1
    
    print(f"Processed {fixed_count} files")

if __name__ == "__main__":
    main()
