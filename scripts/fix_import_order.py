#!/usr/bin/env python3
"""
Fix import order issues caused by the debug print removal script.
Ensures proper Python file structure:
1. Shebang (if present)
2. Module docstring
3. Future imports
4. Regular imports (including logging)
"""

import re
from pathlib import Path
from typing import List, Tuple


def fix_file_structure(file_path: Path) -> bool:
    """Fix the structure of a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if not lines:
            return False
        
        # Categorize lines
        shebang_lines = []
        docstring_lines = []
        future_imports = []
        logging_imports = []
        other_imports = []
        rest_of_file = []
        
        in_docstring = False
        docstring_quote = None
        found_code = False
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Handle shebang
            if i == 0 and line.startswith('#!'):
                shebang_lines.append(line)
                i += 1
                continue
            
            # Skip initial comments (but not shebang)
            if not found_code and stripped.startswith('#') and not line.startswith('#!'):
                i += 1
                continue
            
            # Handle docstrings
            if not found_code and not in_docstring:
                if stripped.startswith('"""') or stripped.startswith("'''"):
                    in_docstring = True
                    docstring_quote = '"""' if '"""' in stripped else "'''"
                    docstring_lines.append(line)
                    # Check if docstring ends on same line
                    if stripped.count(docstring_quote) >= 2:
                        in_docstring = False
                        found_code = True
                    i += 1
                    continue
            
            if in_docstring:
                docstring_lines.append(line)
                if docstring_quote in stripped:
                    in_docstring = False
                    found_code = True
                i += 1
                continue
            
            # Handle imports
            if stripped.startswith('from __future__ import'):
                future_imports.append(line)
                found_code = True
                i += 1
                continue
            
            if (stripped.startswith('from genomevault.utils.logging import') or 
                stripped == 'logger = get_logger(__name__)'):
                logging_imports.append(line)
                found_code = True
                i += 1
                continue
            
            if (stripped.startswith('from ') or stripped.startswith('import ')) and not any(
                kw in stripped for kw in ['__future__', 'genomevault.utils.logging']
            ):
                other_imports.append(line)
                found_code = True
                i += 1
                continue
            
            # Rest of the file
            rest_of_file.extend(lines[i:])
            break
        
        # Rebuild file in correct order
        new_lines = []
        
        # 1. Shebang
        new_lines.extend(shebang_lines)
        
        # 2. Module docstring
        new_lines.extend(docstring_lines)
        
        # 3. Future imports
        if future_imports:
            if new_lines and not new_lines[-1].strip() == '':
                new_lines.append('\n')
            new_lines.extend(future_imports)
        
        # 4. Regular imports (including logging)
        all_imports = other_imports + logging_imports
        if all_imports:
            if new_lines and not new_lines[-1].strip() == '':
                new_lines.append('\n')
            new_lines.extend(all_imports)
        
        # 5. Rest of file
        if rest_of_file:
            if new_lines and not new_lines[-1].strip() == '':
                new_lines.append('\n')
            new_lines.extend(rest_of_file)
        
        # Write back if changed
        new_content = ''.join(new_lines)
        old_content = ''.join(lines)
        
        if new_content != old_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Fix import order in all affected files."""
    directories = ['devtools', 'examples', 'tests']
    
    total_fixed = 0
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            continue
        
        pass  # Debug print removed
        
        for py_file in dir_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            if fix_file_structure(py_file):
                print(f"  Fixed: {py_file}")
                total_fixed += 1
    
    print(f"\nFixed {total_fixed} files")
    return 0


if __name__ == "__main__":
    main()