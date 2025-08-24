#!/usr/bin/env python3
"""
Fix Python file headers to ensure proper order:
1. Shebang (if present)
2. Module docstring
3. from __future__ import annotations
4. Regular imports
"""

from pathlib import Path
import re

def fix_file_header(file_path: Path) -> bool:
    """Fix the header of a Python file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Extract components
        shebang = None
        docstring_lines = []
        future_import = None
        rest_lines = []
        
        i = 0
        
        # Handle shebang
        if lines and lines[0].startswith('#!'):
            shebang = lines[0]
            i = 1
        
        # Handle imports/docstrings at the beginning
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments at the beginning
            if not line or (line.startswith('#') and not line.startswith('#!')):
                i += 1
                continue
            
            # Check for future import
            if line.startswith('from __future__ import'):
                future_import = lines[i]
                i += 1
                continue
                
            # Check for logging import (our added imports)
            if 'from genomevault.utils.logging import' in line:
                # Skip logging imports for now, we'll handle them in rest
                break
                
            # Check for docstring
            if line.startswith('"""') or line.startswith("'''"):
                quote_char = '"""' if line.startswith('"""') else "'''"
                docstring_lines.append(lines[i])
                i += 1
                
                # Handle multi-line docstring
                if line.count(quote_char) < 2:  # Docstring continues
                    while i < len(lines) and quote_char not in lines[i]:
                        docstring_lines.append(lines[i])
                        i += 1
                    if i < len(lines):
                        docstring_lines.append(lines[i])
                        i += 1
                continue
            
            # Everything else goes to rest
            break
        
        # Rest of the file
        rest_lines = lines[i:]
        
        # Rebuild file
        new_lines = []
        
        # 1. Shebang
        if shebang:
            new_lines.append(shebang)
        
        # 2. Docstring
        if docstring_lines:
            if new_lines:
                new_lines.append('')
            new_lines.extend(docstring_lines)
        
        # 3. Future import
        if future_import:
            if new_lines:
                new_lines.append('')
            new_lines.append(future_import)
        
        # 4. Rest of file
        if rest_lines:
            if new_lines:
                new_lines.append('')
            new_lines.extend(rest_lines)
        
        # Write back if changed
        new_content = '\n'.join(new_lines)
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Fix all Python files in target directories."""
    directories = ['devtools', 'examples', 'tests']
    
    for dir_name in directories:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            continue
        
        pass  # Debug print removed
        
        fixed_count = 0
        for py_file in dir_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            
            if fix_file_header(py_file):
                print(f"  Fixed: {py_file}")
                fixed_count += 1
        
        print(f"  Fixed {fixed_count} files in {dir_name}/")

if __name__ == "__main__":
    main()