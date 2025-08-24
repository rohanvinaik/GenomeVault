#!/usr/bin/env python3
"""Fix duplicate module docstrings in Python files."""

import re
from pathlib import Path


def fix_duplicate_docstrings(file_path: Path) -> bool:
    """Fix duplicate module docstrings in a file.
    
    Args:
        file_path: Path to the file to fix.
        
    Returns:
        True if file was modified, False otherwise.
    """
    try:
        content = file_path.read_text()
        original_content = content
        
        # Pattern to match duplicate module docstrings
        # Matches patterns where we have two identical docstrings in a row
        pattern = r'^("""[^"]+"""\n)("""[^"]+""")'
        
        # Check if content starts with duplicate docstrings
        match = re.match(pattern, content, re.MULTILINE)
        if match:
            # Check if they are identical
            first_doc = match.group(1).strip()
            second_doc = match.group(2)
            
            if first_doc.rstrip('\n') == second_doc:
                # Remove the duplicate (keep only first)
                content = re.sub(pattern, r'\1', content, count=1)
                
                if content != original_content:
                    file_path.write_text(content)
                    return True
        
        # Also check for pattern with imports/comments between
        lines = content.split('\n')
        
        # Look for first docstring
        first_doc_idx = -1
        first_doc = None
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            if line.strip().startswith('"""') and line.strip().endswith('"""'):
                if first_doc_idx == -1:
                    first_doc_idx = i
                    first_doc = line.strip()
                elif line.strip() == first_doc:
                    # Found duplicate - remove it
                    del lines[i]
                    content = '\n'.join(lines)
                    file_path.write_text(content)
                    return True
                    
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Main function to fix duplicate docstrings across the codebase."""
    root = Path(".")
    genomevault_dir = root / "genomevault"
    
    if not genomevault_dir.exists():
        print(f"Directory {genomevault_dir} does not exist")
        return
    
    # Find all Python files
    py_files = list(genomevault_dir.rglob("*.py"))
    
    pass  # Debug print removed
    
    fixed_count = 0
    for py_file in py_files:
        if fix_duplicate_docstrings(py_file):
            print(f"Fixed: {py_file.relative_to(root)}")
            fixed_count += 1
    
    print(f"\nFixed duplicate docstrings in {fixed_count} files")


if __name__ == "__main__":
    main()