#!/usr/bin/env python3
"""Remove absolute paths from all files in the repository"""

import os
import re
import json
from pathlib import Path
from typing import Set

def find_files_with_absolute_paths(root_dir: Path) -> Set[Path]:
    """Find all files containing absolute paths"""
    files_with_paths = set()
    
    # Patterns to search for
    patterns = [
        r'/Users/[^/\s]+/',
        r'/home/[^/\s]+/',
        r'C:\\Users\\[^\\]+\\',
    ]
    
    # File extensions to check
    extensions = ['.py', '.md', '.json', '.txt', '.yml', '.yaml', '.sh']
    
    # Directories to skip
    skip_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', '.tox'}
    
    for root, dirs, files in os.walk(root_dir):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            # Check file extension
            if not any(file.endswith(ext) for ext in extensions):
                continue
                
            file_path = Path(root) / file
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern in patterns:
                    if re.search(pattern, content):
                        files_with_paths.add(file_path)
                        break
            except Exception:
                # Skip files that can't be read
                pass
    
    return files_with_paths

def clean_file(file_path: Path) -> bool:
    """Clean absolute paths from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Replace absolute paths with relative or generic paths
        replacements = [
            # User home directories
            (r'/Users/[^/\s]+/genomevault', '.'),
            (r'/Users/[^/\s]+/', '~/'),
            (r'/home/[^/\s]+/genomevault', '.'),
            (r'/home/[^/\s]+/', '~/'),
            (r'C:\\Users\\[^\\]+\\genomevault', '.'),
            (r'C:\\Users\\[^\\]+\\', '~\\'),
            
            # Specific paths in error messages
            (r'from \'[^\']*(/Users/[^/]+/genomevault/[^\']*)', r"from 'genomevault"),
            (r'\(/Users/[^/]+/genomevault/([^)]*)\)', r'(./\1)'),
        ]
        
        for pattern, replacement in replacements:
            content = re.sub(pattern, replacement, content)
        
        # Special handling for JSON files
        if file_path.suffix == '.json':
            try:
                data = json.loads(content)
                content = json.dumps(clean_json_paths(data), indent=2)
            except json.JSONDecodeError:
                pass  # Not valid JSON, use regex replacements
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def clean_json_paths(obj):
    """Recursively clean paths in JSON objects"""
    if isinstance(obj, dict):
        return {k: clean_json_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_paths(item) for item in obj]
    elif isinstance(obj, str):
        # Clean paths in strings
        obj = re.sub(r'/Users/[^/\s]+/genomevault', '.', obj)
        obj = re.sub(r'/Users/[^/\s]+/', '~/', obj)
        obj = re.sub(r'/home/[^/\s]+/genomevault', '.', obj)
        obj = re.sub(r'/home/[^/\s]+/', '~/', obj)
        return obj
    else:
        return obj

def main():
    """Main execution"""
    root_dir = Path(__file__).parent.parent
    
    print("🔍 Searching for files with absolute paths...")
    files_with_paths = find_files_with_absolute_paths(root_dir)
    
    if not files_with_paths:
        print("✅ No files with absolute paths found!")
        return
    
    print(f"Found {len(files_with_paths)} files with absolute paths:")
    for file_path in sorted(files_with_paths):
        relative_path = file_path.relative_to(root_dir)
        print(f"  - {relative_path}")
    
    print("\n🧹 Cleaning absolute paths...")
    cleaned_count = 0
    
    for file_path in files_with_paths:
        if clean_file(file_path):
            relative_path = file_path.relative_to(root_dir)
            print(f"  ✅ Cleaned: {relative_path}")
            cleaned_count += 1
    
    print(f"\n✅ Cleaned {cleaned_count} files")
    
    # Update .gitignore to exclude results with absolute paths
    gitignore_path = root_dir / '.gitignore'
    gitignore_additions = [
        '\n# Exclude results with potential absolute paths',
        'results/',
        'experimental_results/',
        'benchmark_results/**/*.json',
        '*.sig',
        '*.pem',
    ]
    
    with open(gitignore_path, 'r') as f:
        gitignore_content = f.read()
    
    if 'results/' not in gitignore_content:
        with open(gitignore_path, 'a') as f:
            f.write('\n'.join(gitignore_additions))
        print("✅ Updated .gitignore to exclude result artifacts")

if __name__ == "__main__":
    main()