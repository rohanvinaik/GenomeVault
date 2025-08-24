#!/usr/bin/env python3
"""
Remove ALL absolute paths from the entire codebase.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Set

# Patterns that indicate absolute paths
ABSOLUTE_PATH_PATTERNS = [
    r'/Users/[^"\'`\s\)]+',
    r'/home/[^"\'`\s\)]+',
    r'C:\\[^"\'`\s\)]+',
    r'C:/[^"\'`\s\)]+',
    r'/opt/[^"\'`\s\)]+',
    r'/var/[^"\'`\s\)]+',
    r'/tmp/[^"\'`\s\)]+',  # Should use tempfile module
]

# Files to check (including shell scripts and docs)
FILE_EXTENSIONS = ['.py', '.sh', '.bash', '.yml', '.yaml', '.md', '.txt', '.json']

def find_absolute_paths(directory: Path) -> List[Tuple[Path, int, str]]:
    """Find ALL absolute paths in all file types."""
    absolute_paths = []
    
    for pattern_str in ABSOLUTE_PATH_PATTERNS:
        pattern = re.compile(pattern_str)
        
        for ext in FILE_EXTENSIONS:
            for file_path in directory.rglob(f"*{ext}"):
                # Skip virtual environments and dependencies
                skip_dirs = ['venv', '.venv', 'node_modules', 'build', 'dist', '.git']
                if any(skip_dir in str(file_path) for skip_dir in skip_dirs):
                    continue
                
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    for i, line in enumerate(content.splitlines(), 1):
                        if matches := pattern.findall(line):
                            for match in matches:
                                # Skip URLs and documentation references
                                if 'http' in line or 'github.com' in line:
                                    continue
                                absolute_paths.append((file_path, i, match))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    return absolute_paths

def fix_file_paths(file_path: Path) -> bool:
    """Fix absolute paths in a specific file."""
    try:
        content = file_path.read_text()
        original_content = content
        
        # Replacement mappings
        replacements = {
            # User paths
            r'/Users/[^/]+/genomevault': '${GENOMEVAULT_ROOT:-$(pwd)}',
            r'/Users/[^/]+/Desktop': '${HOME}/Desktop',
            r'/Users/[^/]+/Downloads': '${HOME}/Downloads',
            r'/Users/[^/]+': '${HOME}',
            r'/home/[^/]+/genomevault': '${GENOMEVAULT_ROOT:-$(pwd)}',
            r'/home/[^/]+': '${HOME}',
            
            # System paths
            r'/tmp/genomevault': '${TMPDIR:-/tmp}/genomevault',
            r'/tmp': '${TMPDIR:-/tmp}',
            r'/opt/genomevault': '${GENOMEVAULT_ROOT:-/opt/genomevault}',
            r'/var/lib/genomevault': '${GENOMEVAULT_DATA:-/var/lib/genomevault}',
            r'/var/log/genomevault': '${GENOMEVAULT_LOGS:-/var/log/genomevault}',
            
            # Windows paths
            r'C:\\Users\\[^\\]+\\genomevault': '%GENOMEVAULT_ROOT%',
            r'C:\\Users\\[^\\]+': '%USERPROFILE%',
            r'C:/Users/[^/]+/genomevault': '%GENOMEVAULT_ROOT%',
            r'C:/Users/[^/]+': '%USERPROFILE%',
        }
        
        # Apply replacements based on file type
        if file_path.suffix in ['.sh', '.bash']:
            # Shell script replacements
            for pattern, replacement in replacements.items():
                content = re.sub(pattern, replacement, content)
                
        elif file_path.suffix == '.py':
            # Python replacements
            python_replacements = {
                r'Path.home()': 'Path(__file__).parent.parent',
                r'Path.home()]+"': 'Path.home()',
                r'Path.home()': 'Path(__file__).parent.parent',
                r'Path.home()]+"': 'Path.home()',
                r'Path(tempfile.gettempdir())': 'Path(tempfile.gettempdir())',
            }
            
            for pattern, replacement in python_replacements.items():
                content = re.sub(pattern, replacement, content)
            
            # Add imports if needed
            if 'Path(' in content and 'from pathlib import Path' not in content:
                content = 'from pathlib import Path\n' + content
            if 'tempfile.' in content and 'import tempfile' not in content:
                content = 'import tempfile\n' + content
                
        elif file_path.suffix in ['.yml', '.yaml']:
            # YAML replacements
            yaml_replacements = {
                r'/Users/[^/]+/genomevault': '${GENOMEVAULT_ROOT}',
                r'/home/[^/]+/genomevault': '${GENOMEVAULT_ROOT}',
                r'/tmp': '${TMPDIR}',
            }
            
            for pattern, replacement in yaml_replacements.items():
                content = re.sub(pattern, replacement, content)
        
        # Save if changed
        if content != original_content:
            file_path.write_text(content)
            return True
            
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        
    return False

def create_ci_guard():
    """Create GitHub Actions workflow to catch absolute paths."""
    
    ci_content = '''name: Absolute Path Guard

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  check-absolute-paths:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Check for absolute paths
      run: |
        echo "Checking for absolute paths..."
        
        # Define patterns to check
        PATTERNS=(
          "/Users/"
          Path.home()
          "C:\\\\"
          "C:/"
          "/opt/genomevault"
          "/var/lib/genomevault"
        )
        
        # Files to check
        FILES=$(find . -type f \\( \
          -name "*.py" -o \
          -name "*.sh" -o \
          -name "*.bash" -o \
          -name "*.yml" -o \
          -name "*.yaml" -o \
          -name "*.md" \
        \\) | grep -v node_modules | grep -v venv | grep -v .git)
        
        # Check each pattern
        FOUND=0
        for pattern in "${PATTERNS[@]}"; do
          echo "Checking for: $pattern"
          if grep -r "$pattern" $FILES 2>/dev/null; then
            echo "❌ Found absolute path: $pattern"
            FOUND=1
          fi
        done
        
        if [ $FOUND -eq 1 ]; then
          echo "❌ Absolute paths detected! Please use relative or environment-based paths."
          exit 1
        else
          echo "✅ No absolute paths found"
        fi
    
    - name: Run path checking script
      run: |
        if [ -f scripts/check_paths.py ]; then
          python3 scripts/check_paths.py
        fi
'''
    
    ci_path = Path(".github/workflows/absolute_path_guard.yml")
    ci_path.parent.mkdir(parents=True, exist_ok=True)
    ci_path.write_text(ci_content)
    
    print(f"✅ Created CI guard: {ci_path}")
    
    # Also create a pre-commit hook
    pre_commit_content = '''#!/bin/bash
# Pre-commit hook to check for absolute paths

echo "Checking for absolute paths..."

# Check for common absolute path patterns
if grep -r Path.home() \
   --include="*.py" \
   --include="*.sh" \
   --include="*.yml" \
   --exclude-dir=venv \
   --exclude-dir=node_modules .; then
    echo "❌ Absolute paths found! Please fix before committing."
    exit 1
fi

echo "✅ No absolute paths found"
'''
    
    pre_commit_path = Path(".git/hooks/pre-commit")
    if pre_commit_path.parent.exists():
        pre_commit_path.write_text(pre_commit_content)
        pre_commit_path.chmod(0o755)
        print(f"✅ Created pre-commit hook: {pre_commit_path}")

def main():
    """Main execution."""
    print("🔍 Removing ALL absolute paths from codebase")
    print("=" * 50)
    
    project_dir = Path(".")
    
    # Find all absolute paths
    print("\nSearching for absolute paths...")
    absolute_paths = find_absolute_paths(project_dir)
    
    if absolute_paths:
        print(f"\n❌ Found {len(absolute_paths)} absolute paths:")
        
        # Group by file
        by_file = {}
        for file_path, line_num, path in absolute_paths:
            if file_path not in by_file:
                by_file[file_path] = []
            by_file[file_path].append((line_num, path))
        
        # Show summary
        for file_path, occurrences in list(by_file.items())[:10]:
            print(f"\n  {file_path}:")
            for line_num, path in occurrences[:3]:
                print(f"    Line {line_num}: {path}")
        
        # Fix all files
        print("\n🔧 Fixing absolute paths...")
        fixed_count = 0
        for file_path in by_file.keys():
            if fix_file_paths(file_path):
                fixed_count += 1
                print(f"  Fixed: {file_path}")
        
        print(f"\n✅ Fixed {fixed_count} files")
    else:
        print("✅ No absolute paths found!")
    
    # Create CI guard
    print("\n📝 Creating CI guard...")
    create_ci_guard()
    
    print("\n" + "=" * 50)
    print("✅ Complete!")
    print("\nNext steps:")
    print("  1. Review changes: git diff")
    print("  2. Test scripts: bash e2e_demo.sh")
    print("  3. Commit: git add -A && git commit -m 'Remove all absolute paths'")

if __name__ == "__main__":
    sys.exit(main())