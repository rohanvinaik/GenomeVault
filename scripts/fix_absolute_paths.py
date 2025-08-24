#!/usr/bin/env python3
"""
Fix absolute paths in the codebase.

Replaces hardcoded Path.home()Path.home()Path.home()/'  # Home directory references
    ]
    
    combined_pattern = '|'.join(f'({p})' for p in patterns)
    
    for py_file in directory.rglob("*.py"):
        # Skip virtual environments and build directories
        if any(part in py_file.parts for part in ['venv', '.venv', 'node_modules', 'build', 'dist']):
            continue
            
        try:
            content = py_file.read_text()
            for i, line in enumerate(content.splitlines(), 1):
                if matches := re.findall(combined_pattern, line):
                    for match_group in matches:
                        match = next(m for m in match_group if m)
                        absolute_paths.append((py_file, i, match))
        except Exception as e:
            print(f"Error reading {py_file}: {e}")
    
    return absolute_paths

def fix_absolute_paths(directory: Path):
    """Fix absolute paths by replacing with relative/configurable ones."""
    
    replacements = {
        # Map common absolute paths to relative ones
        r'Path.home()/]+/genomevault': '.',
        r'Path.home()/]+/Desktop': 'Path.home() / "Desktop"',
        r'Path.home()/]+/Documents': 'Path.home() / "Documents"',
        r'Path.home()/]+/Downloads': 'Path.home() / "Downloads"',
        r'Path.home()/]+': 'Path.home()',
        r'Path.home()/]+/genomevault': '.',
        r'Path.home()/]+': 'Path.home()',
        r'Path.home() / ': 'Path.home() / ',
    }
    
    files_fixed = set()
    
    for py_file in directory.rglob("*.py"):
        if any(part in py_file.parts for part in ['venv', '.venv', 'node_modules']):
            continue
            
        try:
            content = py_file.read_text()
            original_content = content
            
            for pattern, replacement in replacements.items():
                content = re.sub(pattern, replacement, content)
            
            # Add Path import if needed and content was changed
            if content != original_content:
                if 'from pathlib import Path' not in content and 'Path.' in content:
                    # Add import at the top after other imports
                    lines = content.splitlines()
                    import_added = False
                    
                    for i, line in enumerate(lines):
                        if line.startswith('import ') or line.startswith('from '):
                            continue
                        elif not line.strip() or line.startswith('#'):
                            continue
                        else:
                            # Found first non-import line
                            lines.insert(i, 'from pathlib import Path')
                            lines.insert(i+1, '')
                            import_added = True
                            break
                    
                    if not import_added:
                        lines.insert(0, 'from pathlib import Path')
                        lines.insert(1, '')
                    
                    content = '\n'.join(lines)
                
                py_file.write_text(content)
                files_fixed.add(py_file)
                
        except Exception as e:
            print(f"Error fixing {py_file}: {e}")
    
    return files_fixed

def create_path_config():
    """Create a configuration file for paths."""
    
    config_content = '''"""
Path configuration for GenomeVault.

All paths should be configured here rather than hardcoded.
"""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
KEYS_DIR = PROJECT_ROOT / "keys"
CIRCUITS_DIR = PROJECT_ROOT / "circuits"
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ENCRYPTED_DATA_DIR = DATA_DIR / "encrypted"

# User directories (configurable)
USER_HOME = Path.home()
USER_DESKTOP = USER_HOME / "Desktop"
USER_DOCUMENTS = USER_HOME / "Documents"
USER_DOWNLOADS = USER_HOME / "Downloads"

# Environment-based paths
GENOMEVAULT_HOME = Path(os.environ.get('GENOMEVAULT_HOME', str(PROJECT_ROOT)))
GENOMEVAULT_DATA = Path(os.environ.get('GENOMEVAULT_DATA', str(DATA_DIR)))
GENOMEVAULT_LOGS = Path(os.environ.get('GENOMEVAULT_LOGS', str(LOGS_DIR)))

# Temporary directories
TEMP_DIR = Path(os.environ.get('TMPDIR', '/tmp')) / 'genomevault'
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Create all directories
for dir_path in [
    DATA_DIR, RESULTS_DIR, LOGS_DIR, KEYS_DIR, 
    CIRCUITS_DIR, CONFIGS_DIR, RAW_DATA_DIR,
    PROCESSED_DATA_DIR, ENCRYPTED_DATA_DIR
]:
    dir_path.mkdir(parents=True, exist_ok=True)

def get_data_path(filename: str, data_type: str = 'raw') -> Path:
    """Get path for a data file."""
    if data_type == 'raw':
        return RAW_DATA_DIR / filename
    elif data_type == 'processed':
        return PROCESSED_DATA_DIR / filename
    elif data_type == 'encrypted':
        return ENCRYPTED_DATA_DIR / filename
    else:
        return DATA_DIR / filename

def get_result_path(filename: str) -> Path:
    """Get path for a result file."""
    return RESULTS_DIR / filename

def get_log_path(filename: str) -> Path:
    """Get path for a log file."""
    return LOGS_DIR / filename

def get_config_path(filename: str) -> Path:
    """Get path for a config file."""
    return CONFIGS_DIR / filename
'''
    
    config_path = Path("genomevault/utils/paths.py")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_content)
    
    return config_path

def main():
    """Main execution."""
    print("🔧 Fixing absolute paths in codebase")
    print("=" * 50)
    
    project_dir = Path(".")
    
    # Find absolute paths
    print("\nSearching for absolute paths...")
    absolute_paths = find_absolute_paths(project_dir)
    
    if absolute_paths:
        print(f"\nFound {len(absolute_paths)} absolute paths:")
        for file_path, line_num, path in absolute_paths[:10]:  # Show first 10
            print(f"  {file_path}:{line_num} - {path}")
        
        if len(absolute_paths) > 10:
            print(f"  ... and {len(absolute_paths) - 10} more")
    
    # Fix paths
    print("\nFixing absolute paths...")
    fixed_files = fix_absolute_paths(project_dir)
    
    if fixed_files:
        print(f"\n✅ Fixed {len(fixed_files)} files:")
        for file_path in sorted(fixed_files)[:10]:
            print(f"  {file_path}")
    
    # Create path configuration
    print("\nCreating path configuration...")
    config_path = create_path_config()
    print(f"✅ Created: {config_path}")
    
    print("\n" + "=" * 50)
    print("✅ Path fixing complete!")
    print("\nNext steps:")
    print("  1. Review changes: git diff")
    print("  2. Update imports to use genomevault.utils.paths")
    print("  3. Test the changes: pytest tests/")

if __name__ == "__main__":
    main()