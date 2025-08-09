#!/usr/bin/env python3
"""Add docstrings to all files missing them in the genomevault package."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from add_docstrings_smart import add_docstring_to_file


def process_directory(directory: Path, skip_tests: bool = True) -> dict:
    """Process all Python files in a directory."""
    stats = {
        'files_processed': 0,
        'docstrings_added': 0,
        'files_with_additions': 0,
        'errors': 0
    }
    
    # Get all Python files
    python_files = list(directory.rglob('*.py'))
    
    for filepath in python_files:
        # Skip certain files
        if skip_tests and 'test' in filepath.name:
            continue
        if '__pycache__' in str(filepath):
            continue
        if 'scripts/' in str(filepath):
            continue
        if '.cleanup_backups' in str(filepath):
            continue
            
        rel_path = filepath.relative_to(directory.parent)
        print(f"Processing {rel_path}...", end=' ')
        
        try:
            success, num_added = add_docstring_to_file(filepath)
            if success:
                stats['files_processed'] += 1
                if num_added > 0:
                    print(f"✓ Added {num_added} docstrings")
                    stats['docstrings_added'] += num_added
                    stats['files_with_additions'] += 1
                else:
                    print("- No changes needed")
            else:
                print("✗ Error")
                stats['errors'] += 1
        except Exception as e:
            print(f"✗ Error: {e}")
            stats['errors'] += 1
    
    return stats


def main():
    """Main function to add docstrings to all files."""
    root = Path(__file__).resolve().parents[1]
    genomevault_dir = root / 'genomevault'
    
    print("=" * 60)
    print("Adding Docstrings to GenomeVault")
    print("=" * 60)
    print()
    
    # Process main package
    stats = process_directory(genomevault_dir)
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Files with additions: {stats['files_with_additions']}")
    print(f"Total docstrings added: {stats['docstrings_added']}")
    print(f"Errors: {stats['errors']}")
    
    # Run analysis again to see improvement
    print("\n" + "=" * 60)
    print("Running analysis to check improvement...")
    print("=" * 60)
    
    import subprocess
    result = subprocess.run(
        [sys.executable, 'scripts/analyze_missing_docstrings.py'],
        capture_output=True,
        text=True
    )
    
    # Parse the output to get the new count
    for line in result.stdout.split('\n'):
        if 'Total missing docstrings:' in line:
            print(line)
        elif 'Coverage:' in line:
            print(line)
    
    return stats['docstrings_added']


if __name__ == "__main__":
    total = main()
    sys.exit(0)