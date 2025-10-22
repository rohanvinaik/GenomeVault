#!/usr/bin/env python3
"""
Automated Migration Script for Hardware Backend System

Migrates Python files from legacy HypervectorEncoder to the new
BackendOptimizedEncoder system.

Usage:
    # Migrate specific file
    python scripts/migrate_to_backend_system.py benchmarks/encoding_comparison_benchmark.py

    # Migrate entire directory
    python scripts/migrate_to_backend_system.py benchmarks/ --recursive

    # Dry run (preview changes)
    python scripts/migrate_to_backend_system.py benchmarks/ --dry-run

    # Migrate all benchmark scripts
    python scripts/migrate_to_backend_system.py benchmarks/ --recursive --filter="*.py"
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple
import shutil


class BackendMigrator:
    """Migrates Python files to use the new backend system"""

    def __init__(self, dry_run: bool = False, backup: bool = True):
        self.dry_run = dry_run
        self.backup = backup
        self.files_processed = 0
        self.files_modified = 0

    def migrate_file(self, file_path: Path) -> Tuple[bool, List[str]]:
        """
        Migrate a single Python file

        Returns:
            (modified, changes) - Whether file was modified and list of changes made
        """
        try:
            with open(file_path, 'r') as f:
                original_content = f.read()
        except Exception as e:
            print(f"✗ Error reading {file_path}: {e}")
            return False, []

        content = original_content
        changes = []

        # Pattern 1: Update imports
        if 'from genomevault.hypervector_transform import HypervectorEncoder' in content:
            content = content.replace(
                'from genomevault.hypervector_transform import HypervectorEncoder',
                'from genomevault.hypervector_transform import create_backend_encoder'
            )
            changes.append("Updated import to use create_backend_encoder")

        if 'from genomevault.hypervector_transform import HypervectorConfig' in content:
            # Keep HypervectorConfig import for backward compatibility, but add backend import
            if 'create_backend_encoder' not in content:
                content = content.replace(
                    'from genomevault.hypervector_transform import HypervectorConfig',
                    'from genomevault.hypervector_transform import HypervectorConfig, create_backend_encoder'
                )
                changes.append("Added create_backend_encoder to imports")

        # Pattern 2: Replace HypervectorEncoder initialization
        # Match: encoder = HypervectorEncoder(config)
        encoder_init_pattern = r'(\w+)\s*=\s*HypervectorEncoder\(([^)]*)\)'
        matches = list(re.finditer(encoder_init_pattern, content))
        for match in reversed(matches):  # Reverse to maintain indices
            var_name = match.group(1)
            config_arg = match.group(2).strip()

            # Extract dimension if in config
            dim_match = re.search(r'dimension\s*=\s*(\d+)', config_arg)
            dimension = dim_match.group(1) if dim_match else '8192'

            # Replace with new backend encoder
            new_init = f'{var_name} = create_backend_encoder(dimension={dimension})'
            content = content[:match.start()] + new_init + content[match.end():]
            changes.append(f"Migrated {var_name} initialization to use create_backend_encoder")

        # Pattern 3: Update encode() method calls to encode_single()
        # Match: encoder.encode(data, OmicsType.GENOMIC)
        encode_pattern = r'\.encode\(([^,)]+),\s*OmicsType\.\w+\)'
        matches = list(re.finditer(encode_pattern, content))
        for match in reversed(matches):
            data_arg = match.group(1).strip()
            new_call = f'.encode_single({data_arg})'
            content = content[:match.start()] + new_call + content[match.end():]
            changes.append("Updated encode() to encode_single()")

        # Pattern 4: Remove OmicsType import if no longer needed
        if 'from genomevault.core.constants import OmicsType' in content:
            # Check if OmicsType is still used elsewhere
            if content.count('OmicsType') == 1:  # Only in import
                content = re.sub(
                    r'from genomevault\.core\.constants import OmicsType\n?',
                    '',
                    content
                )
                changes.append("Removed unused OmicsType import")

        # Pattern 5: Add backend configuration if using config
        if 'HypervectorConfig(' in content and 'backend=' not in content:
            # Add a comment about backend configuration
            config_pattern = r'(config\s*=\s*HypervectorConfig\([^)]*\))'
            match = re.search(config_pattern, content)
            if match:
                comment = "\n    # Note: Use create_backend_encoder(backend='auto') to leverage hardware acceleration"
                insert_pos = match.end()
                content = content[:insert_pos] + comment + content[insert_pos:]
                changes.append("Added note about backend configuration")

        # Pattern 6: Update batch processing
        # Match: [encoder.encode(x, OmicsType.GENOMIC) for x in batch]
        batch_pattern = r'\[([^\.]+)\.encode\((\w+),\s*OmicsType\.\w+\)\s+for\s+\2\s+in\s+([^\]]+)\]'
        match = re.search(batch_pattern, content)
        if match:
            encoder_var = match.group(1)
            item_var = match.group(2)
            iterable = match.group(3).strip()
            new_batch = f'{encoder_var}.encode_batch({iterable})'
            content = content[:match.start()] + new_batch + content[match.end():]
            changes.append("Optimized batch encoding to use encode_batch()")

        # Check if anything changed
        if content == original_content:
            return False, []

        # Create backup if requested
        if self.backup and not self.dry_run:
            backup_path = file_path.with_suffix(file_path.suffix + '.backup')
            shutil.copy2(file_path, backup_path)
            changes.append(f"Created backup: {backup_path.name}")

        # Write changes
        if not self.dry_run:
            try:
                with open(file_path, 'w') as f:
                    f.write(content)
            except Exception as e:
                print(f"✗ Error writing {file_path}: {e}")
                return False, changes

        return True, changes

    def migrate_directory(self, directory: Path, recursive: bool = False, pattern: str = "*.py"):
        """Migrate all Python files in a directory"""
        glob_pattern = f"**/{pattern}" if recursive else pattern
        python_files = list(directory.glob(glob_pattern))

        print(f"\nFound {len(python_files)} Python files to process")
        print("=" * 70)

        for file_path in python_files:
            self.files_processed += 1
            modified, changes = self.migrate_file(file_path)

            if modified:
                self.files_modified += 1
                status = "🔄 [DRY RUN]" if self.dry_run else "✓"
                print(f"\n{status} {file_path}")
                for change in changes:
                    print(f"  - {change}")
            else:
                print(f"  ⊙ {file_path} (no changes needed)")

    def print_summary(self):
        """Print migration summary"""
        print("\n" + "=" * 70)
        print("Migration Summary:")
        print(f"  Files processed: {self.files_processed}")
        print(f"  Files modified:  {self.files_modified}")
        if self.dry_run:
            print("\n  ℹ️  This was a dry run. No files were actually modified.")
            print("     Run without --dry-run to apply changes.")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Migrate Python files to use hardware backend system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Migrate single file
  %(prog)s benchmarks/encoding_comparison_benchmark.py

  # Migrate directory
  %(prog)s benchmarks/ --recursive

  # Dry run to preview changes
  %(prog)s benchmarks/ --dry-run --recursive

  # Migrate without creating backups
  %(prog)s benchmarks/ --no-backup
        """
    )

    parser.add_argument(
        'path',
        type=Path,
        help='File or directory to migrate'
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Recursively process directories'
    )
    parser.add_argument(
        '-d', '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create .backup files'
    )
    parser.add_argument(
        '--filter',
        default='*.py',
        help='File pattern to match (default: *.py)'
    )

    args = parser.parse_args()

    # Validate path
    if not args.path.exists():
        print(f"✗ Error: Path does not exist: {args.path}")
        sys.exit(1)

    # Create migrator
    migrator = BackendMigrator(
        dry_run=args.dry_run,
        backup=not args.no_backup
    )

    print("=" * 70)
    print("Hardware Backend Migration Tool")
    print("=" * 70)
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")

    # Process file or directory
    if args.path.is_file():
        modified, changes = migrator.migrate_file(args.path)
        migrator.files_processed = 1
        if modified:
            migrator.files_modified = 1
            status = "🔄 [DRY RUN]" if args.dry_run else "✓"
            print(f"\n{status} {args.path}")
            for change in changes:
                print(f"  - {change}")
        else:
            print(f"\n⊙ {args.path} (no changes needed)")
    else:
        migrator.migrate_directory(
            args.path,
            recursive=args.recursive,
            pattern=args.filter
        )

    # Print summary
    migrator.print_summary()

    return 0 if migrator.files_modified > 0 or args.dry_run else 1


if __name__ == '__main__':
    sys.exit(main())
