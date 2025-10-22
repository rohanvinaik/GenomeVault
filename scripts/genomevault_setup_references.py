#!/usr/bin/env python3
"""
GenomeVault Reference Genome Setup Tool

Interactive wizard for downloading, validating, and configuring reference
genome pools for differential encoding.

Usage:
    python scripts/genomevault_setup_references.py
    python scripts/genomevault_setup_references.py --use-case development
    python scripts/genomevault_setup_references.py --custom ref1 ref2 ref3
    python scripts/genomevault_setup_references.py --validate
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genomevault.differential_encoding import (
    STANDARD_REFERENCES,
    RECOMMENDED_POOLS,
    download_reference_genomes,
    validate_reference_pool,
    setup_default_references,
    get_reference_info,
    SecureReferenceGenomeManager,
)
from genomevault.differential_encoding.reference_setup import (
    print_available_references,
    print_recommended_pools,
    print_validation_results,
)


def progress_callback(name: str, current: int, total: int) -> None:
    """Display progress bar for downloads."""
    if total > 0:
        percent = (current / total) * 100
        bar_length = 40
        filled = int(bar_length * current / total)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r  {name}: [{bar}] {percent:.1f}%", end="", flush=True)
        if current == total:
            print()  # New line when complete


def interactive_setup() -> None:
    """Run interactive setup wizard."""
    print("\n" + "=" * 80)
    print("GENOMEVAULT REFERENCE GENOME SETUP WIZARD")
    print("=" * 80)
    print()
    print("This wizard will help you set up reference genomes for differential encoding.")
    print()

    # Step 1: Choose reference directory
    print("Step 1: Reference Directory")
    print("-" * 80)
    default_dir = Path.home() / ".genomevault" / "references"
    ref_dir_input = input(f"Enter reference directory (default: {default_dir}): ").strip()

    if ref_dir_input:
        ref_dir = Path(ref_dir_input).expanduser().resolve()
    else:
        ref_dir = default_dir

    print(f"Using directory: {ref_dir}")
    print()

    # Step 2: Choose use case or custom
    print("Step 2: Reference Selection")
    print("-" * 80)
    print("Choose a use case or select custom references:")
    print()
    print("  1. Development  - Synthetic test data (fastest)")
    print("  2. Research     - 1000 Genomes data")
    print("  3. Clinical     - gnomAD + 1000 Genomes")
    print("  4. Production   - gnomAD v4")
    print("  5. Custom       - Select specific references")
    print()

    choice = input("Enter choice (1-5): ").strip()

    use_case_map = {
        "1": "development",
        "2": "research",
        "3": "clinical",
        "4": "production",
    }

    if choice in use_case_map:
        use_case = use_case_map[choice]
        sources = RECOMMENDED_POOLS[use_case]
        print(f"\nSelected: {use_case}")
        print(f"References: {', '.join(sources)}")
    elif choice == "5":
        print("\nAvailable references:")
        for i, (name, source) in enumerate(STANDARD_REFERENCES.items(), 1):
            print(f"  {i}. {name} - {source.description}")
        print()
        selected = input("Enter reference numbers (comma-separated, e.g., 1,3): ").strip()
        indices = [int(x.strip()) - 1 for x in selected.split(",")]
        all_refs = list(STANDARD_REFERENCES.keys())
        sources = [all_refs[i] for i in indices if 0 <= i < len(all_refs)]
        print(f"\nSelected: {', '.join(sources)}")
    else:
        print("Invalid choice. Exiting.")
        return

    print()

    # Step 3: Confirm and download
    print("Step 3: Download and Setup")
    print("-" * 80)

    total_size = sum(
        STANDARD_REFERENCES[s].size_mb
        for s in sources
        if s in STANDARD_REFERENCES
    )
    print(f"Total download size: ~{total_size:.1f} MB")
    print()

    confirm = input("Proceed with download? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Setup cancelled.")
        return

    print("\nDownloading references...")
    print()

    try:
        references = download_reference_genomes(
            sources,
            ref_dir,
            progress_callback=progress_callback,
        )

        print()
        print(f"✅ Downloaded {len(references)} reference genomes")
        print()

        # Step 4: Validation
        print("Step 4: Validation")
        print("-" * 80)

        manager = SecureReferenceGenomeManager(reference_dir=ref_dir)
        for ref_genome in references.values():
            manager.pool.add_reference(ref_genome)

        result = validate_reference_pool(manager)
        print_validation_results(result)

        # Summary
        if result.is_valid:
            print("\n" + "=" * 80)
            print("✅ SETUP COMPLETE")
            print("=" * 80)
            print()
            print(f"Reference directory: {ref_dir}")
            print(f"References installed: {len(references)}")
            print()
            print("You can now use differential encoding with these references:")
            print()
            print("  from genomevault.hypervector_transform import UnifiedGenomicEncoder")
            print("  from genomevault.differential_encoding import AnalysisType")
            print()
            print(f"  encoder = UnifiedGenomicEncoder(")
            print(f"      mode=EncodingMode.DIFFERENTIAL,")
            print(f"      reference_dir=Path('{ref_dir}'),")
            print(f"  )")
            print()
        else:
            print("\n" + "=" * 80)
            print("⚠️  SETUP COMPLETED WITH ERRORS")
            print("=" * 80)
            print()
            print(f"Some references failed validation. See errors above.")
            print(f"Reference directory: {ref_dir}")
            print()

    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()


def validate_references(ref_dir: Path) -> None:
    """Validate existing reference pool."""
    print("\n" + "=" * 80)
    print("REFERENCE POOL VALIDATION")
    print("=" * 80)
    print()
    print(f"Validating references in: {ref_dir}")
    print()

    try:
        manager = SecureReferenceGenomeManager(reference_dir=ref_dir)

        if manager.reference_count == 0:
            print("❌ No references found")
            print()
            print("Run setup wizard to download references:")
            print(f"  python {Path(__file__).name}")
            print()
            return

        result = validate_reference_pool(manager)
        print_validation_results(result)

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()


def list_references(ref_dir: Path) -> None:
    """List installed references."""
    print("\n" + "=" * 80)
    print("INSTALLED REFERENCES")
    print("=" * 80)
    print()
    print(f"Reference directory: {ref_dir}")
    print()

    try:
        info = get_reference_info(ref_dir)

        if info["reference_count"] == 0:
            print("No references installed.")
            print()
            print("Run setup wizard to download references:")
            print(f"  python {Path(__file__).name}")
            print()
            return

        print(f"Total references: {info['reference_count']}")
        print()

        for ref_id, ref_info in info["references"].items():
            print(f"📚 {ref_id}")
            print(f"   Assembly: {ref_info['assembly']}")
            print(f"   Variants: {ref_info['variant_count']:,}")
            print(f"   Chromosomes: {', '.join(ref_info['chromosomes'])}")
            print(f"   Hash: {ref_info['hash']}")
            print()

    except Exception as e:
        print(f"❌ Failed to list references: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GenomeVault Reference Genome Setup Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run interactive wizard
  python scripts/genomevault_setup_references.py

  # Quick setup for development
  python scripts/genomevault_setup_references.py --use-case development

  # Custom references
  python scripts/genomevault_setup_references.py --custom synthetic_test

  # Validate existing references
  python scripts/genomevault_setup_references.py --validate

  # List installed references
  python scripts/genomevault_setup_references.py --list
        """
    )

    parser.add_argument(
        "--ref-dir",
        type=Path,
        default=Path.home() / ".genomevault" / "references",
        help="Reference directory (default: ~/.genomevault/references)",
    )

    parser.add_argument(
        "--use-case",
        choices=list(RECOMMENDED_POOLS.keys()),
        help="Setup recommended references for use case",
    )

    parser.add_argument(
        "--custom",
        nargs="+",
        choices=list(STANDARD_REFERENCES.keys()),
        help="Download specific references",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate existing references",
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List installed references",
    )

    parser.add_argument(
        "--show-available",
        action="store_true",
        help="Show available reference sources",
    )

    parser.add_argument(
        "--show-pools",
        action="store_true",
        help="Show recommended reference pools",
    )

    args = parser.parse_args()

    # Expand path
    ref_dir = args.ref_dir.expanduser().resolve()

    # Show available references
    if args.show_available:
        print_available_references()
        return

    # Show recommended pools
    if args.show_pools:
        print_recommended_pools()
        return

    # Validate existing
    if args.validate:
        validate_references(ref_dir)
        return

    # List installed
    if args.list:
        list_references(ref_dir)
        return

    # Quick setup with use case
    if args.use_case:
        print(f"\nSetting up {args.use_case} reference pool...")
        print(f"Reference directory: {ref_dir}")
        print()

        try:
            manager = setup_default_references(
                ref_dir,
                use_case=args.use_case,
                progress_callback=progress_callback,
            )

            result = validate_reference_pool(manager)
            print_validation_results(result)

            if result.is_valid:
                print("\n✅ Setup complete!")
            else:
                print("\n⚠️  Setup completed with errors")

        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        return

    # Custom references
    if args.custom:
        print(f"\nDownloading custom references: {', '.join(args.custom)}")
        print(f"Reference directory: {ref_dir}")
        print()

        try:
            references = download_reference_genomes(
                args.custom,
                ref_dir,
                progress_callback=progress_callback,
            )

            print()
            print(f"✅ Downloaded {len(references)} references")

            manager = SecureReferenceGenomeManager(reference_dir=ref_dir)
            for ref_genome in references.values():
                manager.pool.add_reference(ref_genome)

            result = validate_reference_pool(manager)
            print_validation_results(result)

        except Exception as e:
            print(f"\n❌ Download failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        return

    # Run interactive wizard
    interactive_setup()


if __name__ == "__main__":
    main()
