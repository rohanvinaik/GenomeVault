#!/usr/bin/env python3
"""
GenomeVault Academic Paper Generation Pipeline v2.0

Orchestrates the complete workflow for GenomeVault v2.0.0 with differential encoding:
1. Clean old results (if --clean)
2. Run benchmark experiments (differential encoding primary, HDC/PIR secondary)
3. Generate figures reflecting v2.0 architecture
4. Update paper with results
5. Generate PDF

Usage:
    python scripts/run_full_paper_pipeline.py
    python scripts/run_full_paper_pipeline.py --quick --clean
    python scripts/run_full_paper_pipeline.py --skip-benchmarks
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Paths
ROOT = Path(__file__).parent.parent
SCRIPTS = ROOT / "scripts"
RESULTS = ROOT / "benchmark_results"
PAPER = ROOT / "docs" / "GenomeVault_Academic_Paper.md"
FIGURES = ROOT / "docs" / "paper_figures"
PDF = ROOT / "docs" / "paper_submission" / "GenomeVault_Manuscript.pdf"


def clean_old_results(dry_run=False):
    """Clean old benchmark results and generated files."""
    logger.info("Cleaning old results...")

    items_to_clean = [
        RESULTS / "differential_encoding",
        FIGURES,
        PDF.parent / "*.pdf",
    ]

    for item in items_to_clean:
        if isinstance(item, Path) and item.exists():
            if dry_run:
                logger.info(f"  Would remove: {item}")
            else:
                if item.is_dir():
                    shutil.rmtree(item)
                    logger.info(f"  Removed directory: {item}")
                else:
                    item.unlink()
                    logger.info(f"  Removed file: {item}")

    # Recreate directories
    if not dry_run:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "differential_encoding").mkdir(parents=True, exist_ok=True)
        FIGURES.mkdir(parents=True, exist_ok=True)
        PDF.parent.mkdir(parents=True, exist_ok=True)


def run_benchmarks(quick=False):
    """Run benchmark experiments to generate result data."""
    logger.info("Running benchmark experiments...")

    # Primary benchmark: Differential Encoding (v2.0 core feature)
    logger.info("  [1/3] Running differential encoding benchmarks...")
    diff_encoding_script = SCRIPTS / "run_differential_encoding_benchmarks.py"
    if diff_encoding_script.exists():
        try:
            cmd = [sys.executable, str(diff_encoding_script)]
            if quick:
                cmd.append("--quick")
            subprocess.run(cmd, check=True, cwd=ROOT)
            logger.info("  ✓ Differential encoding benchmarks complete")
        except subprocess.CalledProcessError as e:
            logger.error(f"  ✗ Differential encoding benchmarks failed: {e}")
            return False
    else:
        logger.error(f"  ✗ Differential encoding benchmark script not found: {diff_encoding_script}")
        return False

    # Secondary benchmarks: HDC and PIR performance
    logger.info("  [2/3] Running HDC benchmarks...")
    hdc_script = SCRIPTS / "bench_hdc.py"
    if hdc_script.exists():
        try:
            subprocess.run([sys.executable, str(hdc_script)], check=True, cwd=ROOT)
            logger.info("  ✓ HDC benchmarks complete")
        except subprocess.CalledProcessError as e:
            logger.warning(f"  ⚠ HDC benchmarks failed: {e}")
    else:
        logger.warning(f"  ⚠ HDC benchmark script not found: {hdc_script}")

    logger.info("  [3/3] Running PIR benchmarks...")
    pir_script = SCRIPTS / "bench_pir.py"
    if pir_script.exists():
        try:
            subprocess.run([sys.executable, str(pir_script)], check=True, cwd=ROOT)
            logger.info("  ✓ PIR benchmarks complete")
        except subprocess.CalledProcessError as e:
            logger.warning(f"  ⚠ PIR benchmarks failed: {e}")
    else:
        logger.warning(f"  ⚠ PIR benchmark script not found: {pir_script}")

    return True


def generate_figures():
    """Generate all paper figures."""
    logger.info("Generating figures...")

    script = SCRIPTS / "generate_paper_figures_v2.py"
    if not script.exists():
        logger.error(f"Figure generation script not found: {script}")
        return False

    try:
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)
        logger.info("  Figures generated successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Figure generation failed: {e}")
        return False


def update_paper():
    """Update paper with experimental results."""
    logger.info("Updating paper with results...")

    script = SCRIPTS / "update_paper_with_results.py"
    if not script.exists():
        logger.error(f"Paper update script not found: {script}")
        return False

    try:
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)
        logger.info("  Paper updated successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Paper update failed: {e}")
        return False


def generate_pdf():
    """Generate PDF from markdown paper."""
    logger.info("Generating PDF...")

    script = SCRIPTS / "generate_paper_pdf.py"
    if not script.exists():
        logger.error(f"PDF generation script not found: {script}")
        return False

    try:
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)
        logger.info(f"  PDF generated: {PDF}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"PDF generation failed: {e}")
        return False


def create_manifest():
    """Create manifest of generated files."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "genomevault_version": "2.0.0",
        "architecture": "differential_encoding_core",
        "paper": str(PAPER.relative_to(ROOT)) if PAPER.exists() else None,
        "pdf": str(PDF.relative_to(ROOT)) if PDF.exists() else None,
        "figures": [str(f.relative_to(ROOT)) for f in FIGURES.glob("*.png")] if FIGURES.exists() else [],
        "results": {
            "differential_encoding": str((RESULTS / "differential_encoding" / "latest_results.json").relative_to(ROOT))
            if (RESULTS / "differential_encoding" / "latest_results.json").exists() else None,
            "hdc": [str(f.relative_to(ROOT)) for f in (RESULTS / "hdc").glob("*.json")]
            if (RESULTS / "hdc").exists() else [],
            "pir": [str(f.relative_to(ROOT)) for f in (RESULTS / "pir").glob("*.json")]
            if (RESULTS / "pir").exists() else [],
        }
    }

    manifest_path = ROOT / "docs" / "paper_submission" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest saved: {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="GenomeVault v2.0 Academic Paper Generation Pipeline"
    )
    parser.add_argument(
        '--clean',
        action='store_true',
        help='Clean old results before running'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: reduced iterations for faster benchmarks'
    )
    parser.add_argument(
        '--skip-benchmarks',
        action='store_true',
        help='Skip benchmark execution (use existing results)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without executing'
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("GenomeVault v2.0.0 Academic Paper Generation Pipeline")
    logger.info("Architecture: Differential Encoding Core")
    logger.info("="*70)
    logger.info("")

    # Step 1: Clean old results
    if args.clean:
        clean_old_results(dry_run=args.dry_run)
        logger.info("")

    if args.dry_run:
        logger.info("Dry run complete. Exiting.")
        return 0

    # Step 2: Run benchmarks (unless skipped)
    if not args.skip_benchmarks:
        if not run_benchmarks(quick=args.quick):
            logger.error("Benchmark execution failed")
            return 1
        logger.info("")
    else:
        logger.info("Skipping benchmarks (using existing results)")
        logger.info("")

    # Step 3: Generate figures
    if not generate_figures():
        logger.error("Figure generation failed")
        return 1
    logger.info("")

    # Step 4: Update paper with results
    if not update_paper():
        logger.warning("Paper update had issues (may need manual review)")
    logger.info("")

    # Step 5: Generate PDF
    if not generate_pdf():
        logger.warning("PDF generation failed (check pandoc installation)")
    logger.info("")

    # Step 6: Create manifest
    manifest = create_manifest()
    logger.info("")

    # Summary
    logger.info("="*70)
    logger.info("Pipeline Complete!")
    logger.info("="*70)
    logger.info("")
    logger.info("Generated files:")
    logger.info(f"  Paper: {manifest.get('paper', 'Not found')}")
    logger.info(f"  PDF: {manifest.get('pdf', 'Not found')}")
    logger.info(f"  Figures: {len(manifest.get('figures', []))} files")
    logger.info(f"  Primary Results: {manifest['results'].get('differential_encoding', 'Not found')}")
    logger.info("")
    logger.info(f"Manifest: docs/paper_submission/manifest.json")
    logger.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
