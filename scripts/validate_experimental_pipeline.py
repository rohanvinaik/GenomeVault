#!/usr/bin/env python3
"""
Validate GenomeVault v2.0 Experimental Pipeline Setup

This script validates that all components of the experimental pipeline
are properly configured and can execute successfully.

Usage:
    python scripts/validate_experimental_pipeline.py
    python scripts/validate_experimental_pipeline.py --verbose
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Paths
ROOT = Path(__file__).parent.parent
SCRIPTS = ROOT / "scripts"
RESULTS_DIR = ROOT / "benchmark_results"
FIGURES_DIR = ROOT / "docs" / "paper_figures"
REPORTS_DIR = ROOT / "docs" / "experimental_reports"


class PipelineValidator:
    """Validate experimental pipeline setup and execution"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = 0

    def check_directory(self, path: Path, create: bool = False) -> bool:
        """Check if directory exists or create it"""
        if path.exists():
            logger.info(f"  ✓ Directory exists: {path.relative_to(ROOT)}")
            return True
        elif create:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✓ Created directory: {path.relative_to(ROOT)}")
            return True
        else:
            logger.error(f"  ✗ Directory missing: {path.relative_to(ROOT)}")
            return False

    def check_script(self, script_path: Path) -> bool:
        """Check if script exists and is executable"""
        if not script_path.exists():
            logger.error(f"  ✗ Script missing: {script_path.relative_to(ROOT)}")
            return False

        # Try to get help text
        try:
            result = subprocess.run(
                [sys.executable, str(script_path), "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                logger.info(f"  ✓ Script valid: {script_path.name}")
                return True
            else:
                logger.warning(f"  ⚠ Script has issues: {script_path.name}")
                if self.verbose:
                    logger.warning(f"    Error: {result.stderr[:200]}")
                return False
        except Exception as e:
            logger.error(f"  ✗ Script error: {script_path.name} - {e}")
            return False

    def check_python_imports(self) -> bool:
        """Check required Python packages"""
        logger.info("\nChecking Python dependencies...")

        required = [
            "numpy",
            "matplotlib",
            "seaborn",
            "pandas",
            "json",
            "pathlib",
            "subprocess"
        ]

        all_present = True
        for package in required:
            try:
                __import__(package)
                logger.info(f"  ✓ {package}")
            except ImportError:
                logger.error(f"  ✗ {package} not installed")
                all_present = False

        return all_present

    def test_quick_benchmark(self) -> bool:
        """Run a quick benchmark to test execution"""
        logger.info("\nTesting quick benchmark execution...")

        script = SCRIPTS / "run_differential_encoding_benchmarks.py"
        if not script.exists():
            logger.error(f"  ✗ Benchmark script not found: {script}")
            return False

        try:
            logger.info("  Running quick differential encoding benchmark...")
            result = subprocess.run(
                [sys.executable, str(script), "--quick"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            if result.returncode == 0:
                logger.info("  ✓ Quick benchmark completed successfully")

                # Check if results were created
                results_file = RESULTS_DIR / "differential_encoding" / "latest_results.json"
                if results_file.exists():
                    logger.info(f"  ✓ Results file created: {results_file.relative_to(ROOT)}")
                    return True
                else:
                    logger.warning("  ⚠ Benchmark ran but no results file found")
                    return False
            else:
                logger.error(f"  ✗ Benchmark failed with code {result.returncode}")
                if self.verbose:
                    logger.error(f"    Stderr: {result.stderr[-500:]}")
                return False

        except subprocess.TimeoutExpired:
            logger.error("  ✗ Benchmark timed out (>2 minutes)")
            return False
        except Exception as e:
            logger.error(f"  ✗ Benchmark error: {e}")
            return False

    def test_figure_generation(self) -> bool:
        """Test figure generation"""
        logger.info("\nTesting figure generation...")

        script = SCRIPTS / "generate_paper_figures_v2.py"
        if not script.exists():
            logger.error(f"  ✗ Figure script not found: {script}")
            return False

        try:
            logger.info("  Generating test figures...")
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                logger.info("  ✓ Figure generation completed")

                # Check if figures were created
                png_files = list(FIGURES_DIR.glob("*.png"))
                if png_files:
                    logger.info(f"  ✓ Generated {len(png_files)} PNG figures")
                    return True
                else:
                    logger.warning("  ⚠ No PNG figures found")
                    return False
            else:
                logger.error(f"  ✗ Figure generation failed")
                if self.verbose:
                    logger.error(f"    Error: {result.stderr[-500:]}")
                return False

        except Exception as e:
            logger.error(f"  ✗ Figure generation error: {e}")
            return False

    def test_report_generation(self) -> bool:
        """Test report generation"""
        logger.info("\nTesting report generation...")

        script = SCRIPTS / "generate_experimental_report.py"
        if not script.exists():
            logger.error(f"  ✗ Report script not found: {script}")
            return False

        try:
            logger.info("  Generating test report...")
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                logger.info("  ✓ Report generation completed")

                # Check if reports were created
                md_file = REPORTS_DIR / "latest_experimental_report.md"
                if md_file.exists():
                    logger.info(f"  ✓ Markdown report created")
                    return True
                else:
                    logger.warning("  ⚠ No report file found")
                    return False
            else:
                logger.error(f"  ✗ Report generation failed")
                if self.verbose:
                    logger.error(f"    Error: {result.stderr[-500:]}")
                return False

        except Exception as e:
            logger.error(f"  ✗ Report generation error: {e}")
            return False

    def run_validation(self) -> int:
        """Run complete validation suite"""
        logger.info("="*70)
        logger.info("GenomeVault v2.0 Experimental Pipeline Validation")
        logger.info("="*70)

        checks: List[Tuple[str, callable]] = [
            ("Directory Structure", self.validate_directories),
            ("Script Files", self.validate_scripts),
            ("Python Dependencies", self.check_python_imports),
            ("Quick Benchmark", self.test_quick_benchmark),
            ("Figure Generation", self.test_figure_generation),
            ("Report Generation", self.test_report_generation),
        ]

        for check_name, check_func in checks:
            logger.info(f"\n{'='*70}")
            logger.info(f"CHECK: {check_name}")
            logger.info('='*70)

            try:
                if check_func():
                    self.checks_passed += 1
                    logger.info(f"\n✓ {check_name} PASSED")
                else:
                    self.checks_failed += 1
                    logger.error(f"\n✗ {check_name} FAILED")
            except Exception as e:
                self.checks_failed += 1
                logger.error(f"\n✗ {check_name} CRASHED: {e}")

        # Final summary
        logger.info("\n" + "="*70)
        logger.info("VALIDATION SUMMARY")
        logger.info("="*70)
        logger.info(f"Passed: {self.checks_passed}/{len(checks)}")
        logger.info(f"Failed: {self.checks_failed}/{len(checks)}")
        if self.warnings > 0:
            logger.info(f"Warnings: {self.warnings}")

        if self.checks_failed == 0:
            logger.info("\n✓ All validation checks passed!")
            logger.info("The experimental pipeline is properly configured.")
            return 0
        else:
            logger.error(f"\n✗ {self.checks_failed} validation check(s) failed.")
            logger.error("Please fix the issues before running the full pipeline.")
            return 1

    def validate_directories(self) -> bool:
        """Validate directory structure"""
        logger.info("\nValidating directory structure...")

        dirs_to_check = [
            (ROOT / "scripts", False),
            (ROOT / "benchmarks" / "differential_encoding", False),
            (RESULTS_DIR, True),
            (RESULTS_DIR / "differential_encoding", True),
            (FIGURES_DIR, True),
            (REPORTS_DIR, True),
        ]

        all_ok = True
        for dir_path, create in dirs_to_check:
            if not self.check_directory(dir_path, create=create):
                all_ok = False

        return all_ok

    def validate_scripts(self) -> bool:
        """Validate script files"""
        logger.info("\nValidating script files...")

        scripts_to_check = [
            "run_full_paper_pipeline.py",
            "run_differential_encoding_benchmarks.py",
            "generate_paper_figures_v2.py",
            "generate_experimental_report.py",
        ]

        all_ok = True
        for script_name in scripts_to_check:
            if not self.check_script(SCRIPTS / script_name):
                all_ok = False

        return all_ok


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Validate GenomeVault experimental pipeline setup"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed error messages'
    )
    parser.add_argument(
        '--skip-tests',
        action='store_true',
        help='Skip actual execution tests (only check files)'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    validator = PipelineValidator(verbose=args.verbose)

    if args.skip_tests:
        logger.info("Skipping execution tests (--skip-tests)")
        # Only run structure checks
        success = (
            validator.validate_directories() and
            validator.validate_scripts() and
            validator.check_python_imports()
        )
        return 0 if success else 1
    else:
        return validator.run_validation()


if __name__ == "__main__":
    sys.exit(main())
