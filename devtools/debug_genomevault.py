from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


#!/usr/bin/env python3
"""
Debug script for GenomeVault - identifies and helps fix common issues
"""

import subprocess
import sys
from pathlib import Path


class GenomeVaultDebugger:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.issues = []
        self.fixed = []

    def check_python_version(self):
        """Check if Python version is compatible"""
        logger.debug("🐍 Checking Python version...")
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            logger.debug(
                "  ✅ Python {version.major}.{version.minor}.{version.micro} is compatible"
            )
            return True
        else:
            self.issues.append("Python {version.major}.{version.minor} is too old. Need 3.8+")
            return False

    def check_pydantic(self):
        """Check Pydantic installation and version"""
        logger.debug("📦 Checking Pydantic...")
        try:
            import pydantic

            logger.debug("  ✅ Pydantic {version} is installed")

            # Check if pydantic-settings is installed
            try:
                import pydantic_settings

                logger.debug("  ✅ pydantic-settings is installed")
                return True
            except ImportError:
                logger.exception("Unhandled exception")
                self.issues.append("pydantic-settings is not installed")
                return False
                raise
        except ImportError:
            logger.exception("Unhandled exception")
            self.issues.append("Pydantic is not installed")
            return False
            raise

    def fix_pydantic(self):
        """Fix Pydantic issues"""
        logger.debug("🔧 Fixing Pydantic issues...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "pydantic>=2.0.0",
                ],
                check=True,
            )
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "pydantic-settings>=2.0.0"],
                check=True,
            )
            self.fixed.append("Pydantic upgraded to v2 with pydantic-settings")
            return True
        except subprocess.CalledProcessError:
            logger.exception("Unhandled exception")
            logger.error("  ❌ Failed to fix Pydantic: {e}")
            return False
            raise

    def check_imports(self):
        """Check if all GenomeVault modules can be imported"""
        logger.debug("📦 Checking GenomeVault imports...")

        modules_to_check = [
            "core.config",
            "local_processing.sequencing",
            "local_processing.transcriptomics",
            "local_processing.epigenetics",
            "hypervector_transform.encoding",
            "zk_proofs.circuits.base_circuits",
        ]

        failed_imports = []

        for module in modules_to_check:
            try:
                __import__(module)
                logger.debug("  ✅ {module}")
            except ImportError as e:
                logger.exception("Unhandled exception")
                logger.debug("  ❌ {module}: {e}")
                failed_imports.append((module, str(e)))
                raise

        if failed_imports:
            self.issues.append("Failed to import {len(failed_imports)} modules")
            return False
        return True

    def check_requirements(self):
        """Check if all requirements are installed"""
        logger.debug("📦 Checking requirements...")

        try:
            with open(self.root_dir / "requirements.txt") as f:
                requirements = f.readlines()

            missing = []
            for req in requirements:
                req = req.strip()
                if req and not req.startswith("#"):
                    # Extract package name
                    pkg_name = req.split(">=")[0].split("==")[0].strip()
                    try:
                        __import__(pkg_name.replace("-", "_"))
                    except ImportError:
                        logger.exception("Unhandled exception")
                        missing.append(pkg_name)
                        raise

            if missing:
                self.issues.append("Missing packages: {', '.join(missing)}")
                return False

            logger.debug("  ✅ All requirements satisfied")
            return True

        except FileNotFoundError:
            logger.exception("Unhandled exception")
            self.issues.append("requirements.txt not found")
            return False
            raise

    def run_tests(self):
        """Run basic tests"""
        logger.debug("🧪 Running tests...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_simple.py", "-v"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("  ✅ Tests passed!")
                return True
            else:
                logger.error("  ❌ Tests failed:")
                logger.debug(result.stdout)
                self.issues.append("Tests failed")
                return False
        except subprocess.CalledProcessError:
            logger.exception("Unhandled exception")
            self.issues.append("Failed to run tests: {e}")
            return False
            raise

    def run_diagnostics(self):
        """Run all diagnostics"""
        logger.debug("=" * 60)
        logger.debug("🔍 GenomeVault Diagnostic Tool")
        logger.debug("=" * 60)
        # Removed empty print()

        # Run checks
        self.check_python_version()
        # Removed empty print()

        pydantic_ok = self.check_pydantic()
        if not pydantic_ok:
            self.fix_pydantic()
            pydantic_ok = self.check_pydantic()
        # Removed empty print()

        self.check_requirements()
        # Removed empty print()

        self.check_imports()
        # Removed empty print()

        if not self.issues:
            self.run_tests()

        # Summary
        # Removed empty print()
        logger.debug("=" * 60)
        logger.debug("📊 Summary")
        logger.debug("=" * 60)

        if self.fixed:
            logger.debug("✅ Fixed issues:")
            for fix in self.fixed:
                logger.debug("  - {fix}")
            # Removed empty print()

        if self.issues:
            logger.debug("❌ Remaining issues:")
            for issue in self.issues:
                logger.debug("  - {issue}")
            # Removed empty print()
            logger.debug("🔧 To fix remaining issues, run:")
            logger.debug("  pip install -r requirements.txt")
            logger.debug("  ./fix_dependencies.sh")
        else:
            logger.info("✅ All checks passed! GenomeVault is ready to use.")

        return len(self.issues) == 0


def main():
    debugger = GenomeVaultDebugger()
    success = debugger.run_diagnostics()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
