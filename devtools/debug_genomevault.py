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
        print("🐍 Checking Python version...")
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print("  ✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
            return True
        else:
            self.issues.append("Python {version.major}.{version.minor} is too old. Need 3.8+")
            return False

    def check_pydantic(self):
        """Check Pydantic installation and version"""
        print("📦 Checking Pydantic...")
        try:
            import pydantic

            print("  ✅ Pydantic {version} is installed")

            # Check if pydantic-settings is installed
            try:
                import pydantic_settings

                print("  ✅ pydantic-settings is installed")
                return True
            except ImportError:
                from genomevault.observability.logging import configure_logging

                logger = configure_logging()
                logger.exception("Unhandled exception")
                self.issues.append("pydantic-settings is not installed")
                return False
                raise
        except ImportError:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            self.issues.append("Pydantic is not installed")
            return False
            raise

    def fix_pydantic(self):
        """Fix Pydantic issues"""
        print("🔧 Fixing Pydantic issues...")
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
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            print("  ❌ Failed to fix Pydantic: {e}")
            return False
            raise

    def check_imports(self):
        """Check if all GenomeVault modules can be imported"""
        print("📦 Checking GenomeVault imports...")

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
                print("  ✅ {module}")
            except ImportError as e:
                from genomevault.observability.logging import configure_logging

                logger = configure_logging()
                logger.exception("Unhandled exception")
                print("  ❌ {module}: {e}")
                failed_imports.append((module, str(e)))
                raise

        if failed_imports:
            self.issues.append("Failed to import {len(failed_imports)} modules")
            return False
        return True

    def check_requirements(self):
        """Check if all requirements are installed"""
        print("📦 Checking requirements...")

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
                        from genomevault.observability.logging import configure_logging

                        logger = configure_logging()
                        logger.exception("Unhandled exception")
                        missing.append(pkg_name)
                        raise

            if missing:
                self.issues.append("Missing packages: {', '.join(missing)}")
                return False

            print("  ✅ All requirements satisfied")
            return True

        except FileNotFoundError:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            self.issues.append("requirements.txt not found")
            return False
            raise

    def run_tests(self):
        """Run basic tests"""
        print("🧪 Running tests...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_simple.py", "-v"],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                print("  ✅ Tests passed!")
                return True
            else:
                print("  ❌ Tests failed:")
                print(result.stdout)
                self.issues.append("Tests failed")
                return False
        except subprocess.CalledProcessError:
            from genomevault.observability.logging import configure_logging

            logger = configure_logging()
            logger.exception("Unhandled exception")
            self.issues.append("Failed to run tests: {e}")
            return False
            raise

    def run_diagnostics(self):
        """Run all diagnostics"""
        print("=" * 60)
        print("🔍 GenomeVault Diagnostic Tool")
        print("=" * 60)
        print()

        # Run checks
        self.check_python_version()
        print()

        pydantic_ok = self.check_pydantic()
        if not pydantic_ok:
            self.fix_pydantic()
            pydantic_ok = self.check_pydantic()
        print()

        self.check_requirements()
        print()

        self.check_imports()
        print()

        if not self.issues:
            self.run_tests()

        # Summary
        print()
        print("=" * 60)
        print("📊 Summary")
        print("=" * 60)

        if self.fixed:
            print("✅ Fixed issues:")
            for fix in self.fixed:
                print("  - {fix}")
            print()

        if self.issues:
            print("❌ Remaining issues:")
            for issue in self.issues:
                print("  - {issue}")
            print()
            print("🔧 To fix remaining issues, run:")
            print("  pip install -r requirements.txt")
            print("  ./fix_dependencies.sh")
        else:
            print("✅ All checks passed! GenomeVault is ready to use.")

        return len(self.issues) == 0


def main():
    debugger = GenomeVaultDebugger()
    success = debugger.run_diagnostics()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
