#!/usr/bin/env python3
"""
Verify that Ruff is properly installed and Phase 3 is ready to run.
"""

import json
import subprocess
from pathlib import Path


def check_ruff_installation():
    """Check if Ruff is properly installed."""
    print("🔍 Checking Ruff installation...")

    # Check version
    try:
        result = subprocess.run(
            ["ruff", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ Ruff version: {version}")

            if "0.4." in version or "0.5." in version:
                print("✅ Version is adequate for Phase 3")
                return True
            else:
                print("❌ Version is too old for Phase 3")
                return False
        else:
            print(f"❌ Could not get Ruff version: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error checking Ruff: {e}")
        return False


def test_f821_detection():
    """Test F821 detection with JSON output."""
    print("\\n🧪 Testing F821 detection...")

    try:
        result = subprocess.run(
            ["ruff", "check", ".", "--select", "F821", "--output-format", "json"],
            cwd="/Users/rohanvinaik/genomevault",
            capture_output=True,
            text=True,
            timeout=30,
        )

        print(f"F821 check exit code: {result.returncode}")

        if result.stdout:
            try:
                violations = json.loads(result.stdout)
                print(f"✅ Found {len(violations)} F821 violations")

                if violations:
                    print("\\nSample violations:")
                    for i, v in enumerate(violations[:3]):
                        file_short = v["filename"].replace(
                            "/Users/rohanvinaik/genomevault/", ""
                        )
                        print(
                            f"  {i+1}. {file_short}:{v['location']['row']} - {v['message']}"
                        )

                return True
            except json.JSONDecodeError:
                print("❌ Could not parse JSON output")
                print(f"Raw output: {result.stdout[:200]}")
                return False
        else:
            print("✅ No F821 violations found")
            return True

    except Exception as e:
        print(f"❌ Error testing F821 detection: {e}")
        return False


def test_config_file():
    """Test that .ruff.toml is working."""
    print("\\n📝 Testing .ruff.toml configuration...")

    ruff_config = Path("/Users/rohanvinaik/genomevault/.ruff.toml")

    if not ruff_config.exists():
        print("❌ .ruff.toml not found")
        return False

    content = ruff_config.read_text()
    print(f"✅ .ruff.toml exists ({len(content)} chars)")

    # Test that the config doesn't cause errors
    try:
        result = subprocess.run(
            ["ruff", "check", ".", "--quiet"],
            cwd="/Users/rohanvinaik/genomevault",
            capture_output=True,
            text=True,
            timeout=10,
        )

        if "unknown field" in (result.stderr or ""):
            print("❌ Configuration has unknown fields")
            print(f"Error: {result.stderr}")
            return False
        else:
            print("✅ Configuration is valid")
            return True

    except Exception as e:
        print(f"❌ Error testing configuration: {e}")
        return False


def test_phase3_readiness():
    """Test if Phase 3 is ready to run."""
    print("\\n🎯 Testing Phase 3 readiness...")

    try:
        result = subprocess.run(
            ["python", "enhanced_cleanup.py", "--phase", "3", "--dry-run"],
            cwd="/Users/rohanvinaik/genomevault",
            capture_output=True,
            text=True,
            timeout=15,
        )

        print(f"Phase 3 dry-run exit code: {result.returncode}")

        if result.returncode == 0:
            print("✅ Phase 3 dry-run successful")
            return True
        else:
            print("❌ Phase 3 dry-run failed")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error testing Phase 3: {e}")
        return False


def main():
    print("🚀 Verifying Ruff Installation and Phase 3 Readiness\\n")

    all_good = True

    # Test 1: Ruff installation
    if not check_ruff_installation():
        all_good = False

    # Test 2: F821 detection
    if not test_f821_detection():
        all_good = False

    # Test 3: Configuration
    if not test_config_file():
        all_good = False

    # Test 4: Phase 3 readiness
    if not test_phase3_readiness():
        all_good = False

    print("\\n" + "=" * 50)
    if all_good:
        print("🎉 ALL TESTS PASSED! Phase 3 is ready to run.")
        print("\\nYou can now execute:")
        print("   python enhanced_cleanup.py --phase 3")
        print("\\nThis should process and fix your F821 undefined name errors!")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        print("\\nYou may need to:")
        print("   1. Run: python proper_ruff_upgrade.py")
        print("   2. Manually install: python -m pip install ruff>=0.4.4")
        print("   3. Check your Python environment")


if __name__ == "__main__":
    main()
