#!/usr/bin/env python3
"""
Test the Ruff upgrade functionality in Phase 3.
"""

import subprocess


def test_current_ruff():
    """Test current Ruff version and F821 detection."""
    print("🔍 Testing current Ruff setup...")

    try:
        result = subprocess.run(
            ["ruff", "--version"], capture_output=True, text=True, timeout=10
        )

        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"Current Ruff version: {version}")

            # Check if it's adequate (0.4.0+)
            if "0.4" in version or "0.5" in version or "1." in version:
                print("✅ Ruff version is adequate for F821 fixing")
                return True
            else:
                print("❌ Ruff version is too old - needs upgrade")
                return False
        else:
            print(f"❌ Could not get Ruff version: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error checking Ruff: {e}")
        return False


def test_upgrade_process():
    """Test the upgrade process (dry run)."""
    print("\\n🔧 Testing Ruff upgrade process...")

    try:
        # Test which package managers are available
        managers = []

        # Test pip
        try:
            result = subprocess.run(
                ["pip", "--version"], capture_output=True, timeout=5
            )
            if result.returncode == 0:
                managers.append("pip")
        except:
            pass

        # Test pip3
        try:
            result = subprocess.run(
                ["pip3", "--version"], capture_output=True, timeout=5
            )
            if result.returncode == 0:
                managers.append("pip3")
        except:
            pass

        # Test conda
        try:
            result = subprocess.run(
                ["conda", "--version"], capture_output=True, timeout=5
            )
            if result.returncode == 0:
                managers.append("conda")
        except:
            pass

        print(f"Available package managers: {', '.join(managers)}")

        if managers:
            print("✅ At least one package manager available for upgrade")
            return True
        else:
            print("❌ No package managers available")
            return False

    except Exception as e:
        print(f"❌ Error testing upgrade process: {e}")
        return False


def run_enhanced_phase3():
    """Run the enhanced Phase 3 with Ruff upgrade."""
    print("\\n🚀 Testing Enhanced Phase 3...")

    try:
        result = subprocess.run(
            ["python", "enhanced_cleanup.py", "--phase", "3", "--dry-run"],
            cwd="/Users/rohanvinaik/genomevault",
            capture_output=True,
            text=True,
            timeout=30,
        )

        print(f"Exit code: {result.returncode}")
        if result.stdout:
            print("Output:")
            print(result.stdout)

        if result.stderr:
            print("Errors:")
            print(result.stderr)

        return result.returncode == 0

    except Exception as e:
        print(f"❌ Error running enhanced Phase 3: {e}")
        return False


if __name__ == "__main__":
    print("🎯 Testing Enhanced Phase 3 with Ruff Upgrade\\n")

    # Test 1: Current Ruff version
    current_adequate = test_current_ruff()

    # Test 2: Upgrade capability
    can_upgrade = test_upgrade_process()

    # Test 3: Enhanced Phase 3
    phase3_works = run_enhanced_phase3()

    print("\\n📊 Test Results:")
    print(f"   Current Ruff adequate: {'✅' if current_adequate else '❌'}")
    print(f"   Can upgrade Ruff: {'✅' if can_upgrade else '❌'}")
    print(f"   Phase 3 works: {'✅' if phase3_works else '❌'}")

    if current_adequate:
        print("\\n🎉 Ready to run: python enhanced_cleanup.py --phase 3")
    elif can_upgrade:
        print("\\n🔧 Will upgrade Ruff automatically when running Phase 3")
    else:
        print("\\n⚠️  Manual Ruff upgrade may be needed")
