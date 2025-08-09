#!/usr/bin/env python3
"""
Proper Ruff cleanup and upgrade following the recommended approach.
"""

import subprocess
import sys
from pathlib import Path


def find_all_ruff():
    """Find all ruff installations on PATH."""
    print("🔍 Finding all ruff installations...")

    try:
        result = subprocess.run(["which", "-a", "ruff"], capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and result.stdout:
            paths = result.stdout.strip().split("\n")
            print(f"Found ruff at: {paths}")
            return paths
        else:
            print("No ruff found on PATH")
            return []
    except Exception as e:
        print(f"Error finding ruff: {e}")
        return []


def uninstall_old_ruff():
    """Nuke the old ruff installation."""
    print("🗑️  Uninstalling old ruff...")

    try:
        result = subprocess.run(
            ["python", "-m", "pip", "uninstall", "-y", "ruff"],
            capture_output=True,
            text=True,
            timeout=60,
        )

        print(f"Uninstall exit code: {result.returncode}")
        if result.stdout:
            print(f"Stdout: {result.stdout}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")

        if result.returncode == 0:
            print("✅ Old ruff uninstalled successfully")
            return True
        else:
            print("⚠️  Uninstall completed (may not have been installed)")
            return True  # Continue anyway

    except Exception as e:
        print(f"❌ Error uninstalling ruff: {e}")
        return False


def install_new_ruff():
    """Install ruff >= 0.4.4."""
    print("📦 Installing ruff >= 0.4.4...")

    try:
        result = subprocess.run(
            ["python", "-m", "pip", "install", "ruff>=0.4.4,<0.5"],
            capture_output=True,
            text=True,
            timeout=120,
        )

        print(f"Install exit code: {result.returncode}")
        if result.stdout:
            print(f"Stdout: {result.stdout}")
        if result.stderr:
            print(f"Stderr: {result.stderr}")

        if result.returncode == 0:
            print("✅ New ruff installed successfully")
            return True
        else:
            print("❌ Failed to install new ruff")
            return False

    except Exception as e:
        print(f"❌ Error installing ruff: {e}")
        return False


def confirm_ruff_version():
    """Confirm the new ruff version."""
    print("🔍 Confirming ruff version...")

    try:
        result = subprocess.run(["ruff", "--version"], capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and result.stdout:
            version = result.stdout.strip()
            print(f"✅ Ruff version: {version}")

            if "0.4." in version or "0.5." in version:
                print("✅ Version is adequate for F821 fixing")
                return True
            else:
                print("❌ Version is still too old")
                return False
        else:
            print(f"❌ Could not get ruff version: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error checking ruff version: {e}")
        return False


def test_ruff_functionality():
    """Test that ruff works correctly."""
    print("🧪 Testing ruff functionality...")

    try:
        # Test basic check
        result = subprocess.run(
            ["ruff", "check", ".", "--quiet"],
            cwd="/Users/rohanvinaik/genomevault",
            capture_output=True,
            text=True,
            timeout=15,
        )

        print(f"Basic check exit code: {result.returncode}")

        if "unknown field" in (result.stderr or ""):
            print("❌ Configuration issues remain")
            return False

        # Test F821 JSON output
        result2 = subprocess.run(
            ["ruff", "check", ".", "--select", "F821", "--output-format", "json"],
            cwd="/Users/rohanvinaik/genomevault",
            capture_output=True,
            text=True,
            timeout=30,
        )

        print(f"F821 JSON check exit code: {result2.returncode}")

        if result2.stdout or result2.returncode in [
            0,
            1,
        ]:  # 0 = no errors, 1 = errors found
            print("✅ F821 JSON output works")
            return True
        else:
            print(f"❌ F821 JSON check failed: {result2.stderr}")
            return False

    except Exception as e:
        print(f"❌ Error testing ruff: {e}")
        return False


def update_ruff_config():
    """Update .ruff.toml to use modern features."""
    print("📝 Updating .ruff.toml...")

    ruff_config = Path("/Users/rohanvinaik/genomevault/.ruff.toml")

    if not ruff_config.exists():
        print("❌ .ruff.toml not found")
        return False

    content = ruff_config.read_text()

    # Backup original
    backup_path = ruff_config.with_suffix(".toml.backup")
    backup_path.write_text(content)
    print(f"✅ Backed up original config to {backup_path}")

    # Add [output] section if not present
    if "[output]" not in content:
        new_content = "[output]\nmax-violations = 200\n\n" + content
        ruff_config.write_text(new_content)
        print("✅ Added [output] section with max-violations = 200")
    else:
        print("✅ [output] section already exists")

    return True


def main():
    print("🚀 Proper Ruff Cleanup and Upgrade\n")

    # Step 1: Find existing installations
    find_all_ruff()

    # Step 2: Uninstall old version
    if not uninstall_old_ruff():
        print("\n❌ Failed to uninstall old ruff")
        sys.exit(1)

    # Step 3: Install new version
    if not install_new_ruff():
        print("\n❌ Failed to install new ruff")
        sys.exit(1)

    # Step 4: Confirm version
    if not confirm_ruff_version():
        print("\n❌ New ruff version is not adequate")
        sys.exit(1)

    # Step 5: Test functionality
    if not test_ruff_functionality():
        print("\n❌ Ruff functionality test failed")
        sys.exit(1)

    # Step 6: Update configuration
    if not update_ruff_config():
        print("\n❌ Failed to update configuration")
        sys.exit(1)

    print("\n🎉 SUCCESS! Ruff is now properly upgraded and configured.")
    print("\nYou can now run:")
    print("   python enhanced_cleanup.py --phase 3")
    print("   # This should now work without issues!")


if __name__ == "__main__":
    main()
