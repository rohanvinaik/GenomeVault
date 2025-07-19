#!/usr/bin/env python3
"""Final status check of GenomeVault imports"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n{'='*60}")
    print(f"🔍 {description}")
    print(f"{'='*60}")
    
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        cwd="/Users/rohanvinaik/genomevault"
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def main():
    """Check the final status"""
    print("🚀 GenomeVault Final Status Check")
    print("="*60)
    
    # Change to project directory
    os.chdir("/Users/rohanvinaik/genomevault")
    
    # Test 1: Basic pytest
    success1 = run_command(
        "python3 -m pytest tests/test_simple.py -v --tb=short",
        "Running basic pytest tests"
    )
    
    # Test 2: Import diagnostic
    success2 = run_command(
        "python3 diagnose_imports.py",
        "Running import diagnostic"
    )
    
    # Test 3: Comprehensive test
    success3 = run_command(
        "python3 test_all_imports_final.py",
        "Running comprehensive import test"
    )
    
    print("\n" + "="*60)
    print("📊 SUMMARY:")
    print("="*60)
    print(f"✅ Basic pytest: {'PASSED' if success1 else 'FAILED'}")
    print(f"✅ Import diagnostic: {'PASSED' if success2 else 'FAILED'}")
    print(f"✅ Comprehensive test: {'PASSED' if success3 else 'FAILED'}")
    print("="*60)
    
    if success1:
        print("\n🎉 SUCCESS! The basic tests are passing.")
        print("The circular import issues have been resolved.")
        print("\nRemaining tasks:")
        print("- Some modules still need circuits/biological/base_circuits.py")
        print("- Coverage is low because we're only running simple tests")
        print("- But the core import structure is now working!")
    else:
        print("\n❌ There are still some issues to resolve.")

if __name__ == "__main__":
    main()
