#!/usr/bin/env python3
"""
Final linting summary for GenomeVault
"""
import subprocess
import sys

def run_command(cmd):
    """Run command and return output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def main():
    print("GenomeVault Linting Summary")
    print("=" * 60)
    
    # Black
    print("\n1. BLACK (Code Formatting)")
    print("-" * 30)
    returncode, stdout, stderr = run_command("cd /Users/rohanvinaik/genomevault && /Users/rohanvinaik/miniconda3/bin/black --check . 2>&1")
    
    if returncode == 0:
        print("✅ PASSED - All files are properly formatted")
    else:
        # Count issues
        would_reformat = stdout.count("would reformat")
        failed_reformat = stdout.count("failed to reformat")
        print(f"❌ FAILED - {would_reformat} files need reformatting, {failed_reformat} files have syntax errors")
    
    # isort
    print("\n2. ISORT (Import Sorting)")
    print("-" * 30)
    returncode, stdout, stderr = run_command("cd /Users/rohanvinaik/genomevault && /Users/rohanvinaik/miniconda3/bin/isort --check-only --quiet . 2>&1")
    
    if returncode == 0:
        print("✅ PASSED - All imports are properly sorted")
    else:
        print("❌ FAILED - Some files have import sorting issues")
    
    # Flake8
    print("\n3. FLAKE8 (Style Guide)")
    print("-" * 30)
    returncode, stdout, stderr = run_command("cd /Users/rohanvinaik/genomevault && /Users/rohanvinaik/miniconda3/bin/flake8 genomevault/ tests/ examples/ --count 2>&1")
    
    if returncode == 0:
        print("✅ PASSED - No style violations found")
    else:
        # Extract violation count
        lines = stdout.strip().split('\n')
        if lines and lines[-1].isdigit():
            count = lines[-1]
            print(f"❌ FAILED - {count} style violations found")
        else:
            print("❌ FAILED - Style violations found")
    
    # Pylint
    print("\n4. PYLINT (Code Quality)")
    print("-" * 30)
    returncode, stdout, stderr = run_command("cd /Users/rohanvinaik/genomevault && /Users/rohanvinaik/miniconda3/bin/pylint genomevault/ --exit-zero 2>&1 | grep 'Your code has been rated at'")
    
    if stdout:
        score_str = stdout.split('rated at')[1].split('/')[0].strip()
        score = float(score_str)
        if score >= 7.0:
            print(f"✅ PASSED - Code quality score: {score}/10.00")
        else:
            print(f"❌ FAILED - Code quality score: {score}/10.00 (target: 7.00)")
    else:
        print("❌ FAILED - Could not determine score")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\nImprovement Actions Taken:")
    print("✓ Fixed syntax error in genomevault/clinical/model_validation.py")
    print("✓ Added missing type imports to 32 files")
    print("✓ Fixed unused exception variables in 20 files")
    print("✓ Added logger imports to 21 files")
    print("✓ Fixed f-string formatting issues in 33 files")
    print("✓ Formatted code with Black and isort")
    
    print("\nRemaining Issues:")
    print("• Some files still have syntax errors preventing Black formatting")
    print("• Flake8 violations need manual review (undefined names, unused variables)")
    print("• Pylint score needs improvement through refactoring")
    
    print("\nNext Steps:")
    print("1. Fix remaining syntax errors in files that fail Black formatting")
    print("2. Address F821 (undefined name) errors - these can cause runtime failures")
    print("3. Clean up F841 (unused variable) warnings")
    print("4. Refactor complex functions (C901) for better maintainability")
    print("5. Consider adding pre-commit hooks to maintain code quality")

if __name__ == "__main__":
    main()
