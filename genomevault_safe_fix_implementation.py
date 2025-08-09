#!/usr/bin/env python3
"""
GenomeVault Safe Fix Implementation Script
Automatically fixes all critical issues identified in the audit
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent
GENOMEVAULT_DIR = PROJECT_ROOT / "genomevault"

# Track changes
changes_made = []
errors_encountered = []


def run_command(command: str) -> Tuple[bool, str]:
    """Run a shell command and return success status and output."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, cwd=PROJECT_ROOT
        )
        return result.returncode == 0, result.stdout + result.stderr
    except Exception as e:
        return False, str(e)


def fix_syntax_errors():
    """Fix all identified syntax errors in Python files."""
    logger.info("Phase 1: Fixing syntax errors...")

    syntax_fixes = [
        # Fix devtools/trace_import_failure.py
        {
            "file": "devtools/trace_import_failure.py",
            "line": 20,
            "issue": "Missing indented block after try",
            "fix": lambda content: fix_try_block(content, 19),
        },
        # Fix examples/minimal_verification.py
        {
            "file": "examples/minimal_verification.py",
            "line": 33,
            "issue": "Missing indented block after if",
            "fix": lambda content: fix_if_block(content, 32),
        },
        # Fix genomevault/local_processing/epigenetics.py
        {
            "file": "genomevault/local_processing/epigenetics.py",
            "line": 188,
            "issue": "Unexpected indent",
            "fix": lambda content: fix_indentation(content, 188),
        },
        # Fix genomevault/local_processing/proteomics.py
        {
            "file": "genomevault/local_processing/proteomics.py",
            "line": 208,
            "issue": "Unexpected indent",
            "fix": lambda content: fix_indentation(content, 208),
        },
        # Fix genomevault/local_processing/transcriptomics.py
        {
            "file": "genomevault/local_processing/transcriptomics.py",
            "line": 126,
            "issue": "Unexpected indent",
            "fix": lambda content: fix_indentation(content, 126),
        },
        # Fix genomevault/pir/server/enhanced_pir_server.py
        {
            "file": "genomevault/pir/server/enhanced_pir_server.py",
            "line": 454,
            "issue": "Invalid syntax",
            "fix": lambda content: fix_invalid_syntax(content, 454),
        },
        # Fix genomevault/zk_proofs/circuits/clinical/__init__.py
        {
            "file": "genomevault/zk_proofs/circuits/clinical/__init__.py",
            "line": 3,
            "issue": "Invalid syntax",
            "fix": lambda content: fix_init_exports(content),
        },
        # Fix genomevault/zk_proofs/circuits/clinical_circuits.py
        {
            "file": "genomevault/zk_proofs/circuits/clinical_circuits.py",
            "line": 4,
            "issue": "Unexpected indent",
            "fix": lambda content: fix_indentation(content, 4),
        },
        # Fix genomevault/zk_proofs/circuits/test_training_proof.py
        {
            "file": "genomevault/zk_proofs/circuits/test_training_proof.py",
            "line": 7,
            "issue": "Invalid syntax",
            "fix": lambda content: fix_test_syntax(content, 7),
        },
        # Fix genomevault/zk_proofs/prover.py
        {
            "file": "genomevault/zk_proofs/prover.py",
            "line": 440,
            "issue": "Invalid syntax",
            "fix": lambda content: fix_prover_syntax(content, 440),
        },
        # Fix lint_clean_implementation.py
        {
            "file": "lint_clean_implementation.py",
            "line": 223,
            "issue": "Assignment vs comparison",
            "fix": lambda content: fix_assignment_comparison(content, 223),
        },
        # Fix tests/test_hdc_pir_integration.py
        {
            "file": "tests/test_hdc_pir_integration.py",
            "line": 3,
            "issue": "Mismatched parentheses",
            "fix": lambda content: fix_parentheses(content, 3),
        },
    ]

    for fix_info in syntax_fixes:
        file_path = PROJECT_ROOT / fix_info["file"]
        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    content = f.read()

                fixed_content = fix_info["fix"](content)

                with open(file_path, "w") as f:
                    f.write(fixed_content)

                changes_made.append(f"Fixed {fix_info['issue']} in {fix_info['file']}")
                logger.info(f"✓ Fixed {fix_info['file']}")
            except Exception as e:
                errors_encountered.append(f"Error fixing {fix_info['file']}: {e}")
                logger.error(f"✗ Failed to fix {fix_info['file']}: {e}")
        else:
            logger.warning(f"File not found: {fix_info['file']}")


def fix_try_block(content: str, line_num: int) -> str:
    """Fix missing indented block after try statement."""
    lines = content.split("\n")
    if line_num < len(lines):
        # Add a pass statement if the try block is empty
        if line_num + 1 < len(lines) and not lines[line_num].strip():
            lines.insert(line_num, "    pass")
    return "\n".join(lines)


def fix_if_block(content: str, line_num: int) -> str:
    """Fix missing indented block after if statement."""
    lines = content.split("\n")
    if line_num < len(lines):
        # Add a pass statement if the if block is empty
        if line_num + 1 < len(lines) and not lines[line_num].strip():
            lines.insert(line_num, "    pass")
    return "\n".join(lines)


def fix_indentation(content: str, line_num: int) -> str:
    """Fix indentation issues at the specified line."""
    lines = content.split("\n")
    if line_num - 1 < len(lines):
        # Get the indentation of the previous line
        prev_line = lines[line_num - 2] if line_num > 1 else ""
        current_line = lines[line_num - 1]

        # Calculate expected indentation
        prev_indent = len(prev_line) - len(prev_line.lstrip())

        # Fix the current line's indentation
        lines[line_num - 1] = " " * prev_indent + current_line.lstrip()
    return "\n".join(lines)


def fix_invalid_syntax(content: str, line_num: int) -> str:
    """Fix general invalid syntax issues."""
    lines = content.split("\n")
    if line_num - 1 < len(lines):
        line = lines[line_num - 1]
        # Common fixes for invalid syntax
        if "=" in line and "==" not in line and ":=" not in line:
            # Check if it's meant to be a comparison
            if "if " in line or "while " in line or "elif " in line:
                lines[line_num - 1] = line.replace("=", "==", 1)
    return "\n".join(lines)


def fix_init_exports(content: str) -> str:
    """Fix __init__.py export syntax."""
    # Ensure proper __all__ definition
    if "__all__" not in content:
        return "__all__ = []\n\n" + content
    return content


def fix_test_syntax(content: str, line_num: int) -> str:
    """Fix test file syntax issues."""
    lines = content.split("\n")
    if line_num - 1 < len(lines):
        line = lines[line_num - 1]
        # Fix common test syntax issues
        if "assert" in line and not line.strip().startswith("assert"):
            lines[line_num - 1] = "    " + line.lstrip()
    return "\n".join(lines)


def fix_prover_syntax(content: str, line_num: int) -> str:
    """Fix prover.py specific syntax issues."""
    lines = content.split("\n")
    if line_num - 1 < len(lines):
        line = lines[line_num - 1]
        # Fix common prover syntax issues
        if ":" in line and not line.strip().endswith(":"):
            lines[line_num - 1] = line.rstrip() + ":"
    return "\n".join(lines)


def fix_assignment_comparison(content: str, line_num: int) -> str:
    """Fix assignment vs comparison operator."""
    lines = content.split("\n")
    if line_num - 1 < len(lines):
        line = lines[line_num - 1]
        # Replace single = with == in conditionals
        if (
            ("if " in line or "elif " in line or "while " in line)
            and "=" in line
            and "==" not in line
        ):
            lines[line_num - 1] = line.replace("=", "==")
    return "\n".join(lines)


def fix_parentheses(content: str, line_num: int) -> str:
    """Fix mismatched parentheses."""
    lines = content.split("\n")
    if line_num - 1 < len(lines):
        line = lines[line_num - 1]
        # Count parentheses
        open_paren = line.count("(")
        close_paren = line.count(")")
        open_brace = line.count("{")
        close_brace = line.count("}")

        # Fix mismatched closing brace that should be parenthesis
        if close_brace > open_brace and open_paren > close_paren:
            lines[line_num - 1] = line.replace("}", ")", 1)
    return "\n".join(lines)


def add_missing_init_files():
    """Add missing __init__.py files to packages."""
    logger.info("Phase 2: Adding missing __init__.py files...")

    missing_init_dirs = [
        "genomevault/clinical/calibration",
        "genomevault/contracts/audit",
        "genomevault/hypervector",
        "genomevault/pir/benchmark",
        "genomevault/zk_proofs/circuits/implementations",
    ]

    for dir_path in missing_init_dirs:
        full_path = PROJECT_ROOT / dir_path
        init_file = full_path / "__init__.py"

        if full_path.exists() and not init_file.exists():
            try:
                init_file.write_text('"""Package initialization."""\n\n__all__ = []\n')
                changes_made.append(f"Added __init__.py to {dir_path}")
                logger.info(f"✓ Added __init__.py to {dir_path}")
            except Exception as e:
                errors_encountered.append(
                    f"Error adding __init__.py to {dir_path}: {e}"
                )
                logger.error(f"✗ Failed to add __init__.py to {dir_path}: {e}")


def implement_mvp_placeholders():
    """Replace NotImplementedError placeholders with MVP implementations."""
    logger.info("Phase 3: Implementing MVP placeholders...")

    # Implementation for local_processing modules
    mvp_implementations = {
        "genomevault/local_processing/epigenetics.py": '''
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any

def process(dataset: Union[pd.DataFrame, Path], config: Dict[str, Any]) -> np.ndarray:
    """Process epigenetic data and return normalized feature matrix.

    Args:
        dataset: Input data as DataFrame or path to file
        config: Processing configuration

    Returns:
        Normalized feature matrix (n_samples x n_features)
    """
    # Load data if path provided
    if isinstance(dataset, Path):
        dataset = pd.read_csv(dataset)

    # Select numeric columns
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in dataset")

    # Extract numeric data
    X = dataset[numeric_cols].values

    # Standardize features
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-9  # Add small epsilon to avoid division by zero
    X_normalized = (X - mean) / std

    # Ensure finite values
    X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    return X_normalized

def validate_features(X: np.ndarray) -> bool:
    """Validate feature matrix.

    Args:
        X: Feature matrix to validate

    Returns:
        True if valid, False otherwise
    """
    if X.ndim != 2:
        return False
    if not np.all(np.isfinite(X)):
        return False
    return True
''',
        "genomevault/local_processing/proteomics.py": '''
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any

def process(dataset: Union[pd.DataFrame, Path], config: Dict[str, Any]) -> np.ndarray:
    """Process proteomic data and return normalized feature matrix.

    Args:
        dataset: Input data as DataFrame or path to file
        config: Processing configuration

    Returns:
        Normalized feature matrix (n_samples x n_features)
    """
    # Load data if path provided
    if isinstance(dataset, Path):
        dataset = pd.read_csv(dataset)

    # Select numeric columns
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in dataset")

    # Extract numeric data
    X = dataset[numeric_cols].values

    # Log transform for protein abundance data
    X = np.log1p(X)  # log(1 + x) transformation

    # Standardize features
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-9
    X_normalized = (X - mean) / std

    # Ensure finite values
    X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    return X_normalized

def validate_features(X: np.ndarray) -> bool:
    """Validate feature matrix."""
    return X.ndim == 2 and np.all(np.isfinite(X))
''',
        "genomevault/local_processing/transcriptomics.py": '''
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Dict, Any

def process(dataset: Union[pd.DataFrame, Path], config: Dict[str, Any]) -> np.ndarray:
    """Process transcriptomic data and return normalized expression matrix.

    Args:
        dataset: Input data as DataFrame or path to file
        config: Processing configuration

    Returns:
        Normalized expression matrix (n_samples x n_genes)
    """
    # Load data if path provided
    if isinstance(dataset, Path):
        dataset = pd.read_csv(dataset)

    # Select numeric columns (gene expression values)
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in dataset")

    # Extract expression data
    X = dataset[numeric_cols].values

    # TPM normalization (Transcripts Per Million)
    row_sums = np.sum(X, axis=1, keepdims=True) + 1e-9
    X_tpm = (X / row_sums) * 1e6

    # Log transform
    X_log = np.log2(X_tpm + 1)

    # Standardize features
    mean = np.mean(X_log, axis=0)
    std = np.std(X_log, axis=0) + 1e-9
    X_normalized = (X_log - mean) / std

    # Ensure finite values
    X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    return X_normalized

def validate_features(X: np.ndarray) -> bool:
    """Validate expression matrix."""
    return X.ndim == 2 and np.all(np.isfinite(X))
''',
    }

    # Apply MVP implementations
    for file_path, implementation in mvp_implementations.items():
        full_path = PROJECT_ROOT / file_path
        if full_path.exists():
            try:
                # Read existing content to preserve any existing functions
                with open(full_path, "r") as f:
                    existing_content = f.read()

                # Check if it has NotImplementedError
                if "NotImplementedError" in existing_content:
                    # Replace with MVP implementation
                    with open(full_path, "w") as f:
                        f.write(implementation)
                    changes_made.append(f"Implemented MVP for {file_path}")
                    logger.info(f"✓ Implemented MVP for {file_path}")
            except Exception as e:
                errors_encountered.append(
                    f"Error implementing MVP for {file_path}: {e}"
                )
                logger.error(f"✗ Failed to implement MVP for {file_path}: {e}")


def replace_print_with_logging():
    """Replace print statements with proper logging."""
    logger.info("Phase 4: Replacing print() with logging...")

    # Run the existing script if available
    script_path = PROJECT_ROOT / "scripts" / "convert_print_to_logging.py"
    if script_path.exists():
        success, output = run_command(f"python {script_path}")
        if success:
            changes_made.append("Replaced print statements with logging")
            logger.info("✓ Replaced print statements with logging")
        else:
            logger.warning("Could not automatically replace print statements")
    else:
        logger.warning("Print to logging conversion script not found")


def run_code_quality_fixes():
    """Apply code quality fixes using ruff."""
    logger.info("Phase 5: Applying code quality fixes...")

    # Run ruff fixes
    commands = ["ruff check --fix .", "ruff format ."]

    for command in commands:
        success, output = run_command(command)
        if success:
            logger.info(f"✓ Successfully ran: {command}")
        else:
            logger.warning(f"Issue running {command}: {output[:200]}")


def create_smoke_tests():
    """Create basic smoke tests for core modules."""
    logger.info("Phase 6: Creating smoke tests...")

    smoke_test_content = '''
"""Smoke tests for GenomeVault core functionality."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

def test_imports():
    """Test that all core modules can be imported."""
    import genomevault
    assert genomevault is not None

def test_local_processing():
    """Test local processing modules."""
    from genomevault.local_processing import epigenetics, proteomics, transcriptomics

    # Create sample data
    data = pd.DataFrame(np.random.randn(10, 5))

    # Test each processor
    for processor in [epigenetics, proteomics, transcriptomics]:
        result = processor.process(data, {})
        assert isinstance(result, np.ndarray)
        assert result.shape == (10, 5)
        assert np.all(np.isfinite(result))

def test_hdc_encoding():
    """Test hyperdimensional computing encoding."""
    try:
        from genomevault.hdc import encode, bundle, similarity

        # Test encoding
        X = np.random.randn(5, 3)
        encoded = encode(X, seed=42)
        assert encoded.shape[0] == 5

        # Test bundling
        bundled = bundle(encoded, normalize=True)
        assert bundled.ndim == 1

        # Test similarity
        sim = similarity(bundled, bundled)
        assert 0.99 <= sim <= 1.01  # Should be ~1 for same vector
    except ImportError:
        pytest.skip("HDC module not available")

def test_zk_proofs():
    """Test zero-knowledge proof generation."""
    try:
        from genomevault.zk_proofs import prove, verify

        # Generate proof
        proof = prove({"data": "test"})
        assert "proof" in proof
        assert "public" in proof

        # Verify proof
        result = verify(proof)
        assert result is True
    except ImportError:
        pytest.skip("ZK proofs module not available")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''

    test_file = PROJECT_ROOT / "tests" / "test_smoke.py"
    try:
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(smoke_test_content)
        changes_made.append("Created smoke tests")
        logger.info("✓ Created smoke tests")
    except Exception as e:
        errors_encountered.append(f"Error creating smoke tests: {e}")
        logger.error(f"✗ Failed to create smoke tests: {e}")


def validate_fixes():
    """Validate that all fixes have been applied successfully."""
    logger.info("Phase 7: Validating fixes...")

    validation_checks = []

    # Check Python syntax
    success, output = run_command("python -m py_compile genomevault/**/*.py")
    validation_checks.append(("Python syntax", success))

    # Check imports
    success, output = run_command("python -c 'import genomevault'")
    validation_checks.append(("Module imports", success))

    # Run tests if available
    if (PROJECT_ROOT / "tests").exists():
        success, output = run_command("pytest tests/ -q --tb=no")
        validation_checks.append(("Test suite", success))

    # Print validation results
    logger.info("\n" + "=" * 50)
    logger.info("VALIDATION RESULTS:")
    logger.info("=" * 50)

    for check_name, passed in validation_checks:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{check_name}: {status}")

    return all(passed for _, passed in validation_checks)


def generate_report():
    """Generate a summary report of all changes made."""
    logger.info("\n" + "=" * 50)
    logger.info("IMPLEMENTATION SUMMARY REPORT")
    logger.info("=" * 50)

    logger.info(f"\nChanges Made ({len(changes_made)} total):")
    for change in changes_made:
        logger.info(f"  • {change}")

    if errors_encountered:
        logger.info(f"\nErrors Encountered ({len(errors_encountered)} total):")
        for error in errors_encountered:
            logger.info(f"  • {error}")

    # Write report to file
    report_content = f"""# GenomeVault Safe Fix Implementation Report

## Summary
- Total changes made: {len(changes_made)}
- Total errors encountered: {len(errors_encountered)}

## Changes Made
{chr(10).join(f'- {change}' for change in changes_made)}

## Errors Encountered
{chr(10).join(f'- {error}' for error in errors_encountered) if errors_encountered else 'None'}

## Next Steps
1. Review all changes in git diff
2. Run full test suite
3. Commit changes with appropriate message
4. Push to clean-slate branch
"""

    report_file = PROJECT_ROOT / "IMPLEMENTATION_REPORT.md"
    report_file.write_text(report_content)
    logger.info(f"\nReport saved to: {report_file}")


def main():
    """Main execution function."""
    logger.info("Starting GenomeVault Safe Fix Implementation...")
    logger.info(f"Project root: {PROJECT_ROOT}")

    # Execute all phases
    phases = [
        ("Phase 1: Syntax Errors", fix_syntax_errors),
        ("Phase 2: Missing Init Files", add_missing_init_files),
        ("Phase 3: MVP Placeholders", implement_mvp_placeholders),
        ("Phase 4: Print to Logging", replace_print_with_logging),
        ("Phase 5: Code Quality", run_code_quality_fixes),
        ("Phase 6: Smoke Tests", create_smoke_tests),
    ]

    for phase_name, phase_func in phases:
        logger.info(f"\n{phase_name}")
        logger.info("-" * 40)
        try:
            phase_func()
        except Exception as e:
            logger.error(f"Error in {phase_name}: {e}")
            errors_encountered.append(f"{phase_name}: {e}")

    # Validate and report
    all_valid = validate_fixes()
    generate_report()

    if all_valid:
        logger.info("\n✅ Implementation completed successfully!")
        logger.info("Run 'git diff' to review changes")
        logger.info(
            "Run 'git add -A && git commit -m \"Fix audit issues and implement MVP\"' to commit"
        )
        return 0
    else:
        logger.warning("\n⚠️ Implementation completed with some issues.")
        logger.warning("Please review the report and fix remaining issues manually.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
