# GenomeVault Audit Checklist Implementation Guide

## Quick Start

I've created all the necessary configuration files and scripts to implement the audit checklist. Here's what to do:

### 1. Install Dependencies

```bash
cd /Users/rohanvinaik/genomevault
pip install -e .[dev]
pre-commit install
```

### 2. Run the Implementation Script

This will backup your current config and implement all checklist items:

```bash
python scripts/implement_checklist.py
```

### 3. Validate Implementation

Check that everything is properly set up:

```bash
python scripts/validate_checklist.py
```

## What's Been Created

### Configuration Files
- **pyproject.toml** - Updated to use Hatch build system
- **ruff.toml** - Modern Python linter/formatter config
- **mypy.ini** - Type checking configuration
- **pytest.ini** - Test runner configuration
- **.pre-commit-config.yaml** - Git hooks for code quality
- **.github/workflows/ci.yml** - GitHub Actions CI pipeline

### Code Files
- **genomevault/logging_utils.py** - Centralized logging utilities
- **genomevault/exceptions.py** - Custom exception hierarchy

### Helper Scripts
- **scripts/implement_checklist.py** - Master implementation script
- **scripts/validate_checklist.py** - Validation script
- **scripts/convert_print_to_logging.py** - Find print statements
- **scripts/analyze_complexity.py** - Analyze cyclomatic complexity

## Manual Steps Required

### 1. Convert Print Statements
After running `convert_print_to_logging.py`, manually update files to use logging:

```python
# Add at top of file
from genomevault.logging_utils import get_logger
logger = get_logger(__name__)

# Replace prints
print("Status message")  # → logger.info("Status message")
print(f"Debug: {var}")   # → logger.debug("Debug: %s", var)
print("Error!", e)       # → logger.error("Error!", exc_info=True)
```

### 2. Refactor High Complexity Functions
Review `radon_complexity_report.txt` and `complexity_refactoring_guide.md` to identify functions needing refactoring.

### 3. Update Exception Handling
Replace generic exceptions with specific ones:

```python
# Instead of:
except Exception as e:
    print(f"Error: {e}")

# Use:
from genomevault.exceptions import ConfigError, ValidationError
try:
    ...
except ValidationError as e:
    logger.error("Validation failed: %s", e)
    raise
```

### 4. Add Type Annotations
For public API modules, ensure all functions have type hints:

```python
def process_data(input_file: Path, config: dict[str, Any]) -> pd.DataFrame:
    """Process genomic data with given configuration."""
    ...
```

## Common Commands

```bash
# Format code
ruff format .

# Check code style
ruff check .

# Type check
mypy .

# Run tests with coverage
pytest

# Check cyclomatic complexity
radon cc -s -a genomevault

# Pre-commit checks
pre-commit run --all-files
```

## CI/CD

The GitHub Actions workflow will automatically run on push and PR:
- Python 3.10 and 3.11
- Ruff linting and formatting
- MyPy type checking
- Pytest with coverage

## Next Steps

1. Run `scripts/implement_checklist.py` to apply all changes
2. Review generated reports in project root
3. Fix any issues identified by the validation script
4. Commit changes with: `git add . && git commit -m "Apply audit checklist improvements"`
5. Push to trigger CI: `git push`

## Troubleshooting

If package installation fails:
```bash
# Install build tools
pip install --upgrade pip setuptools wheel hatchling

# Try again
pip install -e .[dev]
```

If pre-commit fails:
```bash
# Update hooks
pre-commit autoupdate

# Run manually
pre-commit run --all-files
```
