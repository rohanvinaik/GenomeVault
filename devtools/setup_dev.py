#!/usr/bin/env python3
"""
One-command development environment setup for GenomeVault.

This script sets up a complete development environment including:
- Virtual environment creation
- Dependency installation with all extras
- Pre-commit hooks setup
- Initial test run to verify installation

Usage:
    python devtools/setup_dev.py [--venv-name VENV_NAME] [--skip-tests]
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple


class Colors:
    """Terminal colors for output formatting."""

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(msg: str) -> None:
    """Print a formatted header message."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{msg}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")


def print_step(msg: str) -> None:
    """Print a formatted step message."""
    print(f"{Colors.OKCYAN}➤ {msg}{Colors.ENDC}")


def print_success(msg: str) -> None:
    """Print a formatted success message."""
    print(f"{Colors.OKGREEN}✓ {msg}{Colors.ENDC}")


def print_error(msg: str) -> None:
    """Print a formatted error message."""
    print(f"{Colors.FAIL}✗ {msg}{Colors.ENDC}")


def print_warning(msg: str) -> None:
    """Print a formatted warning message."""
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")


def run_command(cmd: List[str], description: str, cwd: Optional[Path] = None) -> Tuple[bool, str]:
    """
    Run a command and return success status and output.

    Args:
        cmd: Command to run as list of strings
        description: Description of what the command does
        cwd: Working directory for command

    Returns:
        Tuple of (success, output)
    """
    print_step(description)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, check=True)
        print_success(f"{description} completed")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed: {e}")
        if e.stderr:
            print(f"  Error output: {e.stderr}")
        return False, e.stderr
    except FileNotFoundError:
        print_error(f"Command not found: {' '.join(cmd)}")
        return False, ""


def check_python_version() -> bool:
    """Check if Python version meets requirements."""
    required_version = (3, 9)
    current_version = sys.version_info[:2]

    if current_version < required_version:
        print_error(
            f"Python {required_version[0]}.{required_version[1]}+ is required. "
            f"You have {current_version[0]}.{current_version[1]}"
        )
        return False

    print_success(f"Python {current_version[0]}.{current_version[1]} meets requirements")
    return True


def setup_virtual_environment(venv_name: str) -> Tuple[bool, Path]:
    """
    Create and activate a virtual environment.

    Args:
        venv_name: Name of the virtual environment

    Returns:
        Tuple of (success, venv_path)
    """
    project_root = Path(__file__).parent.parent
    venv_path = project_root / venv_name

    # Check if venv already exists
    if venv_path.exists():
        print_warning(f"Virtual environment '{venv_name}' already exists")
        response = input("Do you want to recreate it? (y/N): ").strip().lower()
        if response != "y":
            print_step("Using existing virtual environment")
            return True, venv_path

        # Remove existing venv
        import shutil

        shutil.rmtree(venv_path)

    # Create virtual environment
    success, _ = run_command(
        [sys.executable, "-m", "venv", str(venv_path)],
        f"Creating virtual environment '{venv_name}'",
    )

    if not success:
        return False, venv_path

    # Get activation command based on OS
    if platform.system() == "Windows":
        activate_cmd = str(venv_path / "Scripts" / "activate.bat")
        pip_path = venv_path / "Scripts" / "pip"
        python_path = venv_path / "Scripts" / "python"
    else:
        activate_cmd = f"source {venv_path / 'bin' / 'activate'}"
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"

    print_success(f"Virtual environment created at {venv_path}")
    print(f"\n  To activate manually, run: {Colors.BOLD}{activate_cmd}{Colors.ENDC}\n")

    return True, venv_path


def install_dependencies(venv_path: Path) -> bool:
    """
    Install project dependencies with all extras.

    Args:
        venv_path: Path to virtual environment

    Returns:
        Success status
    """
    project_root = Path(__file__).parent.parent

    # Get pip path
    if platform.system() == "Windows":
        pip_path = venv_path / "Scripts" / "pip"
    else:
        pip_path = venv_path / "bin" / "pip"

    # Upgrade pip first
    success, _ = run_command([str(pip_path), "install", "--upgrade", "pip"], "Upgrading pip")

    if not success:
        return False

    # Install project with all extras
    print_step("Installing GenomeVault with all extras...")
    extras = ["dev", "ml", "zk", "nanopore", "gpu", "blockchain", "full"]

    for extra in extras:
        success, _ = run_command(
            [str(pip_path), "install", "-e", f".[{extra}]"],
            f"Installing with [{extra}] extras",
            cwd=project_root,
        )
        if not success:
            print_warning(
                f"Failed to install [{extra}] extras (may require additional system dependencies)"
            )

    return True


def setup_pre_commit_hooks(venv_path: Path) -> bool:
    """
    Set up pre-commit hooks.

    Args:
        venv_path: Path to virtual environment

    Returns:
        Success status
    """
    # Get pre-commit path
    if platform.system() == "Windows":
        pre_commit_path = venv_path / "Scripts" / "pre-commit"
    else:
        pre_commit_path = venv_path / "bin" / "pre-commit"

    # Check if pre-commit is installed
    if not pre_commit_path.exists():
        print_warning("pre-commit not found, installing...")
        if platform.system() == "Windows":
            pip_path = venv_path / "Scripts" / "pip"
        else:
            pip_path = venv_path / "bin" / "pip"

        success, _ = run_command([str(pip_path), "install", "pre-commit"], "Installing pre-commit")
        if not success:
            return False

    # Install pre-commit hooks
    project_root = Path(__file__).parent.parent
    success, _ = run_command(
        [str(pre_commit_path), "install"],
        "Installing pre-commit hooks",
        cwd=project_root,
    )

    if success:
        # Run pre-commit on all files to check setup
        print_step("Running pre-commit checks on sample files...")
        run_command(
            [str(pre_commit_path), "run", "--files", "genomevault/__init__.py"],
            "Testing pre-commit setup",
            cwd=project_root,
        )

    return success


def run_initial_tests(venv_path: Path) -> bool:
    """
    Run initial tests to verify installation.

    Args:
        venv_path: Path to virtual environment

    Returns:
        Success status
    """
    project_root = Path(__file__).parent.parent

    # Get pytest path
    if platform.system() == "Windows":
        pytest_path = venv_path / "Scripts" / "pytest"
    else:
        pytest_path = venv_path / "bin" / "pytest"

    # Run fast unit tests
    success, _ = run_command(
        [str(pytest_path), "-m", "not slow and not e2e", "--co", "-q"],
        "Collecting tests (dry run)",
        cwd=project_root,
    )

    if success:
        print_step("Running a few quick tests to verify setup...")
        run_command(
            [str(pytest_path), "-m", "unit", "-k", "test_", "--maxfail=3", "-q"],
            "Running sample unit tests",
            cwd=project_root,
        )

    return True


def create_local_env_file() -> bool:
    """
    Create a .env file with default development settings.

    Returns:
        Success status
    """
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"

    if env_file.exists():
        print_step(".env file already exists")
        return True

    if env_example.exists():
        # Copy from example
        import shutil

        shutil.copy2(env_example, env_file)
        print_success("Created .env from .env.example")
    else:
        # Create basic .env
        content = """# GenomeVault Development Environment
GENOMEVAULT_ENV=development
GENOMEVAULT_DEBUG=true
GENOMEVAULT_LOG_LEVEL=INFO

# API Settings
API_HOST=127.0.0.1
API_PORT=8000

# Performance Settings
GENOMEVAULT_CACHE_SIZE=1000
GENOMEVAULT_MAX_WORKERS=4
"""
        env_file.write_text(content)
        print_success("Created default .env file")

    return True


def print_next_steps(venv_name: str) -> None:
    """Print next steps for the developer."""
    print_header("Setup Complete! Next Steps")

    if platform.system() == "Windows":
        activate_cmd = f"{venv_name}\\Scripts\\activate"
    else:
        activate_cmd = f"source {venv_name}/bin/activate"

    steps = f"""
1. Activate the virtual environment:
   {Colors.BOLD}{activate_cmd}{Colors.ENDC}

2. Run the test suite:
   {Colors.BOLD}pytest tests/{Colors.ENDC}

3. Start the API server:
   {Colors.BOLD}uvicorn genomevault.api.main:app --reload{Colors.ENDC}

4. Run linting and formatting:
   {Colors.BOLD}ruff check genomevault{Colors.ENDC}
   {Colors.BOLD}ruff format genomevault{Colors.ENDC}

5. Check the documentation:
   - README.md for project overview
   - genomevault/MODULE_STRUCTURE.md for code organization
   - CLAUDE.md for AI assistant guidance

6. Run benchmarks:
   {Colors.BOLD}python scripts/bench.py{Colors.ENDC}
    """

    print(steps)
    print(f"\n{Colors.OKGREEN}Happy coding! 🚀{Colors.ENDC}\n")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Set up GenomeVault development environment")
    parser.add_argument(
        "--venv-name",
        default="venv",
        help="Name of virtual environment (default: venv)",
    )
    parser.add_argument("--skip-tests", action="store_true", help="Skip running initial tests")

    args = parser.parse_args()

    print_header("GenomeVault Development Environment Setup")

    # Check Python version
    if not check_python_version():
        return 1

    # Create virtual environment
    success, venv_path = setup_virtual_environment(args.venv_name)
    if not success:
        return 1

    # Install dependencies
    if not install_dependencies(venv_path):
        print_error("Failed to install dependencies")
        return 1

    # Set up pre-commit hooks
    if not setup_pre_commit_hooks(venv_path):
        print_warning("Pre-commit setup failed (non-critical)")

    # Create .env file
    create_local_env_file()

    # Run initial tests
    if not args.skip_tests:
        if not run_initial_tests(venv_path):
            print_warning("Some tests failed (this is expected for initial setup)")

    # Print next steps
    print_next_steps(args.venv_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
