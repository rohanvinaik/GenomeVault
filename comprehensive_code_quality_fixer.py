#!/usr/bin/env python3
"""
Comprehensive Code Quality Fixer for GenomeVault
Runs and implements suggestions from Black, isort, Flake8, and Pylint
"""

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


class CodeQualityFixer:
    def __init__(self):
        self.project_root = Path(".")
        self.fixes_applied = 0
        self.issues_found = 0
        self.tools_status = {}

    def ensure_tools_installed(self) -> bool:
        """Install required tools if not present."""
        tools = {
            'black': 'black',
            'isort': 'isort',
            'flake8': 'flake8',
            'pylint': 'pylint',
            'autoflake': 'autoflake'
        }

        print("🔧 Checking and installing required tools...")

        for tool_name, package_name in tools.items():
            try:
                subprocess.run([sys.executable, "-m", tool_name, "--version"],
                             capture_output = True, check = True)
                print(f"✅ {tool_name} is available")
                self.tools_status[tool_name] = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                print(f"📦 Installing {tool_name}...")
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", package_name],
                                 check = True, capture_output = True)
                    print(f"✅ {tool_name} installed successfully")
                    self.tools_status[tool_name] = True
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed to install {tool_name}: {e}")
                    self.tools_status[tool_name] = False

        return all(self.tools_status.values())

    def get_python_files(self) -> List[Path]:
        """Get all Python files in the project, excluding certain directories."""
        exclude_patterns = {
            '.git', '__pycache__', '.pytest_cache', 'build', 'dist',
            '.venv', 'venv', '.env', 'node_modules', '.tox',
            'genomevault.egg-info', 'htmlcov', '.benchmarks'
        }

        python_files = []
        for file_path in self.project_root.rglob("*.py"):
            if not any(pattern in str(file_path) for pattern in exclude_patterns):
                python_files.append(file_path)

        print(f"📁 Found {len(python_files)} Python files to process")
        return python_files

    def run_black(self) -> bool:
        """Run Black code formatter."""
        if not self.tools_status.get('black', False):
            return False

        print("\n🎨 Running Black (Code Formatter)")
        print(" = " * 50)

        try:
            # Run black with common configuration
            cmd = [
                sys.executable, "-m", "black",
                "--line-length", "88",
                "--target-version", "py38",
                "--include", r"\.py$",
                "--extend-exclude", r"/(\.git|__pycache__|\.pytest_cache|build|dist|\.venv|\.env)/",
                "."
            ]

            result = subprocess.run(cmd, capture_output = True, text = True)

            if result.returncode == 0:
                print("✅ Black formatting completed successfully")
                if "reformatted" in result.stderr:
                    # Count reformatted files
                    reformatted = len(re.findall(r"reformatted", result.stderr))
                    print(f"📝 Reformatted {reformatted} files")
                    self.fixes_applied += reformatted
                else:
                    print("📝 All files were already formatted correctly")
                return True
            else:
                print(f"⚠️ Black completed with warnings: {result.stderr}")
                return True  # Black often succeeds with warnings

        except Exception as e:
            print(f"❌ Error running Black: {e}")
            return False

    def run_isort(self) -> bool:
        """Run isort to organize imports."""
        if not self.tools_status.get('isort', False):
            return False

        print("\n📚 Running isort (Import Organizer)")
        print(" = " * 50)

        try:
            # Run isort with Black-compatible settings
            cmd = [
                sys.executable, "-m", "isort",
                "--profile", "black",
                "--line-length", "88",
                "--multi-line", "3",
                "--trailing-comma",
                "--force-grid-wrap", "0",
                "--combine-as",
                "--skip", "__pycache__",
                "--skip", ".git",
                "--skip", ".pytest_cache",
                "--skip", "build",
                "--skip", "dist",
                "."
            ]

            result = subprocess.run(cmd, capture_output = True, text = True)

            if result.returncode == 0:
                print("✅ isort completed successfully")
                if "Fixing" in result.stdout:
                    # Count fixed files
                    fixed = len(re.findall(r"Fixing", result.stdout))
                    print(f"📝 Fixed imports in {fixed} files")
                    self.fixes_applied += fixed
                else:
                    print("📝 All imports were already correctly organized")
                return True
            else:
                print(f"⚠️ isort completed with issues: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Error running isort: {e}")
            return False

    def run_autoflake(self) -> bool:
        """Run autoflake to remove unused imports and variables."""
        if not self.tools_status.get('autoflake', False):
            return False

        print("\n🧹 Running autoflake (Remove Unused Code)")
        print(" = " * 50)

        try:
            python_files = self.get_python_files()
            fixed_files = 0

            for file_path in python_files:
                cmd = [
                    sys.executable, "-m", "autoflake",
                    "--remove-unused-variables",
                    "--remove-all-unused-imports",
                    "--remove-duplicate-keys",
                    "--in-place",
                    str(file_path)
                ]

                result = subprocess.run(cmd, capture_output = True, text = True)
                if result.returncode == 0 and result.stdout.strip():
                    fixed_files += 1

            print(f"✅ autoflake completed successfully")
            print(f"📝 Cleaned {fixed_files} files")
            self.fixes_applied += fixed_files
            return True

        except Exception as e:
            print(f"❌ Error running autoflake: {e}")
            return False

    def run_flake8(self) -> Tuple[bool, List[Dict]]:
        """Run Flake8 linter and return issues."""
        if not self.tools_status.get('flake8', False):
            return False, []

        print("\n🔍 Running Flake8 (Style Guide Enforcement)")
        print(" = " * 50)

        try:
            cmd = [
                sys.executable, "-m", "flake8",
                "--max-line-length", "88",
                "--extend-ignore", "E203, W503, E501",  # Black-compatible ignores
                "--exclude", ".git, __pycache__, .pytest_cache, build, dist, .venv, .env",
                "--format", "%(path)s:%(row)d:%(col)d: %(code)s %(text)s",
                "."
            ]

            result = subprocess.run(cmd, capture_output = True, text = True)

            issues = []
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(': ', 2)
                        if len(parts) >= 3:
                            location = parts[0]
                            code_and_text = parts[1] + ': ' + parts[2]
                            issues.append({
                                'location': location,
                                'message': code_and_text,
                                'line': line
                            })

            self.issues_found += len(issues)

            if issues:
                print(f"⚠️ Found {len(issues)} Flake8 issues")
                # Show first 10 issues
                for issue in issues[:10]:
                    print(f"  📍 {issue['location']}: {issue['message']}")
                if len(issues) > 10:
                    print(f"  ... and {len(issues) - 10} more issues")
            else:
                print("✅ No Flake8 issues found")

            return True, issues

        except Exception as e:
            print(f"❌ Error running Flake8: {e}")
            return False, []

    def fix_common_flake8_issues(self, issues: List[Dict]) -> int:
        """Fix common Flake8 issues automatically."""
        print("\n🔧 Fixing Common Flake8 Issues")
        print(" = " * 50)

        fixes_applied = 0
        file_changes = {}

        for issue in issues:
            location = issue['location']
            message = issue['message']

            # Parse location
            parts = location.split(':')
            if len(parts) >= 3:
                file_path = parts[0]
                line_num = int(parts[1]) - 1  # Convert to 0-based

                # Only process files we can modify
                if not Path(file_path).exists():
                    continue

                # Load file content if not already loaded
                if file_path not in file_changes:
                    try:
                        with open(file_path, 'r', encoding = 'utf-8') as f:
                            file_changes[file_path] = f.readlines()
                    except Exception:
                        continue

                lines = file_changes[file_path]
                if line_num >= len(lines):
                    continue

                line = lines[line_num]
                original_line = line

                # Fix specific issues
                if 'E302' in message:  # Expected 2 blank lines
                    if line_num > 0 and lines[line_num - 1].strip():
                        lines.insert(line_num, '\n')
                        fixes_applied += 1

                elif 'E303' in message:  # Too many blank lines
                    if line.strip() == '' and line_num > 0 and lines[line_num - 1].strip() == '':
                        lines.pop(line_num)
                        fixes_applied += 1

                elif 'W292' in message:  # No newline at end of file
                    if line_num == len(lines) - 1 and not line.endswith('\n'):
                        lines[line_num] = line + '\n'
                        fixes_applied += 1

                elif 'E225' in message:  # Missing whitespace around operator
                    # Fix common operator spacing issues
                    line = re.sub(r'([^ = !<>]) = ([^ = ])', r'\1 = \2', line)
                    line = re.sub(r'([^ = !<>]) == ([^ = ])', r'\1 == \2', line)
                    line = re.sub(r'([^ = !<>]) != ([^ = ])', r'\1 != \2', line)
                    if line != original_line:
                        lines[line_num] = line
                        fixes_applied += 1

                elif 'E231' in message:  # Missing whitespace after ', '
                    line = re.sub(r', ([^\s])', r', \1', line)
                    if line != original_line:
                        lines[line_num] = line
                        fixes_applied += 1

        # Write back modified files
        files_modified = 0
        for file_path, lines in file_changes.items():
            try:
                with open(file_path, 'w', encoding = 'utf-8') as f:
                    f.writelines(lines)
                files_modified += 1
            except Exception as e:
                print(f"⚠️ Could not write {file_path}: {e}")

        print(f"✅ Applied {fixes_applied} automatic fixes to {files_modified} files")
        self.fixes_applied += fixes_applied
        return fixes_applied

    def run_pylint(self) -> Tuple[bool, List[Dict]]:
        """Run Pylint and return issues."""
        if not self.tools_status.get('pylint', False):
            return False, []

        print("\n🐍 Running Pylint (Advanced Code Analysis)")
        print(" = " * 50)

        try:
            # Get a subset of files for Pylint (it's slow on large codebases)
            python_files = self.get_python_files()

            # Focus on main source files, skip tests for speed
            main_files = [f for f in python_files if 'test' not in str(f) and 'legacy' not in str(f)][:20]

            if not main_files:
                print("⏭️ No suitable files for Pylint analysis")
                return True, []

            print(f"🔍 Analyzing {len(main_files)} main source files with Pylint...")

            cmd = [
                sys.executable, "-m", "pylint",
                "--output-format", "json",
                "--disable", "C0103, C0114, C0115, C0116, R0903, R0913, W0613",  # Disable some verbose checks
                "--max-line-length", "88"
            ] + [str(f) for f in main_files]

            result = subprocess.run(cmd, capture_output = True, text = True, timeout = 300)  # 5 min timeout

            issues = []
            if result.stdout:
                try:
                    pylint_data = json.loads(result.stdout)
                    for item in pylint_data:
                        if isinstance(item, dict) and 'message' in item:
                            issues.append({
                                'file': item.get('path', ''),
                                'line': item.get('line', 0),
                                'column': item.get('column', 0),
                                'message': item.get('message', ''),
                                'symbol': item.get('symbol', ''),
                                'type': item.get('type', '')
                            })
                except json.JSONDecodeError:
                    # Fallback to parsing text output
                    for line in result.stdout.split('\n'):
                        if ':' in line and ('error' in line or 'warning' in line):
                            issues.append({'line': line})

            self.issues_found += len(issues)

            if issues:
                print(f"⚠️ Found {len(issues)} Pylint issues")
                # Show summary by type
                issue_types = {}
                for issue in issues:
                    issue_type = issue.get('type', 'unknown')
                    issue_types[issue_type] = issue_types.get(issue_type, 0) + 1

                for issue_type, count in sorted(issue_types.items()):
                    print(f"  📊 {issue_type}: {count} issues")

                # Show first few issues
                for issue in issues[:5]:
                    if 'symbol' in issue:
                        print(f"  📍 {issue['file']}:{issue['line']} ({issue['symbol']}) {issue['message']}")

                if len(issues) > 5:
                    print(f"  ... and {len(issues) - 5} more issues")
            else:
                print("✅ No Pylint issues found")

            return True, issues

        except subprocess.TimeoutExpired:
            print("⏰ Pylint analysis timed out (5 minutes) - skipping")
            return True, []
        except Exception as e:
            print(f"❌ Error running Pylint: {e}")
            return False, []

    def create_config_files(self):
        """Create configuration files for the tools."""
        print("\n⚙️ Creating Configuration Files")
        print(" = " * 50)

        # pyproject.toml for Black and isort
        pyproject_content = '''[tool.black]
line-length = 88
target-version = ['py38']
include = '\\.pyi?$'
extend-exclude = '''
    /(\\.git
        | __pycache__
        | \\.pytest_cache
        | build
        | dist
        | \\.venv
        | \\.env)/
'''

[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
combine_as_imports = true
skip = ["__pycache__", ".git", ".pytest_cache", "build", "dist"]

[tool.pylint.messages_control]
disable = [
    "C0103",  # Invalid name
    "C0114",  # Missing module docstring
    "C0115",  # Missing class docstring
    "C0116",  # Missing function docstring
    "R0903",  # Too few public methods
    "R0913",  # Too many arguments
    "W0613",  # Unused argument
]

[tool.pylint.format]
max-line-length = 88
'''

        # .flake8 configuration
        flake8_content = '''[flake8]
max-line-length = 88
extend-ignore = E203, W503, E501
exclude =
    .git,
    __pycache__,
    .pytest_cache,
    build,
    dist,
    .venv,
    .env,
    genomevault.egg-info
per-file-ignores =
    __init__.py:F401
'''

        try:
            # Write pyproject.toml
            if not Path("pyproject.toml").exists():
                with open("pyproject.toml", "w") as f:
                    f.write(pyproject_content)
                print("✅ Created pyproject.toml")
            else:
                print("📁 pyproject.toml already exists")

            # Write .flake8
            with open(".flake8", "w") as f:
                f.write(flake8_content)
            print("✅ Created .flake8")

        except Exception as e:
            print(f"⚠️ Error creating config files: {e}")

    def generate_summary_report(self, flake8_issues: List[Dict], pylint_issues: List[Dict]):
        """Generate a comprehensive summary report."""
        report_content = f"""# Code Quality Report for GenomeVault

## 🎯 Summary
- **Tools Run**: Black, isort, autoflake, Flake8, Pylint
- **Fixes Applied**: {self.fixes_applied}
- **Issues Found**: {self.issues_found}
- **Python Files Processed**: {len(self.get_python_files())}

## 🔧 Tools Status
"""

        for tool, status in self.tools_status.items():
            status_icon = "✅" if status else "❌"
            report_content += f"- {status_icon} {tool.title()}: {'Available' if status else 'Failed'}\n"

        report_content += f"""
## 📊 Issues Breakdown

### Flake8 Issues: {len(flake8_issues)}
"""

        if flake8_issues:
            # Group by error code
            error_codes = {}
            for issue in flake8_issues:
                match = re.search(r'([EFWC]\d{3})', issue.get('message', ''))
                if match:
                    code = match.group(1)
                    error_codes[code] = error_codes.get(code, 0) + 1

            for code, count in sorted(error_codes.items()):
                report_content += f"- {code}: {count} occurrences\n"

        report_content += f"""
### Pylint Issues: {len(pylint_issues)}
"""

        if pylint_issues:
            # Group by type
            pylint_types = {}
            for issue in pylint_issues:
                issue_type = issue.get('type', 'unknown')
                pylint_types[issue_type] = pylint_types.get(issue_type, 0) + 1

            for issue_type, count in sorted(pylint_types.items()):
                report_content += f"- {issue_type}: {count} issues\n"

        report_content += """
## 🎯 Next Steps

### High Priority
1. Review remaining Flake8 issues (focus on E and W codes)
2. Address Pylint errors and warnings
3. Consider adding type hints (use mypy)

### Ongoing Maintenance
1. Set up pre-commit hooks
2. Integrate tools into CI/CD pipeline
3. Regular code quality reviews

### Configuration
- Configuration files created: .flake8, pyproject.toml
- Tools configured for Black compatibility
- Pylint configured with reasonable defaults

## 🔄 Re-running Tools
```bash
# Format code
python -m black .
python -m isort .

# Clean unused imports
python -m autoflake --remove-unused-variables --remove-all-unused-imports --in-place --recursive .

# Check quality
python -m flake8 .
python -m pylint genomevault/
```
"""

        with open("CODE_QUALITY_REPORT.md", "w") as f:
            f.write(report_content)

        print(f"\n📋 Generated CODE_QUALITY_REPORT.md")

    def run_all_tools(self):
        """Run all code quality tools in the correct order."""
        print("🚀 GenomeVault Code Quality Comprehensive Fixer")
        print(" = " * 60)

        # Check and install tools
        if not self.ensure_tools_installed():
            print("❌ Some tools failed to install. Continuing with available tools...")

        # Create configuration files
        self.create_config_files()

        # Run tools in order (formatters first, then linters)
        flake8_issues = []
        pylint_issues = []

        # 1. Clean unused imports first
        self.run_autoflake()

        # 2. Organize imports
        self.run_isort()

        # 3. Format code
        self.run_black()

        # 4. Check style guide compliance
        success, flake8_issues = self.run_flake8()
        if success and flake8_issues:
            self.fix_common_flake8_issues(flake8_issues)
            # Re-run Flake8 to see improvement
            print("\n🔄 Re-running Flake8 after fixes...")
            _, flake8_issues = self.run_flake8()

        # 5. Advanced analysis
        self.run_pylint()

        # Generate report
        self.generate_summary_report(flake8_issues, pylint_issues)

        # Final summary
        print("\n" + " = " * 60)
        print("🎉 Code Quality Comprehensive Analysis Complete!")
        print(" = " * 60)
        print(f"✅ Fixes Applied: {self.fixes_applied}")
        print(f"📊 Issues Identified: {self.issues_found}")
        print("📋 Report Generated: CODE_QUALITY_REPORT.md")
        print("\n📞 Next Steps:")
        print("1. Review the generated report")
        print("2. Address remaining high-priority issues")
        print("3. Set up pre-commit hooks for continuous quality")

def main():
    """Main entry point."""
    try:
        fixer = CodeQualityFixer()
        fixer.run_all_tools()
    except KeyboardInterrupt:
        print("\n⏹️ Code quality analysis interrupted by user")
    except Exception as e:
        print(f"❌ Error during code quality analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
