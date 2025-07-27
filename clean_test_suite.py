#!/usr/bin/env python3
"""
Comprehensive test suite and experiments cleaner for GenomeVault project.
Fixes import errors, indentation issues, and runs linters.
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from typing import List, Tuple, Dict
import ast
import autopep8
import isort
from collections import defaultdict

class TestSuiteCleaner:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.test_dirs = [
            project_root / "tests",
            project_root / "experiments",
            Path.home() / "experiments"  # External experiments directory
        ]
        self.issues_found = defaultdict(list)
        self.fixed_files = []
        
    def find_python_files(self) -> List[Path]:
        """Find all Python files in test and experiment directories."""
        python_files = []
        for test_dir in self.test_dirs:
            if test_dir.exists():
                python_files.extend(test_dir.rglob("*.py"))
        return python_files
    
    def fix_indentation_errors(self, content: str, filepath: Path) -> str:
        """Fix common indentation errors."""
        lines = content.split('\n')
        fixed_lines = []
        in_function = False
        function_indent = 0
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                fixed_lines.append(line)
                continue
                
            # Detect function/method definitions
            if re.match(r'^(def|class)\s+', line.strip()):
                in_function = True
                function_indent = len(line) - len(line.lstrip())
                fixed_lines.append(line)
                continue
            
            # Fix duplicate docstrings (common issue)
            if i > 0 and '"""' in line and '"""' in lines[i-1]:
                # Skip duplicate docstring
                self.issues_found[filepath].append(f"Line {i+1}: Removed duplicate docstring")
                continue
                
            # Fix incorrect indentation for docstrings
            if in_function and '"""' in line and line.strip().startswith('"""'):
                # Ensure proper indentation for docstrings
                proper_indent = function_indent + 4
                line = ' ' * proper_indent + line.strip()
                
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
    
    def fix_import_errors(self, content: str, filepath: Path) -> str:
        """Fix common import errors."""
        lines = content.split('\n')
        fixed_lines = []
        imports_section = []
        past_imports = False
        
        for line in lines:
            # Collect imports
            if line.strip().startswith(('import ', 'from ')) and not past_imports:
                imports_section.append(line)
            elif line.strip() and not line.strip().startswith('#') and imports_section:
                past_imports = True
                # Sort and deduplicate imports
                imports_section = list(set(imports_section))
                imports_section.sort()
                fixed_lines.extend(imports_section)
                fixed_lines.append('')  # Add blank line after imports
                fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        
        # Fix relative imports for test files
        content = '\n'.join(fixed_lines)
        if 'tests' in str(filepath):
            # Add sys.path manipulation if not present
            if 'sys.path.insert' not in content:
                sys_path_fix = '''import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
'''
                # Insert after initial imports
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith(('import', 'from', '#')):
                        lines.insert(i, sys_path_fix)
                        break
                content = '\n'.join(lines)
                self.issues_found[filepath].append("Added sys.path fix for imports")
        
        return content
    
    def fix_syntax_errors(self, content: str, filepath: Path) -> str:
        """Fix common syntax errors."""
        # Fix duplicate function definitions (TODO comments issue)
        lines = content.split('\n')
        fixed_lines = []
        skip_next = False
        
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
                
            # Check for duplicate docstring pattern
            if i < len(lines) - 2:
                if ('"""TODO:' in line and 
                    '"""' in lines[i+1] and 
                    lines[i+1].strip().startswith('"""')):
                    # Skip the TODO line
                    self.issues_found[filepath].append(f"Line {i+1}: Removed TODO docstring")
                    continue
            
            fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def run_autopep8(self, content: str, filepath: Path) -> str:
        """Run autopep8 to fix PEP8 issues."""
        try:
            fixed_content = autopep8.fix_code(
                content,
                options={
                    'aggressive': 2,
                    'max_line_length': 100,
                    'ignore': ['E501'],  # Ignore line too long
                }
            )
            if fixed_content != content:
                self.issues_found[filepath].append("Applied autopep8 formatting")
            return fixed_content
        except Exception as e:
            self.issues_found[filepath].append(f"autopep8 error: {str(e)}")
            return content
    
    def run_isort(self, content: str, filepath: Path) -> str:
        """Run isort to fix import ordering."""
        try:
            fixed_content = isort.code(content, profile="black", line_length=100)
            if fixed_content != content:
                self.issues_found[filepath].append("Applied isort import sorting")
            return fixed_content
        except Exception as e:
            self.issues_found[filepath].append(f"isort error: {str(e)}")
            return content
    
    def validate_python_syntax(self, content: str, filepath: Path) -> bool:
        """Validate Python syntax using ast."""
        try:
            ast.parse(content)
            return True
        except SyntaxError as e:
            self.issues_found[filepath].append(f"Syntax error: {e}")
            return False
    
    def fix_file(self, filepath: Path) -> bool:
        """Fix a single Python file."""
        print(f"Processing: {filepath}")
        
        try:
            # Read file
            with open(filepath, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Skip empty files
            if not original_content.strip():
                return True
            
            # Apply fixes in sequence
            content = original_content
            content = self.fix_import_errors(content, filepath)
            content = self.fix_indentation_errors(content, filepath)
            content = self.fix_syntax_errors(content, filepath)
            content = self.run_isort(content, filepath)
            content = self.run_autopep8(content, filepath)
            
            # Validate syntax
            if not self.validate_python_syntax(content, filepath):
                print(f"  ❌ Syntax errors remain in {filepath}")
                return False
            
            # Write back if changed
            if content != original_content:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixed_files.append(filepath)
                print(f"  ✅ Fixed {filepath}")
            else:
                print(f"  ✓ No changes needed for {filepath}")
            
            return True
            
        except Exception as e:
            self.issues_found[filepath].append(f"Error processing file: {e}")
            print(f"  ❌ Error processing {filepath}: {e}")
            return False
    
    def run_flake8(self, files: List[Path]) -> Dict[Path, List[str]]:
        """Run flake8 on files and collect issues."""
        flake8_issues = defaultdict(list)
        
        for filepath in files:
            try:
                result = subprocess.run(
                    ['flake8', '--max-line-length=100', '--ignore=E501,W503', str(filepath)],
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    flake8_issues[filepath] = result.stdout.strip().split('\n')
            except Exception as e:
                print(f"flake8 error for {filepath}: {e}")
        
        return flake8_issues
    
    def run_pylint(self, files: List[Path]) -> Dict[Path, List[str]]:
        """Run pylint on files and collect issues."""
        pylint_issues = defaultdict(list)
        
        for filepath in files:
            try:
                result = subprocess.run(
                    ['pylint', '--disable=C0114,C0115,C0116,R0903,R0801', str(filepath)],
                    capture_output=True,
                    text=True
                )
                if result.stdout:
                    pylint_issues[filepath] = [
                        line for line in result.stdout.strip().split('\n')
                        if line and not line.startswith('---')
                    ]
            except Exception as e:
                print(f"pylint error for {filepath}: {e}")
        
        return pylint_issues
    
    def generate_report(self, files: List[Path], flake8_issues: Dict, pylint_issues: Dict):
        """Generate a comprehensive report."""
        report_path = self.project_root / "test_suite_cleanup_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Test Suite Cleanup Report\n\n")
            f.write(f"Total files processed: {len(files)}\n")
            f.write(f"Files fixed: {len(self.fixed_files)}\n\n")
            
            # Fixed files
            if self.fixed_files:
                f.write("## Fixed Files\n\n")
                for filepath in sorted(self.fixed_files):
                    f.write(f"- {filepath.relative_to(self.project_root)}\n")
                    if filepath in self.issues_found:
                        for issue in self.issues_found[filepath]:
                            f.write(f"  - {issue}\n")
                f.write("\n")
            
            # Remaining flake8 issues
            if flake8_issues:
                f.write("## Remaining Flake8 Issues\n\n")
                for filepath, issues in sorted(flake8_issues.items()):
                    if issues:
                        f.write(f"### {filepath.relative_to(self.project_root)}\n")
                        for issue in issues[:5]:  # Show first 5 issues
                            f.write(f"- {issue}\n")
                        if len(issues) > 5:
                            f.write(f"- ... and {len(issues) - 5} more issues\n")
                        f.write("\n")
            
            # Remaining pylint issues
            if pylint_issues:
                f.write("## Remaining Pylint Issues\n\n")
                for filepath, issues in sorted(pylint_issues.items()):
                    if issues:
                        f.write(f"### {filepath.relative_to(self.project_root)}\n")
                        for issue in issues[:5]:  # Show first 5 issues
                            f.write(f"- {issue}\n")
                        if len(issues) > 5:
                            f.write(f"- ... and {len(issues) - 5} more issues\n")
                        f.write("\n")
        
        print(f"\n📄 Report generated: {report_path}")
    
    def clean_all(self):
        """Main cleaning process."""
        print("🧹 Starting test suite cleanup...\n")
        
        # Find all Python files
        python_files = self.find_python_files()
        print(f"Found {len(python_files)} Python files to process\n")
        
        # Fix each file
        for filepath in python_files:
            self.fix_file(filepath)
        
        print("\n🔍 Running linters...\n")
        
        # Run linters
        flake8_issues = self.run_flake8(python_files)
        pylint_issues = self.run_pylint(python_files)
        
        # Generate report
        self.generate_report(python_files, flake8_issues, pylint_issues)
        
        print("\n✅ Cleanup complete!")
        print(f"Fixed {len(self.fixed_files)} files")
        print(f"Flake8 issues remaining: {sum(len(issues) for issues in flake8_issues.values())}")
        print(f"Pylint issues remaining: {sum(len(issues) for issues in pylint_issues.values())}")


def main():
    """Main entry point."""
    # Check if we have the required tools
    required_tools = ['flake8', 'pylint']
    missing_tools = []
    
    for tool in required_tools:
        try:
            subprocess.run([tool, '--version'], capture_output=True)
        except FileNotFoundError:
            missing_tools.append(tool)
    
    if missing_tools:
        print(f"⚠️  Missing tools: {', '.join(missing_tools)}")
        print("Installing missing tools...")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_tools)
    
    # Also ensure we have autopep8 and isort
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'autopep8', 'isort'], 
                   capture_output=True)
    
    # Run the cleaner
    project_root = Path(__file__).parent
    cleaner = TestSuiteCleaner(project_root)
    cleaner.clean_all()


if __name__ == "__main__":
    main()
