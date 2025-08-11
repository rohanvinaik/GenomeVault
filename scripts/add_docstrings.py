#!/usr/bin/env python3
"""Automatically add docstrings to Python files following Google style."""

import ast
import re
import sys
from pathlib import Path
from typing import Optional, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DocstringGenerator(ast.NodeVisitor):
    """Generate docstrings for Python code elements."""

    def __init__(self, source_lines: List[str]):
        self.source_lines = source_lines
        self.insertions = []  # List of (line_number, docstring) tuples

    def generate_module_docstring(self, filepath: str) -> str:
        """Generate a module-level docstring."""
        module_name = Path(filepath).stem
        if module_name == "__init__":
            package_name = Path(filepath).parent.name
            return f'"""Package initialization for {package_name}."""'
        else:
            # Infer purpose from module name
            module_name_readable = module_name.replace("_", " ").title()
            return f'"""{module_name_readable} module for genomic data processing."""'

    def generate_class_docstring(self, node: ast.ClassDef) -> str:
        """Generate a class docstring."""
        class_name = node.name

        # Check for dataclass
        is_dataclass = any(
            isinstance(dec, ast.Name) and dec.id == "dataclass" for dec in node.decorator_list
        )

        if is_dataclass:
            return f'"""Data class representing {self._humanize_name(class_name)}."""'

        # Check for common patterns
        if "Error" in class_name or "Exception" in class_name:
            return f'"""Exception raised for {self._humanize_name(class_name.replace("Error", "").replace("Exception", ""))} errors."""'

        if "Manager" in class_name:
            return f'"""Manages {self._humanize_name(class_name.replace("Manager", ""))} operations."""'

        if "Client" in class_name:
            return f'"""Client for {self._humanize_name(class_name.replace("Client", ""))} operations."""'

        if "Server" in class_name:
            return f'"""Server for {self._humanize_name(class_name.replace("Server", ""))} operations."""'

        return f'"""Class for {self._humanize_name(class_name)} functionality."""'

    def generate_function_docstring(self, node: ast.FunctionDef) -> str:
        """Generate a function docstring."""
        func_name = node.name
        args = self._extract_args(node)
        returns = self._extract_return_type(node)

        # Special cases
        if func_name == "__init__":
            docstring = '"""Initialize the instance.'
            if args:
                docstring += "\n\n    Args:"
                for arg in args:
                    docstring += f"\n        {arg}: Parameter for initialization."
            docstring += '\n    """'
            return docstring

        if func_name == "__str__":
            return '"""Return string representation of the object."""'

        if func_name == "__repr__":
            return '"""Return detailed string representation of the object."""'

        # Generate based on function name patterns
        if func_name.startswith("get_"):
            action = f"Get {self._humanize_name(func_name[4:])}"
        elif func_name.startswith("set_"):
            action = f"Set {self._humanize_name(func_name[4:])}"
        elif func_name.startswith("is_"):
            action = f"Check if {self._humanize_name(func_name[3:])}"
        elif func_name.startswith("has_"):
            action = f"Check if has {self._humanize_name(func_name[4:])}"
        elif func_name.startswith("validate_"):
            action = f"Validate {self._humanize_name(func_name[9:])}"
        elif func_name.startswith("process_"):
            action = f"Process {self._humanize_name(func_name[8:])}"
        elif func_name.startswith("handle_"):
            action = f"Handle {self._humanize_name(func_name[7:])}"
        elif func_name.startswith("create_"):
            action = f"Create {self._humanize_name(func_name[7:])}"
        elif func_name.startswith("update_"):
            action = f"Update {self._humanize_name(func_name[7:])}"
        elif func_name.startswith("delete_"):
            action = f"Delete {self._humanize_name(func_name[7:])}"
        elif func_name.startswith("load_"):
            action = f"Load {self._humanize_name(func_name[5:])}"
        elif func_name.startswith("save_"):
            action = f"Save {self._humanize_name(func_name[5:])}"
        else:
            action = f"{self._humanize_name(func_name)}"

        docstring = f'"""{action}.'

        if args:
            docstring += "\n\n    Args:"
            for arg in args:
                arg_desc = self._infer_arg_description(arg)
                docstring += f"\n        {arg}: {arg_desc}"

        if returns:
            docstring += "\n\n    Returns:"
            docstring += f"\n        {returns}"

        # Check for raises
        has_raise = self._check_for_raises(node)
        if has_raise:
            docstring += "\n\n    Raises:"
            docstring += "\n        Exception: If an error occurs."

        docstring += '\n    """'
        return docstring

    def _extract_args(self, node: ast.FunctionDef) -> List[str]:
        """Extract argument names from function definition."""
        args = []
        for arg in node.args.args:
            if arg.arg not in ["self", "cls"]:
                args.append(arg.arg)
        return args

    def _extract_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type from function definition."""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return node.returns.id
            elif isinstance(node.returns, ast.Constant):
                if node.returns.value is None:
                    return None
            return "The result of the operation"

        # Check for return statements
        for child in ast.walk(node):
            if isinstance(child, ast.Return) and child.value:
                return "The processed result"

        return None

    def _check_for_raises(self, node: ast.FunctionDef) -> bool:
        """Check if function raises exceptions."""
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                return True
        return False

    def _infer_arg_description(self, arg_name: str) -> str:
        """Infer argument description from its name."""
        common_patterns = {
            "filepath": "Path to the file",
            "filename": "Name of the file",
            "data": "Input data to process",
            "config": "Configuration parameters",
            "logger": "Logger instance",
            "timeout": "Timeout in seconds",
            "verbose": "Enable verbose output",
            "debug": "Enable debug mode",
            "force": "Force operation",
            "recursive": "Process recursively",
            "index": "Index value",
            "key": "Key for lookup",
            "value": "Value to set",
            "message": "Message to process",
            "error": "Error information",
            "result": "Result to process",
            "options": "Optional parameters",
            "params": "Parameters dictionary",
            "kwargs": "Additional keyword arguments",
            "args": "Additional arguments",
        }

        for pattern, description in common_patterns.items():
            if pattern in arg_name.lower():
                return description

        return f"The {self._humanize_name(arg_name)}"

    def _humanize_name(self, name: str) -> str:
        """Convert snake_case or CamelCase to human readable."""
        # Handle snake_case
        if "_" in name:
            return name.replace("_", " ")

        # Handle CamelCase
        result = re.sub(r"([A-Z])", r" \1", name)
        return result.strip().lower()

    def visit_Module(self, node):
        """Visit module node."""
        if not ast.get_docstring(node):
            # Module docstring should be at line 1 (after shebang and encoding)
            insert_line = 0
            for i, line in enumerate(self.source_lines):
                if line.strip() and not line.startswith("#") and not line.startswith('"""'):
                    insert_line = i
                    break

            docstring = self.generate_module_docstring(self.filepath)
            self.insertions.append((insert_line, docstring))

        self.generic_visit(node)

    def visit_ClassDef(self, node):
        """Visit class definition."""
        if not ast.get_docstring(node):
            # Add docstring after class definition line
            docstring = self.generate_class_docstring(node)
            indent = self._get_indent(node.lineno - 1)
            self.insertions.append((node.lineno, f"{indent}    {docstring}"))

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Visit function definition."""
        if not node.name.startswith("_") or node.name in [
            "__init__",
            "__str__",
            "__repr__",
        ]:
            if not ast.get_docstring(node):
                docstring = self.generate_function_docstring(node)
                indent = self._get_indent(node.lineno - 1)
                self.insertions.append((node.lineno, f"{indent}    {docstring}"))

        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition."""
        if not node.name.startswith("_"):
            if not ast.get_docstring(node):
                docstring = self.generate_function_docstring(node)
                docstring = docstring.replace('"""', '"""Async function to ', 1)
                indent = self._get_indent(node.lineno - 1)
                self.insertions.append((node.lineno, f"{indent}    {docstring}"))

        self.generic_visit(node)

    def _get_indent(self, line_idx: int) -> str:
        """Get indentation for a line."""
        if line_idx < len(self.source_lines):
            line = self.source_lines[line_idx]
            return line[: len(line) - len(line.lstrip())]
        return ""


def add_docstrings_to_file(filepath: Path) -> bool:
    """Add docstrings to a Python file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
            source_lines = source.splitlines()

        tree = ast.parse(source, filename=str(filepath))

        generator = DocstringGenerator(source_lines)
        generator.filepath = str(filepath)
        generator.visit(tree)

        if not generator.insertions:
            return False

        # Sort insertions by line number in reverse order
        generator.insertions.sort(key=lambda x: x[0], reverse=True)

        # Apply insertions
        lines = source_lines.copy()
        for line_no, docstring in generator.insertions:
            # Insert after the function/class definition line
            if line_no < len(lines):
                # Check if next line is already a docstring
                next_line_idx = line_no
                while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                    next_line_idx += 1

                if next_line_idx < len(lines) and '"""' not in lines[next_line_idx]:
                    lines.insert(line_no, docstring)

        # Write back
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return True

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Main function to add docstrings."""
    root = Path(__file__).resolve().parents[1]

    # Priority files to process
    priority_patterns = [
        "genomevault/core/exceptions.py",
        "genomevault/utils/metrics.py",
        "genomevault/hypervector_transform/binding_operations.py",
        "genomevault/zk_proofs/circuits/base_circuits.py",
        "genomevault/pir/core.py",
        "genomevault/utils/monitoring.py",
    ]

    processed = 0
    for pattern in priority_patterns:
        filepath = root / pattern
        if filepath.exists():
            print(f"Processing {filepath.relative_to(root)}...")
            if add_docstrings_to_file(filepath):
                processed += 1
                print(f"  ✓ Added docstrings")
            else:
                print(f"  - No changes needed")

    print(f"\nProcessed {processed} files")


if __name__ == "__main__":
    main()
