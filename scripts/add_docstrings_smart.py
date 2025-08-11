#!/usr/bin/env python3
"""Smart docstring generator that adds Google-style docstrings to Python files."""

import ast
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import re

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def infer_purpose_from_name(name: str, context: str = "") -> str:
    """Infer the purpose of a function/class from its name."""
    # Convert snake_case to readable
    readable = name.replace("_", " ")

    # Common patterns
    patterns = {
        r"^get_(.+)": "Retrieve {}",
        r"^set_(.+)": "Set {}",
        r"^is_(.+)": "Check if {}",
        r"^has_(.+)": "Check if has {}",
        r"^validate_(.+)": "Validate {}",
        r"^process_(.+)": "Process {}",
        r"^handle_(.+)": "Handle {}",
        r"^create_(.+)": "Create {}",
        r"^update_(.+)": "Update {}",
        r"^delete_(.+)": "Delete {}",
        r"^load_(.+)": "Load {}",
        r"^save_(.+)": "Save {}",
        r"^encode_(.+)": "Encode {}",
        r"^decode_(.+)": "Decode {}",
        r"^encrypt_(.+)": "Encrypt {}",
        r"^decrypt_(.+)": "Decrypt {}",
        r"^parse_(.+)": "Parse {}",
        r"^format_(.+)": "Format {}",
        r"^convert_(.+)": "Convert {}",
        r"^generate_(.+)": "Generate {}",
        r"^calculate_(.+)": "Calculate {}",
        r"^compute_(.+)": "Compute {}",
        r"^verify_(.+)": "Verify {}",
        r"^check_(.+)": "Check {}",
        r"^init_(.+)": "Initialize {}",
        r"^setup_(.+)": "Setup {}",
        r"^cleanup_(.+)": "Cleanup {}",
        r"^register_(.+)": "Register {}",
        r"^unregister_(.+)": "Unregister {}",
    }

    for pattern, template in patterns.items():
        match = re.match(pattern, name)
        if match:
            return template.format(match.group(1).replace("_", " "))

    # Context-specific patterns
    if context == "genomevault":
        if "hypervector" in name:
            return f"Perform hypervector {readable} operation"
        elif "zk" in name or "proof" in name:
            return f"Handle zero-knowledge {readable}"
        elif "pir" in name:
            return f"Execute PIR {readable}"
        elif "federated" in name:
            return f"Coordinate federated {readable}"
        elif "clinical" in name:
            return f"Process clinical {readable}"

    return readable.capitalize()


def generate_docstring_for_function(node: ast.FunctionDef, context: str = "") -> str:
    """Generate a docstring for a function."""
    name = node.name

    # Special methods
    if name == "__init__":
        docstring = '"""Initialize instance.\n'
    elif name == "__str__":
        return '"""Return string representation."""'
    elif name == "__repr__":
        return '"""Return detailed representation for debugging."""'
    elif name == "__enter__":
        return '"""Enter context manager."""'
    elif name == "__exit__":
        return '"""Exit context manager."""'
    elif name.startswith("__") and name.endswith("__"):
        return f'"""Implement {name} special method."""'
    else:
        purpose = infer_purpose_from_name(name, context)
        docstring = f'"""{purpose}.\n'

    # Add Args section if there are arguments
    args = []
    for arg in node.args.args:
        if arg.arg not in ["self", "cls"]:
            args.append(arg.arg)

    if args:
        docstring += "\n    Args:\n"
        for arg in args:
            # Infer type and description
            arg_desc = infer_arg_description(arg)
            docstring += f"        {arg}: {arg_desc}\n"

    # Add Returns section if function returns something
    has_return = any(isinstance(n, ast.Return) and n.value for n in ast.walk(node))
    if has_return and name != "__init__":
        return_desc = infer_return_description(node)
        docstring += "\n    Returns:\n"
        docstring += f"        {return_desc}\n"

    # Add Raises section if function raises exceptions
    has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(node))
    if has_raise:
        docstring += "\n    Raises:\n"
        # Try to identify specific exceptions
        exceptions = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Raise):
                if n.exc and isinstance(n.exc, ast.Call):
                    if isinstance(n.exc.func, ast.Name):
                        exceptions.add(n.exc.func.id)

        if exceptions:
            for exc in sorted(exceptions):
                docstring += f"        {exc}: When operation fails.\n"
        else:
            docstring += "        Exception: When operation fails.\n"

    docstring = docstring.rstrip() + '\n    """'
    return docstring


def infer_arg_description(arg_name: str) -> str:
    """Infer argument description from name."""
    # Common parameter patterns
    patterns = {
        "filepath": "Path to the file.",
        "filename": "Name of the file.",
        "directory": "Directory path.",
        "path": "File or directory path.",
        "data": "Input data to process.",
        "config": "Configuration dictionary.",
        "params": "Parameters dictionary.",
        "options": "Optional settings.",
        "timeout": "Timeout in seconds.",
        "verbose": "Enable verbose output.",
        "debug": "Enable debug mode.",
        "logger": "Logger instance.",
        "callback": "Callback function.",
        "handler": "Handler function.",
        "index": "Index position.",
        "key": "Dictionary key.",
        "value": "Value to set.",
        "message": "Message string.",
        "error": "Error object.",
        "exception": "Exception object.",
        "result": "Operation result.",
        "response": "Server response.",
        "request": "Client request.",
        "query": "Query string.",
        "url": "URL string.",
        "host": "Host address.",
        "port": "Port number.",
        "encoding": "Character encoding.",
        "dtype": "Data type.",
        "shape": "Array shape.",
        "dim": "Dimension value.",
        "size": "Size value.",
        "count": "Number of items.",
        "limit": "Maximum limit.",
        "offset": "Starting offset.",
        "threshold": "Threshold value.",
        "epsilon": "Small epsilon value.",
        "alpha": "Alpha parameter.",
        "beta": "Beta parameter.",
        "model": "Model instance.",
        "weights": "Model weights.",
        "features": "Feature array.",
        "labels": "Label array.",
        "batch_size": "Batch size for processing.",
        "epochs": "Number of training epochs.",
        "learning_rate": "Learning rate for optimization.",
    }

    # Check exact matches first
    if arg_name in patterns:
        return patterns[arg_name]

    # Check partial matches
    for pattern, desc in patterns.items():
        if pattern in arg_name:
            return desc.replace(pattern, arg_name)

    # Genomic-specific patterns
    if "genome" in arg_name:
        return "Genomic data."
    elif "variant" in arg_name:
        return "Genetic variant information."
    elif "snp" in arg_name:
        return "SNP data."
    elif "vcf" in arg_name:
        return "VCF file data."
    elif "hypervector" in arg_name or "hv" in arg_name:
        return "Hypervector representation."
    elif "proof" in arg_name:
        return "Zero-knowledge proof."
    elif "circuit" in arg_name:
        return "ZK circuit specification."

    # Default
    return f'{arg_name.replace("_", " ").capitalize()}.'


def infer_return_description(node: ast.FunctionDef) -> str:
    """Infer return description from function."""
    name = node.name

    # Pattern-based inference
    if name.startswith("get_"):
        return f'The {name[4:].replace("_", " ")}.'
    elif name.startswith("is_") or name.startswith("has_"):
        return "True if condition is met, False otherwise."
    elif name.startswith("create_") or name.startswith("generate_"):
        return f'Newly created {name.split("_", 1)[1].replace("_", " ")}.'
    elif name.startswith("calculate_") or name.startswith("compute_"):
        return "Calculated result."
    elif name.startswith("validate_"):
        return "Validation result."
    elif name.startswith("process_"):
        return "Processed data."
    elif name.startswith("load_"):
        return "Loaded data."
    elif name.startswith("save_"):
        return "Save operation success status."

    # Check return annotation
    if node.returns:
        if isinstance(node.returns, ast.Name):
            type_name = node.returns.id
            if type_name == "bool":
                return "Boolean result."
            elif type_name == "str":
                return "String result."
            elif type_name == "int":
                return "Integer result."
            elif type_name == "float":
                return "Float result."
            elif type_name == "dict":
                return "Dictionary result."
            elif type_name == "list":
                return "List result."
            elif type_name == "None":
                return "None."
            else:
                return f"{type_name} instance."

    return "Operation result."


def generate_docstring_for_class(node: ast.ClassDef, context: str = "") -> str:
    """Generate a docstring for a class."""
    name = node.name

    # Check if it's an exception
    if name.endswith("Error") or name.endswith("Exception"):
        base = name.replace("Error", "").replace("Exception", "")
        return f'"""Exception raised for {base.lower()} errors."""'

    # Check for common patterns
    if name.endswith("Manager"):
        base = name.replace("Manager", "")
        return f'"""Manage {base.lower()} operations and state."""'

    if name.endswith("Client"):
        base = name.replace("Client", "")
        return f'"""Client for {base.lower()} operations."""'

    if name.endswith("Server"):
        base = name.replace("Server", "")
        return f'"""Server handling {base.lower()} requests."""'

    if name.endswith("Handler"):
        base = name.replace("Handler", "")
        return f'"""Handle {base.lower()} events and operations."""'

    if name.endswith("Processor"):
        base = name.replace("Processor", "")
        return f'"""Process {base.lower()} data."""'

    if name.endswith("Builder"):
        base = name.replace("Builder", "")
        return f'"""Build {base.lower()} objects."""'

    if name.endswith("Factory"):
        base = name.replace("Factory", "")
        return f'"""Factory for creating {base.lower()} instances."""'

    # Check if it's a dataclass
    is_dataclass = any(
        (isinstance(d, ast.Name) and d.id == "dataclass")
        or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
        for d in node.decorator_list
    )

    if is_dataclass:
        return f'"""Data container for {name.lower().replace("_", " ")} information."""'

    # Context-specific
    if context == "genomevault":
        if "Hypervector" in name:
            return (
                f'"""Hypervector-based {name.replace("Hypervector", "").lower()} implementation."""'
            )
        elif "ZK" in name or "Proof" in name:
            return f'"""Zero-knowledge proof {name.replace("ZK", "").replace("Proof", "").lower()} component."""'
        elif "PIR" in name:
            return (
                f'"""Private information retrieval {name.replace("PIR", "").lower()} component."""'
            )

    # Default
    return f'"""{name.replace("_", " ")} implementation."""'


def add_docstring_to_file(filepath: Path) -> Tuple[bool, int]:
    """Add missing docstrings to a file.

    Returns:
        Tuple of (success, number_added).
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        # Parse the file
        tree = ast.parse(content)

        # Determine context from filepath
        context = "genomevault" if "genomevault" in str(filepath) else ""

        lines = content.splitlines()
        insertions = []

        # Check module docstring
        if not ast.get_docstring(tree):
            module_name = filepath.stem
            if module_name == "__init__":
                module_doc = f'"""Package initialization for {filepath.parent.name}."""'
            else:
                module_doc = f'"""{module_name.replace("_", " ").title()} module."""'

            # Find where to insert (after any from __future__ imports)
            insert_line = 0
            for i, line in enumerate(lines):
                if (
                    line.strip()
                    and not line.startswith("#")
                    and not line.startswith("from __future__")
                ):
                    insert_line = i
                    break

            insertions.append((insert_line, module_doc, 0))

        # Visit all nodes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not ast.get_docstring(node):
                    docstring = generate_docstring_for_class(node, context)
                    # Find the line after the class definition
                    class_line = node.lineno - 1
                    indent = len(lines[class_line]) - len(lines[class_line].lstrip())
                    insertions.append((node.lineno, docstring, indent + 4))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip private methods unless they're special methods
                if not node.name.startswith("_") or node.name in [
                    "__init__",
                    "__str__",
                    "__repr__",
                    "__enter__",
                    "__exit__",
                ]:
                    if not ast.get_docstring(node):
                        docstring = generate_docstring_for_function(node, context)
                        if isinstance(node, ast.AsyncFunctionDef):
                            docstring = docstring.replace('"""', '"""Async operation to ', 1)

                        # Find the line after the function definition
                        func_line = node.lineno - 1
                        indent = len(lines[func_line]) - len(lines[func_line].lstrip())
                        insertions.append((node.lineno, docstring, indent + 4))

        if not insertions:
            return True, 0

        # Sort insertions by line number in reverse
        insertions.sort(key=lambda x: x[0], reverse=True)

        # Apply insertions
        for line_no, docstring, indent in insertions:
            indent_str = " " * indent
            indented_docstring = "\n".join(
                indent_str + line if line.strip() else line for line in docstring.splitlines()
            )
            lines.insert(line_no, indented_docstring)

        # Write back
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        return True, len(insertions)

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        import traceback

        traceback.print_exc()
        return False, 0


def main():
    """Main function to add docstrings."""
    root = Path(__file__).resolve().parents[1]

    # Files to process (from analysis)
    priority_files = [
        "genomevault/core/exceptions.py",
        "genomevault/utils/metrics.py",
        "genomevault/hypervector_transform/binding_operations.py",
        "genomevault/hypervector_transform/hierarchical.py",
        "genomevault/zk_proofs/circuits/base_circuits.py",
        "genomevault/utils/monitoring.py",
        "genomevault/pir/core.py",
    ]

    total_added = 0

    for file_path in priority_files:
        full_path = root / file_path
        if full_path.exists():
            print(f"Processing {file_path}...")
            success, num_added = add_docstring_to_file(full_path)
            if success:
                if num_added > 0:
                    print(f"  ✓ Added {num_added} docstrings")
                    total_added += num_added
                else:
                    print(f"  - No docstrings needed")
            else:
                print(f"  ✗ Error processing file")

    print(f"\nTotal docstrings added: {total_added}")


if __name__ == "__main__":
    main()
