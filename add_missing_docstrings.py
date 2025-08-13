#!/usr/bin/env python3
"""Add missing docstrings to functions and classes in the codebase."""

import ast
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


class DocstringAnalyzer(ast.NodeVisitor):
    """AST visitor to find functions and classes without docstrings."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.missing_docstrings = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions and check for docstrings."""
        if not ast.get_docstring(node):
            self.missing_docstrings.append({
                'type': 'function',
                'name': node.name,
                'line': node.lineno,
                'is_method': self._is_method(node),
                'is_private': node.name.startswith('_'),
                'is_dunder': node.name.startswith('__') and node.name.endswith('__'),
                'args': self._get_args(node),
                'returns': self._get_return_type(node),
                'decorators': [d.id if hasattr(d, 'id') else str(d) for d in node.decorator_list]
            })
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions and check for docstrings."""
        if not ast.get_docstring(node):
            self.missing_docstrings.append({
                'type': 'async_function',
                'name': node.name,
                'line': node.lineno,
                'is_method': self._is_method(node),
                'is_private': node.name.startswith('_'),
                'is_dunder': node.name.startswith('__') and node.name.endswith('__'),
                'args': self._get_args(node),
                'returns': self._get_return_type(node),
                'decorators': [d.id if hasattr(d, 'id') else str(d) for d in node.decorator_list]
            })
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions and check for docstrings."""
        if not ast.get_docstring(node):
            self.missing_docstrings.append({
                'type': 'class',
                'name': node.name,
                'line': node.lineno,
                'is_private': node.name.startswith('_'),
                'bases': [self._get_base_name(base) for base in node.bases],
                'decorators': [d.id if hasattr(d, 'id') else str(d) for d in node.decorator_list]
            })
        self.generic_visit(node)

    def _is_method(self, node: ast.FunctionDef) -> bool:
        """Check if function is a method (has self or cls as first arg)."""
        if node.args.args:
            first_arg = node.args.args[0].arg
            return first_arg in ('self', 'cls')
        return False

    def _get_args(self, node: ast.FunctionDef) -> List[str]:
        """Get list of argument names."""
        return [arg.arg for arg in node.args.args]

    def _get_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Get return type annotation if present."""
        if node.returns:
            return ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        return None

    def _get_base_name(self, base: ast.expr) -> str:
        """Get base class name."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return ast.unparse(base) if hasattr(ast, 'unparse') else str(base)
        return str(base)


def generate_docstring(info: Dict[str, Any]) -> str:
    """Generate appropriate docstring based on function/class information.

    Args:
        info: Dictionary containing information about the function/class.

    Returns:
        Generated docstring text.
    """
    if info['type'] == 'class':
        return generate_class_docstring(info)
    else:
        return generate_function_docstring(info)


def generate_class_docstring(info: Dict[str, Any]) -> str:
    """Generate docstring for a class.

    Args:
        info: Dictionary containing class information.

    Returns:
        Generated class docstring.
    """
    name = info['name']

    # Generate appropriate description based on class name
    if 'Error' in name or 'Exception' in name:
        desc = f"Exception raised for {name.replace('Error', '').replace('Exception', '')} errors."
    elif 'Manager' in name:
        desc = f"Manages {name.replace('Manager', '')} operations and state."
    elif 'Handler' in name:
        desc = f"Handles {name.replace('Handler', '')} processing and events."
    elif 'Client' in name:
        desc = f"Client for interacting with {name.replace('Client', '')} services."
    elif 'Server' in name:
        desc = f"Server for handling {name.replace('Server', '')} requests."
    elif 'Config' in name:
        desc = f"Configuration settings for {name.replace('Config', '')}."
    elif 'Model' in name:
        desc = f"Model representing {name.replace('Model', '')} data structure."
    elif 'View' in name:
        desc = f"View component for {name.replace('View', '')} display."
    elif 'Controller' in name:
        desc = f"Controller for {name.replace('Controller', '')} logic."
    elif 'Service' in name:
        desc = f"Service layer for {name.replace('Service', '')} operations."
    elif 'Repository' in name:
        desc = f"Repository for {name.replace('Repository', '')} data access."
    elif 'Factory' in name:
        desc = f"Factory for creating {name.replace('Factory', '')} instances."
    elif 'Builder' in name:
        desc = f"Builder for constructing {name.replace('Builder', '')} objects."
    elif 'Validator' in name:
        desc = f"Validator for {name.replace('Validator', '')} validation rules."
    elif 'Parser' in name:
        desc = f"Parser for {name.replace('Parser', '')} data formats."
    elif 'Serializer' in name:
        desc = f"Serializer for {name.replace('Serializer', '')} data conversion."
    elif 'Adapter' in name:
        desc = f"Adapter for {name.replace('Adapter', '')} interface compatibility."
    elif 'Strategy' in name:
        desc = f"Strategy implementation for {name.replace('Strategy', '')} algorithm."
    elif 'Observer' in name:
        desc = f"Observer for {name.replace('Observer', '')} event monitoring."
    elif 'Decorator' in name:
        desc = f"Decorator for {name.replace('Decorator', '')} enhancement."
    elif info['bases'] and 'ABC' in str(info['bases']):
        desc = f"Abstract base class for {name} implementations."
    elif info['bases'] and 'Enum' in str(info['bases']):
        desc = f"Enumeration of {name} values."
    elif info['bases']:
        desc = f"{name} implementation extending {', '.join(info['bases'])}."
    else:
        desc = f"{name} class implementation."

    # Add attributes section for dataclasses
    if 'dataclass' in info.get('decorators', []):
        return f'''"""{desc}

    Attributes:
        # TODO: Document attributes
    """'''

    return f'"""{desc}"""'


def generate_function_docstring(info: Dict[str, Any]) -> str:
    """Generate docstring for a function/method.

    Args:
        info: Dictionary containing function information.

    Returns:
        Generated function docstring.
    """
    name = info['name']
    args = info['args']
    returns = info['returns']
    is_method = info['is_method']
    is_async = info['type'] == 'async_function'

    # Generate description based on function name patterns
    if name.startswith('test_'):
        desc = f"Test {name[5:].replace('_', ' ')}."
    elif name.startswith('get_'):
        desc = f"Get {name[4:].replace('_', ' ')}."
    elif name.startswith('set_'):
        desc = f"Set {name[4:].replace('_', ' ')}."
    elif name.startswith('is_'):
        desc = f"Check if {name[3:].replace('_', ' ')}."
    elif name.startswith('has_'):
        desc = f"Check if has {name[4:].replace('_', ' ')}."
    elif name.startswith('can_'):
        desc = f"Check if can {name[4:].replace('_', ' ')}."
    elif name.startswith('should_'):
        desc = f"Determine if should {name[7:].replace('_', ' ')}."
    elif name.startswith('create_'):
        desc = f"Create {name[7:].replace('_', ' ')}."
    elif name.startswith('build_'):
        desc = f"Build {name[6:].replace('_', ' ')}."
    elif name.startswith('make_'):
        desc = f"Make {name[5:].replace('_', ' ')}."
    elif name.startswith('generate_'):
        desc = f"Generate {name[9:].replace('_', ' ')}."
    elif name.startswith('calculate_'):
        desc = f"Calculate {name[10:].replace('_', ' ')}."
    elif name.startswith('compute_'):
        desc = f"Compute {name[8:].replace('_', ' ')}."
    elif name.startswith('process_'):
        desc = f"Process {name[8:].replace('_', ' ')}."
    elif name.startswith('handle_'):
        desc = f"Handle {name[7:].replace('_', ' ')}."
    elif name.startswith('parse_'):
        desc = f"Parse {name[6:].replace('_', ' ')}."
    elif name.startswith('validate_'):
        desc = f"Validate {name[9:].replace('_', ' ')}."
    elif name.startswith('verify_'):
        desc = f"Verify {name[7:].replace('_', ' ')}."
    elif name.startswith('check_'):
        desc = f"Check {name[6:].replace('_', ' ')}."
    elif name.startswith('load_'):
        desc = f"Load {name[5:].replace('_', ' ')}."
    elif name.startswith('save_'):
        desc = f"Save {name[5:].replace('_', ' ')}."
    elif name.startswith('read_'):
        desc = f"Read {name[5:].replace('_', ' ')}."
    elif name.startswith('write_'):
        desc = f"Write {name[6:].replace('_', ' ')}."
    elif name.startswith('update_'):
        desc = f"Update {name[7:].replace('_', ' ')}."
    elif name.startswith('delete_'):
        desc = f"Delete {name[7:].replace('_', ' ')}."
    elif name.startswith('remove_'):
        desc = f"Remove {name[7:].replace('_', ' ')}."
    elif name.startswith('add_'):
        desc = f"Add {name[4:].replace('_', ' ')}."
    elif name.startswith('append_'):
        desc = f"Append {name[7:].replace('_', ' ')}."
    elif name.startswith('insert_'):
        desc = f"Insert {name[7:].replace('_', ' ')}."
    elif name.startswith('find_'):
        desc = f"Find {name[5:].replace('_', ' ')}."
    elif name.startswith('search_'):
        desc = f"Search for {name[7:].replace('_', ' ')}."
    elif name.startswith('filter_'):
        desc = f"Filter {name[7:].replace('_', ' ')}."
    elif name.startswith('sort_'):
        desc = f"Sort {name[5:].replace('_', ' ')}."
    elif name.startswith('merge_'):
        desc = f"Merge {name[6:].replace('_', ' ')}."
    elif name.startswith('split_'):
        desc = f"Split {name[6:].replace('_', ' ')}."
    elif name.startswith('join_'):
        desc = f"Join {name[5:].replace('_', ' ')}."
    elif name.startswith('connect_'):
        desc = f"Connect to {name[8:].replace('_', ' ')}."
    elif name.startswith('disconnect_'):
        desc = f"Disconnect from {name[11:].replace('_', ' ')}."
    elif name.startswith('start_'):
        desc = f"Start {name[6:].replace('_', ' ')}."
    elif name.startswith('stop_'):
        desc = f"Stop {name[5:].replace('_', ' ')}."
    elif name.startswith('run_'):
        desc = f"Run {name[4:].replace('_', ' ')}."
    elif name.startswith('execute_'):
        desc = f"Execute {name[8:].replace('_', ' ')}."
    elif name.startswith('init_'):
        desc = f"Initialize {name[5:].replace('_', ' ')}."
    elif name.startswith('setup_'):
        desc = f"Setup {name[6:].replace('_', ' ')}."
    elif name.startswith('cleanup_'):
        desc = f"Cleanup {name[8:].replace('_', ' ')}."
    elif name.startswith('reset_'):
        desc = f"Reset {name[6:].replace('_', ' ')}."
    elif name.startswith('clear_'):
        desc = f"Clear {name[6:].replace('_', ' ')}."
    elif name.startswith('encode_'):
        desc = f"Encode {name[7:].replace('_', ' ')}."
    elif name.startswith('decode_'):
        desc = f"Decode {name[7:].replace('_', ' ')}."
    elif name.startswith('encrypt_'):
        desc = f"Encrypt {name[8:].replace('_', ' ')}."
    elif name.startswith('decrypt_'):
        desc = f"Decrypt {name[8:].replace('_', ' ')}."
    elif name.startswith('serialize_'):
        desc = f"Serialize {name[10:].replace('_', ' ')}."
    elif name.startswith('deserialize_'):
        desc = f"Deserialize {name[12:].replace('_', ' ')}."
    elif name.startswith('transform_'):
        desc = f"Transform {name[10:].replace('_', ' ')}."
    elif name.startswith('convert_'):
        desc = f"Convert {name[8:].replace('_', ' ')}."
    elif name.startswith('map_'):
        desc = f"Map {name[4:].replace('_', ' ')}."
    elif name.startswith('reduce_'):
        desc = f"Reduce {name[7:].replace('_', ' ')}."
    elif name.startswith('filter_'):
        desc = f"Filter {name[7:].replace('_', ' ')}."
    elif name == '__init__':
        desc = "Initialize the instance."
    elif name == '__str__':
        desc = "Return string representation."
    elif name == '__repr__':
        desc = "Return detailed string representation."
    elif name == '__eq__':
        desc = "Check equality with another object."
    elif name == '__lt__':
        desc = "Check if less than another object."
    elif name == '__le__':
        desc = "Check if less than or equal to another object."
    elif name == '__gt__':
        desc = "Check if greater than another object."
    elif name == '__ge__':
        desc = "Check if greater than or equal to another object."
    elif name == '__hash__':
        desc = "Return hash value."
    elif name == '__len__':
        desc = "Return length."
    elif name == '__getitem__':
        desc = "Get item by key or index."
    elif name == '__setitem__':
        desc = "Set item by key or index."
    elif name == '__delitem__':
        desc = "Delete item by key or index."
    elif name == '__contains__':
        desc = "Check if contains item."
    elif name == '__iter__':
        desc = "Return iterator."
    elif name == '__next__':
        desc = "Get next item."
    elif name == '__enter__':
        desc = "Enter context manager."
    elif name == '__exit__':
        desc = "Exit context manager."
    elif name == '__call__':
        desc = "Call the instance as a function."
    elif name == '__new__':
        desc = "Create new instance."
    elif name == '__del__':
        desc = "Cleanup when instance is deleted."
    elif name == 'setUp':
        desc = "Set up test fixtures."
    elif name == 'tearDown':
        desc = "Tear down test fixtures."
    elif name == 'setUpClass':
        desc = "Set up class-level test fixtures."
    elif name == 'tearDownClass':
        desc = "Tear down class-level test fixtures."
    elif is_async:
        desc = f"Asynchronously {name.replace('_', ' ')}."
    else:
        desc = f"{name.replace('_', ' ').capitalize()}."

    # Build docstring parts
    docstring_parts = [f'"""{desc}']

    # Add Args section
    if args:
        # Filter out self/cls
        filtered_args = [arg for arg in args if arg not in ('self', 'cls')]
        if filtered_args:
            docstring_parts.append("\n    Args:")
            for arg in filtered_args:
                # Try to infer type from name
                arg_type = infer_type_from_name(arg)
                docstring_parts.append(f"        {arg}: {arg_type}")

    # Add Returns section
    if returns:
        docstring_parts.append(f"\n    Returns:\n        {returns}")
    elif (name.startswith('is_')
        or name.startswith('has_')
        or name.startswith('can_')
        or name.startswith('should_')):
        docstring_parts.append(
            "\n    Returns:\n        bool: True if condition is met, "
            "False otherwise."
        )
    elif name.startswith('get_'):
        docstring_parts.append("\n    Returns:\n        The requested value.")
    elif (name.startswith('create_')
        or name.startswith('build_')
        or name.startswith('make_')
        or name.startswith('generate_')):
        docstring_parts.append("\n    Returns:\n        The created instance.")
    elif name == '__init__':
        # __init__ doesn't return anything
        pass
    elif name == '__str__' or name == '__repr__':
        docstring_parts.append("\n    Returns:\n        str: String representation.")
    elif name == '__len__':
        docstring_parts.append("\n    Returns:\n        int: Length of the object.")
    elif name == '__hash__':
        docstring_parts.append("\n    Returns:\n        int: Hash value.")
    elif name == '__iter__':
        docstring_parts.append("\n    Returns:\n        Iterator object.")
    elif not name.startswith('_'):
        # Public function without clear return pattern
        docstring_parts.append("\n    Returns:\n        Result of the operation.")

    docstring_parts.append('    """')
    return ''.join(docstring_parts)


def infer_type_from_name(arg_name: str) -> str:
    """Infer type from argument name patterns.

    Args:
        arg_name: Name of the argument.

    Returns:
        Inferred type description.
    """
    name_lower = arg_name.lower()

    if name_lower in ('path', 'filepath', 'file_path', 'filename', 'file_name', 'dir', 'directory'):
        return "Path to file or directory."
    elif name_lower in ('url', 'uri', 'endpoint'):
        return "URL or endpoint address."
    elif name_lower in ('host', 'hostname', 'server'):
        return "Host name or address."
    elif name_lower in ('port', 'port_number'):
        return "Port number."
    elif name_lower in ('timeout', 'duration', 'interval', 'delay'):
        return "Timeout duration in seconds."
    elif name_lower in ('size', 'length', 'count', 'num', 'number', 'n'):
        return "Number or count value."
    elif name_lower in ('index', 'idx', 'i', 'j', 'k'):
        return "Index position."
    elif name_lower in ('id', 'identifier', 'uid', 'uuid'):
        return "Unique identifier."
    elif name_lower in ('name', 'title', 'label'):
        return "Name or label string."
    elif name_lower in ('message', 'msg', 'text', 'content'):
        return "Message or text content."
    elif name_lower in ('data', 'payload', 'body'):
        return "Data payload."
    elif name_lower in ('config', 'configuration', 'settings', 'options', 'params', 'parameters'):
        return "Configuration dictionary or object."
    elif name_lower in ('key', 'secret', 'token', 'password', 'pwd'):
        return "Secret key or token."
    elif name_lower in ('value', 'val'):
        return "Value to process."
    elif name_lower in ('result', 'output', 'response'):
        return "Result or output data."
    elif name_lower in ('error', 'exception', 'err', 'e'):
        return "Error or exception object."
    elif name_lower in ('callback', 'handler', 'func', 'function'):
        return "Callback function."
    elif name_lower in ('flag', 'enabled', 'disabled', 'active'):
        return "Boolean flag."
    elif name_lower in ('mode', 'type', 'kind'):
        return "Operation mode or type."
    elif name_lower in ('format', 'fmt'):
        return "Format specification."
    elif name_lower in ('encoding', 'charset'):
        return "Character encoding."
    elif name_lower in ('verbose', 'debug', 'quiet', 'silent'):
        return "Verbosity flag."
    elif (name_lower.startswith('is_')
        or name_lower.startswith('has_')
        or name_lower.startswith('use_')):
        return "Boolean flag."
    elif name_lower.endswith('_list') or name_lower.endswith('s'):
        return "List of items."
    elif name_lower.endswith('_dict') or name_lower.endswith('_map'):
        return "Dictionary mapping."
    elif name_lower.endswith('_set'):
        return "Set of items."
    elif name_lower.endswith('_queue'):
        return "Queue of items."
    elif name_lower.endswith('_stack'):
        return "Stack of items."
    elif name_lower.endswith('_tree'):
        return "Tree structure."
    elif name_lower.endswith('_graph'):
        return "Graph structure."
    else:
        return "Parameter value."


def add_docstring_to_file(filepath: str, missing_docs: List[Dict]) -> int:
    """Add docstrings to a file.

    Args:
        filepath: Path to the file.
        missing_docs: List of missing docstring information.

    Returns:
        Number of docstrings added.
    """
    if not missing_docs:
        return 0

    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Sort by line number in reverse to avoid offset issues
        missing_docs.sort(key=lambda x: x['line'], reverse=True)

        count = 0
        for doc_info in missing_docs:
            line_num = doc_info['line'] - 1  # Convert to 0-based

            # Find the function/class definition line
            if line_num < len(lines):
                def_line = lines[line_num]
                indent = len(def_line) - len(def_line.lstrip())

                # Generate appropriate docstring
                docstring = generate_docstring(doc_info)

                # Find where to insert the docstring (after the definition line and any decorators)
                insert_line = line_num + 1
                while insert_line < len(lines) and lines[insert_line].strip().startswith('@'):
                    insert_line += 1

                # Skip past the function signature if it spans multiple lines
                if insert_line < len(lines) and ':' not in lines[line_num]:
                    while insert_line < len(lines) and ':' not in lines[insert_line - 1]:
                        insert_line += 1

                # Add proper indentation to docstring
                indented_docstring = '\n'.join(
                    ' ' * (indent + 4) + line if line else ''
                    for line in docstring.split('\n')
                ) + '\n'

                # Insert the docstring
                if insert_line < len(lines):
                    lines.insert(insert_line, indented_docstring)
                    count += 1

        # Write back the modified content
        if count > 0:
            with open(filepath, 'w') as f:
                f.writelines(lines)

        return count

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0


def process_directory(directory: str, limit: int = None) -> Tuple[int, List[str]]:
    """Process all Python files in a directory.

    Args:
        directory: Directory path to process.
        limit: Maximum number of files to process.

    Returns:
        Tuple of (total docstrings added, list of processed files).
    """
    total_added = 0
    processed_files = []
    file_count = 0

    for root, dirs, files in os.walk(directory):
        # Skip virtual environments and cache
        dirs[:] = [d for d in dirs if d not in {'.venv', \
            'venv', \
            '__pycache__', \
            '.git', \
            'node_modules'}]

        for file in files:
            if file.endswith('.py'):
                if limit and file_count >= limit:
                    return total_added, processed_files

                filepath = os.path.join(root, file)

                try:
                    with open(filepath, 'r') as f:
                        source = f.read()

                    tree = ast.parse(source)
                    analyzer = DocstringAnalyzer(filepath)
                    analyzer.visit(tree)

                    if analyzer.missing_docstrings:
                        added = add_docstring_to_file(filepath, analyzer.missing_docstrings)
                        if added > 0:
                            total_added += added
                            processed_files.append((filepath, added))
                            file_count += 1
                            print(f"Added {added} docstrings to {filepath}")

                except Exception as e:
                    # Skip files with parsing errors
                    pass

    return total_added, processed_files


def main():
    """Main function to add missing docstrings."""
    print("=" * 70)
    print("ADDING MISSING DOCSTRINGS")
    print("=" * 70)

    # Priority directories to process
    directories = [
        'genomevault',
        'tests',
        'examples',
        'scripts',
    ]

    total_added = 0
    all_processed = []

    for directory in directories:
        if os.path.exists(directory):
            print(f"\nProcessing {directory}...")
            added, \
                processed = process_directory(directory, \
                limit=50)  # Process up to 50 files per directory
            total_added += added
            all_processed.extend(processed)

    # Summary
    print("\n" + "=" * 70)
    print(f"Total docstrings added: {total_added}")

    if all_processed:
        print("\nTop files modified:")
        for filepath, count in sorted(all_processed, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {filepath}: {count} docstrings")

    print("=" * 70)


if __name__ == "__main__":
    main()
