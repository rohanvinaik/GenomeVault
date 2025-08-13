#!/usr/bin/env python3
"""Add comprehensive docstrings to all functions and classes missing them."""

import ast
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def analyze_function_body(node: ast.FunctionDef) -> Dict[str, Any]:
    """Analyze function body to understand what it does.

    Args:
        node: AST function node.

    Returns:
        Dictionary with analysis results.
    """
    analysis = {
        'raises': [],
        'returns_value': False,
        'has_loops': False,
        'has_conditionals': False,
        'calls_methods': [],
        'modifies_self': False,
        'is_generator': False,
        'is_property': False,
        'is_staticmethod': False,
        'is_classmethod': False,
    }

    # Check decorators
    for decorator in node.decorator_list:
        if isinstance(decorator, ast.Name):
            if decorator.id == 'property':
                analysis['is_property'] = True
            elif decorator.id == 'staticmethod':
                analysis['is_staticmethod'] = True
            elif decorator.id == 'classmethod':
                analysis['is_classmethod'] = True

    # Analyze body
    for stmt in ast.walk(node):
        if isinstance(stmt, ast.Raise):
            if hasattr(stmt, 'exc') and stmt.exc:
                if isinstance(stmt.exc, ast.Call) and hasattr(stmt.exc.func, 'id'):
                    analysis['raises'].append(stmt.exc.func.id)
                elif isinstance(stmt.exc, ast.Name):
                    analysis['raises'].append(stmt.exc.id)
        elif isinstance(stmt, ast.Return):
            if stmt.value is not None:
                analysis['returns_value'] = True
        elif isinstance(stmt, ast.Yield) or isinstance(stmt, ast.YieldFrom):
            analysis['is_generator'] = True
            analysis['returns_value'] = True
        elif isinstance(stmt, (ast.For, ast.While)):
            analysis['has_loops'] = True
        elif isinstance(stmt, ast.If):
            analysis['has_conditionals'] = True
        elif isinstance(stmt, ast.Attribute):
            if hasattr(stmt.value, 'id') and stmt.value.id == 'self':
                # Check if it's being assigned to
                parent = getattr(stmt, 'parent', None)
                if isinstance(parent, ast.Assign):
                    analysis['modifies_self'] = True

    return analysis


def generate_comprehensive_docstring(
    name: str,
    args: List[str],
    returns: Optional[str],
    is_method: bool,
    is_async: bool,
    node_type: str,
    analysis: Dict[str, Any],
    class_name: Optional[str] = None,
    filepath: Optional[str] = None
) -> str:
    """Generate a comprehensive docstring based on function analysis.

    Args:
        name: Function name.
        args: List of argument names.
        returns: Return type annotation.
        is_method: Whether it's a method.
        is_async: Whether it's async.
        node_type: Type of node (function/method/class).
        analysis: Analysis results from function body.
        class_name: Name of containing class if applicable.
        filepath: Path to file for context.

    Returns:
        Generated docstring.
    """
    if node_type == 'class':
        return generate_class_docstring_comprehensive(name, filepath)

    # Determine function purpose from name and context
    desc = infer_function_purpose(name, class_name, filepath, analysis)

    # Build docstring
    parts = [f'"""{desc}']

    # Add extended description for complex functions
    if analysis['has_loops'] and analysis['has_conditionals']:
        parts.append(f"\n    \n    This {'method' if is_method else 'function'} "
                     f"implements complex logic")
        parts.append(f"    with conditional branching and iteration.")
    elif analysis['is_generator']:
        parts.append(f"\n    \n    Yields items from the {'method' if is_method else 'function'}'s iteration.")

    # Args section
    filtered_args = [arg for arg in args if arg not in ('self', 'cls')]
    if filtered_args:
        parts.append("\n    \n    Args:")
        for arg in filtered_args:
            arg_desc = generate_arg_description(arg, name, class_name)
            parts.append(f"        {arg}: {arg_desc}")

    # Returns section
    if analysis['is_generator']:
        parts.append("\n    \n    Yields:")
        parts.append("        Items from the iteration.")
    elif analysis['returns_value'] or returns:
        parts.append("\n    \n    Returns:")
        if returns:
            return_desc = generate_return_description(returns, name)
            parts.append(f"        {return_desc}")
        else:
            return_desc = infer_return_type(name, analysis)
            parts.append(f"        {return_desc}")
    elif name != '__init__':
        # Check if it might return something based on name
        if (
            any(name.startswith(p) for p in ['get_', 'create_', 'build_', 'generate_',  \
                'compute_', 'calculate_'])
        ):
            parts.append("\n    \n    Returns:")
            parts.append("        The computed result.")

    # Raises section
    if analysis['raises']:
        parts.append("\n    \n    Raises:")
        for exc in set(analysis['raises']):
            exc_desc = generate_exception_description(exc)
            parts.append(f"        {exc}: {exc_desc}")

    # Add example for complex or public API functions
    if not name.startswith('_') and not is_method and analysis['has_loops']:
        parts.append("\n    \n    Example:")
        parts.append(f"        >>> result = {name}()")
        parts.append("        >>> print(result)")

    parts.append('\n    """')
    return ''.join(parts)


def infer_function_purpose(
    name: str,
    class_name: Optional[str],
    filepath: Optional[str],
    analysis: Dict[str, Any]
) -> str:
    """Infer the purpose of a function from its name and context.

    Args:
        name: Function name.
        class_name: Containing class name.
        filepath: File path for context.
        analysis: Function body analysis.

    Returns:
        Description of function purpose.
    """
    # Special methods
    if name == '__init__':
        return f"Initialize {class_name or 'instance'}."
    elif name == '__str__':
        return "Return human-readable string representation."
    elif name == '__repr__':
        return "Return detailed string representation for debugging."
    elif name == '__eq__':
        return "Check equality with another instance."
    elif name == '__hash__':
        return "Compute hash value for use in collections."
    elif name == '__len__':
        return "Return the length of the container."
    elif name == '__getitem__':
        return "Retrieve item by key or index."
    elif name == '__setitem__':
        return "Set item value by key or index."
    elif name == '__enter__':
        return "Enter the runtime context."
    elif name == '__exit__':
        return "Exit the runtime context."
    elif name == '__call__':
        return "Make instance callable as a function."

    # Context from filepath
    if filepath:
        if 'test' in filepath.lower():
            if name.startswith('test_'):
                return f"Test {name[5:].replace('_', ' ')}."
            elif name == 'setUp':
                return "Set up test fixtures before each test method."
            elif name == 'tearDown':
                return "Clean up after each test method."
        elif 'hypervector' in filepath or 'hv' in filepath:
            if 'encode' in name:
                return "Encode data into hypervector representation."
            elif 'decode' in name:
                return "Decode hypervector back to original representation."
            elif 'bind' in name:
                return "Bind hypervectors using XOR operation."
            elif 'bundle' in name:
                return "Bundle multiple hypervectors through addition."
        elif 'zk' in filepath or 'proof' in filepath:
            if 'prove' in name:
                return "Generate zero-knowledge proof."
            elif 'verify' in name:
                return "Verify zero-knowledge proof."
            elif 'witness' in name:
                return "Generate witness for the circuit."
        elif 'crypto' in filepath:
            if 'encrypt' in name:
                return "Encrypt data using cryptographic algorithm."
            elif 'decrypt' in name:
                return "Decrypt encrypted data."
            elif 'sign' in name:
                return "Generate cryptographic signature."
            elif 'verify' in name:
                return "Verify cryptographic signature."

    # Property methods
    if analysis.get('is_property'):
        if name.startswith('_'):
            return f"Get {name[1:].replace('_', ' ')} property value."
        return f"Get {name.replace('_', ' ')} property value."

    # Standard prefixes
    prefixes = {
        'get_': 'Retrieve',
        'set_': 'Set',
        'is_': 'Check if',
        'has_': 'Check if has',
        'can_': 'Check if can',
        'should_': 'Determine if should',
        'create_': 'Create new',
        'build_': 'Build',
        'make_': 'Make',
        'generate_': 'Generate',
        'calculate_': 'Calculate',
        'compute_': 'Compute',
        'process_': 'Process',
        'handle_': 'Handle',
        'parse_': 'Parse',
        'validate_': 'Validate',
        'verify_': 'Verify',
        'check_': 'Check',
        'load_': 'Load',
        'save_': 'Save',
        'read_': 'Read',
        'write_': 'Write',
        'update_': 'Update',
        'delete_': 'Delete',
        'remove_': 'Remove',
        'add_': 'Add',
        'append_': 'Append',
        'insert_': 'Insert',
        'find_': 'Find',
        'search_': 'Search for',
        'filter_': 'Filter',
        'sort_': 'Sort',
        'merge_': 'Merge',
        'split_': 'Split',
        'transform_': 'Transform',
        'convert_': 'Convert',
        'encode_': 'Encode',
        'decode_': 'Decode',
        'serialize_': 'Serialize',
        'deserialize_': 'Deserialize',
        'init_': 'Initialize',
        'setup_': 'Set up',
        'cleanup_': 'Clean up',
        'reset_': 'Reset',
        'clear_': 'Clear',
        'start_': 'Start',
        'stop_': 'Stop',
        'run_': 'Run',
        'execute_': 'Execute',
    }

    for prefix, verb in prefixes.items():
        if name.startswith(prefix):
            subject = name[len(prefix):].replace('_', ' ')
            return f"{verb} {subject}."

    # Default based on analysis
    if analysis.get('modifies_self'):
        return f"Update internal state for {name.replace('_', ' ')}."
    elif analysis.get('returns_value'):
        return f"Compute and return {name.replace('_', ' ')}."
    else:
        return f"Perform {name.replace('_', ' ')} operation."


def generate_arg_description(arg: str, func_name: str, class_name: Optional[str]) -> str:
    """Generate description for a function argument.

    Args:
        arg: Argument name.
        func_name: Function name for context.
        class_name: Class name for context.

    Returns:
        Argument description.
    """
    # Common argument patterns
    arg_patterns = {
        'data': 'Input data to process.',
        'value': 'Value to set or process.',
        'key': 'Key for lookup or storage.',
        'index': 'Index position in collection.',
        'item': 'Item to add or process.',
        'element': 'Element to process.',
        'node': 'Node in data structure.',
        'path': 'File system path.',
        'filepath': 'Path to file.',
        'filename': 'Name of file.',
        'url': 'URL to access.',
        'host': 'Host name or address.',
        'port': 'Port number.',
        'timeout': 'Timeout in seconds.',
        'config': 'Configuration settings.',
        'options': 'Optional parameters.',
        'params': 'Parameters dictionary.',
        'kwargs': 'Additional keyword arguments.',
        'args': 'Positional arguments.',
        'message': 'Message content.',
        'error': 'Error information.',
        'exception': 'Exception instance.',
        'callback': 'Callback function.',
        'handler': 'Handler function.',
        'name': 'Name identifier.',
        'id': 'Unique identifier.',
        'size': 'Size parameter.',
        'count': 'Number of items.',
        'limit': 'Maximum limit.',
        'offset': 'Starting offset.',
        'verbose': 'Enable verbose output.',
        'debug': 'Enable debug mode.',
        'force': 'Force operation.',
        'recursive': 'Apply recursively.',
        'encoding': 'Character encoding.',
        'format': 'Output format.',
        'mode': 'Operation mode.',
        'dtype': 'Data type.',
        'shape': 'Array shape.',
        'axis': 'Axis to operate on.',
        'dim': 'Dimension parameter.',
    }

    # Check exact match
    if arg in arg_patterns:
        return arg_patterns[arg]

    # Check patterns
    if arg.endswith('_path'):
        return f"Path to {arg[:-5].replace('_', ' ')}."
    elif arg.endswith('_file'):
        return f"File containing {arg[:-5].replace('_', ' ')}."
    elif arg.endswith('_dir') or arg.endswith('_directory'):
        return f"Directory for {arg.split('_')[0]}."
    elif arg.endswith('_list'):
        return f"List of {arg[:-5].replace('_', ' ')} items."
    elif arg.endswith('_dict'):
        return f"Dictionary of {arg[:-5].replace('_', ' ')} mappings."
    elif arg.endswith('_set'):
        return f"Set of {arg[:-4].replace('_', ' ')} items."
    elif arg.startswith('is_'):
        return f"Whether {arg[3:].replace('_', ' ')}."
    elif arg.startswith('has_'):
        return f"Whether has {arg[4:].replace('_', ' ')}."
    elif arg.startswith('use_'):
        return f"Whether to use {arg[4:].replace('_', ' ')}."
    elif arg.startswith('enable_'):
        return f"Whether to enable {arg[7:].replace('_', ' ')}."
    elif arg.startswith('num_'):
        return f"Number of {arg[4:].replace('_', ' ')}."
    elif arg.startswith('max_'):
        return f"Maximum {arg[4:].replace('_', ' ')}."
    elif arg.startswith('min_'):
        return f"Minimum {arg[4:].replace('_', ' ')}."

    # Context-specific
    if 'hypervector' in func_name.lower() or 'hv' in arg.lower():
        if 'dim' in arg:
            return "Hypervector dimension."
        elif 'vector' in arg:
            return "Hypervector data."
    elif 'proof' in func_name.lower() or 'zk' in func_name.lower():
        if 'witness' in arg:
            return "Witness data for proof."
        elif 'statement' in arg:
            return "Public statement to prove."

    # Default
    return f"{arg.replace('_', ' ').capitalize()} parameter."


def generate_return_description(return_type: str, func_name: str) -> str:
    """Generate return value description.

    Args:
        return_type: Return type annotation.
        func_name: Function name for context.

    Returns:
        Return value description.
    """
    if 'bool' in return_type:
        if func_name.startswith('is_') or func_name.startswith('has_'):
            return "bool: True if condition met, False otherwise."
        return "bool: Operation success status."
    elif 'str' in return_type:
        return "str: Result string."
    elif 'int' in return_type:
        return "int: Computed integer value."
    elif 'float' in return_type:
        return "float: Computed floating point value."
    elif 'List' in return_type or 'list' in return_type:
        return "List: Result list."
    elif 'Dict' in return_type or 'dict' in return_type:
        return "Dict: Result dictionary."
    elif 'Tuple' in return_type or 'tuple' in return_type:
        return "Tuple: Result values."
    elif 'None' in return_type:
        return "None"
    elif 'Optional' in return_type:
        return "Optional result value, None if not found."
    else:
        return f"{return_type}: Result instance."


def infer_return_type(func_name: str, analysis: Dict[str, Any]) -> str:
    """Infer return type from function name and analysis.

    Args:
        func_name: Function name.
        analysis: Function body analysis.

    Returns:
        Inferred return type description.
    """
    if func_name.startswith('is_') or func_name.startswith('has_') or func_name.startswith('can_'):
        return "bool: True if condition is satisfied, False otherwise."
    elif func_name.startswith('get_'):
        return "Retrieved value."
    elif func_name.startswith('create_') or func_name.startswith('build_'):
        return "Created instance."
    elif func_name.startswith('calculate_') or func_name.startswith('compute_'):
        return "Computed result."
    elif analysis.get('is_generator'):
        return "Generator yielding results."
    elif analysis.get('returns_value'):
        return "Processed result."
    else:
        return "None"


def generate_exception_description(exc_name: str) -> str:
    """Generate description for an exception.

    Args:
        exc_name: Exception class name.

    Returns:
        Exception description.
    """
    exceptions = {
        'ValueError': 'If input values are invalid.',
        'TypeError': 'If input types are incorrect.',
        'KeyError': 'If required key is not found.',
        'IndexError': 'If index is out of range.',
        'AttributeError': 'If attribute is not found.',
        'NotImplementedError': 'If feature is not implemented.',
        'RuntimeError': 'If runtime error occurs.',
        'IOError': 'If I/O operation fails.',
        'FileNotFoundError': 'If file is not found.',
        'PermissionError': 'If permission is denied.',
        'ConnectionError': 'If connection fails.',
        'TimeoutError': 'If operation times out.',
        'AssertionError': 'If assertion fails.',
        'ImportError': 'If import fails.',
        'MemoryError': 'If out of memory.',
        'OSError': 'If OS error occurs.',
        'StopIteration': 'When iteration completes.',
        'GeneratorExit': 'When generator is closed.',
        'KeyboardInterrupt': 'If interrupted by user.',
        'SystemExit': 'If system exit is triggered.',
        'Exception': 'If an error occurs.',
    }

    return exceptions.get(exc_name, f"If {exc_name} occurs.")


def generate_class_docstring_comprehensive(name: str, filepath: Optional[str]) -> str:
    """Generate comprehensive class docstring.

    Args:
        name: Class name.
        filepath: File path for context.

    Returns:
        Class docstring.
    """
    # Determine class type from name and context
    if 'Error' in name or 'Exception' in name:
        base = name.replace('Error', '').replace('Exception', '')
        return f'"""Exception raised when {base.lower()} operation fails."""'

    # Context from filepath
    if filepath:
        if 'hypervector' in filepath or 'hv' in filepath:
            if 'Encoder' in name:
                return f'"""Encoder for converting data to hypervector representation."""'
            elif 'Decoder' in name:
                return f'"""Decoder for converting hypervectors back to original format."""'
            elif 'Engine' in name:
                return f'"""Engine for hypervector operations and computations."""'
        elif 'zk' in filepath or 'proof' in filepath:
            if 'Circuit' in name:
                return f'"""Zero-knowledge circuit for {name.replace("Circuit", "")} proofs."""'
            elif 'Prover' in name:
                return f'"""Prover for generating {name.replace("Prover", "")} proofs."""'
            elif 'Verifier' in name:
                return f'"""Verifier for validating {name.replace("Verifier", "")} proofs."""'
        elif 'crypto' in filepath:
            if 'Key' in name:
                return f'"""Cryptographic key management for {name.replace("Key", "")}."""'
            elif 'Cipher' in name:
                return f'"""Cipher implementation for {name.replace("Cipher", "")} encryption."""'
        elif 'test' in filepath.lower():
            return f'"""Test suite for {name.replace("Test", "")} functionality."""'

    # Pattern-based
    patterns = {
        'Manager': 'Manages {} resources and operations.',
        'Handler': 'Handles {} events and processing.',
        'Controller': 'Controls {} logic and flow.',
        'Service': 'Provides {} services.',
        'Client': 'Client for {} interactions.',
        'Server': 'Server for {} operations.',
        'Factory': 'Factory for creating {} instances.',
        'Builder': 'Builder for constructing {} objects.',
        'Parser': 'Parses {} data formats.',
        'Validator': 'Validates {} according to rules.',
        'Serializer': 'Serializes {} data structures.',
        'Adapter': 'Adapts {} interfaces.',
        'Strategy': 'Strategy pattern for {} algorithms.',
        'Observer': 'Observer for {} events.',
        'Decorator': 'Decorator for {} enhancements.',
        'Repository': 'Repository for {} data access.',
        'Model': 'Model representing {} data.',
        'View': 'View for {} presentation.',
        'Config': 'Configuration for {} settings.',
        'Engine': 'Engine for {} processing.',
        'Pipeline': 'Pipeline for {} data flow.',
        'Processor': 'Processes {} data.',
        'Generator': 'Generates {} output.',
        'Converter': 'Converts between {} formats.',
        'Transformer': 'Transforms {} data.',
    }

    for pattern, template in patterns.items():
        if pattern in name:
            subject = name.replace(pattern, '').strip()
            if not subject:
                subject = 'related'
            return f'"""{template.format(subject.lower())}"""'

    return f'"""{name} implementation."""'


def add_docstrings_to_file(filepath: str) -> int:
    """Add comprehensive docstrings to a Python file.

    Args:
        filepath: Path to the Python file.

    Returns:
        Number of docstrings added.
    """
    try:
        with open(filepath, 'r') as f:
            source = f.read()
            lines = source.splitlines()

        tree = ast.parse(source)

        # Track what needs docstrings
        missing = []

        class DocstringVisitor(ast.NodeVisitor):
            def __init__(self):
                self.current_class = None

            def visit_ClassDef(self, node):
                if not ast.get_docstring(node):
                    missing.append({
                        'type': 'class',
                        'node': node,
                        'name': node.name,
                        'line': node.lineno
                    })
                old_class = self.current_class
                self.current_class = node.name
                self.generic_visit(node)
                self.current_class = old_class

            def visit_FunctionDef(self, node):
                if not ast.get_docstring(node):
                    analysis = analyze_function_body(node)
                    missing.append({
                        'type': 'function',
                        'node': node,
                        'name': node.name,
                        'line': node.lineno,
                        'class': self.current_class,
                        'analysis': analysis
                    })
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node):
                if not ast.get_docstring(node):
                    analysis = analyze_function_body(node)
                    missing.append({
                        'type': 'async_function',
                        'node': node,
                        'name': node.name,
                        'line': node.lineno,
                        'class': self.current_class,
                        'analysis': analysis
                    })
                self.generic_visit(node)

        visitor = DocstringVisitor()
        visitor.visit(tree)

        if not missing:
            return 0

        # Sort by line number in reverse
        missing.sort(key=lambda x: x['line'], reverse=True)

        # Add docstrings
        modified_lines = lines.copy()
        added = 0

        for item in missing:
            node = item['node']
            line_idx = item['line'] - 1

            # Generate docstring
            if item['type'] == 'class':
                docstring = generate_class_docstring_comprehensive(item['name'], filepath)
            else:
                args = [arg.arg for arg in node.args.args]
                returns = ast.unparse(node.returns) if node.returns and \
                    hasattr(ast, 'unparse') else None
                is_method = args and args[0] in ('self', 'cls')
                is_async = item['type'] == 'async_function'

                docstring = generate_comprehensive_docstring(
                    item['name'],
                    args,
                    returns,
                    is_method,
                    is_async,
                    item['type'],
                    item.get('analysis', {}),
                    item.get('class'),
                    filepath
                )

            # Find indentation
            def_line = modified_lines[line_idx]
            indent = len(def_line) - len(def_line.lstrip())

            # Find where to insert (after the def line)
            insert_idx = line_idx + 1

            # Skip past multi-line function signature
            while insert_idx < len(modified_lines) and ':' not in modified_lines[insert_idx - 1]:
                insert_idx += 1

            # Add indented docstring
            docstring_lines = docstring.split('\n')
            for i, doc_line in enumerate(docstring_lines):
                if doc_line:
                    indented_line = ' ' * (indent + 4) + doc_line
                else:
                    indented_line = ''
                modified_lines.insert(insert_idx + i, indented_line)

            added += 1

        # Write back
        if added > 0:
            with open(filepath, 'w') as f:
                f.write('\n'.join(modified_lines))

        return added

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0


def main():
    """Main function to add comprehensive docstrings."""
    print("=" * 70)
    print("ADDING COMPREHENSIVE DOCSTRINGS")
    print("=" * 70)

    # Priority files to process
    priority_files = [
        'genomevault/hypervector/engine.py',
        'genomevault/zk_proofs/core/prover.py',
        'genomevault/crypto/keys.py',
        'genomevault/api/main.py',
        'genomevault/federated/server.py',
    ]

    total_added = 0

    # Process priority files first
    for filepath in priority_files:
        if os.path.exists(filepath):
            print(f"Processing {filepath}...")
            added = add_docstrings_to_file(filepath)
            if added > 0:
                print(f"  Added {added} docstrings")
                total_added += added

    # Process remaining files in key directories
    directories = ['genomevault', 'tests', 'examples']
    files_processed = 0
    max_files = 100  # Limit to avoid timeout

    for directory in directories:
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                dirs[:] = [d for d in dirs if d not in {'.venv', '__pycache__', '.git'}]

                for file in files:
                    if file.endswith('.py') and files_processed < max_files:
                        filepath = os.path.join(root, file)
                        if filepath not in priority_files:
                            added = add_docstrings_to_file(filepath)
                            if added > 0:
                                print(f"Added {added} docstrings to {filepath}")
                                total_added += added
                                files_processed += 1

    print("\n" + "=" * 70)
    print(f"Total comprehensive docstrings added: {total_added}")
    print("=" * 70)


if __name__ == "__main__":
    main()
