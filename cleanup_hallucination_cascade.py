#!/usr/bin/env python3
"""
Comprehensive cleanup script to fix all __init__.py files 
to match actual module contents, breaking the hallucination cascade
"""

import os
import ast
import re
from pathlib import Path

def extract_module_exports(filepath):
    """Extract all classes, functions, and constants from a Python file"""
    exports = {
        'classes': [],
        'functions': [],
        'constants': [],
        'imports': []
    }
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Parse the AST
        tree = ast.parse(content)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                exports['classes'].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                # Skip private functions
                if not node.name.startswith('_'):
                    exports['functions'].append(node.name)
            elif isinstance(node, ast.Assign):
                # Look for constant assignments
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        exports['constants'].append(target.id)
        
        return exports
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return exports

def fix_init_file(package_path):
    """Fix __init__.py for a package based on actual module contents"""
    init_path = package_path / "__init__.py"
    
    print(f"\n{'='*60}")
    print(f"Fixing {package_path.name}/__init__.py")
    print(f"{'='*60}")
    
    # Scan all Python files in the package
    all_exports = {}
    for py_file in package_path.glob("*.py"):
        if py_file.name == "__init__.py" or py_file.name.startswith('_'):
            continue
        
        module_name = py_file.stem
        exports = extract_module_exports(py_file)
        if any(exports.values()):
            all_exports[module_name] = exports
            print(f"\n{module_name}.py exports:")
            for category, items in exports.items():
                if items:
                    print(f"  {category}: {', '.join(items[:5])}{'...' if len(items) > 5 else ''}")
    
    # Generate new __init__.py content
    new_content = f'''"""
{package_path.name.replace('_', ' ').title()} Package
Auto-generated to match actual module contents
"""

'''
    
    # Generate imports
    for module_name, exports in all_exports.items():
        items_to_import = exports['classes'] + exports['functions']
        if items_to_import:
            # Limit imports to prevent huge lines
            if len(items_to_import) > 10:
                new_content += f"from . import {module_name}\n"
            else:
                new_content += f"from .{module_name} import (\n"
                for item in items_to_import:
                    new_content += f"    {item},\n"
                new_content += ")\n"
        new_content += "\n"
    
    # Generate __all__
    new_content += "__all__ = [\n"
    for module_name, exports in all_exports.items():
        items = exports['classes'] + exports['functions']
        if len(items) > 10:
            new_content += f"    # {module_name} module\n"
            new_content += f"    '{module_name}',\n"
        else:
            for item in items:
                new_content += f"    '{item}',\n"
    new_content += "]\n"
    
    # Write the new __init__.py
    with open(init_path, 'w') as f:
        f.write(new_content)
    
    print(f"\n✅ Fixed {init_path}")

def main():
    """Fix all package __init__.py files"""
    print("🧹 Comprehensive __init__.py cleanup")
    print("Breaking the hallucination cascade!")
    
    # Get the project root
    project_root = Path.cwd()
    
    # List of packages to fix
    packages = [
        'core',
        'utils',
        'local_processing',
        'hypervector_transform',
        'zk_proofs',
        'pir',
        'blockchain',
        'api',
        'advanced_analysis',
    ]
    
    for package_name in packages:
        package_path = project_root / package_name
        if package_path.exists() and package_path.is_dir():
            fix_init_file(package_path)
        else:
            print(f"\n⚠️  Package {package_name} not found, skipping...")
    
    print("\n" + "="*60)
    print("✅ Cleanup complete!")
    print("="*60)
    
    # Test imports
    print("\n🧪 Testing basic imports...")
    try:
        from core.config import get_config
        print("✅ core.config imports work")
        
        from utils import get_logger
        print("✅ utils imports work")
        
        print("\n✅ Basic imports successful!")
    except Exception as e:
        print(f"\n❌ Import test failed: {e}")

if __name__ == "__main__":
    main()
