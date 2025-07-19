#!/usr/bin/env python3
"""
Deep tail-chasing cleanup script
Recursively finds all import mismatches and fixes them
"""

import os
import ast
import re
from pathlib import Path
from typing import Set, Dict, List, Tuple

class TailChasingDetector:
    def __init__(self, root_path: Path):
        self.root = root_path
        self.all_definitions = {}  # module_path -> set of defined names
        self.all_imports = {}      # module_path -> set of imported names
        self.phantom_imports = {}  # module_path -> set of non-existent imports
        
    def scan_module(self, filepath: Path) -> Set[str]:
        """Extract all definitions from a module"""
        definitions = set()
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    definitions.add(node.name)
                elif isinstance(node, ast.FunctionDef):
                    if not node.name.startswith('_'):
                        definitions.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            definitions.add(target.id)
        except Exception as e:
            print(f"Error scanning {filepath}: {e}")
            
        return definitions
    
    def find_imports(self, filepath: Path) -> List[Tuple[str, List[str]]]:
        """Find all imports in a file"""
        imports = []
        
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and node.module.startswith('.'):
                        # Relative import
                        module = node.module
                        items = [alias.name for alias in node.names]
                        imports.append((module, items))
                    elif node.level > 0:
                        # from . import x
                        module = '.' * node.level
                        items = [alias.name for alias in node.names]
                        imports.append((module, items))
        except Exception as e:
            print(f"Error finding imports in {filepath}: {e}")
            
        return imports
    
    def resolve_relative_import(self, from_file: Path, import_path: str) -> Path:
        """Resolve a relative import to an absolute path"""
        from_dir = from_file.parent
        
        if import_path == '.':
            return from_dir
        elif import_path.startswith('..'):
            levels = import_path.count('.')
            target = from_dir
            for _ in range(levels):
                target = target.parent
            rest = import_path.lstrip('.')
            if rest:
                for part in rest.split('.'):
                    target = target / part
            return target
        else:
            # .module
            module_name = import_path[1:]  # Remove leading dot
            return from_dir / module_name
    
    def scan_all(self):
        """Scan entire codebase"""
        print("🔍 Scanning codebase for tail-chasing imports...")
        
        for py_file in self.root.rglob("*.py"):
            if '__pycache__' in str(py_file):
                continue
                
            rel_path = py_file.relative_to(self.root)
            
            # Get definitions
            definitions = self.scan_module(py_file)
            self.all_definitions[str(rel_path)] = definitions
            
            # Get imports
            imports = self.find_imports(py_file)
            self.all_imports[str(rel_path)] = imports
            
    def find_phantom_imports(self):
        """Find all imports that don't exist"""
        print("\n🔍 Finding phantom imports...")
        
        for file_path, imports in self.all_imports.items():
            file_full_path = self.root / file_path
            phantoms = []
            
            for module_path, items in imports:
                if module_path.startswith('.'):
                    # Resolve relative import
                    try:
                        target_path = self.resolve_relative_import(file_full_path, module_path)
                        
                        # Check if it's a module file
                        if target_path.suffix == '':
                            target_file = target_path.with_suffix('.py')
                        else:
                            target_file = target_path
                        
                        # Get relative path
                        if target_file.exists():
                            rel_target = target_file.relative_to(self.root)
                            target_defs = self.all_definitions.get(str(rel_target), set())
                            
                            for item in items:
                                if item not in target_defs:
                                    phantoms.append((module_path, item))
                                    
                    except Exception as e:
                        print(f"  Error resolving {module_path} from {file_path}: {e}")
                        
            if phantoms:
                self.phantom_imports[file_path] = phantoms
                
    def generate_fixes(self):
        """Generate fixes for phantom imports"""
        print("\n🔧 Generating fixes...")
        
        fixes = {}
        
        for file_path, phantoms in self.phantom_imports.items():
            print(f"\n📄 {file_path} has {len(phantoms)} phantom imports:")
            for module, item in phantoms:
                print(f"   ❌ from {module} import {item}")
            
            # Read the file
            full_path = self.root / file_path
            with open(full_path, 'r') as f:
                content = f.read()
            
            # Remove phantom imports
            for module, item in phantoms:
                # Try different import patterns
                patterns = [
                    rf'from {re.escape(module)} import .*{re.escape(item)}.*\n',
                    rf'{re.escape(item)},?\s*\n?',
                ]
                
                for pattern in patterns:
                    content = re.sub(pattern, '', content)
            
            # Clean up empty imports
            content = re.sub(r'from [.\w]+ import \(\s*\)', '', content)
            content = re.sub(r'\n\n\n+', '\n\n', content)
            
            fixes[full_path] = content
            
        return fixes
    
    def apply_fixes(self, fixes: Dict[Path, str], dry_run: bool = True):
        """Apply the fixes"""
        if dry_run:
            print("\n🔍 DRY RUN - Changes that would be made:")
        else:
            print("\n✍️  Applying fixes...")
            
        for filepath, new_content in fixes.items():
            if dry_run:
                print(f"\n📝 Would fix {filepath.relative_to(self.root)}")
            else:
                with open(filepath, 'w') as f:
                    f.write(new_content)
                print(f"✅ Fixed {filepath.relative_to(self.root)}")

def main():
    print("🌀 Deep Tail-Chasing Cleanup 🌀")
    print("=" * 60)
    
    root = Path.cwd()
    detector = TailChasingDetector(root)
    
    # Scan everything
    detector.scan_all()
    print(f"\n📊 Scanned {len(detector.all_definitions)} modules")
    
    # Find phantom imports
    detector.find_phantom_imports()
    
    if detector.phantom_imports:
        print(f"\n⚠️  Found phantom imports in {len(detector.phantom_imports)} files")
        
        # Generate fixes
        fixes = detector.generate_fixes()
        
        # Apply fixes (dry run first)
        detector.apply_fixes(fixes, dry_run=True)
        
        response = input("\n🤔 Apply these fixes? (y/n): ")
        if response.lower() == 'y':
            detector.apply_fixes(fixes, dry_run=False)
            print("\n✅ Tail-chasing cleanup complete!")
        else:
            print("\n👍 Dry run complete, no changes made")
    else:
        print("\n✅ No phantom imports found!")
    
    # Now fix __init__.py files specifically
    print("\n" + "=" * 60)
    print("🔧 Fixing __init__.py files...")
    
    init_files = list(root.rglob("*/__init__.py"))
    for init_file in init_files:
        if '__pycache__' in str(init_file):
            continue
            
        fix_init_file(init_file, detector.all_definitions)

def fix_init_file(init_path: Path, all_definitions: Dict[str, Set[str]]):
    """Fix a specific __init__.py file"""
    package_path = init_path.parent
    package_name = package_path.name
    
    if package_name in ['__pycache__', '.git', 'tests']:
        return
        
    print(f"\n📦 Fixing {package_name}/__init__.py")
    
    # Find all modules in this package
    module_exports = {}
    
    for py_file in package_path.glob("*.py"):
        if py_file.name == '__init__.py' or py_file.name.startswith('_'):
            continue
            
        rel_path = py_file.relative_to(Path.cwd())
        module_name = py_file.stem
        
        if str(rel_path) in all_definitions:
            exports = all_definitions[str(rel_path)]
            if exports:
                module_exports[module_name] = exports
    
    if not module_exports:
        print(f"  ⚠️  No exports found")
        return
        
    # Generate new __init__.py content
    content = f'''"""
{package_name.replace('_', ' ').title()} Package
"""

'''
    
    # Add imports
    for module, exports in sorted(module_exports.items()):
        export_list = sorted(list(exports))
        # Skip very large modules
        if len(export_list) > 15:
            content += f"# Too many exports in {module}, import module directly\n"
            content += f"from . import {module}\n\n"
        else:
            content += f"from .{module} import (\n"
            for item in export_list:
                content += f"    {item},\n"
            content += ")\n\n"
    
    # Generate __all__
    content += "__all__ = [\n"
    for module, exports in sorted(module_exports.items()):
        export_list = sorted(list(exports))
        if len(export_list) > 15:
            content += f"    '{module}',\n"
        else:
            for item in export_list:
                content += f"    '{item}',\n"
    content += "]\n"
    
    # Write the file
    with open(init_path, 'w') as f:
        f.write(content)
    
    print(f"  ✅ Fixed with {sum(len(e) for e in module_exports.values())} exports")

if __name__ == "__main__":
    main()
