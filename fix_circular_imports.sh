#!/bin/bash
# Fix circular import dependencies in GenomeVault

echo "🔧 Fixing circular import dependencies in GenomeVault..."
echo "=================================================="

cd /Users/rohanvinaik/genomevault

# Step 1: Check what's actually in core/config.py
echo "📋 Step 1: Checking core/config.py for base_credits..."
if grep -q "base_credits" core/config.py; then
    echo "✓ Found base_credits in config.py"
    BASE_CREDITS_NAME=$(grep -E "(base_credits|BASE_CREDITS)" core/config.py | grep -v "^#" | head -1 | awk -F'=' '{print $1}' | xargs)
    echo "  Actual name: $BASE_CREDITS_NAME"
else
    echo "✗ base_credits not found in config.py"
    BASE_CREDITS_NAME=""
fi

# Step 2: Fix core/__init__.py
echo -e "\n📋 Step 2: Fixing core/__init__.py..."
cat > core/__init__.py << 'EOF'
"""Core functionality for GenomeVault."""

from .config import Config, get_config

__all__ = [
    "Config",
    "get_config",
]
EOF
echo "✓ Fixed core/__init__.py"

# Step 3: Fix utils/__init__.py to avoid circular imports
echo -e "\n📋 Step 3: Fixing utils/__init__.py..."
cat > utils/__init__.py << 'EOF'
"""Utility functions for GenomeVault."""

# Import specific utilities without importing from core
# This breaks the circular dependency chain

__all__ = []

# Try to import each utility module
try:
    from .encryption import AESGCMCipher
    __all__.append("AESGCMCipher")
except ImportError:
    pass

try:
    from .hashing import secure_hash
    __all__.append("secure_hash")
except ImportError:
    pass

try:
    from .logging import get_logger
    __all__.append("get_logger")
except ImportError:
    pass

# For config access, users should import directly:
# from core.config import get_config
EOF
echo "✓ Fixed utils/__init__.py"

# Step 4: Fix root __init__.py to use lazy loading
echo -e "\n📋 Step 4: Fixing root __init__.py..."
cat > __init__.py << 'EOF'
"""GenomeVault - Privacy-preserving genomic analysis platform."""

__version__ = "0.1.0"

# Avoid circular imports by not importing submodules at package level
# Users should import directly from submodules:
# from genomevault.local_processing import SequencingPipeline
# from genomevault.core.config import get_config

__all__ = ["__version__"]
EOF
echo "✓ Fixed root __init__.py"

# Step 5: Fix local_processing/__init__.py
echo -e "\n📋 Step 5: Fixing local_processing/__init__.py..."
cat > local_processing/__init__.py << 'EOF'
"""Local processing modules for GenomeVault."""

# Use lazy imports to avoid circular dependencies
__all__ = []

# Define what should be available
_modules = {
    "SequencingPipeline": ".sequencing",
    "TranscriptomicsProcessor": ".transcriptomics",
    "EpigeneticsAnalyzer": ".epigenetics",
    "ProteomicsHandler": ".proteomics",
    "PhenotypeIntegrator": ".phenotypes",
}

def __getattr__(name):
    """Lazy import implementation."""
    if name in _modules:
        module_path = _modules[name]
        from importlib import import_module
        module = import_module(module_path, package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return list(_modules.keys())
EOF
echo "✓ Fixed local_processing/__init__.py"

# Step 6: Create a test script to verify imports
echo -e "\n📋 Step 6: Creating import test script..."
cat > test_imports.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify all imports are working correctly."""

import sys
import traceback

def test_import(import_statement, description):
    """Test a single import statement."""
    print(f"\n🧪 Testing: {description}")
    print(f"   Import: {import_statement}")
    try:
        exec(import_statement)
        print("   ✅ Success!")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {type(e).__name__}: {e}")
        if "--verbose" in sys.argv:
            traceback.print_exc()
        return False

def main():
    """Run all import tests."""
    print("=" * 60)
    print("GenomeVault Import Tests")
    print("=" * 60)
    
    tests = [
        ("import genomevault", "Package import"),
        ("from core.config import Config, get_config", "Core config imports"),
        ("from utils.encryption import AESGCMCipher", "Utils encryption import"),
        ("from utils.logging import get_logger", "Utils logging import"),
        ("from utils.hashing import secure_hash", "Utils hashing import"),
        ("from local_processing.sequencing import SequencingPipeline", "Local processing import"),
        ("import hypervector_transform", "Hypervector module"),
        ("import zk_proofs", "ZK proofs module"),
        ("import pir", "PIR module"),
        ("import blockchain", "Blockchain module"),
        ("import api", "API module"),
    ]
    
    passed = 0
    failed = 0
    
    for import_stmt, desc in tests:
        if test_import(import_stmt, desc):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
EOF
chmod +x test_imports.py
echo "✓ Created test_imports.py"

# Step 7: Fix imports in key files that might have issues
echo -e "\n📋 Step 7: Fixing imports in key modules..."

# Fix sequencing.py imports
if [ -f "local_processing/sequencing.py" ]; then
    echo "   Fixing local_processing/sequencing.py..."
    # Create a backup
    cp local_processing/sequencing.py local_processing/sequencing.py.bak
    
    # Fix the imports at the top of the file
    python3 << 'PYTHON_FIX'
import re

with open('local_processing/sequencing.py', 'r') as f:
    content = f.read()

# Fix the import section
old_import = "from utils import AESGCMCipher, get_config, get_logger, secure_hash"
new_imports = """from utils.encryption import AESGCMCipher
from utils.logging import get_logger
from utils.hashing import secure_hash
from core.config import get_config"""

content = content.replace(old_import, new_imports)

# Also fix any other problematic imports
content = re.sub(r'from utils import get_config', 'from core.config import get_config', content)

with open('local_processing/sequencing.py', 'w') as f:
    f.write(content)

print("   ✓ Fixed imports in sequencing.py")
PYTHON_FIX
fi

# Step 8: Create a circular dependency detector
echo -e "\n📋 Step 8: Creating circular dependency detector..."
cat > detect_circular_imports.py << 'EOF'
#!/usr/bin/env python3
"""Detect circular imports in the GenomeVault project."""

import ast
import os
from pathlib import Path
from collections import defaultdict

def find_imports(filepath):
    """Extract imports from a Python file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        tree = ast.parse(content)
    except:
        return []
    
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    
    return imports

def find_cycles(graph):
    """Find cycles in a directed graph using DFS."""
    def dfs(node, path, visited):
        if node in path:
            cycle_start = path.index(node)
            return [path[cycle_start:] + [node]]
        
        if node in visited:
            return []
        
        visited.add(node)
        cycles = []
        
        for neighbor in graph.get(node, []):
            cycles.extend(dfs(neighbor, path + [node], visited))
        
        return cycles
    
    all_cycles = []
    visited = set()
    
    for node in graph:
        if node not in visited:
            cycles = dfs(node, [], visited)
            all_cycles.extend(cycles)
    
    # Remove duplicate cycles
    unique_cycles = []
    for cycle in all_cycles:
        normalized = tuple(sorted(cycle))
        if normalized not in [tuple(sorted(c)) for c in unique_cycles]:
            unique_cycles.append(cycle)
    
    return unique_cycles

def main():
    """Build dependency graph and find circular imports."""
    print("🔍 Scanning for circular imports...")
    print("=" * 60)
    
    # Build dependency graph
    graph = defaultdict(list)
    module_to_file = {}
    
    for py_file in Path('.').rglob('*.py'):
        if any(skip in str(py_file) for skip in ['test', '__pycache__', '.bak', 'build', 'dist']):
            continue
            
        # Convert file path to module name
        module = str(py_file).replace('/', '.').replace('\\', '.').replace('.py', '')
        if module.startswith('.'):
            module = module[1:]
        
        module_to_file[module] = py_file
        imports = find_imports(py_file)
        
        for imp in imports:
            # Only track internal imports
            if any(imp.startswith(prefix) for prefix in ['core', 'utils', 'local_processing', 'hypervector', 'zk_proofs', 'pir', 'blockchain', 'api']):
                graph[module].append(imp)
    
    # Find cycles
    cycles = find_cycles(graph)
    
    if cycles:
        print("❌ Circular dependencies found:")
        for i, cycle in enumerate(cycles, 1):
            print(f"\nCycle {i}:")
            for j, module in enumerate(cycle):
                print(f"  {'└─>' if j == len(cycle)-1 else '├─>'} {module}")
                if module in module_to_file:
                    print(f"  {'   ' if j == len(cycle)-1 else '│  '}    File: {module_to_file[module]}")
    else:
        print("✅ No circular dependencies found!")
    
    print("\n" + "=" * 60)
    return len(cycles)

if __name__ == "__main__":
    import sys
    sys.exit(main())
EOF
chmod +x detect_circular_imports.py
echo "✓ Created detect_circular_imports.py"

# Step 9: Run the tests
echo -e "\n📋 Step 9: Running import tests..."
echo "=================================================="
python3 test_imports.py

echo -e "\n📋 Step 10: Checking for circular dependencies..."
echo "=================================================="
python3 detect_circular_imports.py

echo -e "\n📋 Step 11: Running pytest..."
echo "=================================================="
python3 -m pytest tests/test_simple.py -v --tb=short

echo -e "\n✅ Import fix process complete!"
echo "=================================================="
echo ""
echo "🔍 Summary of changes:"
echo "   - Fixed core/__init__.py to remove non-existent base_credits import"
echo "   - Updated utils/__init__.py to avoid circular imports from core"
echo "   - Modified root __init__.py to use minimal imports"
echo "   - Created lazy loading for local_processing module"
echo "   - Fixed import statements in sequencing.py"
echo ""
echo "📚 Next steps:"
echo "   1. Review the test results above"
echo "   2. If there are still errors, check the specific module imports"
echo "   3. Use detect_circular_imports.py to find any remaining cycles"
echo "   4. Run: python3 test_imports.py --verbose for detailed error traces"
