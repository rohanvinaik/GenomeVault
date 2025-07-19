"""Fix import issues in the project."""
import os
import sys
from pathlib import Path

# Change to genomevault directory
os.chdir('/Users/rohanvinaik/genomevault')

# Fix 1: Update core/config.py to use pydantic v2
config_file = Path('core/config.py')
if config_file.exists():
    content = config_file.read_text()
    # Replace BaseSettings import for Pydantic v2
    content = content.replace(
        'from pydantic import BaseSettings, Field, validator',
        'from pydantic import Field\nfrom pydantic_settings import BaseSettings\nfrom pydantic import field_validator'
    )
    content = content.replace('@validator', '@field_validator')
    config_file.write_text(content)
    print("✅ Fixed pydantic imports in core/config.py")

# Fix 2: Create a proper setup.py if it doesn't exist or update it
setup_content = '''from setuptools import setup, find_packages

setup(
    name="genomevault",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "."},
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
    ],
)
'''
Path('setup.py').write_text(setup_content)
print("✅ Updated setup.py")

# Fix 3: Install the package in development mode
os.system('pip install -e .')
print("✅ Installed genomevault in development mode")

# Fix 4: Update conftest.py to properly set up the path
conftest_content = '''"""Pytest configuration and fixtures."""
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# This allows imports like "from genomevault.x import y" to work
import genomevault
'''
Path('tests/conftest.py').write_text(conftest_content)
print("✅ Updated tests/conftest.py")

# Fix 5: Fix relative imports in local_processing/epigenetics.py
epigenetics_file = Path('local_processing/epigenetics.py')
if epigenetics_file.exists():
    content = epigenetics_file.read_text()
    # Change relative import to absolute
    content = content.replace(
        'from ..core.config import get_config',
        'from core.config import get_config'
    )
    epigenetics_file.write_text(content)
    print("✅ Fixed imports in local_processing/epigenetics.py")

# Fix 6: Fix imports in hypervector_transform/binding.py
binding_file = Path('hypervector_transform/binding.py')
if binding_file.exists():
    content = binding_file.read_text()
    # Change to absolute import
    content = content.replace(
        'from core.exceptions import BindingError',
        'from genomevault.core.exceptions import BindingError'
    )
    binding_file.write_text(content)
    print("✅ Fixed imports in hypervector_transform/binding.py")

print("\n🔧 Now installing pydantic-settings...")
os.system('pip install pydantic-settings')

print("\n✅ All fixes applied! Try running pytest again.")
