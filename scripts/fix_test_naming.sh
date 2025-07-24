#!/bin/bash
# fix_test_naming.sh
# Script to rename test files to follow lane-specific naming convention

echo "Starting test file renaming to follow lane-specific naming convention..."

# Function to rename file if it exists
rename_if_exists() {
    local old_name=$1
    local new_name=$2
    local dir=$3
    
    if [ -f "$dir/$old_name" ]; then
        mv "$dir/$old_name" "$dir/$new_name"
        echo "  ✓ Renamed: $old_name → $new_name"
    else
        echo "  - Skipped: $old_name (not found)"
    fi
}

# Fix unit tests
echo -e "\nFixing unit test names..."
rename_if_exists "test_hypervector.py" "test_hdc_hypervector.py" "tests/unit"
rename_if_exists "test_hypervector_encoding.py" "test_hdc_hypervector_encoding.py" "tests/unit"
rename_if_exists "test_pir.py" "test_pir_basic.py" "tests/unit"

# Fix e2e tests
echo -e "\nFixing e2e test names..."
rename_if_exists "test_pir_integration.py" "test_pir_e2e.py" "tests/e2e"
rename_if_exists "test_zk_integration.py" "test_zk_e2e.py" "tests/e2e"

# Fix zk tests
echo -e "\nFixing zk test names..."
rename_if_exists "test_property_circuits.py" "test_zk_property_circuits.py" "tests/zk"

# Update any imports that might reference the old names
echo -e "\nChecking for import references to update..."

# Function to update imports
update_imports() {
    local old_module=$1
    local new_module=$2
    
    # Find all Python files and update imports
    find . -name "*.py" -type f -not -path "./.git/*" -not -path "./.pytest_cache/*" | while read -r file; do
        if grep -q "$old_module" "$file"; then
            sed -i '' "s/$old_module/$new_module/g" "$file"
            echo "  ✓ Updated imports in: $file"
        fi
    done
}

# Update imports for renamed modules
update_imports "test_hypervector" "test_hdc_hypervector"
update_imports "test_hypervector_encoding" "test_hdc_hypervector_encoding"
update_imports "from tests.unit.test_pir import" "from tests.unit.test_pir_basic import"
update_imports "test_pir_integration" "test_pir_e2e"
update_imports "test_zk_integration" "test_zk_e2e"
update_imports "test_property_circuits" "test_zk_property_circuits"

echo -e "\n✅ Test file renaming complete!"
echo "Next steps:"
echo "1. Run 'make test' to ensure all tests still pass"
echo "2. Commit the changes with: git add -A && git commit -m 'Standardize test file naming convention'"
