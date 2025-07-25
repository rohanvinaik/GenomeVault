#!/bin/bash

echo "Fixing Black formatting issues for SNP dial implementation..."

# Ensure we're using the right Black version
echo "Black version:"
python -m black --version

# Format each file with proper settings
echo -e "\nFormatting files..."

# Format the Python files with Black's default settings (it will read from pyproject.toml)
python -m black genomevault/api/routers/query_tuned.py
python -m black genomevault/hypervector/positional.py
python -m black genomevault/hypervector/encoding/genomic.py
python -m black genomevault/pir/client/batched_query_builder.py
python -m black test_snp_dial.py

# Also run isort to fix import ordering
echo -e "\nFixing import order with isort..."
python -m isort genomevault/api/routers/query_tuned.py
python -m isort genomevault/hypervector/positional.py
python -m isort genomevault/hypervector/encoding/genomic.py
python -m isort genomevault/pir/client/batched_query_builder.py
python -m isort test_snp_dial.py

# Check if Black would still reformat
echo -e "\nChecking with black --check..."
python -m black --check genomevault/api/routers/query_tuned.py genomevault/hypervector/positional.py genomevault/hypervector/encoding/genomic.py genomevault/pir/client/batched_query_builder.py test_snp_dial.py

echo -e "\nDone! Files should now be properly formatted."
