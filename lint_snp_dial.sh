#!/bin/bash

echo "Running linters on SNP dial implementation files..."

# Files to check
FILES=(
    "genomevault/hypervector/positional.py"
    "genomevault/hypervector/encoding/genomic.py"
    "genomevault/api/routers/query_tuned.py"
    "genomevault/pir/client/batched_query_builder.py"
)

# Run black formatter
echo "Running black formatter..."
python -m black "${FILES[@]}" --line-length 100

# Run isort
echo "Running isort..."
python -m isort "${FILES[@]}"

# Run flake8
echo "Running flake8..."
python -m flake8 "${FILES[@]}"

# Run mypy (type checking)
echo "Running mypy..."
python -m mypy "${FILES[@]}" --ignore-missing-imports

echo "Linting complete!"
