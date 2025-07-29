#!/bin/bash
# Run linters and tests for core exceptions implementation

cd /Users/rohanvinaik/genomevault

echo "Running Black formatter..."
black genomevault/core/exceptions.py tests/core/test_exceptions.py

echo "Running isort..."
isort genomevault/core/exceptions.py tests/core/test_exceptions.py

echo "Running flake8..."
flake8 genomevault/core/exceptions.py tests/core/test_exceptions.py

echo "Running pytest..."
python -m pytest tests/core/test_exceptions.py -v

echo "Done!"
