#!/bin/bash
cd /Users/rohanvinaik/genomevault
echo "Running ruff check after removing global ignores..."
ruff check . --output-format=text > /Users/rohanvinaik/genomevault/task3_ruff_output/ruff_errors.txt 2>&1
echo "Ruff check completed. Check ruff_errors.txt for results."
