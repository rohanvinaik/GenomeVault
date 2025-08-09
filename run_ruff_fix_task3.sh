#!/bin/bash
cd /Users/rohanvinaik/genomevault
echo "Running ruff with --fix option..."
ruff check . --fix > /Users/rohanvinaik/genomevault/task3_ruff_output/ruff_fixed.txt 2>&1
echo "Ruff fix completed. Check ruff_fixed.txt for results."
