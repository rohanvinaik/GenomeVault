#!/bin/bash
cd /Users/rohanvinaik/genomevault
echo "Running pytest to check test status..."
pytest -q > /Users/rohanvinaik/genomevault/task5_pytest_output.txt 2>&1
echo "Pytest completed. Check task5_pytest_output.txt for results."
