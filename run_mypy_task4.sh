#!/bin/bash
cd /Users/rohanvinaik/genomevault
echo "Running mypy baseline check..."
mypy genomevault pir zk_proofs > /Users/rohanvinaik/genomevault/task4_mypy_output.txt 2>&1
echo "Mypy check completed. Check task4_mypy_output.txt for results."
