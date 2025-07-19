#!/bin/bash
# Run the minimal test and capture output

cd /Users/rohanvinaik/genomevault

echo "Running minimal test..."
python3 test_minimal.py 2>&1 | tee test_output.log

echo -e "\n\n=== TEST OUTPUT SAVED TO test_output.log ==="
