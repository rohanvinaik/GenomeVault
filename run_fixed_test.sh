#!/bin/bash
# Run the fixed test and capture output

cd /Users/rohanvinaik/genomevault

echo "Running fixed minimal test..."
python3 test_minimal_fixed.py 2>&1 | tee test_output_fixed.log

echo -e "\n\n=== TEST OUTPUT SAVED TO test_output_fixed.log ==="
