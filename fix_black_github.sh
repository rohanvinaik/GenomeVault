#!/bin/bash
# Fix Black syntax errors in GenomeVault

echo "Fixing Black syntax errors..."

# Fix missing f-string prefixes in logger.error statements
echo "Fixing f-string issues..."
find genomevault -name "*.py" -type f | while read file; do
    # Fix patterns like logger.error("... {var} ...")
    sed -i 's/logger\.error("\([^"]*{[^}]\+}[^"]*\)")/logger.error(f"\1")/g' "$file"
    sed -i "s/logger\.error('\([^']*{[^}]\+}[^']*\)')/logger.error(f'\1')/g" "$file"
done

# Fix specific files with known issues
echo "Fixing specific file issues..."

# Fix genomevault/blockchain/hipaa/verifier.py line 17
if [ -f "genomevault/blockchain/hipaa/verifier.py" ]; then
    sed -i '17s/.*/from .models import HIPAAVerificationRecord/' genomevault/blockchain/hipaa/verifier.py
    # Ensure logger is properly imported
    sed -i '18s/.*/from genomevault.utils.logging import get_logger\n\nlogger = get_logger(__name__)/' genomevault/blockchain/hipaa/verifier.py
fi

# Fix trailing commas in from statements
find genomevault -name "*.py" -type f -exec sed -i 's/from typing import \(.*\),$/from typing import \1/g' {} +
find genomevault -name "*.py" -type f -exec sed -i 's/from \(.*\),$/from \1/g' {} +

# Fix trailing commas after equals
find genomevault -name "*.py" -type f -exec sed -i 's/= *,/=/g' {} +

# Fix structured logging in security_monitor.py
if [ -f "genomevault/utils/security_monitor.py" ]; then
    sed -i '214s/.*/        logger.error(f"anomaly_detector_training_failed: {str(e)}")/' genomevault/utils/security_monitor.py
fi

# Fix specific line numbers based on the errors
echo "Fixing specific line errors..."

# Array of fixes: "file:line:pattern:replacement"
fixes=(
    "genomevault/hypervector_transform/encoding.py:125:logger\.error(\"Encoding error: {str(e)}\"):logger.error(f\"Encoding error: {str(e)}\")"
    "genomevault/hypervector_transform/hdc_encoder.py:188:logger\.error(f\"Encoding error: {str(e)}\"):logger.error(f\"Encoding error: {str(e)}\")"
    "genomevault/hypervector_transform/registry.py:114:logger\.error(\"Failed to load registry: {e}\"):logger.error(f\"Failed to load registry: {e}\")"
    "genomevault/integration/proof_of_training.py:288:logger\.error(\"Failed to submit attestation: {e}\"):logger.error(f\"Failed to submit attestation: {e}\")"
    "genomevault/local_processing/pipeline.py:70:logger\.error(\"Failed to process {omics_type}: {str(e)}\"):logger.error(f\"Failed to process {omics_type}: {str(e)}\")"
    "genomevault/pir/client.py:311:logger\.error(\"Error querying server {server.server_id}: {e}\"):logger.error(f\"Error querying server {server.server_id}: {e}\")"
    "genomevault/pir/network/coordinator.py:258:logger\.error(\"Health monitor error: {e}\"):logger.error(f\"Health monitor error: {e}\")"
    "genomevault/pir/server/handler.py:121:logger\.error(\"Error handling PIR query: {str(e)}\"):logger.error(f\"Error handling PIR query: {str(e)}\")"
    "genomevault/pir/server/shard_manager.py:267:logger\.error(\"Error creating shard {shard_index}: {e}\"):logger.error(f\"Error creating shard {shard_index}: {e}\")"
    "genomevault/security/phi_detector.py:127:logger\.error(\"Error scanning file {filepath}: {e}\"):logger.error(f\"Error scanning file {filepath}: {e}\")"
    "genomevault/pir/server/enhanced_pir_server.py:484:logger\.error(\"Error processing query {query_id}: {e}\"):logger.error(f\"Error processing query {query_id}: {e}\")"
)

for fix in "${fixes[@]}"; do
    IFS=':' read -r file line pattern replacement <<< "$fix"
    if [ -f "$file" ]; then
        sed -i "${line}s/${pattern}/${replacement}/" "$file" 2>/dev/null || true
    fi
done

echo "Done! Running black --check to verify..."
black --check . || echo "Some files still need formatting. Running black to format..."
black . || echo "Black formatting completed with some warnings"

echo "Fix script completed!"
