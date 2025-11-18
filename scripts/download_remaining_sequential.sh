#!/bin/bash
# Sequential download of remaining samples (African and South Asian)
# Downloads one sample at a time to avoid resource conflicts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/sequential_download_${TIMESTAMP}.log"

echo "======================================================================" | tee -a "$LOG_FILE"
echo "Sequential Genomic Download - Remaining Samples" | tee -a "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Define samples to download (2 African + 2 South Asian)
SAMPLES=(
    "african:ERR3239756"
    "african:ERR3239778"
    "south_asian:ERR3239912"
    "south_asian:ERR3239934"
)

SUCCESS_COUNT=0
FAIL_COUNT=0

for SAMPLE_SPEC in "${SAMPLES[@]}"; do
    POOL=$(echo "$SAMPLE_SPEC" | cut -d: -f1)
    ACCESSION=$(echo "$SAMPLE_SPEC" | cut -d: -f2)

    echo "======================================================================" | tee -a "$LOG_FILE"
    echo "[$(date)] Downloading: $ACCESSION ($POOL)" | tee -a "$LOG_FILE"
    echo "======================================================================" | tee -a "$LOG_FILE"

    # Download this one sample
    python3 "$SCRIPT_DIR/download_genomic_data_automated.py" \
        --accession "$ACCESSION" \
        --output-dir "data/downloaded/fastq/${POOL}" \
        2>&1 | tee -a "$LOG_FILE"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ SUCCESS: $ACCESSION completed" | tee -a "$LOG_FILE"
        ((SUCCESS_COUNT++))
    else
        echo "❌ FAILED: $ACCESSION (exit code: $EXIT_CODE)" | tee -a "$LOG_FILE"
        ((FAIL_COUNT++))
    fi

    echo "" | tee -a "$LOG_FILE"

    # Small delay between downloads
    if [ $SUCCESS_COUNT -lt ${#SAMPLES[@]} ]; then
        echo "Waiting 10 seconds before next download..." | tee -a "$LOG_FILE"
        sleep 10
    fi
done

echo "======================================================================" | tee -a "$LOG_FILE"
echo "Download Summary" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "Successful: $SUCCESS_COUNT / ${#SAMPLES[@]}" | tee -a "$LOG_FILE"
echo "Failed: $FAIL_COUNT / ${#SAMPLES[@]}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "======================================================================" | tee -a "$LOG_FILE"
