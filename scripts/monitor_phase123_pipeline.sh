#!/bin/bash
#
# Monitor Phase 1-3 optimized k=13 enhanced privacy pipeline
#

OUTPUT_DIR="benchmark_results/enhanced_privacy_k13_phase123_optimized"
LOG_FILE="logs/phase123_optimized_deployment.log"
LAYER2_DIR="$OUTPUT_DIR/layer2_reference_pool"

echo "========================================================================"
echo "Phase 1-3 Optimized Pipeline Monitor"
echo "========================================================================"
echo ""

# Check if pipeline is running
PID=$(ps aux | grep "deploy_phase123_optimized_pipeline.py" | grep -v grep | awk '{print $2}')

if [ -z "$PID" ]; then
    echo "⚠️  Pipeline not running"
else
    echo "✅ Pipeline running (PID: $PID)"
fi

echo ""
echo "------------------------------------------------------------------------"
echo "Progress Summary"
echo "------------------------------------------------------------------------"
echo ""

# Count completed references
if [ -d "$LAYER2_DIR" ]; then
    COMPLETED_REFS=$(ls -1 "$LAYER2_DIR"/*.vcf.gz 2>/dev/null | wc -l | tr -d ' ')
    echo "References completed: $COMPLETED_REFS / 12"
    echo ""

    # Show completed references with sizes
    if [ $COMPLETED_REFS -gt 0 ]; then
        echo "Completed VCF files:"
        ls -lh "$LAYER2_DIR"/*.vcf.gz 2>/dev/null | awk '{print "  " $9 " - " $5}'
        echo ""
    fi

    # Show current work in progress
    CURRENT_REF=$(ls -1 "$LAYER2_DIR"/*.bam 2>/dev/null | grep -v "sorted.bam" | head -1)
    if [ -n "$CURRENT_REF" ]; then
        echo "Currently processing:"
        ls -lh "$LAYER2_DIR"/*.bam 2>/dev/null | grep -v "sorted.bam" | head -3
        echo ""
    fi
fi

echo "------------------------------------------------------------------------"
echo "Performance Metrics"
echo "------------------------------------------------------------------------"
echo ""

# Extract timing from log
if [ -f "$LOG_FILE" ]; then
    # Get minimap2 index build time
    INDEX_TIME=$(grep "Index built in" "$LOG_FILE" | tail -1 | grep -oE "[0-9]+\.[0-9]+" | head -1)
    if [ -n "$INDEX_TIME" ]; then
        echo "Minimap2 index: ${INDEX_TIME}s (built once, reused for all refs)"
    fi

    # Get alignment times
    ALIGN_TIMES=$(grep "Alignment complete in" "$LOG_FILE" | grep -oE "[0-9]+\.[0-9]+" || true)
    if [ -n "$ALIGN_TIMES" ]; then
        AVG_ALIGN=$(echo "$ALIGN_TIMES" | awk '{s+=$1; c++} END {if(c>0) printf "%.1f", s/c}')
        if [ -n "$AVG_ALIGN" ]; then
            echo "Average alignment+sort: ${AVG_ALIGN}s per reference"
        fi
    fi

    # Get variant calling times
    VARIANT_TIMES=$(grep "Variant calling complete in" "$LOG_FILE" | grep -oE "[0-9]+\.[0-9]+" || true)
    if [ -n "$VARIANT_TIMES" ]; then
        AVG_VARIANT=$(echo "$VARIANT_TIMES" | awk '{s+=$1; c++} END {if(c>0) printf "%.1f", s/c}')
        if [ -n "$AVG_VARIANT" ]; then
            echo "Average variant calling: ${AVG_VARIANT}s per reference"
        fi
    fi

    echo ""
fi

echo "------------------------------------------------------------------------"
echo "Optimization Status"
echo "------------------------------------------------------------------------"
echo ""

if [ -f "$LOG_FILE" ]; then
    echo "Phase 1:"
    grep -q "Using sambamba" "$LOG_FILE" && echo "  ✅ Sambamba parallel sorting (10 threads, 8GB RAM)" || echo "  ❌ Sambamba"
    grep -q "parallel_bcftools" "$LOG_FILE" && echo "  ✅ Parallel BCFtools (5 threads)" || echo "  ⚠️  Parallel BCFtools"
    grep -q "Using cached minimap2 index" "$LOG_FILE" && echo "  ✅ Minimap2 index caching (reusing index)" || echo "  🔨 Minimap2 index caching (building...)"
    grep -q "Metal.*backend" "$LOG_FILE" && echo "  ✅ Metal GPU HDC" || echo "  ⚠️  Metal GPU HDC"

    echo ""
    echo "Phase 3:"
    grep -q "Chromosome-partitioned sort" "$LOG_FILE" && echo "  ✅ Chromosome-partitioned sorting" || echo "  ⚠️  Chromosome-partitioned sorting"
    grep -q "parallel.*VCF" "$LOG_FILE" && echo "  ✅ Parallel VCF parsing" || echo "  ⚠️  Parallel VCF parsing"
fi

echo ""
echo "------------------------------------------------------------------------"
echo "Latest Log Entries"
echo "------------------------------------------------------------------------"
echo ""

if [ -f "$LOG_FILE" ]; then
    tail -15 "$LOG_FILE"
fi

echo ""
echo "========================================================================"
echo "To monitor continuously: watch -n 10 ./scripts/monitor_phase123_pipeline.sh"
echo "To view full log: tail -f $LOG_FILE"
echo "========================================================================"
