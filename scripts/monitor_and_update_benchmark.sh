#!/bin/bash
# Continuous monitoring script for benchmark report updates

REPORT_FILE="benchmark_results/enhanced_privacy_pipeline/ENHANCED_PRIVACY_PIPELINE_BENCHMARK_REPORT.md"
METRICS_FILE="benchmark_results/enhanced_privacy_pipeline/benchmark_metrics_summary.json"
LOG_FILE="pipeline_resume_final.log"
PIPELINE_PID=3289

echo "Starting continuous benchmark monitoring..."
echo "Report: $REPORT_FILE"
echo "Metrics: $METRICS_FILE"
echo "Pipeline PID: $PIPELINE_PID"
echo ""

update_count=0

while ps -p $PIPELINE_PID > /dev/null 2>&1; do
    update_count=$((update_count + 1))
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    echo "[$timestamp] Update #$update_count"
    
    # Get Layer 3 status
    if ps -p 3304 > /dev/null 2>&1; then
        elapsed=$(ps -o etime -p 3304 | tail -1 | xargs)
        cputime=$(ps -o cputime -p 3304 | tail -1 | xargs)
        cpu_pct=$(ps -o %cpu -p 3304 | tail -1 | xargs)
        
        # Get output size
        layer3_size=$(du -sh benchmark_results/enhanced_privacy_pipeline/layer3_query 2>/dev/null | awk '{print $1}')
        temp_count=$(ls benchmark_results/enhanced_privacy_pipeline/layer3_query/*.tmp.* 2>/dev/null | wc -l | xargs)
        
        echo "  Layer 3 Status:"
        echo "    Elapsed: $elapsed"
        echo "    CPU Time: $cputime"
        echo "    CPU%: $cpu_pct"
        echo "    Output: $layer3_size"
        echo "    Temp files: $temp_count"
        
        # Append to update log
        echo "[$timestamp] Layer3: elapsed=$elapsed, cpu=$cputime, size=$layer3_size, files=$temp_count" >> benchmark_results/enhanced_privacy_pipeline/benchmark_updates.log
    else
        echo "  Layer 3: Completed or not running"
        
        # Check for Layer 4
        if grep -q "LAYER 4" "$LOG_FILE" 2>/dev/null; then
            echo "  Layer 4: DETECTED!"
            layer4_start=$(grep "LAYER 4" "$LOG_FILE" | head -1)
            echo "    $layer4_start"
        fi
    fi
    
    # Check for completion
    if grep -q "Pipeline completed" "$LOG_FILE" 2>/dev/null || grep -q "SUCCESS" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "✅ PIPELINE COMPLETED!"
        echo "Final benchmark report update needed."
        break
    fi
    
    # Check for errors
    if grep -q "ERROR\|FAILED\|Exception" "$LOG_FILE" 2>/dev/null | tail -5; then
        echo "⚠️  Errors detected in log file"
    fi
    
    echo ""
    
    # Sleep for 5 minutes between updates
    sleep 300
done

echo ""
echo "Monitoring complete. Total updates: $update_count"
echo "Check benchmark_results/enhanced_privacy_pipeline/benchmark_updates.log for full history"
