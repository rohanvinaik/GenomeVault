#!/bin/bash
#
# Monitor Parallel Reference Generation
#
# Tracks both Ref2 regeneration and Ref3 generation simultaneously
#

echo "========================================================================"
echo "Parallel Reference Generation Monitor"
echo "========================================================================"
date
echo ""

# === Ref2 Regeneration ===
echo "────────────────────────────────────────────────────────────────────────"
echo "Ref2 Regeneration (Chunks 1-21, threads=1)"
echo "────────────────────────────────────────────────────────────────────────"

if ps aux | grep -q "[r]egenerate_missing_chunks.sh"; then
    echo "✓ Regeneration script running"
    
    NEAT_PID=$(ps aux | grep "[n]eat read-simulator.*chunk_0" | awk '{print $2}')
    if [ -n "$NEAT_PID" ]; then
        ps aux | grep "[n]eat read-simulator.*chunk_0" | awk '{print "  NEAT: PID " $2 " | CPU: " $3 "% | TIME: " $10}'
    fi
    
    COMPLETED=$(find /Users/rohanvinaik/genomevault/benchmark_results/differential_encoding_samples/temp/chunks_1-21_regenerated -name "sample2_r*.fastq.gz" 2>/dev/null | wc -l | tr -d ' ')
    CHUNKS_DONE=$((COMPLETED / 2))
    PERCENT=$((CHUNKS_DONE * 100 / 21))
    
    printf "  Progress: %d/21 chunks (%d%%) [" $CHUNKS_DONE $PERCENT
    BARS=$((PERCENT / 5))
    for i in $(seq 1 20); do
        if [ $i -le $BARS ]; then printf "="; else printf " "; fi
    done
    printf "]\n"
else
    echo "✓ Regeneration complete"
fi

echo ""

# === Ref3 Generation ===
echo "────────────────────────────────────────────────────────────────────────"
echo "Ref3 Generation (Full Pipeline, threads=10)"
echo "────────────────────────────────────────────────────────────────────────"

if ps aux | grep -q "[g]enerate_ref3_parallel.sh"; then
    echo "✓ Ref3 script running"
    
    # Check which phase
    if ps aux | grep -q "simuG.pl.*-seed 300"; then
        ps aux | grep "simuG.pl.*-seed 300" | grep -v grep | awk '{print "  simuG: PID " $2 " | CPU: " $3 "%"}'
        echo "  Phase: Variant generation"
    elif ps aux | grep -q "neat read-simulator.*sample3"; then
        ps aux | grep "neat read-simulator.*sample3" | grep -v grep | head -1 | awk '{print "  NEAT: PID " $2 " | CPU: " $3 "% | TIME: " $10}'
        
        # Try to count chunks
        TEMP_DIR=$(find /var/folders -path "*/tmp*/splits" -type d 2>/dev/null | grep -v tmp5fmcup8p | tail -1 | xargs dirname 2>/dev/null)
        if [ -n "$TEMP_DIR" ]; then
            CHUNKS=$(find "$TEMP_DIR" -name "sample3_r1.fastq.gz" -type f 2>/dev/null | wc -l | tr -d ' ')
            echo "  Phase: Sequencing reads generation ($CHUNKS/102 chunks)"
        else
            echo "  Phase: Sequencing reads generation"
        fi
    else
        tail -3 benchmark_results/ref3_parallel_generation.log | head -1
    fi
else
    if [ -f /Users/rohanvinaik/genomevault/benchmark_results/differential_encoding_samples/references/ref3/sample3_r1.fastq.gz ]; then
        echo "✓ Ref3 complete"
        ls -lh /Users/rohanvinaik/genomevault/benchmark_results/differential_encoding_samples/references/ref3/*.fastq.gz 2>/dev/null | awk '{print "  " $9 ": " $5}'
    else
        echo "⚠️  Ref3 script not running (may be between phases)"
    fi
fi

echo ""
echo "========================================================================"
echo "CPU Usage Summary"
echo "========================================================================"
ps aux | grep -E "(neat|simuG)" | grep -v grep | wc -l | awk '{print "Active processes: " $1}'
echo ""

echo "Monitor logs:"
echo "  Ref2: tail -f benchmark_results/chunk_regeneration.log"
echo "  Ref3: tail -f benchmark_results/ref3_parallel_generation.log"
echo "========================================================================"
