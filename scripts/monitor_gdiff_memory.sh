#!/bin/bash
# Monitor GDiff benchmark memory and CPU usage

echo "GDiff k=3 Benchmark Monitor"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "=== GDiff k=3 Benchmark Monitor - $(date +%H:%M:%S) ==="
    echo ""

    # Check if process is running
    if ps aux | grep -q "[r]un_k3_whole_genome_benchmark"; then
        echo "Status: ✅ RUNNING"

        # Main process
        ps aux | grep "[r]un_k3_whole_genome_benchmark" | \
            awk '{printf "Main: PID %s | CPU %s%% | MEM %s%% (%.1f MB)\n", $2, $3, $4, $6/1024}'

        # Workers
        worker_count=$(ps aux | grep "spawn_main" | grep -v grep | wc -l | tr -d ' ')
        worker_cpu=$(ps aux | grep "spawn_main" | grep -v grep | awk '{sum+=$3} END {printf "%.1f", sum}')
        echo "Workers: $worker_count active | Aggregate CPU: ${worker_cpu}%"

        # Hardware (compact, one line)
        # Use powermetrics (requires passwordless sudo setup)
        cpu_temp=$(timeout 2 sudo powermetrics --samplers smc -i1 -n1 2>/dev/null | grep -i "CPU die temperature" | awk '{print $4}' || echo "?")

        if command -v istats &> /dev/null; then
            fan_speed=$(istats fan speed 2>/dev/null | grep "Fan 0" | awk '{print $4}')
        else
            fan_speed=""
        fi

        mem_free=$(vm_stat | perl -ne '/page size of (\d+)/ and $size=$1; /Pages free.*?(\d+)/ and printf("%.1f", $1 * $size / 1073741824);')
        battery=$(pmset -g batt 2>/dev/null | grep -o "[0-9]*%" | head -1)

        if [ -n "$fan_speed" ]; then
            echo "Hardware: CPU ${cpu_temp}°C | Fan ${fan_speed} RPM | RAM Free ${mem_free} GB | Battery ${battery:-N/A}"
        else
            echo "Hardware: CPU ${cpu_temp}°C | RAM Free ${mem_free} GB | Battery ${battery:-N/A}"
        fi

        echo ""
        echo "=== Recent Progress ==="
        tail -8 benchmark_results/k3_whole_genome_benchmark/run.log 2>/dev/null | grep -v "^$"
    else
        echo "Status: ⚠️  NOT RUNNING or COMPLETED"
        echo ""
        echo "=== Final Log Lines ==="
        tail -10 benchmark_results/k3_whole_genome_benchmark/run.log 2>/dev/null
        break
    fi

    sleep 10
done
