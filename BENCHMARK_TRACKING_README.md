# Benchmark Tracking - Quick Start

## 📊 Active Monitoring Commands

### For Live Visual Monitoring (Recommended)
```bash
python3 scripts/graphical_pipeline_tracker.py
```
**What you'll see:** Beautiful color-coded dashboard with progress bars, CPU/memory usage, active workers, and speedup calculations. Updates every 5 seconds.

### For Simple Status Checks
```bash
./scripts/monitor_phase123_pipeline.sh
```

### For Detailed Benchmarking
```bash
# One-time snapshot
python3 scripts/track_pipeline_benchmarks.py

# Auto-updating every 30 seconds
./scripts/auto_track_benchmarks.sh
```

## 📈 After Pipeline Completes

Generate comprehensive final report:
```bash
python3 scripts/generate_final_benchmark_report.py
```

## 📁 Where to Find Results

- **Live metrics:** `benchmark_results/enhanced_privacy_k13_phase123_optimized/phase123_benchmarks_*.json`
- **Final report:** `benchmark_results/PHASE123_FINAL_REPORT_*.md`
- **Log files:** `logs/phase123_optimized_deployment.log`

## 🎯 Expected Performance

- **Baseline:** 7.5 hours per reference (90 hours total)
- **Phase 1-3 Optimized:** 12-18 min per reference (2.4-3.6 hours total)
- **Speedup:** 25-37.5×

## 💾 Files Created

All benchmark tracking tools automatically save data to:
- JSON files (detailed metrics)
- Markdown reports (human-readable)
- Log files (complete history)

See `MONITORING_AND_BENCHMARKING_GUIDE.md` for full documentation.
