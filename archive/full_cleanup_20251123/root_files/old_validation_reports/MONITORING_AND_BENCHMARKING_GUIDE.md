# Phase 1-3 Pipeline Monitoring & Benchmarking Guide

Complete guide for tracking the optimized k=13 enhanced privacy pipeline.

---

## 📊 Benchmark Tracking Tools

### 1. Real-Time Graphical Monitor

**Best for:** Live visual monitoring with progress bars and resource usage

```bash
python3 scripts/graphical_pipeline_tracker.py
```

**Features:**
- ✅ Color-coded progress bars
- ✅ Real-time CPU/memory usage (all worker processes)
- ✅ Active worker process display
- ✅ Visual reference grid (12 boxes)
- ✅ Speedup calculations
- ✅ Time remaining estimates
- ✅ Auto-refreshing every 5 seconds

**What you'll see:**
- Total CPU: ~660% (minimap2 using 6.6 cores)
- Total Memory: ~30% (19GB for whole-genome)
- Active Workers: minimap2, sambamba, bcftools
- Progress: X / 12 references completed
- Estimated completion time

---

### 2. Simple Status Monitor

**Best for:** Quick status checks

```bash
./scripts/monitor_phase123_pipeline.sh
```

**Features:**
- ✅ Completed references count
- ✅ Current work in progress
- ✅ Performance metrics
- ✅ Optimization status
- ✅ Latest log entries

---

### 3. Benchmark Tracker

**Best for:** Detailed performance analysis

```bash
# One-time snapshot
python3 scripts/track_pipeline_benchmarks.py

# Auto-updating (every 30 seconds)
./scripts/auto_track_benchmarks.sh
```

**Tracks:**
- ✅ Per-reference timing (alignment, sorting, variant calling)
- ✅ Average/min/max statistics
- ✅ Speedup vs baseline (37.5× expected)
- ✅ Time remaining projections
- ✅ Saves JSON benchmarks for analysis

**Output:**
- `benchmark_results/enhanced_privacy_k13_phase123_optimized/phase123_benchmarks_YYYYMMDD_HHMMSS.json`

---

### 4. Live Log Monitoring

**Best for:** Debugging and detailed progress

```bash
# Live tail
tail -f logs/phase123_optimized_deployment.log

# Last 50 lines
tail -50 logs/phase123_optimized_deployment.log
```

---

### 5. Process Monitor

**Best for:** Resource usage verification

```bash
# All pipeline processes
ps aux | grep -E "minimap2|sambamba|bcftools|samtools|deploy_phase123" | grep -v grep

# Detailed resource usage
ps aux | grep minimap2 | grep -v grep | awk '{print "CPU: " $3 "% | Memory: " $4 "%"}'
```

---

## 📈 Generating Final Report

After pipeline completes (or at any time):

```bash
python3 scripts/generate_final_benchmark_report.py
```

**Generates:**
- ✅ Comprehensive Markdown report
- ✅ Baseline vs optimized comparison
- ✅ Per-reference detailed timing
- ✅ Speedup analysis
- ✅ Optimization breakdown
- ✅ Hardware utilization stats

**Output:**
- `benchmark_results/PHASE123_FINAL_REPORT_YYYYMMDD_HHMMSS.md`

---

## 🎯 Expected Benchmarks

### Baseline (Before Optimization)
- **Per Reference:** 7.5 hours
- **Total (k=12):** 90 hours
- **CPU Usage:** ~100% (single-threaded bottlenecks)

### Phase 1-3 Optimized (Target)
- **Per Reference:** 12-18 minutes
- **Total (k=12):** 2.4-3.6 hours
- **CPU Usage:** ~660% (multi-threaded)
- **Speedup:** 25-37.5×

### Key Metrics to Track

| Metric | Baseline | Phase 1-3 Target |
|--------|----------|------------------|
| Index Build | 60s × 12 refs | 57s × 1 (cached) |
| Alignment+Sort | ~6.5 hours | ~8-12 minutes |
| Variant Calling | ~1 hour | ~2-3 minutes |
| **Total/Reference** | **7.5 hours** | **12-18 minutes** |

---

## 🔍 Monitoring Checklist

### During Pipeline Execution

- [ ] Verify all optimizations active (check log)
- [ ] Monitor CPU usage (~660% expected)
- [ ] Monitor memory usage (~30% peak)
- [ ] Track per-reference completion time
- [ ] Verify index caching (only builds once)
- [ ] Check for errors in log

### After Completion

- [ ] Generate final benchmark report
- [ ] Compare against baseline estimates
- [ ] Validate speedup calculations
- [ ] Check all 12 references completed
- [ ] Verify VCF files generated and indexed
- [ ] Save benchmark JSON for documentation

---

## 📁 Benchmark Files Location

```
benchmark_results/enhanced_privacy_k13_phase123_optimized/
├── phase123_benchmarks_YYYYMMDD_HHMMSS.json    # Detailed metrics
├── PHASE123_FINAL_REPORT_YYYYMMDD_HHMMSS.md    # Final report
├── layer2_reference_pool/
│   ├── ref1.vcf.gz
│   ├── ref2.vcf.gz
│   └── ... (12 references)
└── index_cache/
    └── consensus.mmi  # Cached minimap2 index

logs/
├── phase123_optimized_deployment.log           # Main pipeline log
└── benchmark_tracking.log                      # Auto-tracker log
```

---

## 🚀 Quick Reference Commands

```bash
# Start graphical monitor
python3 scripts/graphical_pipeline_tracker.py

# Check quick status
./scripts/monitor_phase123_pipeline.sh

# Get latest benchmarks
python3 scripts/track_pipeline_benchmarks.py

# Auto-update benchmarks (background)
nohup ./scripts/auto_track_benchmarks.sh > logs/auto_benchmark.log 2>&1 &

# Generate final report
python3 scripts/generate_final_benchmark_report.py

# View live log
tail -f logs/phase123_optimized_deployment.log

# Check pipeline process
ps aux | grep deploy_phase123 | grep -v grep
```

---

## 💡 Pro Tips

1. **Start auto-benchmark tracker** in background for continuous monitoring:
   ```bash
   nohup ./scripts/auto_track_benchmarks.sh > logs/auto_benchmark.log 2>&1 &
   ```

2. **Use graphical tracker** for visual monitoring (updates every 5s)

3. **Run benchmark tracker** after each reference completes to track progress

4. **Generate final report** when pipeline completes for comprehensive analysis

5. **Keep benchmark JSON files** for historical comparison and documentation

---

## 📞 Troubleshooting

### Low CPU Usage
- Check if minimap2 is running: `ps aux | grep minimap2`
- Expected: 600-800% CPU (6-8 cores)
- If lower: Check thread settings in script

### High Memory Usage
- Normal: 30-40% (19-26 GB) for whole-genome
- If >50%: Monitor with `top` or Activity Monitor
- M1 Max has 64GB - plenty of headroom

### No Progress
- Check pipeline is running: `ps aux | grep deploy_phase123`
- Check for errors: `tail -50 logs/phase123_optimized_deployment.log`
- Verify disk space: `df -h`

---

**Last Updated:** 2025-10-26
**Pipeline Version:** Phase 1-3 Optimized k=13 Enhanced Privacy Pipeline
