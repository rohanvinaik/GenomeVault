#!/usr/bin/env python3
"""Analyze PIR scaling behavior from benchmarks"""

import numpy as np
import matplotlib.pyplot as plt

# Data points from benchmarks
data = [
    (100_000, 200),     # 100K rows: 200-500ms (using lower bound)
    (1_000_000, 918),   # 1M rows: 918ms
    # 10M in progress
]

# Calculate scaling
rows = np.array([d[0] for d in data])
times = np.array([d[1] for d in data])

# Fit log-log to determine scaling exponent
log_rows = np.log10(rows)
log_times = np.log10(times)

# Linear regression in log-log space
slope, intercept = np.polyfit(log_rows, log_times, 1)

print("=" * 60)
print("PIR SCALING ANALYSIS")
print("=" * 60)

print(f"\nData points:")
for r, t in data:
    print(f"  • {r:,} rows: {t:.0f}ms")

print(f"\nScaling exponent: {slope:.3f}")

if slope < 1.0:
    print(f"✅ SUB-LINEAR scaling (O(n^{slope:.2f}))")
    print("   Performance scales better than linearly with data size")
elif slope == 1.0:
    print(f"➖ LINEAR scaling (O(n))")
    print("   Performance scales proportionally with data size")
else:
    print(f"⚠️  SUPER-LINEAR scaling (O(n^{slope:.2f}))")
    print("   Performance degrades faster than linearly")

# Predict 10M performance
predicted_10m = 10 ** (slope * np.log10(10_000_000) + intercept)
print(f"\nPredicted 10M query time: {predicted_10m:.0f}ms")

# Calculate efficiency metrics
print(f"\nEfficiency Analysis:")
print(f"  • 100K → 1M (10x data): {times[1]/times[0]:.1f}x slower")
print(f"  • Expected if linear: 10x")
print(f"  • Expected if sub-linear O(n^{slope:.2f}): {10**slope:.1f}x")

# IT-PIR theoretical analysis
print(f"\nIT-PIR Theoretical Bounds:")
print(f"  • Communication: O(n^(1/{3})) for 3 servers")
print(f"  • Computation: O(n) for XOR operations")
print(f"  • Overall: O(n) dominated by computation")
print(f"  • Our observed: O(n^{slope:.2f})")

if slope < 1.0:
    print(f"\n🎉 Achieving sub-linear scaling through:")
    print(f"   • Metal GPU acceleration (parallel XOR)")
    print(f"   • Efficient memory access patterns")
    print(f"   • Multi-core query processing")
    
# Extrapolate to larger scales
print(f"\nExtrapolated Performance:")
scales = [100_000_000, 1_000_000_000]
for s in scales:
    pred = 10 ** (slope * np.log10(s) + intercept)
    print(f"  • {s:,} rows: {pred/1000:.1f} seconds")
    
print("\n" + "=" * 60)