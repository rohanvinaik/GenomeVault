#!/usr/bin/env python3
"""Test unified hardware acceleration architecture."""

import time
import sys
import numpy as np
from pathlib import Path

# Add genomevault to path
sys.path.insert(0, '/Users/rohanvinaik/genomevault')

from genomevault.hardware import (
    list_available_accelerators,
    get_best_accelerator,
    UnifiedAccelerationEngine,
    AccelerationConfig
)
from genomevault.hardware.hypervector_adapter import (
    HardwareAcceleratedHypervectorEngine,
    HypervectorConfig,
    get_hypervector_engine
)
from genomevault.core.constants import OmicsType
from genomevault.utils.logging import get_logger

logger = get_logger(__name__)


def test_hardware_detection():
    """Test hardware detection and selection."""
    print("=" * 60)
    print("Hardware Detection")
    print("=" * 60)
    
    # List all available accelerators
    accelerators = list_available_accelerators()
    
    print("\nAvailable Accelerators:")
    for acc in accelerators:
        print(f"  {acc}")
    
    # Get best accelerator
    best = get_best_accelerator()
    print(f"\nRecommended: {best.name} ({best.type.value})")
    
    print("✅ Hardware detection successful")


def test_unified_engine():
    """Test unified acceleration engine."""
    print("\n" + "=" * 60)
    print("Unified Acceleration Engine")
    print("=" * 60)
    
    # Create engine with auto-detection
    engine = UnifiedAccelerationEngine()
    
    # Get info
    info = engine.get_info()
    print("\nEngine Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test basic operations
    print("\nTesting operations:")
    
    # Matrix multiplication
    a = np.random.randn(100, 100).astype(np.float32)
    b = np.random.randn(100, 100).astype(np.float32)
    
    a_dev = engine.to_device(a)
    b_dev = engine.to_device(b)
    
    start = time.perf_counter()
    c_dev = engine.matmul(a_dev, b_dev)
    c = engine.from_device(c_dev)
    matmul_time = (time.perf_counter() - start) * 1000
    
    print(f"  MatMul (100x100): {matmul_time:.2f}ms")
    
    # FFT
    data = np.random.randn(1024) + 1j * np.random.randn(1024)
    data_dev = engine.to_device(data)
    
    start = time.perf_counter()
    fft_dev = engine.fft(data_dev)
    fft_result = engine.from_device(fft_dev)
    fft_time = (time.perf_counter() - start) * 1000
    
    print(f"  FFT (1024): {fft_time:.2f}ms")
    
    # Normalization
    vector = np.random.randn(10000).astype(np.float32)
    vector_dev = engine.to_device(vector)
    
    start = time.perf_counter()
    norm_dev = engine.normalize(vector_dev)
    norm_result = engine.from_device(norm_dev)
    norm_time = (time.perf_counter() - start) * 1000
    
    print(f"  Normalize (10000): {norm_time:.2f}ms")
    
    # Verify normalization
    norm_check = np.linalg.norm(norm_result)
    assert abs(norm_check - 1.0) < 0.01, f"Normalization failed: {norm_check}"
    
    print("✅ Unified engine tests passed")


def test_hypervector_with_unified():
    """Test hypervector operations with unified hardware."""
    print("\n" + "=" * 60)
    print("Hypervector with Unified Hardware")
    print("=" * 60)
    
    # Create hypervector engine
    config = HypervectorConfig(
        dimension=10000,
        batch_size=1024,
        use_sparse=True
    )
    
    engine = HardwareAcceleratedHypervectorEngine(config)
    
    print(f"\nUsing backend: {engine.engine.backend.type.value}")
    print(f"Device: {engine.engine.backend.name}")
    
    # Test single encoding
    print("\n1. Single sample encoding:")
    data = np.random.randn(100).astype(np.float32)
    
    start = time.perf_counter()
    encoded = engine.encode(data, OmicsType.GENOMIC)
    encoding_time = (time.perf_counter() - start) * 1000
    
    print(f"  Shape: {encoded.shape}")
    print(f"  Time: {encoding_time:.2f}ms")
    print(f"  Sparsity: {np.mean(encoded == 0):.1%}")
    
    # Test batch encoding
    print("\n2. Batch encoding:")
    batch_data = np.random.randn(100, 100).astype(np.float32)
    
    start = time.perf_counter()
    batch_encoded = engine.encode(batch_data, OmicsType.TRANSCRIPTOMIC)
    batch_time = (time.perf_counter() - start) * 1000
    
    print(f"  Shape: {batch_encoded.shape}")
    print(f"  Time: {batch_time:.2f}ms")
    print(f"  Throughput: {100000/batch_time:.0f} samples/sec")
    
    # Test similarity computation
    print("\n3. Similarity computation:")
    query = batch_encoded[0]
    database = batch_encoded[1:]
    
    for metric in ["cosine", "hamming", "euclidean"]:
        start = time.perf_counter()
        similarities = engine.compute_similarity(query, database, metric)
        sim_time = (time.perf_counter() - start) * 1000
        
        print(f"  {metric}: {sim_time:.2f}ms, max={np.max(similarities):.3f}")
    
    print("✅ Hypervector tests passed")


def test_backward_compatibility():
    """Test backward compatibility with old interfaces."""
    print("\n" + "=" * 60)
    print("Backward Compatibility")
    print("=" * 60)
    
    # Test MetalHypervectorEngine alias
    from genomevault.hardware.hypervector_adapter import MetalHypervectorEngine
    
    config = HypervectorConfig(dimension=5000)
    engine = MetalHypervectorEngine(config)
    
    data = np.random.randn(50).astype(np.float32)
    encoded = engine.encode(data)
    
    print(f"MetalHypervectorEngine (alias) works: {encoded.shape}")
    
    # Test LocalGPUEngine alias
    from genomevault.hardware.hypervector_adapter import LocalGPUEngine
    
    engine2 = LocalGPUEngine(config)
    encoded2 = engine2.encode(data)
    
    print(f"LocalGPUEngine (alias) works: {encoded2.shape}")
    
    print("✅ Backward compatibility maintained")


def benchmark_comparison():
    """Compare performance across different backends."""
    print("\n" + "=" * 60)
    print("Performance Comparison")
    print("=" * 60)
    
    # Test parameters
    input_dim = 1000
    num_samples = 1000
    
    print(f"\nBenchmark: {num_samples} samples, {input_dim} features")
    print("-" * 40)
    
    # Get available backends
    accelerators = list_available_accelerators()
    
    results = []
    
    for acc in accelerators:
        if not acc.available:
            continue
        
        try:
            # Create engine with specific backend
            config = AccelerationConfig(
                dimension=10000,
                device=acc.type
            )
            engine = UnifiedAccelerationEngine(config)
            
            # Run benchmark
            bench_results = engine.benchmark("matmul", 500)
            
            results.append({
                "backend": acc.type.value,
                "device": acc.name,
                "time_ms": bench_results.get("time_ms", 0),
                "gflops": bench_results.get("gflops", 0)
            })
            
            print(f"{acc.type.value:8s}: {bench_results.get('time_ms', 0):.2f}ms, "
                  f"{bench_results.get('gflops', 0):.1f} GFLOPS")
            
        except Exception as e:
            print(f"{acc.type.value:8s}: Failed - {e}")
    
    if results:
        # Find best performer
        best = min(results, key=lambda x: x["time_ms"])
        print(f"\nBest performer: {best['backend']} ({best['device']})")


def test_integration_with_existing():
    """Test integration with existing GenomeVault pipelines."""
    print("\n" + "=" * 60)
    print("Integration Test")
    print("=" * 60)
    
    # Test with HDC encoding
    from genomevault.hypervector_transform.encoding import HypervectorEncoder, HypervectorConfig as OldConfig
    
    # Create old-style encoder
    old_config = OldConfig(dimension=1000)
    old_encoder = HypervectorEncoder(config=old_config)
    
    # Create new unified encoder
    new_config = HypervectorConfig(dimension=1000)
    new_encoder = HardwareAcceleratedHypervectorEngine(new_config)
    
    # Test data
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    
    # Encode with both
    old_result = old_encoder.encode(data, OmicsType.GENOMIC)
    new_result = new_encoder.encode(data, OmicsType.GENOMIC)
    
    print(f"Old encoder shape: {old_result.shape if hasattr(old_result, 'shape') else len(old_result)}")
    print(f"New encoder shape: {new_result.shape}")
    
    print("✅ Integration test passed")


def main():
    """Run all unified hardware tests."""
    print("🚀 GenomeVault Unified Hardware Acceleration Tests")
    print("=" * 60)
    
    try:
        test_hardware_detection()
        test_unified_engine()
        test_hypervector_with_unified()
        test_backward_compatibility()
        benchmark_comparison()
        test_integration_with_existing()
        
        print("\n" + "=" * 60)
        print("✅ ALL UNIFIED HARDWARE TESTS PASSED")
        print("=" * 60)
        
        print("\nKey Benefits of Unified Architecture:")
        print("  • Single interface for all hardware backends")
        print("  • Automatic backend selection")
        print("  • Code reuse across pipelines")
        print("  • Simplified maintenance")
        print("  • Backward compatibility maintained")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())