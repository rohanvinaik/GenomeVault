#!/usr/bin/env python3
"""Minimal test to verify basic functionality"""

def test_basic_imports():
    """Test that basic imports work"""
    # Core imports
    from core.config import Config, get_config
    from core.exceptions import GenomeVaultError
    
    # Utils imports
    from utils.logging import get_logger
    from utils.encryption import AESGCMCipher, secure_hash
    
    # Get instances
    config = get_config()
    logger = get_logger(__name__)
    
    # Basic operations
    assert config is not None
    assert logger is not None
    
    # Test secure_hash
    test_data = b"test data"
    hash_result = secure_hash(test_data)
    assert len(hash_result) == 64  # SHA256 produces 64 hex chars
    
    print("✅ All basic imports and operations work!")
    return True

def test_hypervector():
    """Test hypervector functionality"""
    from hypervector_transform.encoding import HypervectorEncoder
    from hypervector_transform.binding import circular_bind
    import torch
    
    # Create encoder
    encoder = HypervectorEncoder(dimension=1000)
    
    # Create test vectors
    v1 = torch.randn(1000)
    v2 = torch.randn(1000)
    
    # Test binding
    result = circular_bind([v1, v2])
    assert result.shape[0] == 1000
    
    print("✅ Hypervector operations work!")
    return True

def test_sequencing():
    """Test sequencing processor"""
    from local_processing.sequencing import SequencingProcessor
    
    # Create processor
    processor = SequencingProcessor()
    
    # Just verify it initializes
    assert processor is not None
    
    print("✅ Sequencing processor initializes!")
    return True

if __name__ == "__main__":
    import sys
    
    print("🧪 Running minimal functionality tests...")
    print("=" * 50)
    
    all_passed = True
    
    tests = [
        ("Basic imports", test_basic_imports),
        ("Hypervector", test_hypervector),
        ("Sequencing", test_sequencing),
    ]
    
    for name, test_func in tests:
        try:
            print(f"\nTesting {name}...")
            test_func()
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)
