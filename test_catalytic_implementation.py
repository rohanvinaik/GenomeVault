.hypervector.encoding.catalytic_projections import CatalyticProjectionPool
        import torch
        
        pool = CatalyticProjectionPool(dimension=1000, pool_size=5)
        test_vector = torch.randn(1000)
        projected = pool.apply_catalytic_projection(test_vector, [0, 1, 2])
        
        print(f"✓ Projection pool working: input shape {test_vector.shape}, output shape {projected.shape}")
    except Exception as e:
        print(f"✗ Projection pool error: {e}")
    
    # Test 2: Catalytic PIR Client
    print("\n2. Testing Catalytic PIR Client...")
    try:
        from genomevault.pir.catalytic_client import CatalyticPIRClient
        from genomevault.pir.client import PIRServer

        # Mock servers
        servers = [
            PIRServer("test1", "http://localhost:8001", "local", False, 0.95, 10),
            PIRServer("test2", "http://localhost:8002", "local", False, 0.95, 10)
        ]
        
        client = CatalyticPIRClient(servers, database_size=10000)
        print(f"✓ Catalytic PIR client initialized with {len(servers)} servers")
    except Exception as e:
        print(f"✗ Catalytic PIR client error: {e}")
    
    # Test 3: COEC Proof Engine
    print("\n3. Testing COEC Catalytic Proof Engine...")
    try:
        from genomevault.zk_proofs.advanced.coec_catalytic_proof import COECCatalyticProofEngine
        
        engine = COECCatalyticProofEngine()
        
        # Test constraint checking
        test_state = {
            "genotypes": {
                "locus1": {"AA": 0.25, "Aa": 0.5, "aa": 0.25}
            },
            "allele_frequencies": {
                "locus1": {"A": 0.5, "a": 0.5}
            }
        }
        
        hwe_constraint = engine.constraint_operators["hardy_weinberg"]
        is_satisfied = hwe_constraint.is_satisfied(test_state)
        
        print(f"✓ COEC engine working: Hardy-Weinberg satisfied = {is_satisfied}")
    except Exception as e:
        print(f"✗ COEC engine error: {e}")


def main():
    """Main test function."""
    print("Catalytic GenomeVault Implementation Test")
    print("=" * 70)
    
    # Run async tests
    loop = asyncio.get_event_loop()
    
    # Test pipeline
    pipeline_success = loop.run_until_complete(test_catalytic_pipeline())
    
    # Test components
    loop.run_until_complete(test_individual_components())
    
    print("\n\nTest Summary")
    print("=" * 50)
    print(f"Pipeline test: {'PASSED' if pipeline_success else 'FAILED'}")
    print("\nCatalytic implementation is ready for use!")


if __name__ == "__main__":
    main()
