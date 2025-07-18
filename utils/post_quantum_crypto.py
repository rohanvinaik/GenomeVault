(plaintext),
        'success': decrypted == plaintext
    }
    
    return results


# Example usage
if __name__ == "__main__":
    # Initialize hybrid system
    crypto = HybridPostQuantumCrypto()
    
    # Example: Encrypt genomic data
    genomic_data = b"ACGTACGTACGT" * 1000
    
    # Generate recipient keypair
    recipient_keypair = crypto.kyber.generate_keypair()
    
    # Encrypt
    encrypted = crypto.encrypt(genomic_data, recipient_keypair.public_key)
    print(f"Encrypted size: {len(encrypted.to_bytes())} bytes")
    
    # Decrypt
    decrypted = crypto.decrypt(encrypted, recipient_keypair.private_key)
    assert decrypted == genomic_data
    print("Decryption successful!")
    
    # Benchmark
    print("\nBenchmarking post-quantum crypto...")
    results = benchmark_post_quantum_crypto()
    
    for algo, metrics in results.items():
        print(f"\n{algo}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
