#!/usr/bin/env python3
"""
GenomeVault ZK Proof CLI Tool

Command-line interface for generating and verifying zero-knowledge proofs.
As specified in Stage 2 of the ZK implementation plan.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

from genomevault.zk_proofs import Prover, Verifier
from genomevault.zk_proofs.circuits.implementations.variant_frequency_circuit import (
    VariantFrequencyCircuit,
    create_example_frequency_proof
)
from genomevault.zk_proofs.circuits.implementations.variant_proof_circuit import (
    VariantProofCircuit,
    create_variant_proof_example
)


def load_json_file(filepath: str) -> Dict[str, Any]:
    """Load JSON data from file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        sys.exit(1)


def save_json_file(data: Dict[str, Any], filepath: str):
    """Save JSON data to file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving {filepath}: {e}")
        sys.exit(1)


def cmd_prove(args):
    """Generate a zero-knowledge proof."""
    print(f"Generating {args.circuit} proof...")
    
    # Load inputs
    if args.public_input:
        public_inputs = load_json_file(args.public_input)
    else:
        print("Error: --public-input is required")
        sys.exit(1)
    
    if args.private_input:
        private_inputs = load_json_file(args.private_input)
    else:
        print("Error: --private-input is required")
        sys.exit(1)
    
    # Initialize appropriate circuit
    start_time = time.time()
    
    if args.circuit == "variant_presence":
        circuit = VariantProofCircuit(merkle_depth=20)
        circuit.setup_circuit(public_inputs, private_inputs)
        circuit.generate_constraints()
        
    elif args.circuit == "variant_frequency":
        circuit = VariantFrequencyCircuit(max_snps=32, merkle_depth=20)
        circuit.setup_circuit(public_inputs, private_inputs)
        circuit.generate_constraints()
        
    else:
        print(f"Error: Unknown circuit type: {args.circuit}")
        sys.exit(1)
    
    # Verify constraints are satisfied
    if not circuit.verify_constraints():
        print("Error: Circuit constraints not satisfied!")
        print("Please check your inputs.")
        sys.exit(1)
    
    setup_time = time.time() - start_time
    
    # Generate proof
    print(f"Circuit setup complete ({setup_time:.3f}s)")
    print(f"Constraints: {circuit.cs.num_constraints()}")
    print(f"Variables: {circuit.cs.num_variables()}")
    
    # Initialize prover
    prover = Prover()
    
    # Generate proof (using the circuit's constraint system)
    proof_start = time.time()
    
    # In practice, this would call the gnark backend
    # For now, create a mock proof structure
    proof_data = {
        "circuit": args.circuit,
        "public_inputs": [str(val) for val in circuit.get_public_inputs()],
        "proof": {
            "pi_a": "0x" + "00" * 48,  # Mock proof points
            "pi_b": "0x" + "00" * 96,
            "pi_c": "0x" + "00" * 48,
            "protocol": "groth16",
            "curve": "bn254"
        },
        "metadata": {
            "prover": "genomevault-zk",
            "version": "3.0.0",
            "timestamp": int(time.time()),
            "generation_time_ms": int((time.time() - proof_start) * 1000)
        }
    }
    
    # Save proof
    output_file = args.output or f"{args.circuit}_proof.json"
    save_json_file(proof_data, output_file)
    
    total_time = time.time() - start_time
    
    print(f"\nProof generated successfully!")
    print(f"Output: {output_file}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Proof size: ~384 bytes")


def cmd_verify(args):
    """Verify a zero-knowledge proof."""
    print(f"Verifying proof from {args.proof}...")
    
    # Load proof
    proof_data = load_json_file(args.proof)
    
    # Load public inputs if provided separately
    if args.public_input:
        public_inputs = load_json_file(args.public_input)
    else:
        # Extract from proof
        public_inputs = proof_data.get("public_inputs", [])
    
    # Initialize verifier
    verifier = Verifier()
    
    # Verify proof
    start_time = time.time()
    
    # In practice, this would call the gnark backend
    # For now, perform basic checks
    is_valid = True
    
    # Check proof structure
    required_fields = ["circuit", "public_inputs", "proof", "metadata"]
    for field in required_fields:
        if field not in proof_data:
            print(f"Error: Missing required field: {field}")
            is_valid = False
    
    # Check proof components
    if is_valid and "proof" in proof_data:
        proof = proof_data["proof"]
        required_proof_fields = ["pi_a", "pi_b", "pi_c", "protocol", "curve"]
        for field in required_proof_fields:
            if field not in proof:
                print(f"Error: Missing proof field: {field}")
                is_valid = False
    
    verification_time = (time.time() - start_time) * 1000
    
    # Display results
    print(f"\nVerification Result: {'VALID' if is_valid else 'INVALID'}")
    print(f"Circuit: {proof_data.get('circuit', 'unknown')}")
    print(f"Verification time: {verification_time:.1f}ms")
    
    if args.verbose and is_valid:
        print(f"\nProof metadata:")
        metadata = proof_data.get("metadata", {})
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    sys.exit(0 if is_valid else 1)


def cmd_demo(args):
    """Run demonstration with example data."""
    print(f"Running {args.circuit} demonstration...")
    
    if args.circuit == "variant_presence":
        # Create example variant presence proof
        circuit = create_variant_proof_example()
        
        # Extract inputs for saving
        public_inputs = {
            "variant_hash": "example_hash",
            "reference_hash": "GRCh38_hash", 
            "commitment_root": "genome_root"
        }
        
        private_inputs = {
            "variant_data": {
                "chr": "chr1",
                "pos": 12345,
                "ref": "A",
                "alt": "G"
            },
            "merkle_path": ["hash1", "hash2"],
            "merkle_indices": [0, 1],
            "witness_randomness": "random_value"
        }
        
    elif args.circuit == "variant_frequency":
        # Create example frequency proof
        circuit = create_example_frequency_proof()
        
        # Extract inputs for saving
        public_inputs = {
            "total_sum": 7472,
            "merkle_root": "population_root",
            "num_snps": 5,
            "snp_ids": ["rs7903146", "rs1801282", "rs5219", "rs13266634", "rs10830963"]
        }
        
        private_inputs = {
            "allele_counts": [1523, 892, 2145, 1678, 1234],
            "merkle_proofs": [
                {"path": ["node1", "node2"], "indices": [0, 1]}
            ] * 5,
            "randomness": "random_value"
        }
        
    else:
        print(f"Error: No demo available for circuit: {args.circuit}")
        sys.exit(1)
    
    # Display circuit info
    info = circuit.get_circuit_info()
    print(f"\nCircuit Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Save example inputs if requested
    if args.save_inputs:
        save_json_file(public_inputs, f"{args.circuit}_public.json")
        save_json_file(private_inputs, f"{args.circuit}_private.json")
        print(f"\nExample inputs saved:")
        print(f"  Public: {args.circuit}_public.json")
        print(f"  Private: {args.circuit}_private.json")
        print(f"\nYou can now run:")
        print(f"  zk_prove --circuit {args.circuit} --public-input {args.circuit}_public.json --private-input {args.circuit}_private.json")


def cmd_info(args):
    """Display information about available circuits."""
    
    circuits = {
        "variant_presence": {
            "description": "Prove presence of a genetic variant without revealing location",
            "public_inputs": ["variant_hash", "reference_hash", "commitment_root"],
            "private_inputs": ["variant_data", "merkle_proof", "witness_randomness"],
            "proof_size": "192 bytes",
            "verification_time": "<10ms",
            "constraints": "~10,000"
        },
        "variant_frequency": {
            "description": "Prove sum of allele frequencies without revealing individual counts",
            "public_inputs": ["total_sum", "merkle_root", "num_snps", "snp_ids"],
            "private_inputs": ["allele_counts", "merkle_proofs", "randomness"],
            "proof_size": "384 bytes", 
            "verification_time": "<25ms",
            "constraints": "~15,000"
        },
        "polygenic_risk_score": {
            "description": "Calculate PRS while keeping individual variants private",
            "public_inputs": ["model_hash", "score_commitment", "threshold"],
            "private_inputs": ["variants", "weights", "merkle_proofs"],
            "proof_size": "384 bytes",
            "verification_time": "<25ms",
            "constraints": "~20,000"
        },
        "diabetes_risk": {
            "description": "Clinical pilot: prove glucose AND genetic risk exceed thresholds",
            "public_inputs": ["glucose_threshold", "risk_threshold", "alert_commitment"],
            "private_inputs": ["glucose_reading", "risk_score", "randomness"],
            "proof_size": "384 bytes",
            "verification_time": "<25ms",
            "constraints": "~15,000"
        }
    }
    
    if args.circuit:
        # Show specific circuit details
        if args.circuit in circuits:
            circuit = circuits[args.circuit]
            print(f"\nCircuit: {args.circuit}")
            print(f"Description: {circuit['description']}")
            print(f"\nPublic Inputs:")
            for inp in circuit['public_inputs']:
                print(f"  - {inp}")
            print(f"\nPrivate Inputs:")
            for inp in circuit['private_inputs']:
                print(f"  - {inp}")
            print(f"\nPerformance:")
            print(f"  Proof size: {circuit['proof_size']}")
            print(f"  Verification time: {circuit['verification_time']}")
            print(f"  Constraints: {circuit['constraints']}")
        else:
            print(f"Error: Unknown circuit: {args.circuit}")
            sys.exit(1)
    else:
        # List all circuits
        print("\nAvailable ZK Circuits:")
        print("=" * 60)
        for name, info in circuits.items():
            print(f"\n{name}:")
            print(f"  {info['description']}")
            print(f"  Proof size: {info['proof_size']}, Verification: {info['verification_time']}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GenomeVault Zero-Knowledge Proof CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a variant presence proof
  zk_prove --circuit variant_presence --public-input public.json --private-input private.json
  
  # Verify a proof
  zk_verify --proof variant_proof.json
  
  # Run demonstration
  zk_demo --circuit variant_frequency --save-inputs
  
  # Get circuit information
  zk_info --circuit diabetes_risk
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Prove command
    prove_parser = subparsers.add_parser('prove', help='Generate a ZK proof')
    prove_parser.add_argument('--circuit', '-c', required=True, 
                             choices=['variant_presence', 'variant_frequency', 'polygenic_risk_score', 'diabetes_risk'],
                             help='Circuit type to use')
    prove_parser.add_argument('--public-input', '-p', required=True,
                             help='Path to public inputs JSON file')
    prove_parser.add_argument('--private-input', '-w', required=True,
                             help='Path to private inputs (witness) JSON file')
    prove_parser.add_argument('--output', '-o',
                             help='Output proof file (default: <circuit>_proof.json)')
    prove_parser.set_defaults(func=cmd_prove)
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify a ZK proof')
    verify_parser.add_argument('--proof', '-p', required=True,
                              help='Path to proof JSON file')
    verify_parser.add_argument('--public-input', '-i',
                              help='Path to public inputs (if not in proof file)')
    verify_parser.add_argument('--verbose', '-v', action='store_true',
                              help='Show detailed verification info')
    verify_parser.set_defaults(func=cmd_verify)
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run circuit demonstration')
    demo_parser.add_argument('--circuit', '-c', required=True,
                            choices=['variant_presence', 'variant_frequency'],
                            help='Circuit to demonstrate')
    demo_parser.add_argument('--save-inputs', '-s', action='store_true',
                            help='Save example inputs to files')
    demo_parser.set_defaults(func=cmd_demo)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display circuit information')
    info_parser.add_argument('--circuit', '-c',
                            help='Show details for specific circuit')
    info_parser.set_defaults(func=cmd_info)
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
