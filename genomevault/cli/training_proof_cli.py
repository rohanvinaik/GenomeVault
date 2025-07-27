"""
CLI Tool for Training Proof Verification

This module provides command-line tools for verifying ML model training proofs
and attestations in GenomeVault.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import click
import matplotlib.pyplot as plt
from tabulate import tabulate

from genomevault.blockchain.contracts.training_attestation import TrainingAttestationContract
from genomevault.hypervector.visualization.projector import ModelEvolutionVisualizer
from genomevault.local_processing.model_snapshot import SnapshotVerifier
from genomevault.utils.logging import get_logger
from genomevault.zk_proofs.circuits.multi_modal_training_proof import MultiModalTrainingProof
from genomevault.zk_proofs.circuits.training_proof import TrainingProofCircuit

logger = get_logger(__name__)


@click.group()
def training_proof_cli() -> None:
       """TODO: Add docstring for training_proof_cli"""
     """GenomeVault Training Proof Verification CLI"""
    pass


@training_proof_cli.command()
@click.option("--proof-file", "-p", required=True, help="Path to proof file")
@click.option("--snapshot-dir", "-s", required=True, help="Directory containing model snapshots")
@click.option("--output", "-o", help="Output file for verification report")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def verify_proof(proof_file: str, snapshot_dir: str, output: Optional[str], verbose: bool) -> None:
       """TODO: Add docstring for verify_proof"""
     """Verify a training proof against model snapshots"""

    click.echo("🔍 Verifying training proof...")

    # Load proof
    with open(proof_file, "r") as f:
        proof_data = json.load(f)

    # Load snapshots
    snapshot_path = Path(snapshot_dir)
    snapshot_files = sorted(snapshot_path.glob("snapshot_*/metadata.json"))

    if not snapshot_files:
        click.echo("❌ No snapshots found in directory", err=True)
        return

    click.echo(f"Found {len(snapshot_files)} snapshots")

    # Verify snapshot chain
    snapshots = []
    for snap_file in snapshot_files:
        with open(snap_file, "r") as f:
            snapshots.append(json.load(f))

    # Basic verification
    verification_results = {"proof_valid": True, "checks": {}}

    # Check 1: Snapshot count matches
    expected_snapshots = proof_data.get("public_inputs", {}).get("num_snapshots", 0)
    actual_snapshots = len(snapshots)

    snapshot_check = expected_snapshots == actual_snapshots
    verification_results["checks"]["snapshot_count"] = {
        "passed": snapshot_check,
        "expected": expected_snapshots,
        "actual": actual_snapshots,
    }

    if not snapshot_check:
        click.echo(
            f"❌ Snapshot count mismatch: expected {expected_snapshots}, got {actual_snapshots}"
        )
        verification_results["proof_valid"] = False
    else:
        click.echo(f"✅ Snapshot count verified: {actual_snapshots}")

    # Check 2: Final model hash
    if snapshots:
        final_snapshot = snapshots[-1]
        expected_hash = proof_data.get("public_inputs", {}).get("final_model_hash", "")
        actual_hash = final_snapshot.get("model_hash", "")

        hash_check = expected_hash == actual_hash
        verification_results["checks"]["final_model_hash"] = {
            "passed": hash_check,
            "expected": expected_hash[:16] + "...",
            "actual": actual_hash[:16] + "...",
        }

        if not hash_check:
            click.echo(f"❌ Final model hash mismatch")
            verification_results["proof_valid"] = False
        else:
            click.echo(f"✅ Final model hash verified")

    # Check 3: Training timeline
    start_time = proof_data.get("public_inputs", {}).get("training_start_time", 0)
    end_time = proof_data.get("public_inputs", {}).get("training_end_time", 0)

    if snapshots:
        actual_start = snapshots[0].get("timestamp", 0)
        actual_end = snapshots[-1].get("timestamp", 0)

        timeline_check = (
            abs(start_time - actual_start) < 60  # Within 1 minute
            and abs(end_time - actual_end) < 60
        )

        verification_results["checks"]["timeline"] = {
            "passed": timeline_check,
            "start_delta": abs(start_time - actual_start),
            "end_delta": abs(end_time - actual_end),
        }

        if not timeline_check:
            click.echo(f"❌ Timeline mismatch")
            verification_results["proof_valid"] = False
        else:
            click.echo(f"✅ Training timeline verified")

    # Check 4: Merkle root
    if "snapshot_merkle_root" in proof_data.get("commitments", {}):
        # Compute merkle root from snapshots
        snapshot_hashes = [s.get("model_hash", "") for s in snapshots]
        computed_root = compute_merkle_root(snapshot_hashes)
        expected_root = proof_data["commitments"]["snapshot_merkle_root"]

        merkle_check = computed_root == expected_root
        verification_results["checks"]["merkle_root"] = {
            "passed": merkle_check,
            "expected": expected_root[:16] + "...",
            "computed": computed_root[:16] + "...",
        }

        if not merkle_check:
            click.echo(f"❌ Merkle root mismatch")
            verification_results["proof_valid"] = False
        else:
            click.echo(f"✅ Merkle root verified")

    # Summary
    click.echo("\n📊 Verification Summary:")

    if verbose:
        # Detailed table
        table_data = []
        for check_name, check_result in verification_results["checks"].items():
            status = "✅ PASS" if check_result["passed"] else "❌ FAIL"
            table_data.append([check_name, status, json.dumps(check_result, indent=2)])

        click.echo(tabulate(table_data, headers=["Check", "Status", "Details"]))

    if verification_results["proof_valid"]:
        click.echo("\n✅ Proof verification PASSED")
    else:
        click.echo("\n❌ Proof verification FAILED")

    # Save report
    if output:
        with open(output, "w") as f:
            json.dump(verification_results, f, indent=2)
        click.echo(f"\n💾 Verification report saved to {output}")

    return verification_results["proof_valid"]


@training_proof_cli.command()
@click.option("--snapshot-dir", "-s", required=True, help="Directory containing model snapshots")
@click.option(
    "--output-dir", "-o", default="./drift_analysis", help="Output directory for visualizations"
)
@click.option("--threshold", "-t", default=0.15, help="Drift detection threshold")
def analyze_drift(snapshot_dir: str, output_dir: str, threshold: float) -> None:
       """TODO: Add docstring for analyze_drift"""
     """Analyze semantic drift in model training"""

    click.echo("📈 Analyzing semantic drift...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load snapshots
    snapshot_path = Path(snapshot_dir)
    snapshot_files = sorted(snapshot_path.glob("snapshot_*/"))

    if not snapshot_files:
        click.echo("❌ No snapshots found", err=True)
        return

    # Load hypervectors
    hypervectors = []
    labels = []

    for snap_dir in snapshot_files:
        hypervector_file = snap_dir / "hypervector.npy"
        metadata_file = snap_dir / "metadata.json"

        if hypervector_file.exists() and metadata_file.exists():
            import numpy as np

            hypervector = np.load(hypervector_file)
            hypervectors.append(hypervector)

            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                labels.append(f"Epoch {metadata['epoch']}")

    click.echo(f"Loaded {len(hypervectors)} hypervectors")

    # Create visualizer
    visualizer = ModelEvolutionVisualizer(str(output_path))

    # Visualize semantic space
    click.echo("Creating semantic space visualization...")
    visualizer.visualize_semantic_space(
        hypervectors,
        labels,
        title="Model Training Semantic Evolution",
        save_path=str(output_path / "semantic_evolution.png"),
    )

    # Detect drift
    click.echo("Detecting semantic drift...")
    drift_scores, anomalies = visualizer.detect_semantic_drift(hypervectors, threshold=threshold)

    # Plot drift analysis
    visualizer.plot_drift_analysis(
        drift_scores,
        anomalies,
        labels[1:],  # Skip first label
        save_path=str(output_path / "drift_analysis.png"),
    )

    # Analyze trajectory
    projections = visualizer.projections.get("umap", np.array(hypervectors))
    trajectory_metrics = visualizer.analyze_trajectory_smoothness(projections)

    # Print summary
    click.echo("\n📊 Drift Analysis Summary:")
    click.echo(f"Total snapshots: {len(hypervectors)}")
    click.echo(f"Anomalies detected: {len(anomalies)}")
    click.echo(f"Average drift: {np.mean(drift_scores):.4f}")
    click.echo(f"Max drift: {np.max(drift_scores):.4f}")
    click.echo(f"Trajectory smoothness: {trajectory_metrics['smoothness']:.3f}")
    click.echo(f"Path efficiency: {trajectory_metrics['efficiency']:.3f}")

    # Save detailed report
    report = {
        "summary": {
            "total_snapshots": len(hypervectors),
            "anomalies": len(anomalies),
            "avg_drift": float(np.mean(drift_scores)),
            "max_drift": float(np.max(drift_scores)),
            "anomaly_epochs": [labels[i] for i in anomalies],
        },
        "trajectory": trajectory_metrics,
        "drift_scores": [float(d) for d in drift_scores],
    }

    with open(output_path / "drift_report.json", "w") as f:
        json.dump(report, f, indent=2)

    click.echo(f"\n💾 Analysis saved to {output_path}")


@training_proof_cli.command()
@click.option(
    "--contract-address", "-c", required=True, help="Training attestation contract address"
)
@click.option("--attestation-id", "-a", required=True, help="Attestation ID to query")
@click.option("--chain-id", "-n", default=1, help="Blockchain network ID")
@click.option("--verify", "-v", is_flag=True, help="Verify attestation on-chain")
def check_attestation(contract_address: str, attestation_id: str, chain_id: int, verify: bool) -> None:
       """TODO: Add docstring for check_attestation"""
     """Check training attestation on blockchain"""

    click.echo(f"🔗 Checking attestation {attestation_id}...")

    # Create contract instance (mock for CLI demo)
    contract = TrainingAttestationContract(contract_address, chain_id)

    # Get attestation
    attestation = contract.get_attestation(attestation_id)

    if not attestation:
        click.echo(f"❌ Attestation {attestation_id} not found", err=True)
        return

    # Display attestation info
    click.echo("\n📋 Attestation Details:")
    click.echo(f"Status: {attestation['status']}")
    click.echo(f"Model Hash: {attestation['model_hash'][:32]}...")
    click.echo(f"Dataset Hash: {attestation['dataset_hash'][:32]}...")
    click.echo(f"Submitter: {attestation['submitter']}")
    click.echo(f"Timestamp: {attestation['timestamp']}")

    # Verification summary
    click.echo(f"\n✅ Positive verifications: {attestation['positive_verifications']}")
    click.echo(f"❌ Negative verifications: {attestation['negative_verifications']}")

    if verify:
        # Get detailed verifications
        verifications = contract.get_verifications(attestation_id)

        if verifications:
            click.echo("\n🔍 Verification History:")

            table_data = []
            for v in verifications:
                result = "✅" if v["result"] else "❌"
                table_data.append(
                    [
                        v["verifier"][:16] + "...",
                        result,
                        v["timestamp"],
                        v["notes"][:50] if v["notes"] else "",
                    ]
                )

            click.echo(tabulate(table_data, headers=["Verifier", "Result", "Timestamp", "Notes"]))

    # Check if disputed
    if attestation["status"] == "disputed":
        click.echo("\n⚠️  WARNING: This attestation has been disputed!")


@training_proof_cli.command()
@click.option("--session-id", "-s", required=True, help="Training session ID")
@click.option("--proof-file", "-p", required=True, help="Path to save proof")
@click.option("--snapshot-dir", "-d", required=True, help="Directory containing snapshots")
@click.option("--dataset-hash", "-h", required=True, help="Hash of training dataset")
def generate_proof(session_id: str, proof_file: str, snapshot_dir: str, dataset_hash: str) -> None:
       """TODO: Add docstring for generate_proof"""
     """Generate a training proof from snapshots"""

    click.echo("🔨 Generating training proof...")

    # Load snapshots
    from genomevault.local_processing.model_snapshot import ModelSnapshotLogger

    snapshot_logger = ModelSnapshotLogger(session_id, Path(snapshot_dir).parent)

    # Export proof data
    proof_data = snapshot_logger.export_for_proof()

    if not proof_data.get("snapshot_hashes"):
        click.echo("❌ No snapshots found for proof generation", err=True)
        return

    # Create proof circuit
    circuit = TrainingProofCircuit(max_snapshots=len(proof_data["snapshot_hashes"]))

    # Setup inputs
    summary = proof_data["summary"]
    public_inputs = {
        "final_model_hash": proof_data["snapshot_hashes"][-1],
        "training_metadata": {
            "start_time": summary["start_time"],
            "end_time": summary["end_time"],
            "dataset_hash": dataset_hash,
        },
    }

    private_inputs = {
        "snapshot_hashes": proof_data["snapshot_hashes"],
        "model_commit": hashlib.sha256(f"{session_id}_model".encode()).hexdigest(),
        "io_sequence_commit": hashlib.sha256(f"{session_id}_io".encode()).hexdigest(),
        "training_snapshots": proof_data["snapshots"],
    }

    # Setup and generate proof
    circuit.setup(public_inputs, private_inputs)
    proof = circuit.generate_proof()

    # Add session info
    proof["session_id"] = session_id
    proof["generation_time"] = int(time.time())

    # Save proof
    with open(proof_file, "w") as f:
        json.dump(proof, f, indent=2)

    click.echo(f"✅ Proof generated and saved to {proof_file}")
    click.echo(f"   Snapshots: {len(proof_data['snapshot_hashes'])}")
    click.echo(f"   Duration: {summary['duration_seconds']}s")
    click.echo(f"   Merkle root: {proof['commitments']['snapshot_merkle_root'][:32]}...")


def compute_merkle_root(hashes: list) -> str:
       """TODO: Add docstring for compute_merkle_root"""
     """Compute Merkle root of hashes"""
    if not hashes:
        return "0" * 64

    current_level = hashes.copy()

    while len(current_level) > 1:
        next_level = []

        for i in range(0, len(current_level), 2):
            if i + 1 < len(current_level):
                left, right = current_level[i], current_level[i + 1]
            else:
                left, right = current_level[i], current_level[i]

            combined = hashlib.sha256(f"{left}{right}".encode()).hexdigest()
            next_level.append(combined)

        current_level = next_level

    return current_level[0]


if __name__ == "__main__":
    training_proof_cli()
