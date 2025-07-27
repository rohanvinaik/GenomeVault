"""
Training script for KAN Hybrid Architecture

This script demonstrates how to train the KAN models on genomic data
to achieve optimal compression while maintaining reconstruction quality.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from genomevault.hypervector.encoding.genomic import GenomicEncoder
from genomevault.hypervector.kan import KANCompressor, KANHybridEncoder


class GenomicDataset(Dataset):
    """Dataset for training KAN models on genomic data"""

    def __init__(self, num_samples: int = 10000, variants_per_sample: int = 100) -> None:
            """TODO: Add docstring for __init__"""
    self.num_samples = num_samples
        self.variants_per_sample = variants_per_sample

        # Pre-generate hypervectors for training
        print(f"Generating {num_samples} training samples...")
        self.encoder = GenomicEncoder(dimension=10000)
        self.samples = []

        for i in tqdm(range(num_samples)):
            # Generate random variants
            variants = self._generate_variants(variants_per_sample)
            # Encode to hypervector
            hv = self.encoder.encode_genome(variants)
            self.samples.append(hv)

    def _generate_variants(self, n: int) -> List[Dict]:
           """TODO: Add docstring for _generate_variants"""
     """Generate random genomic variants"""
        variants = []
        for _ in range(n):
            variants.append(
                {
                    "chromosome": f"chr{np.random.randint(1, 23)}",
                    "position": np.random.randint(1, 250000000),
                    "ref": np.random.choice(["A", "T", "G", "C"]),
                    "alt": np.random.choice(["A", "T", "G", "C"]),
                    "type": "SNP",
                }
            )
        return variants

    def __len__(self) -> None:
            """TODO: Add docstring for __len__"""
    return self.num_samples

    def __getitem__(self, idx) -> None:
            """TODO: Add docstring for __getitem__"""
    return self.samples[idx]


def train_kan_compressor(
    compressor: KANCompressor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    device: str = "cpu",
) -> Dict[str, List[float]]:
       """TODO: Add docstring for train_kan_compressor"""
     """
    Train KAN compressor model

    Returns:
        Training history dictionary
    """
    compressor = compressor.to(device)

    # Loss function - combination of reconstruction and sparsity
    reconstruction_loss = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(compressor.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # Training history
    history = {"train_loss": [], "val_loss": [], "compression_ratio": []}

    print("\nStarting training...")
    for epoch in range(num_epochs):
        # Training phase
        compressor.train()
        train_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch = batch.to(device)

            # Forward pass
            compressed = compressor.encode(batch)
            reconstructed = compressor.decode(compressed)

            # Compute losses
            recon_loss = reconstruction_loss(reconstructed, batch)

            # Sparsity loss to encourage efficient representation
            sparsity_loss = torch.mean(torch.abs(compressed)) * 0.01

            # Total loss
            loss = recon_loss + sparsity_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation phase
        compressor.eval()
        val_losses = []
        compression_ratios = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # Forward pass
                compressed = compressor.encode(batch)
                reconstructed = compressor.decode(compressed)

                # Compute loss
                loss = reconstruction_loss(reconstructed, batch)
                val_losses.append(loss.item())

                # Compute compression ratio
                original_size = batch.numel() * batch.element_size()
                compressed_bytes = compressor.compress_to_bytes(batch[:1])
                compressed_size = len(compressed_bytes)
                compression_ratios.append(original_size / compressed_size)

        # Update learning rate
        scheduler.step()

        # Record history
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_compression = np.mean(compression_ratios)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["compression_ratio"].append(avg_compression)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
                f"Val Loss={avg_val_loss:.4f}, "
                f"Compression={avg_compression:.1f}x"
            )

    return history


def evaluate_kan_model(
    compressor: KANCompressor, test_loader: DataLoader, device: str = "cpu"
) -> Dict[str, float]:
       """TODO: Add docstring for evaluate_kan_model"""
     """
    Evaluate trained KAN model

    Returns:
        Evaluation metrics
    """
    compressor = compressor.to(device)
    compressor.eval()

    reconstruction_errors = []
    compression_ratios = []
    encoding_times = []
    decoding_times = []

    print("\nEvaluating model...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)

            # Test compression pipeline
            import time

            # Encoding
            start = time.time()
            compressed_bytes = compressor.compress_to_bytes(batch)
            encoding_time = time.time() - start

            # Decoding
            start = time.time()
            reconstructed = compressor.decompress_from_bytes(compressed_bytes, batch.shape[0])
            decoding_time = time.time() - start

            # Metrics
            mse = torch.mean((batch - reconstructed) ** 2).item()
            reconstruction_errors.append(mse)

            original_size = batch.numel() * batch.element_size()
            compression_ratio = original_size / len(compressed_bytes)
            compression_ratios.append(compression_ratio)

            encoding_times.append(encoding_time)
            decoding_times.append(decoding_time)

    # Aggregate metrics
    metrics = {
        "mean_reconstruction_error": np.mean(reconstruction_errors),
        "std_reconstruction_error": np.std(reconstruction_errors),
        "mean_compression_ratio": np.mean(compression_ratios),
        "mean_encoding_time": np.mean(encoding_times),
        "mean_decoding_time": np.mean(decoding_times),
        "throughput_encode": 1.0 / np.mean(encoding_times),
        "throughput_decode": 1.0 / np.mean(decoding_times),
    }

    return metrics


def visualize_training_history(history: Dict[str, List[float]]) -> None:
       """TODO: Add docstring for visualize_training_history"""
     """Visualize training history"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Loss curves
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Compression ratio
    axes[1].plot(history["compression_ratio"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Compression Ratio")
    axes[1].set_title("Compression Ratio Over Time")
    axes[1].grid(True)

    # Loss vs Compression trade-off
    axes[2].scatter(history["compression_ratio"], history["val_loss"])
    axes[2].set_xlabel("Compression Ratio")
    axes[2].set_ylabel("Validation Loss")
    axes[2].set_title("Loss vs Compression Trade-off")
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig("kan_training_history.png", dpi=150)
    plt.show()


def analyze_learned_functions(compressor: KANCompressor, num_functions: int = 5) -> None:
       """TODO: Add docstring for analyze_learned_functions"""
     """Analyze and visualize learned KAN functions"""
    print("\nAnalyzing learned functions...")

    # Extract first layer if it's a KAN layer
    first_layer = None
    for module in compressor.encoder.modules():
        if hasattr(module, "get_symbolic_expression"):
            first_layer = module
            break

    if first_layer is None:
        print("No KAN layers found with symbolic expression support")
        return

    # Visualize some learned functions
    fig, axes = plt.subplots(2, num_functions, figsize=(15, 6))

    for i in range(min(num_functions, first_layer.in_features)):
        for j in range(min(2, first_layer.out_features)):
            x, y = first_layer.get_symbolic_expression(j, i)
            axes[j, i].plot(x, y, linewidth=2)
            axes[j, i].set_title(f"φ_{{{j},{i}}}(x)")
            axes[j, i].grid(True, alpha=0.3)
            axes[j, i].set_xlabel("Input")
            axes[j, i].set_ylabel("Output")

    plt.suptitle("Learned KAN Functions (First Layer)", fontsize=14)
    plt.tight_layout()
    plt.savefig("kan_learned_functions.png", dpi=150)
    plt.show()


def main() -> None:
       """TODO: Add docstring for main"""
     """Main training pipeline"""
    # Configuration
    config = {
        "input_dim": 10000,
        "compressed_dim": 100,
        "batch_size": 32,
        "num_epochs": 50,
        "learning_rate": 1e-3,
        "num_train_samples": 5000,
        "num_val_samples": 1000,
        "num_test_samples": 1000,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    print("KAN Compressor Training Pipeline")
    print("=" * 50)
    print(f"Configuration: {config}")

    # Create datasets
    print("\n1. Creating datasets...")
    train_dataset = GenomicDataset(config["num_train_samples"])
    val_dataset = GenomicDataset(config["num_val_samples"])
    test_dataset = GenomicDataset(config["num_test_samples"])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize model
    print("\n2. Initializing KAN compressor...")
    compressor = KANCompressor(
        input_dim=config["input_dim"],
        compressed_dim=config["compressed_dim"],
        num_layers=3,
        use_linear=True,  # Use LinearKAN for faster training
    )

    # Train model
    print("\n3. Training model...")
    history = train_kan_compressor(
        compressor,
        train_loader,
        val_loader,
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        device=config["device"],
    )

    # Evaluate model
    print("\n4. Evaluating model...")
    metrics = evaluate_kan_model(compressor, test_loader, device=config["device"])

    print("\nEvaluation Results:")
    print(
        f"Mean Reconstruction Error: {metrics['mean_reconstruction_error']:.6f} "
        f"± {metrics['std_reconstruction_error']:.6f}"
    )
    print(f"Mean Compression Ratio: {metrics['mean_compression_ratio']:.1f}x")
    print(f"Encoding Throughput: {metrics['throughput_encode']:.1f} samples/sec")
    print(f"Decoding Throughput: {metrics['throughput_decode']:.1f} samples/sec")

    # Visualize results
    print("\n5. Visualizing results...")
    visualize_training_history(history)
    analyze_learned_functions(compressor)

    # Save model
    print("\n6. Saving trained model...")
    torch.save(
        {
            "model_state_dict": compressor.state_dict(),
            "config": config,
            "metrics": metrics,
            "history": history,
        },
        "kan_compressor_trained.pth",
    )

    print("\nTraining completed successfully!")
    print("Model saved to: kan_compressor_trained.pth")

    # Test on real example
    print("\n7. Testing on example data...")
    test_variants = [
        {"chromosome": "chr1", "position": 1000000, "ref": "A", "alt": "G", "type": "SNP"},
        {"chromosome": "chr2", "position": 2000000, "ref": "C", "alt": "T", "type": "SNP"},
        {"chromosome": "chr3", "position": 3000000, "ref": "G", "alt": "A", "type": "SNP"},
    ]

    encoder = GenomicEncoder(dimension=10000)
    test_hv = encoder.encode_genome(test_variants).unsqueeze(0)

    compressed_bytes = compressor.compress_to_bytes(test_hv)
    reconstructed = compressor.decompress_from_bytes(compressed_bytes, 1)

    similarity = torch.cosine_similarity(test_hv, reconstructed).item()
    print(f"\nTest reconstruction similarity: {similarity:.4f}")
    print(f"Compressed size: {len(compressed_bytes)} bytes")
    print(f"Original size: {test_hv.numel() * test_hv.element_size()} bytes")
    print(
        f"Achieved compression: {test_hv.numel() * test_hv.element_size() / len(compressed_bytes):.1f}x"
    )


if __name__ == "__main__":
    main()
