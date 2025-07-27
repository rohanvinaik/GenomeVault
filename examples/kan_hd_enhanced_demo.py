"""
KAN-HD Hybrid Enhancement Demo

This script demonstrates the enhanced features of the KAN-HD hybrid architecture
based on the insights from the KAN-HD hybrid project knowledge.

Features demonstrated:
1. Adaptive compression with 10-500x ratios
2. Hierarchical multi-modal encoding
3. Scientific interpretability analysis
4. Federated learning capabilities
5. Real-time performance tuning
6. Privacy-preserving transformations
"""
import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

# Import enhanced KAN-HD components
from genomevault.hypervector.kan import (
    CompressionStrategy,
    DataModality,
    EncodingSpecification,
    EnhancedKANHybridEncoder,
    FederatedKANCoordinator,
    FederatedKANParticipant,
    FederationConfig,
    HierarchicalHypervectorEncoder,
    InterpretableKANHybridEncoder,
)


class KANHDDemo:
    """Comprehensive demo of KAN-HD enhanced features"""
    """Comprehensive demo of KAN-HD enhanced features"""
    """Comprehensive demo of KAN-HD enhanced features"""

    def __init__(self) -> None:
    def __init__(self) -> None:
        """Initialize the KAN-HD demo"""
        """Initialize the KAN-HD demo"""
        """Initialize the KAN-HD demo"""
        self.results = {}
        print("🧬 KAN-HD Hybrid Enhancement Demo")
        print("=" * 50)

        def demo_adaptive_compression(self) -> None:
        def demo_adaptive_compression(self) -> None:
        """Demo 1: Adaptive compression strategies"""
        """Demo 1: Adaptive compression strategies"""
        """Demo 1: Adaptive compression strategies"""
        print("\n📊 Demo 1: Adaptive Compression Strategies")
        print("-" * 40)

        # Create enhanced encoder with different strategies
        strategies = [
            CompressionStrategy.ADAPTIVE,
            CompressionStrategy.FIXED,
            CompressionStrategy.OPTIMAL,
        ]

        # Mock genomic data
        mock_variants = [
            {"chromosome": "chr1", "position": 100000, "ref": "A", "alt": "G"},
            {"chromosome": "chr1", "position": 100100, "ref": "C", "alt": "T"},
            {"chromosome": "chr2", "position": 200000, "ref": "G", "alt": "A"},
            {"chromosome": "chr2", "position": 200100, "ref": "T", "alt": "C"},
        ]

        compression_results = {}

        for strategy in strategies:
            print(f"\n🔧 Testing {strategy.value} compression strategy...")

            encoder = EnhancedKANHybridEncoder(
                base_dim=1000,  # Smaller for demo
                compressed_dim=100,
                compression_strategy=strategy,
                enable_interpretability=True,
            )

            # Encode data
            try:
                compressed_data = encoder.encode_genomic_data(
                    mock_variants, compression_ratio=50.0, privacy_level="sensitive"
                )

                # Get performance metrics
                performance = encoder.get_performance_summary()

                compression_results[strategy.value] = {
                    "compressed_size": compressed_data.numel(),
                    "performance": performance,
                    "compression_ratio_achieved": performance.get(
                        "recent_avg_compression_ratio", 50.0
                    ),
                }

                print(f"   ✅ Compressed to {compressed_data.numel()} dimensions")
                print(
                    f"   📈 Compression ratio: {compression_results[strategy.value]['compression_ratio_achieved']:.1f}x"
                )

            except Exception as e:
                print(f"   ❌ Error: {e}")
                compression_results[strategy.value] = {"error": str(e)}

                self.results["adaptive_compression"] = compression_results

        # Find best strategy
        best_strategy = max(
            compression_results.keys(),
            key=lambda k: compression_results[k].get("compression_ratio_achieved", 0),
        )
        print(f"\n🏆 Best strategy: {best_strategy}")

                def demo_hierarchical_encoding(self) -> None:
                def demo_hierarchical_encoding(self) -> None:
        """Demo 2: Hierarchical multi-modal encoding"""
        """Demo 2: Hierarchical multi-modal encoding"""
        """Demo 2: Hierarchical multi-modal encoding"""
        print("\n🏗️ Demo 2: Hierarchical Multi-Modal Encoding")
        print("-" * 45)

        # Create hierarchical encoder
        hierarchical_encoder = HierarchicalHypervectorEncoder(
            base_dim=1000, enable_adaptive_dim=True
        )

        # Prepare multi-modal data
        data_dict = {
            "genomic_variants": torch.randn(32, 100),  # Mock variant data
            "gene_expression": torch.randn(32, 500),  # Mock expression data
            "epigenetic": torch.randn(32, 200),  # Mock methylation data
        }

        # Create encoding specifications
        specifications = {
            "genomic_variants": EncodingSpecification(
                modality=DataModality.GENOMIC_VARIANTS,
                target_dimension=1000,
                compression_ratio=50.0,
                privacy_level="sensitive",
                interpretability_required=True,
            ),
            "gene_expression": EncodingSpecification(
                modality=DataModality.GENE_EXPRESSION,
                target_dimension=1500,
                compression_ratio=100.0,
                privacy_level="highly_sensitive",
                interpretability_required=True,
            ),
            "epigenetic": EncodingSpecification(
                modality=DataModality.EPIGENETIC,
                target_dimension=2000,
                compression_ratio=75.0,
                privacy_level="sensitive",
                interpretability_required=True,
            ),
        }

        print("🔬 Encoding multi-modal genomic data...")

        try:
            # Encode multi-modal data
            encoded_vectors = hierarchical_encoder.encode_multimodal_data(data_dict, specifications)

            print(f"   ✅ Encoded {len(encoded_vectors)} modalities:")
            for modality, vector in encoded_vectors.items():
                print(
                    f"      - {modality}: Base={vector.base_vector.shape[-1]}D, "
                    f"Mid={vector.mid_vector.shape[-1]}D, High={vector.high_vector.shape[-1]}D"
                )

            # Bind modalities
            print("\n🔗 Binding modalities...")
            bound_vector = hierarchical_encoder.bind_multimodal_vectors(
                encoded_vectors, "hierarchical"
            )

            print(f"   ✅ Bound vector dimensions:")
            print(f"      - Base: {bound_vector.base_vector.shape[-1]}D")
            print(f"      - Mid: {bound_vector.mid_vector.shape[-1]}D")
            print(f"      - High: {bound_vector.high_vector.shape[-1]}D")

            # Extract interpretable patterns
            patterns = hierarchical_encoder.extract_interpretable_patterns(bound_vector)

            print(f"\n🧠 Interpretable patterns discovered:")
            print(f"   - Sparsity ratio: {patterns['sparsity_ratio']:.3f}")
            print(f"   - Interpretability score: {patterns['interpretability_score']:.3f}")
            print(f"   - Estimated clusters: {patterns['cluster_structure']['estimated_clusters']}")

                self.results["hierarchical_encoding"] = {
                "modalities_encoded": len(encoded_vectors),
                "bound_dimensions": {
                    "base": bound_vector.base_vector.shape[-1],
                    "mid": bound_vector.mid_vector.shape[-1],
                    "high": bound_vector.high_vector.shape[-1],
                },
                "patterns": patterns,
            }

        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["hierarchical_encoding"] = {"error": str(e)}

            def demo_scientific_interpretability(self) -> None:
            def demo_scientific_interpretability(self) -> None:
        """Demo 3: Scientific interpretability analysis"""
        """Demo 3: Scientific interpretability analysis"""
        """Demo 3: Scientific interpretability analysis"""
        print("\n🔬 Demo 3: Scientific Interpretability Analysis")
        print("-" * 46)

        # Create interpretable encoder
        interpretable_encoder = InterpretableKANHybridEncoder(base_dim=1000, compressed_dim=100)

        print("🧪 Analyzing KAN functions for biological patterns...")

        try:
            # Run interpretability analysis
            analysis_results = interpretable_encoder.analyze_interpretability()

            # Generate scientific report
            scientific_report = interpretable_encoder.generate_scientific_report()  # noqa: F841

            # Extract key insights
            total_functions = sum(
                len(analysis["discovered_functions"]) for analysis in analysis_results.values()
            )

            avg_interpretability = (
                np.mean(
                    [analysis["interpretability_score"] for analysis in analysis_results.values()]
                )
                if analysis_results
                else 0.0
            )

            discovered_types = set()
            all_insights = []

            for analysis in analysis_results.values():
                for func in analysis["discovered_functions"].values():
                    discovered_types.add(func.function_type.value)
                all_insights.extend(analysis["biological_insights"])

            print(f"   ✅ Analysis completed:")
            print(f"      - Functions analyzed: {total_functions}")
            print(f"      - Avg interpretability score: {avg_interpretability:.3f}")
            print(f"      - Function types discovered: {len(discovered_types)}")
            print(f"      - Biological insights: {len(all_insights)}")

            if discovered_types:
                print(f"\n🧬 Discovered function types:")
                for func_type in list(discovered_types)[:5]:  # Show first 5
                    print(f"      - {func_type}")

            if all_insights:
                print(f"\n💡 Key biological insights:")
                for insight in all_insights[:3]:  # Show first 3
                    print(f"      - {insight}")

            # Export results for further analysis
            export_path = (
                f"/tmp/kan_hd_scientific_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            interpretable_encoder.export_discovered_functions(export_path)

                self.results["scientific_interpretability"] = {
                "functions_analyzed": total_functions,
                "interpretability_score": avg_interpretability,
                "function_types": list(discovered_types),
                "insights_count": len(all_insights),
                "export_path": export_path,
            }

        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["scientific_interpretability"] = {"error": str(e)}

            def demo_federated_learning(self) -> None:
            def demo_federated_learning(self) -> None:
        """Demo 4: Federated learning capabilities"""
        """Demo 4: Federated learning capabilities"""
        """Demo 4: Federated learning capabilities"""
        print("\n🌐 Demo 4: Federated Learning Capabilities")
        print("-" * 44)

        print("🏥 Setting up federated learning with multiple institutions...")

        try:
            # Create federation coordinator
            federation_config = FederationConfig(
                min_participants=2,
                max_participants=5,
                privacy_budget=1.0,
                convergence_threshold=1e-3,
                max_rounds=10,
            )

            coordinator = FederatedKANCoordinator(
                base_dim=1000, compressed_dim=100, federation_config=federation_config
            )

            # Create participants (different institutions)
            participants = []
            institutions = ["hospital_a", "clinic_b", "research_c"]

            for i, institution in enumerate(institutions):
                participant = FederatedKANParticipant(
                    participant_id=f"participant_{i+1}",
                    institution_type=institution,
                    base_dim=1000,
                    compressed_dim=100,
                )
                participants.append(participant)

                # Register with coordinator
                registration = coordinator.register_participant(
                    participant_id=f"participant_{i+1}",
                    institution_type=institution,
                    data_characteristics={
                        "sample_count": 1000 + i * 500,
                        "data_quality": "high",
                        "region": f"region_{i+1}",
                    },
                )

                print(f"   ✅ Registered {institution}: {registration['participant_token'][:8]}...")

            # Simulate federated training rounds
            print(f"\n🔄 Simulating federated training...")

            for round_num in range(3):  # Simulate 3 rounds
                print(f"\n   Round {round_num + 1}:")

                # Each participant trains locally and sends updates
                for i, participant in enumerate(participants):
                    # Mock local genomic data
                    mock_local_data = [
                        {
                            "chromosome": f"chr{j}",
                            "position": 100000 + i * 1000 + j * 100,
                            "ref": "A",
                            "alt": "G",
                        }
                        for j in range(1, 6)  # 5 variants per participant
                    ]

                    # Train and get update
                    update = participant.train_local_round(
                        mock_local_data, num_epochs=2, learning_rate=0.001
                    )

                    # Send update to coordinator
                    ack = coordinator.receive_update(update)  # noqa: F841
                    print(
                        f"      {institutions[i]}: Update sent (privacy cost: {update.privacy_guarantee:.4f})"
                    )

                # Aggregate updates
                if len(coordinator.update_history) >= federation_config.min_participants:
                    aggregation_result = coordinator.aggregate_updates()
                    print(
                        f"      📊 Aggregation completed: convergence = {aggregation_result['convergence_metric']:.6f}"
                    )

            # Get federation statistics
            fed_stats = coordinator.get_federation_statistics()

            print(f"\n📈 Federation Statistics:")
            print(f"   - Current round: {fed_stats['current_round']}")
            print(f"   - Participants: {fed_stats['participants']}")
            print(f"   - Estimated rounds remaining: {fed_stats['estimated_rounds_remaining']}")

                    self.results["federated_learning"] = {
                "participants": len(participants),
                "rounds_completed": fed_stats["current_round"],
                "convergence_achieved": (
                    fed_stats["convergence_history"][-1]
                    if fed_stats["convergence_history"]
                    else None
                ),
                "privacy_budgets": fed_stats["privacy_budgets"],
            }

        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["federated_learning"] = {"error": str(e)}

            def demo_performance_tuning(self) -> None:
            def demo_performance_tuning(self) -> None:
        """Demo 5: Real-time performance tuning"""
        """Demo 5: Real-time performance tuning"""
        """Demo 5: Real-time performance tuning"""
        print("\n⚡ Demo 5: Real-Time Performance Tuning")
        print("-" * 42)

        # Create enhanced encoder
        encoder = EnhancedKANHybridEncoder(
            base_dim=1000,
            compressed_dim=100,
            compression_strategy=CompressionStrategy.ADAPTIVE,
            enable_interpretability=True,
        )

        print("🎛️ Testing performance tuning scenarios...")

        # Scenario 1: Optimize for latency
        print("\n   Scenario 1: Optimize for low latency")
        tuning_result_1 = encoder.tune_performance(
            target_latency_ms=100.0, target_compression_ratio=None
        )
        print(f"      Tuning applied: {tuning_result_1}")

        # Scenario 2: Optimize for compression ratio
        print("\n   Scenario 2: Optimize for high compression")
        tuning_result_2 = encoder.tune_performance(
            target_latency_ms=None, target_compression_ratio=200.0
        )
        print(f"      Tuning applied: {tuning_result_2}")

        # Get performance summary
        performance_summary = encoder.get_performance_summary()

        print(f"\n📊 Performance Summary:")
        if performance_summary:
            print(f"   - Current strategy: {performance_summary['current_strategy']}")
            print(f"   - Total operations: {performance_summary['total_operations']}")
            print(
                f"   - Interpretability enabled: {performance_summary['interpretability_enabled']}"
            )

            self.results["performance_tuning"] = {
            "tuning_results": [tuning_result_1, tuning_result_2],
            "performance_summary": performance_summary,
        }

            def demo_privacy_guarantees(self) -> None:
            def demo_privacy_guarantees(self) -> None:
        """Demo 6: Privacy guarantee computation"""
        """Demo 6: Privacy guarantee computation"""
        """Demo 6: Privacy guarantee computation"""
        print("\n🔒 Demo 6: Privacy Guarantee Computation")
        print("-" * 42)

        # Create enhanced encoder
        encoder = EnhancedKANHybridEncoder(
            base_dim=1000, compressed_dim=100, enable_interpretability=True
        )

        # Test different privacy levels
        privacy_levels = ["public", "sensitive", "highly_sensitive"]
        privacy_results = {}

        # Mock original data
        original_data = torch.randn(1000)

        for privacy_level in privacy_levels:
            print(f"\n🛡️ Testing {privacy_level} privacy level...")

            try:
                # Mock variants for encoding
                mock_variants = [
                    {"chromosome": "chr1", "position": 100000, "ref": "A", "alt": "G"},
                    {"chromosome": "chr1", "position": 100100, "ref": "C", "alt": "T"},
                ]

                # Encode with specific privacy level
                encoded_data = encoder.encode_genomic_data(
                    mock_variants, privacy_level=privacy_level
                )

                # Compute privacy guarantees
                privacy_metrics = encoder.compute_privacy_guarantee(original_data, encoded_data)

                privacy_results[privacy_level] = privacy_metrics

                print(f"   📊 Privacy metrics:")
                print(f"      - Privacy score: {privacy_metrics['privacy_score']:.3f}")
                print(
                    f"      - Reconstruction difficulty: {privacy_metrics['reconstruction_difficulty']:.3f}"
                )
                print(f"      - Information leakage: {privacy_metrics['information_leakage']:.3f}")

            except Exception as e:
                print(f"   ❌ Error: {e}")
                privacy_results[privacy_level] = {"error": str(e)}

                self.results["privacy_guarantees"] = privacy_results

        # Find most secure level
        best_privacy = max(
            privacy_results.keys(), key=lambda k: privacy_results[k].get("privacy_score", 0)
        )
        print(f"\n🏆 Most secure level: {best_privacy}")

                def run_full_demo(self) -> None:
                def run_full_demo(self) -> None:
        """Run all demos"""
        """Run all demos"""
        """Run all demos"""
        print("🚀 Running comprehensive KAN-HD enhancement demo...")

        # Run all demos
        demos = [
                    self.demo_adaptive_compression,
                    self.demo_hierarchical_encoding,
                    self.demo_scientific_interpretability,
                    self.demo_federated_learning,
                    self.demo_performance_tuning,
                    self.demo_privacy_guarantees,
        ]

        for demo in demos:
            try:
                demo()
            except Exception as e:
                print(f"❌ Demo failed: {e}")

        # Save results
                self.save_results()

        # Print summary
                self.print_summary()

                def save_results(self) -> None:
                def save_results(self) -> None:
        """Save demo results to file"""
        """Save demo results to file"""
        """Save demo results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/tmp/kan_hd_demo_results_{timestamp}.json"

        try:
            with open(filename, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")

            def print_summary(self) -> None:  # noqa: C901
        """Print comprehensive demo summary"""
        print("\n" + "=" * 60)
        print("📋 KAN-HD Enhancement Demo Summary")
        print("=" * 60)

        for demo_name, results in self.results.items():
            print(f"\n🔹 {demo_name.replace('_', ' ').title()}:")

            if "error" in results:
                print(f"   ❌ Failed: {results['error']}")
            else:
                # Print key metrics for each demo
                if demo_name == "adaptive_compression":
                    strategies = [k for k in results.keys() if "error" not in results[k]]
                    print(f"   ✅ Tested {len(strategies)} compression strategies")

                elif demo_name == "hierarchical_encoding":
                    if "modalities_encoded" in results:
                        print(f"   ✅ Encoded {results['modalities_encoded']} modalities")
                        print(f"   📊 Final dimensions: {results['bound_dimensions']}")

                elif demo_name == "scientific_interpretability":
                    if "functions_analyzed" in results:
                        print(f"   ✅ Analyzed {results['functions_analyzed']} functions")
                        print(
                            f"   🧠 Interpretability score: {results['interpretability_score']:.3f}"
                        )

                elif demo_name == "federated_learning":
                    if "participants" in results:
                        print(f"   ✅ {results['participants']} participants")
                        print(f"   🔄 {results['rounds_completed']} rounds completed")

                elif demo_name == "performance_tuning":
                    if "tuning_results" in results:
                        print(f"   ✅ Applied {len(results['tuning_results'])} tuning optimizations")

                elif demo_name == "privacy_guarantees":
                    levels_tested = len([k for k in results.keys() if "error" not in results[k]])
                    print(f"   ✅ Tested {levels_tested} privacy levels")

        print(f"\n🎉 Demo completed! All KAN-HD enhancements demonstrated.")
        print(f"🔬 Enhanced GenomeVault now supports:")
        print(f"   - 10-500x adaptive compression")
        print(f"   - Multi-modal hierarchical encoding")
        print(f"   - Scientific interpretability analysis")
        print(f"   - Federated learning capabilities")
        print(f"   - Real-time performance tuning")
        print(f"   - Advanced privacy guarantees")


if __name__ == "__main__":
    # Run the comprehensive demo
    demo = KANHDDemo()
    demo.run_full_demo()
