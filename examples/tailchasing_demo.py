"""
Demonstration of chromatin-inspired performance visualization.

This script shows how to use the tailchasing reporting module to visualize
performance bottlenecks using Hi-C style contact matrices and polymer physics
concepts.
"""

import random

import numpy as np

from tailchasing.core.reporting import (
    HiCHeatmapGenerator,
    PolymerMetricsReport,
    TAD,
    generate_comparative_matrices,
    integrate_chromatin_analysis,
)


def create_sample_data():
    """Create sample performance data for demonstration."""

    # Create sample TADs representing functional modules
    tads = [
        TAD(start=0, end=25, name="Database_Layer", activity_level=0.9),
        TAD(start=25, end=50, name="Business_Logic", activity_level=0.7),
        TAD(start=50, end=75, name="API_Handlers", activity_level=0.8),
        TAD(start=75, end=100, name="Frontend_Components", activity_level=0.6),
    ]

    # Create sample contact matrix (interaction frequencies between code regions)
    np.random.seed(42)  # For reproducible results
    contact_matrix = np.random.exponential(scale=2.0, size=(20, 20))

    # Add some structure - stronger interactions within TAD boundaries
    for i in range(0, 5):  # Database layer
        for j in range(0, 5):
            contact_matrix[i, j] *= 2.0

    for i in range(5, 10):  # Business logic
        for j in range(5, 10):
            contact_matrix[i, j] *= 1.8

    for i in range(10, 15):  # API handlers
        for j in range(10, 15):
            contact_matrix[i, j] *= 1.5

    for i in range(15, 20):  # Frontend
        for j in range(15, 20):
            contact_matrix[i, j] *= 1.3

    # Make matrix symmetric
    contact_matrix = (contact_matrix + contact_matrix.T) / 2

    # Create sample interactions
    interactions = []
    for i in range(20):
        for j in range(i + 1, 20):
            if contact_matrix[i, j] > 1.0:  # Only include significant interactions
                # Scale positions to 0-100 range
                pos1 = int((i / 20) * 100)
                pos2 = int((j / 20) * 100)
                strength = contact_matrix[i, j] / contact_matrix.max()
                interactions.append((pos1, pos2, strength))

    # Create sample fix strategies
    fix_strategies = [
        {
            "name": "Database Connection Pooling",
            "impact_score": 0.8,
            "complexity": 0.3,
            "confidence": 0.9,
            "description": "Implement connection pooling to reduce database overhead",
        },
        {
            "name": "API Response Caching",
            "impact_score": 0.7,
            "complexity": 0.4,
            "confidence": 0.8,
            "description": "Cache frequently requested API responses",
        },
        {
            "name": "Frontend Bundle Optimization",
            "impact_score": 0.6,
            "complexity": 0.6,
            "confidence": 0.7,
            "description": "Optimize JavaScript bundles and lazy loading",
        },
        {
            "name": "Business Logic Refactoring",
            "impact_score": 0.9,
            "complexity": 0.8,
            "confidence": 0.6,
            "description": "Refactor complex business logic into smaller components",
        },
        {
            "name": "Asynchronous Processing",
            "impact_score": 0.85,
            "complexity": 0.7,
            "confidence": 0.75,
            "description": "Move heavy computations to background tasks",
        },
    ]

    # Create sample timeline events
    timeline_data = [
        {
            "timestamp": 0.0,
            "name": "Request Initialization",
            "duration": 5.0,
            "impact": 0.2,
            "status": "completed",
        },
        {
            "timestamp": 5.0,
            "name": "Database Query",
            "duration": 45.0,
            "impact": 0.9,
            "status": "completed",
        },
        {
            "timestamp": 50.0,
            "name": "Business Logic Processing",
            "duration": 120.0,
            "impact": 0.8,
            "status": "completed",
        },
        {
            "timestamp": 170.0,
            "name": "Response Serialization",
            "duration": 15.0,
            "impact": 0.3,
            "status": "completed",
        },
        {
            "timestamp": 185.0,
            "name": "Network Transfer",
            "duration": 30.0,
            "impact": 0.5,
            "status": "completed",
        },
    ]

    return tads, contact_matrix, interactions, fix_strategies, timeline_data


def demonstrate_heatmap_generation():
    """Demonstrate Hi-C heatmap generation."""
    print("=" * 60)
    print("Hi-C HEATMAP GENERATION DEMO")
    print("=" * 60)

    tads, contact_matrix, interactions, fix_strategies, timeline_data = create_sample_data()

    generator = HiCHeatmapGenerator()

    # Basic heatmap
    print("-" * 30)
    heatmap = generator.generate_contact_heatmap(contact_matrix, title="Performance Contact Matrix")
    print(heatmap)

    # Heatmap with TAD boundaries
    print("-" * 35)
    tad_heatmap = generator.show_tad_boundaries(contact_matrix, {tad.name: tad for tad in tads})
    print(tad_heatmap)

    # Risk highlighting
    print("-" * 25)
    # Create some risk scores for demonstration
    risk_scores = {}
    for i in range(contact_matrix.shape[0]):
        for j in range(contact_matrix.shape[1]):
            if contact_matrix[i, j] > np.median(contact_matrix):
                risk = min(
                    1.0,
                    contact_matrix[i, j] / contact_matrix.max() + random.uniform(-0.2, 0.2),
                )
                risk_scores[(i, j)] = max(0.0, risk)

    risk_heatmap = generator.highlight_thrash_clusters(contact_matrix, risk_scores)
    print(risk_heatmap)


def demonstrate_polymer_metrics():
    """Demonstrate polymer metrics reporting."""
    print("POLYMER METRICS DEMO")
    print("=" * 60)

    tads, contact_matrix, interactions, fix_strategies, timeline_data = create_sample_data()

    reporter = PolymerMetricsReport()

    # Calculate polymer distances
    print("-" * 30)
    distances = reporter.calculate_polymer_distances(tads, interactions)
    print("Intra-TAD Distances:")
    for tad_name, metrics in distances["intra_tad_distances"].items():
        print(f"  {tad_name}: mean={metrics['mean']:.2f}, count={metrics['count']}")

    for pair_name, metrics in distances["inter_tad_distances"].items():
        print(f"  {pair_name}: mean={metrics['mean']:.2f}, count={metrics['count']}")

    # Contact probabilities
    print("-" * 35)
    contact_probs = reporter.calculate_contact_probabilities(interactions)
    stats = contact_probs["statistics"]
    print(f"Mean contact distance: {stats['mean_contact_distance']:.2f}")
    print(f"Contact decay rate: {stats['contact_decay_rate']:.4f}")
    print(f"Short-range fraction: {stats['short_range_fraction']:.3f}")
    print(f"Long-range fraction: {stats['long_range_fraction']:.3f}")

    # Thrash reduction predictions
    print("-" * 33)
    predictions = reporter.predict_thrash_reduction(fix_strategies)
    for strategy, pred in predictions.items():
        print(f"\n{strategy}:")
        print(f"  Estimated reduction: {pred['estimated_reduction']:.2%}")
        print(f"  Implementation risk: {pred['implementation_risk']:.2f}")
        print(f"  ROI score: {pred['roi_score']:.2f}")
        print(f"  Priority: {pred['recommended_priority']}")

    # Timeline visualization
    print("-" * 32)
    timeline_viz = reporter.visualize_replication_timing(timeline_data)
    print(timeline_viz)

    # Comprehensive report
    print("-" * 35)
    comprehensive = reporter.generate_comprehensive_report(
        tads, interactions, fix_strategies, timeline_data
    )

    summary = comprehensive["summary_metrics"]
    print(f"Overall health score: {summary['overall_health_score']:.3f}")
    print(f"Optimization potential: {summary['optimization_potential']:.3f}")
    print(f"Stability index: {summary['stability_index']:.3f}")


def demonstrate_integration():
    """Demonstrate integration with existing reporting."""
    print("INTEGRATION DEMO")
    print("=" * 60)

    tads, contact_matrix, interactions, fix_strategies, timeline_data = create_sample_data()

    # Simulate existing report
    existing_report = {
        "performance_metrics": {
            "response_time": 250.5,
            "throughput": 1200,
            "error_rate": 0.02,
        },
        "resource_usage": {"cpu_percent": 75.3, "memory_mb": 512, "disk_io": 45.2},
        "timestamp": "2025-01-15T10:30:00Z",
    }

    # Integrate chromatin analysis
    enhanced_report = integrate_chromatin_analysis(
        existing_report,
        contact_matrix,
        tads,
        interactions,
        fix_strategies,
        timeline_data,
    )

    print("-" * 30)
    print("Original sections:", list(existing_report.keys()))
    print("Enhanced sections:", list(enhanced_report.keys()))

    print("-" * 32)
    chromatin = enhanced_report["chromatin_analysis"]

    # Contact matrix summary
    matrix_summary = chromatin["contact_matrix_summary"]
    print(f"Matrix dimensions: {matrix_summary['dimensions']}")
    print(f"Total contacts: {matrix_summary['total_contacts']}")
    print(f"Max contact strength: {matrix_summary['max_contact_strength']:.2f}")

    # TAD analysis
    tad_analysis = chromatin["tad_analysis"]
    print(f"Total TADs: {tad_analysis['total_tads']}")
    print(f"Average TAD size: {tad_analysis['average_tad_size']:.1f}")

    # Risk analysis
    risk_analysis = chromatin["risk_analysis"]
    print(f"High-risk positions: {risk_analysis['high_risk_positions']}")
    print(f"Average risk score: {risk_analysis['average_risk_score']:.3f}")

    for level, count in risk_analysis["risk_distribution"].items():
        print(f"  {level}: {count}")


def demonstrate_comparative_analysis():
    """Demonstrate before/after comparative analysis."""
    print("COMPARATIVE ANALYSIS DEMO")
    print("=" * 60)

    tads, before_matrix, interactions, fix_strategies, timeline_data = create_sample_data()

    # Simulate "after" matrix with improvements
    # Apply different improvement levels to different regions
    after_matrix = before_matrix.copy()

    # Database layer improvement (30% reduction)
    after_matrix[0:5, 0:5] *= 0.7

    # Business logic improvement (20% reduction)
    after_matrix[5:10, 5:10] *= 0.8

    # API handlers slight improvement (10% reduction)
    after_matrix[10:15, 10:15] *= 0.9

    # Frontend no improvement (actually slight increase due to new features)
    after_matrix[15:20, 15:20] *= 1.05

    # Generate comparative analysis
    comparative = generate_comparative_matrices(
        before_matrix, after_matrix, tads, "Multi-layer Optimization"
    )

    print("-" * 33)
    metrics = comparative["metrics"]
    print(f"Total contacts before: {metrics['total_contacts_before']}")
    print(f"Total contacts after: {metrics['total_contacts_after']}")
    print(f"Absolute reduction: {metrics['absolute_reduction']}")
    print(f"Reduction percentage: {metrics['reduction_percentage']:.1f}%")
    print(f"Improvement score: {metrics['improvement_score']:.3f}")

    print("-" * 30)
    for tad_name, improvement in comparative["tad_specific_improvements"].items():
        print(f"{tad_name}:")
        print(f"  Reduction: {improvement['reduction_percentage']:.1f}%")
        print(f"  Before: {improvement['contacts_before']}")
        print(f"  After: {improvement['contacts_after']}")

    print("-" * 25)
    print("✓ Before heatmap generated")
    print("✓ After heatmap generated")
    print("✓ Difference heatmap generated")
    pass  # Debug print removed


def main():
    """Run all demonstrations."""
    print("CHROMATIN-INSPIRED PERFORMANCE VISUALIZATION DEMO")
    print("=" * 50)
    print("This demo shows how biological concepts from chromatin organization")
    print("can be applied to visualize and analyze software performance.")
    print("=" * 50)

    try:
        demonstrate_heatmap_generation()
        demonstrate_polymer_metrics()
        demonstrate_integration()
        demonstrate_comparative_analysis()

        print("DEMO COMPLETE")
        print("=" * 60)
        print("• Hi-C style heatmaps reveal interaction patterns")
        print("• TAD boundaries help identify functional modules")
        print("• Polymer metrics quantify system organization")
        print("• Risk analysis highlights critical bottlenecks")
        print("• Comparative analysis measures optimization effectiveness")
        pass  # Debug print removed

    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        print("Make sure you have installed the rich library:")
        print("pip install rich>=13.0.0")
        raise


if __name__ == "__main__":
    main()
