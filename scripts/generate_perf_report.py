#!/usr/bin/env python3
"""
Generate performance report from benchmark results

Aggregates benchmark results and generates a comprehensive performance report
with visualizations and analysis.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class PerformanceReportGenerator:
    """Generate performance reports from benchmark data"""

    def __init__(self, input_dir: Path, output_dir: Path) -> None:
            """TODO: Add docstring for __init__"""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plot style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["figure.dpi"] = 150

    def load_benchmark_data(self) -> Dict[str, List[Dict]]:
           """TODO: Add docstring for load_benchmark_data"""
     """Load all benchmark JSON files"""
        data = {"hdc": [], "pir": [], "zk": []}

        for lane in data.keys():
            lane_dir = self.input_dir / lane
            if lane_dir.exists():
                for json_file in lane_dir.glob("*.json"):
                    try:
                        with open(json_file, "r") as f:
                            benchmark_data = json.load(f)
                            data[lane].append(benchmark_data)
                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")

        return data

    def analyze_hdc_performance(self, hdc_data: List[Dict]) -> Dict[str, Any]:
           """TODO: Add docstring for analyze_hdc_performance"""
     """Analyze HDC benchmark data"""
        if not hdc_data:
            return {}

        # Get latest benchmark
        latest = max(hdc_data, key=lambda x: x.get("timestamp", ""))

        analysis = {"timestamp": latest.get("timestamp"), "summary": {}}

        # Extract key metrics
        if "benchmarks" in latest:
            benchmarks = latest["benchmarks"]

            # Encoding throughput
            if "encoding_throughput" in benchmarks:
                throughput_data = benchmarks["encoding_throughput"]["data"]
                # Get clinical tier performance
                clinical_perf = throughput_data.get("dim_10000", {}).get("features_1000", {})
                analysis["summary"]["encoding_ops_per_sec"] = clinical_perf.get(
                    "throughput_ops_per_sec", 0
                )

            # Memory usage
            if "memory_usage" in benchmarks:
                mem_data = benchmarks["memory_usage"]["data"]
                clinical_mem = mem_data.get("clinical", {})
                analysis["summary"]["memory_kb"] = clinical_mem.get("memory_kb", 0)
                analysis["summary"]["compression_ratio"] = clinical_mem.get("compression_ratio", 0)

            # Binding performance
            if "binding_operations" in benchmarks:
                binding_data = benchmarks["binding_operations"]["data"]
                best_binding = max(
                    (k for k, v in binding_data.items() if v.get("supported", False)),
                    key=lambda x: binding_data[x].get("throughput_ops_per_sec", 0),
                    default=None,
                )
                if best_binding:
                    analysis["summary"]["best_binding_type"] = best_binding
                    analysis["summary"]["binding_ops_per_sec"] = binding_data[best_binding][
                        "throughput_ops_per_sec"
                    ]

        return analysis

    def generate_hdc_plots(self, hdc_data: List[Dict]) -> None:
           """TODO: Add docstring for generate_hdc_plots"""
     """Generate HDC performance plots"""
        if not hdc_data:
            return

        latest = max(hdc_data, key=lambda x: x.get("timestamp", ""))

        if "benchmarks" not in latest:
            return

        benchmarks = latest["benchmarks"]

        # 1. Throughput comparison plot
        if "encoding_throughput" in benchmarks:
        self._plot_throughput_comparison(benchmarks["encoding_throughput"])

        # 2. Memory usage by tier
        if "memory_usage" in benchmarks:
        self._plot_memory_comparison(benchmarks["memory_usage"])

        # 3. Scalability plot
        if "scalability" in benchmarks:
        self._plot_scalability(benchmarks["scalability"])

    def _plot_throughput_comparison(self, throughput_data: Dict) -> None:
           """TODO: Add docstring for _plot_throughput_comparison"""
     """Plot encoding throughput comparison"""
        data = throughput_data["data"]

        dimensions = []
        throughputs = []

        for dim_key in sorted(data.keys()):
            dim = int(dim_key.split("_")[1])
            # Use 1000 feature size as reference
            features_data = data[dim_key]
            if "features_1000" in features_data:
                throughput = features_data["features_1000"]["throughput_ops_per_sec"]
                dimensions.append(dim)
                throughputs.append(throughput)

        plt.figure()
        plt.plot(dimensions, throughputs, "o-", linewidth=2, markersize=8)
        plt.xlabel("Hypervector Dimension")
        plt.ylabel("Throughput (operations/second)")
        plt.title("HDC Encoding Throughput vs Dimension")
        plt.grid(True, alpha=0.3)

        # Add annotations
        for x, y in zip(dimensions, throughputs):
            plt.annotate(
                f"{y:.0f}", xy=(x, y), xytext=(0, 5), textcoords="offset points", ha="center"
            )

        plt.tight_layout()
        plt.savefig(self.output_dir / "hdc_throughput.png")
        plt.close()

    def _plot_memory_comparison(self, memory_data: Dict) -> None:
           """TODO: Add docstring for _plot_memory_comparison"""
     """Plot memory usage comparison"""
        data = memory_data["data"]

        tiers = list(data.keys())
        memory_kb = [data[tier]["memory_kb"] for tier in tiers]
        compression_ratios = [data[tier]["compression_ratio"] for tier in tiers]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Memory usage
        bars1 = ax1.bar(tiers, memory_kb, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax1.set_xlabel("Compression Tier")
        ax1.set_ylabel("Memory Usage (KB)")
        ax1.set_title("Memory Usage by Compression Tier")

        # Add value labels
        for bar, kb in zip(bars1, memory_kb):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f"{kb:.0f} KB",
                ha="center",
                va="bottom",
            )

        # Compression ratios
        bars2 = ax2.bar(tiers, compression_ratios, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax2.set_xlabel("Compression Tier")
        ax2.set_ylabel("Compression Ratio")
        ax2.set_title("Compression Ratio by Tier")

        # Add value labels
        for bar, ratio in zip(bars2, compression_ratios):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{ratio:.1f}x",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(self.output_dir / "hdc_memory_comparison.png")
        plt.close()

    def _plot_scalability(self, scalability_data: Dict) -> None:
           """TODO: Add docstring for _plot_scalability"""
     """Plot batch processing scalability"""
        data = scalability_data["data"]

        batch_sizes = []
        speedups = []

        for batch_key in sorted(data.keys(), key=lambda x: int(x.split("_")[1])):
            batch_size = int(batch_key.split("_")[1])
            speedup = data[batch_key]["speedup"]
            batch_sizes.append(batch_size)
            speedups.append(speedup)

        plt.figure()
        plt.plot(batch_sizes, speedups, "o-", linewidth=2, markersize=8, label="Actual")
        plt.plot(batch_sizes, batch_sizes, "--", alpha=0.5, label="Ideal linear")
        plt.xlabel("Batch Size")
        plt.ylabel("Speedup")
        plt.title("HDC Batch Processing Scalability")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "hdc_scalability.png")
        plt.close()

    def generate_summary_report(self, all_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
           """TODO: Add docstring for generate_summary_report"""
     """Generate overall summary report"""
        report = {"generated_at": datetime.now().isoformat(), "lanes": {}}

        # HDC summary
        if all_data["hdc"]:
            hdc_analysis = self.analyze_hdc_performance(all_data["hdc"])
            report["lanes"]["hdc"] = hdc_analysis

        # PIR summary (placeholder)
        if all_data["pir"]:
            report["lanes"]["pir"] = {"summary": {"status": "Benchmarks available"}}

        # ZK summary (placeholder)
        if all_data["zk"]:
            report["lanes"]["zk"] = {"summary": {"status": "Benchmarks available"}}

        return report

    def generate_html_report(self, summary: Dict[str, Any]) -> None:
           """TODO: Add docstring for generate_html_report"""
     """Generate HTML performance report"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GenomeVault Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background: #f0f0f0;
            border-radius: 5px;
        }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #1f77b4; }}
        .metric-label {{ font-size: 14px; color: #666; }}
        img {{ max-width: 100%; margin: 20px 0; }}
        .lane-section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <h1>GenomeVault Performance Report</h1>
    <p>Generated: {summary['generated_at']}</p>

    <div class="lane-section">
        <h2>HDC (Hyperdimensional Computing)</h2>
"""

        if "hdc" in summary["lanes"]:
            hdc_data = summary["lanes"]["hdc"]
            if "summary" in hdc_data:
                hdc_summary = hdc_data["summary"]

                html_content += f"""
        <div class="metric">
            <div class="metric-value">{hdc_summary.get('encoding_ops_per_sec', 0):.0f}</div>
            <div class="metric-label">Encodings/sec</div>
        </div>

        <div class="metric">
            <div class="metric-value">{hdc_summary.get('memory_kb', 0):.0f} KB</div>
            <div class="metric-label">Memory per Vector</div>
        </div>

        <div class="metric">
            <div class="metric-value">{hdc_summary.get('compression_ratio', 0):.1f}x</div>
            <div class="metric-label">Compression Ratio</div>
        </div>
"""

        # Add plots
        for plot in ["hdc_throughput.png", "hdc_memory_comparison.png", "hdc_scalability.png"]:
            if (self.output_dir / plot).exists():
                html_content += f'<img src="{plot}" alt="{plot}">\n'

        html_content += """
    </div>

    <div class="lane-section">
        <h2>PIR (Private Information Retrieval)</h2>
        <p>PIR benchmarks available - see detailed reports</p>
    </div>

    <div class="lane-section">
        <h2>ZK (Zero-Knowledge Proofs)</h2>
        <p>ZK benchmarks available - see detailed reports</p>
    </div>

</body>
</html>
"""

        # Save HTML report
        html_path = self.output_dir / "performance_report.html"
        with open(html_path, "w") as f:
            f.write(html_content)

        print(f"HTML report saved to: {html_path}")

    def generate_report(self) -> None:
           """TODO: Add docstring for generate_report"""
     """Generate complete performance report"""
        print("Loading benchmark data...")
        all_data = self.load_benchmark_data()

        print("Analyzing performance...")
        summary = self.generate_summary_report(all_data)

        # Save summary JSON
        summary_path = self.output_dir / "performance_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print("Generating plots...")
        self.generate_hdc_plots(all_data["hdc"])

        print("Generating HTML report...")
        self.generate_html_report(summary)

        print(f"\nPerformance report generated in: {self.output_dir}")
        return summary


    def main() -> None:
       """TODO: Add docstring for main"""
     """Main entry point"""
    parser = argparse.ArgumentParser(description="Generate performance report")
    parser.add_argument("--input", default="benchmarks", help="Input benchmark directory")
    parser.add_argument("--output", default="docs/perf", help="Output report directory")

    args = parser.parse_args()

    generator = PerformanceReportGenerator(args.input, args.output)
    generator.generate_report()


if __name__ == "__main__":
    main()
