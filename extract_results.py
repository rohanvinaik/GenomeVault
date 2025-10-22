import json

with open('benchmark_results/full_pipeline_results/pipeline_run_20251021_180403/pipeline_results.json', 'r') as f:
    data = json.load(f)

# Extract summary metrics without the large result fields
summary = {
    "timestamp": data.get("timestamp"),
    "input_format": data.get("input_format"),
    "quick_mode": data.get("quick_mode"),
    "performance_metrics": data.get("performance_metrics"),
    "summary": data.get("summary")
}

# Get compact stage info without full results
if "stages" in data:
    summary["stages"] = []
    for stage in data["stages"]:
        compact_stage = {
            "stage": stage.get("stage"),
            "status": stage.get("status"),
            "metrics": stage.get("metrics")
        }
        summary["stages"].append(compact_stage)

print(json.dumps(summary, indent=2))
