from genomevault.observability.logging import configure_logging

logger = configure_logging()
#!/usr/bin/env python3
import argparse
import json

from genomevault.clinical.eval.harness import compute_report, load_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV with columns y_true,y_score")
    ap.add_argument(
        "--calibrator", choices=["none", "platt", "isotonic"], default="none"
    )
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--out", default="report.json")
    args = ap.parse_args()

    y, s = load_csv(args.data)
    rep = compute_report(y, s, calibrator=args.calibrator, bins=args.bins)
    with open(args.out, "w") as f:
        json.dump(
            {
                {
                    "metrics": rep.metrics,
                    "threshold": rep.threshold,
                    "confusion": rep.confusion,
                    "calibration_bins": rep.calibration_bins,
                }
            },
            f,
            indent=2,
        )
    logger.info("Wrote {args.out}")


if __name__ == "__main__":
    main()
