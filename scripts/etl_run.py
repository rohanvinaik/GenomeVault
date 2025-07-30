#!/usr/bin/env python3
import argparse, json
from genomevault.pipelines.etl import run_etl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV")
    ap.add_argument("--contract", default=None, help="Path to contract JSON (optional)")
    ap.add_argument("--out", default=None, help="Optional path to write normalized CSV")
    ap.add_argument("--report", default="etl_report.json", help="Where to write the report JSON")
    args = ap.parse_args()

    res = run_etl(args.input, contract_path=args.contract, out_csv=args.out)
    with open(args.report, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.report} (ok={res['ok']}, rows={res['rows']})")

if __name__ == "__main__":
    main()
