# Clinical Calibration & Evaluation (Phase 4)

This phase adds:
- **Calibration metrics**: AUROC, Average Precision (AP), Brier, ECE/MCE, confusion metrics, Youden's J
- **Calibrators**: Platt (logistic), Isotonic (PAV)
- **Evaluation harness**: programmatic (Python) and CLI
- **API**: `/clinical/eval`

## Usage (Python)
```python
from genomevault.clinical.eval.harness import compute_report
rep = compute_report(y_true, y_score, calibrator="platt", bins=10)
print(rep.metrics, rep.threshold)
```

## CLI
```bash
python scripts/clinical_eval_run.py --data /path/to/preds.csv --calibrator platt --bins 10 --out report.json
```

CSV format requires columns: `y_true,y_score`.

## API
```http
POST /clinical/eval
{{
  "y_true": [0,1,0,1],
  "y_score": [0.1, 0.8, 0.2, 0.9],
  "calibrator": "isotonic",
  "bins": 10
}}
```

## Notes
- The harness calibrates and evaluates on the same input set for simplicity; use CV or a holdout set for unbiased calibration metrics.
- Isotonic PAV here is minimal; for large datasets consider specialized libraries.
- All metrics are **dependency-light** (NumPy only).
