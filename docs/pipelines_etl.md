# ETL Pipelines

Basic CSV ETL flow:
1. **Load** CSV with pandas
2. **Transform**: lowercase columns, rename common synonyms
3. **Validate** with contracts
4. **Output** normalized CSV and a JSON report

## Programmatic
```python
from genomevault.pipelines.etl import run_etl
res = run_etl("variants.csv", contract_path="etc/contracts/variant_table.json", out_csv="out/variants_norm.csv")
```

## CLI
```bash
python scripts/etl_run.py --input path/to/variants.csv --contract etc/contracts/variant_table.json --out out/variants_norm.csv --report report.json
```

If the ledger is available, ETL runs are recorded as ledger events.
