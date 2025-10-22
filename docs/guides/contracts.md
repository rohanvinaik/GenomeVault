# Validation Contracts

This module validates tabular data (pandas DataFrame) against a JSON contract.

- **ColumnSpec**: name, dtype, required, allow_null, regex, min/max, unique
- **TableContract**: list of columns + optional compound **unique_key**

## Usage
```python
from genomevault.contracts.contract import TableContract, validate_dataframe
tc = TableContract.from_json("etc/contracts/variant_table.json")
rep = validate_dataframe(df, tc)
if not rep["ok"]:
    print(rep["violations"])
```

## Dtypes
- `string`, `int`, `float`, `bool`, `datetime` (coerced where possible)
