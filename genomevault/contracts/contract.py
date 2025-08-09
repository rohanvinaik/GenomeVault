from __future__ import annotations

"""Contract module."""
"""Contract module."""
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ColumnSpec:
    """Data container for columnspec information."""
    name: str
    dtype: str  # 'string' | 'int' | 'float' | 'bool' | 'datetime'
    required: bool = True
    allow_null: bool = False
    regex: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    unique: bool = False


@dataclass
class TableContract:
    """Data container for tablecontract information."""
    name: str
    columns: list[ColumnSpec] = field(default_factory=list)
    unique_key: list[str] | None = None  # compound unique key (list of columns)

    @staticmethod
    def from_json(path: str) -> TableContract:
        """From json.

            Args:
                path: File or directory path.

            Returns:
                TableContract instance.
            """
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        cols = [ColumnSpec(**c) for c in obj["columns"]]
        return TableContract(
            name=obj.get("name", Path(path).stem),
            columns=cols,
            unique_key=obj.get("unique_key"),
        )

    def to_json(self, path: str) -> None:
        """To json.

            Args:
                path: File or directory path.
            """
        obj = {
            "name": self.name,
            "columns": [c.__dict__ for c in self.columns],
            "unique_key": self.unique_key,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


_DTYPE_MAP = {
    "string": "string",
    "int": "Int64",
    "float": "float64",
    "bool": "boolean",
    "datetime": "datetime64[ns]",
}


def _coerce_types(df: pd.DataFrame, contract: TableContract) -> pd.DataFrame:
    out = df.copy()
    for col in contract.columns:
        if col.name not in out.columns:
            # missing columns will be validated later
            continue
        if col.dtype == "datetime":
            out[col.name] = pd.to_datetime(out[col.name], errors="coerce")
        else:
            out[col.name] = out[col.name].astype(_DTYPE_MAP[col.dtype], errors="ignore")
    return out


def _dtype_ok(series: pd.Series, spec: ColumnSpec) -> bool:
    if spec.dtype == "datetime":
        return pd.api.types.is_datetime64_any_dtype(series)
    if spec.dtype == "string":
        return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(
            series
        )
    if spec.dtype == "int":
        return pd.api.types.is_integer_dtype(series)
    if spec.dtype == "float":
        return pd.api.types.is_float_dtype(series) or pd.api.types.is_integer_dtype(
            series
        )
    if spec.dtype == "bool":
        return pd.api.types.is_bool_dtype(series)
    return True


def validate_dataframe(df: pd.DataFrame, contract: TableContract) -> dict[str, Any]:
    """Return a validation report with booleans and violation details."""
    report: dict[str, Any] = {
        "ok": True,
        "columns": {},
        "row_count": int(df.shape[0]),
        "violations": [],
    }
    df = _coerce_types(df, contract)

    for spec in contract.columns:
        colrep: dict[str, Any] = {"present": spec.name in df.columns}
        if spec.name not in df.columns:
            if spec.required:
                report["ok"] = False
                report["violations"].append(
                    {"type": "missing_column", "column": spec.name}
                )
            report["columns"][spec.name] = colrep
            continue

        s = df[spec.name]
        colrep["dtype_ok"] = _dtype_ok(s, spec)

        if spec.required and s.isna().all():
            report["ok"] = False
            report["violations"].append({"type": "all_null", "column": spec.name})

        if not spec.allow_null and s.isna().any():
            report["ok"] = False
            n = int(s.isna().sum())
            report["violations"].append(
                {"type": "nulls", "column": spec.name, "count": n}
            )

        if spec.regex:
            pat = re.compile(spec.regex)
            bad = (
                s.dropna()
                .astype(str)
                .map(lambda v: bool(pat.fullmatch(v)) if isinstance(v, str) else False)
            )
            if not bad.all():
                nbad = int((~bad).sum())
                if nbad > 0:
                    report["ok"] = False
                    report["violations"].append(
                        {"type": "regex", "column": spec.name, "count": nbad}
                    )

        if spec.minimum is not None:
            mask = s.dropna() < spec.minimum
            if mask.any():
                report["ok"] = False
                report["violations"].append(
                    {
                        "type": "min",
                        "column": spec.name,
                        "count": int(mask.sum()),
                        "min": spec.minimum,
                    }
                )

        if spec.maximum is not None:
            mask = s.dropna() > spec.maximum
            if mask.any():
                report["ok"] = False
                report["violations"].append(
                    {
                        "type": "max",
                        "column": spec.name,
                        "count": int(mask.sum()),
                        "max": spec.maximum,
                    }
                )

        if spec.unique:
            dups = s.dropna().duplicated(keep=False)
            if dups.any():
                report["ok"] = False
                report["violations"].append(
                    {"type": "unique", "column": spec.name, "count": int(dups.sum())}
                )

        report["columns"][spec.name] = colrep

    # compound key
    if contract.unique_key:
        subset = [c for c in contract.unique_key if c in df.columns]
        if len(subset) == len(contract.unique_key):
            dups = df[subset].astype(str).duplicated(keep=False)
            if dups.any():
                report["ok"] = False
                report["violations"].append(
                    {"type": "unique_key", "columns": subset, "count": int(dups.sum())}
                )
        else:
            report["ok"] = False
            report["violations"].append(
                {"type": "unique_key_missing_columns", "columns": contract.unique_key}
            )

    return report
