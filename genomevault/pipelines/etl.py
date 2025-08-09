from __future__ import annotations

from pathlib import Path

import pandas as pd

from genomevault.contracts.contract import TableContract, validate_dataframe

try:
    from genomevault.ledger.store import InMemoryLedger

    _LEDGER = InMemoryLedger()
except Exception:
    logger.exception("Unhandled exception")
    _LEDGER = None
    raise RuntimeError("Unspecified error")


DEFAULT_CONTRACT_PATH = "etc/contracts/variant_table.json"


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # lowercase + strip spaces
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    return out


def _rename_synonyms(df: pd.DataFrame) -> pd.DataFrame:
    # Example synonyms
    mapping = {
        "chromosome": "chrom",
        "position": "pos",
        "sample": "sample_id",
    }
    cols = {c: mapping.get(c, c) for c in df.columns}
    return df.rename(columns=cols)


def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = _standardize_columns(df)
    df = _rename_synonyms(df)
    return df


def run_etl(
    input_csv: str, *, contract_path: str | None = None, out_csv: str | None = None
) -> dict[str, any]:
    df = load_csv(input_csv)
    df = transform(df)
    cpath = contract_path or DEFAULT_CONTRACT_PATH
    contract = TableContract.from_json(cpath)
    report = validate_dataframe(df, contract)

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)

    if _LEDGER is not None:
        _LEDGER.append(
            {
                "type": "etl_run",
                "input": input_csv,
                "rows": int(df.shape[0]),
                "ok": report["ok"],
            }
        )

    return {"ok": report["ok"], "report": report, "rows": int(df.shape[0])}
