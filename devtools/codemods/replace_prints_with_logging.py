#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import libcst as cst
import libcst.matchers as m

ROOT = Path(__file__).resolve().parents[2]


class ReplacePrints(cst.CSTTransformer):
    def leave_Call(self, orig: cst.Call, updated: cst.Call):
        if m.matches(updated.func, m.Name("print")):
            arg = updated.args[0].value.code if updated.args else '"<no-args>"'
            return cst.parse_expression("logger.info(" + arg + ")")
        return updated


def main():
    for p in ROOT.rglob("*.py"):
        code = p.read_text(encoding="utf-8")
        try:
            mod = cst.parse_module(code)
            # ensure logger import at top if print exists
            if "print(" in code and "configure_logging" not in code:
                code = (
                    "from genomevault.observability.logging import configure_logging\nlogger = configure_logging()\n"
                    + code
                )
                p.write_text(code, encoding="utf-8")
                mod = cst.parse_module(code)
            new = mod.visit(ReplacePrints())
            if new.code != code:
                p.write_text(new.code, encoding="utf-8")
        except Exception:
            continue


if __name__ == "__main__":
    main()
