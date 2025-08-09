#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import libcst as cst

ROOT = Path(__file__).resolve().parents[2]


class FixExceptions(cst.CSTTransformer):
    def leave_ExceptHandler(
        self, original_node: cst.ExceptHandler, updated_node: cst.ExceptHandler
    ):
        # bare except -> Exception as e
        if updated_node.type is None:
            updated_node = updated_node.with_changes(
                type=cst.Name("Exception"), name=cst.AsName(cst.Name("e"))
            )
        # ensure body logs and re-raises
        body = list(updated_node.body.body)
        src = "".join(x.code for x in body)
        needs_log = "logger." not in src
        needs_reraise = "raise" not in src
        new_body = body[:]
        if needs_log:
            new_body.insert(
                0,
                cst.parse_statement(
                    "from genomevault.observability.logging import configure_logging\nlogger = configure_logging()\nlogger.exception('Unhandled exception')\n"
                ),
            )
        if needs_reraise:
            new_body.append(cst.parse_statement("raise"))
        return updated_node.with_changes(body=updated_node.body.with_changes(body=tuple(new_body)))


def main():
    py_files = [p for p in ROOT.rglob("*.py") if "/venv/" not in str(p)]
    mods = 0
    for p in py_files:
        code = p.read_text(encoding="utf-8")
        try:
            tree = cst.parse_module(code)
            new = tree.visit(FixExceptions())
            if new.code != code:
                p.write_text(new.code, encoding="utf-8")
                mods += 1
        except Exception:
            continue
    print(f"Modified files: {mods}")


if __name__ == "__main__":
    main()
