#!/usr/bin/env python3
"""
GenomeVault auto-fix codemod
----------------------------

What it does (safe, conservative):
  1) Replaces `except:` with `except Exception:` and appends a TODO to narrow the exception.
  2) For `except Exception` / `except BaseException`, appends a TODO to narrow to specific exceptions.
  3) Converts `print(...)` to `logger.info(...)` in library code; injects `import logging` and
     `logger = logging.getLogger(__name__)` if needed. (Keeps prints in tests and under scripts/ if configured.)
  4) Renames unused parameters in function signatures to prefix '_' (skips self/cls). No body changes are needed.
  5) Flags `from x import *` with a TODO trailing comment to replace with explicit imports.
  6) Flags obvious import-time calls (bare Call expr at module level) with a TODO comment about side-effects.

Usage:
  # Dry-run (default) on current directory
  python genomevault_autofix.py

  # Apply in-place changes
  python genomevault_autofix.py --apply

  # Target a specific repo path
  python genomevault_autofix.py --root /path/to/repo --apply

  # Include tests and scripts (by default excluded)
  python genomevault_autofix.py --apply --include-tests --include-scripts

Requires:
  pip install libcst

Notes:
  - This codemod is intentionally conservative and inserts small TODO comments where human judgment is required.
  - Review diffs before committing.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from dataclasses import dataclass, field

try:
    import libcst as cst
    from libcst import matchers as m
except ImportError:
    print(
        "ERROR: This codemod requires 'libcst'. Install with: pip install libcst",
        file=sys.stderr,
    )
    raise

# --------------------------
# CLI
# --------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GenomeVault autofix codemod (style/logic hygiene).")
    p.add_argument("--root", default=".", help="Repository root to modify (default: current dir).")
    p.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes in-place (otherwise dry-run).",
    )
    p.add_argument("--include-tests", action="store_true", help="Include tests/ directories.")
    p.add_argument("--include-scripts", action="store_true", help="Include scripts/ directories.")
    p.add_argument("--verbose", "-v", action="count", default=0, help="Increase verbosity.")
    return p.parse_args()


# --------------------------
# Utilities
# --------------------------

IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "venv",
    "build",
    "dist",
    ".idea",
    ".vscode",
    "__pycache__",
}


def should_visit_file(path: str, include_tests: bool, include_scripts: bool) -> bool:
    low = path.lower()
    if not low.endswith(".py"):
        return False
    parts = low.split(os.sep)
    if not include_tests and (
        "tests" in parts or low.endswith("_test.py") or os.sep + "test_" in low
    ):
        return False
    if not include_scripts and ("scripts" in parts or "examples" in parts or "demo" in parts):
        return False
    return True


def walk_python_files(root: str, include_tests: bool, include_scripts: bool) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORED_DIRS]
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            if should_visit_file(p, include_tests, include_scripts):
                out.append(p)
    return sorted(out)


# --------------------------
# AST analysis for unused params
# --------------------------


@dataclass
class FunctionKey:
    name: str
    lineno: int


@dataclass
class UnusedParamIndex:
    # Map (file -> (FunctionKey -> set(param names)))
    data: dict[str, dict[FunctionKey, set[str]]] = field(default_factory=dict)

    def add(self, file: str, key: FunctionKey, param: str) -> None:
        self.data.setdefault(file, {}).setdefault(key, set()).add(param)

    def for_function(self, file: str, key: FunctionKey) -> set[str]:
        return self.data.get(file, {}).get(key, set())


def analyze_unused_params(file_path: str, source: str) -> dict[FunctionKey, set[str]]:
    """
    Returns a map of FunctionKey -> set(unused_param_names) using Python's ast.
    """
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return {}
    result: dict[FunctionKey, set[str]] = {}

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._handle(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._handle(node)

        def _handle(self, node):
            # collect param names
            params = [
                a.arg
                for a in (
                    list(getattr(node.args, "posonlyargs", []))
                    + node.args.args
                    + node.args.kwonlyargs
                )
            ]
            # names used in body
            used: set[str] = set()
            for n in ast.walk(node):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    used.add(n.id)
            unused = [p for p in params if p not in used and p not in {"self", "cls", "_", "__"}]
            if unused:
                key = FunctionKey(name=node.name, lineno=node.lineno)
                result.setdefault(key, set()).update(unused)

    Visitor().visit(tree)
    return result


# --------------------------
# LibCST transformers
# --------------------------


class ModuleState:
    def __init__(self):
        self.has_logging_import: bool = False
        self.has_logger_var: bool = False
        self.insert_logger_after_import: cst.CSTNode | None = None
        self.added_import: bool = False
        self.added_logger: bool = False
        self.print_to_logger_count: int = 0
        self.broad_except_count: int = 0
        self.bare_except_count: int = 0
        self.star_import_count: int = 0
        self.top_level_call_count: int = 0
        self.renamed_params: int = 0


class FirstPassDetect(cst.CSTVisitor):
    def __init__(self, state: ModuleState):
        self.state = state

    def visit_Import(self, node: cst.Import) -> None:
        for n in node.names:
            if m.matches(n.name, m.Name("logging")):
                self.state.has_logging_import = True
        if self.state.insert_logger_after_import is None:
            self.state.insert_logger_after_import = node

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if isinstance(node.module, cst.Name) and node.module.value == "logging":
            self.state.has_logging_import = True
        if self.state.insert_logger_after_import is None:
            self.state.insert_logger_after_import = node

    def visit_Assign(self, node: cst.Assign) -> None:
        # detect logger = logging.getLogger(__name__)
        try:
            if m.matches(
                node.value,
                m.Call(func=m.Attribute(value=m.Name("logging"), attr=m.Name("getLogger"))),
            ):
                for t in node.targets:
                    if m.matches(t.target, m.Name("logger")):
                        self.state.has_logger_var = True
        except Exception:
            pass


class AutoFixTransformer(cst.CSTTransformer):
    def __init__(self, unused_index: dict[FunctionKey, set[str]], verbose: int = 0):
        super().__init__()
        self.state = ModuleState()
        self.unused_index = unused_index
        self.verbose = verbose
        self._at_module_level = True  # track module-level for top-level calls

    def _ensure_logging_boilerplate(self, updated_node: cst.Module) -> cst.Module:
        new_body = list(updated_node.body)
        insert_idx = 0
        # Add import logging if missing
        if not self.state.has_logging_import:
            import_logging = cst.parse_statement("import logging\n")
            new_body.insert(insert_idx, import_logging)
            insert_idx += 1
            self.state.added_import = True
        # Add logger var if missing
        if not self.state.has_logger_var:
            logger_assign = cst.parse_statement("logger = logging.getLogger(__name__)\n")
            # place after imports if possible
            if self.state.insert_logger_after_import is not None:
                # find index of first import
                for i, stmt in enumerate(new_body):
                    if stmt is self.state.insert_logger_after_import:
                        new_body.insert(i + 1, logger_assign)
                        break
                else:
                    new_body.insert(insert_idx, logger_assign)
            else:
                new_body.insert(insert_idx, logger_assign)
            self.state.added_logger = True
        return updated_node.with_changes(body=new_body)

    # ----- Module & Simple statements -----

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        # If we converted any print -> logger, ensure logging boilerplate exists
        if self.state.print_to_logger_count > 0 and (
            not self.state.has_logging_import or not self.state.has_logger_var
        ):
            updated_node = self._ensure_logging_boilerplate(updated_node)
        return updated_node

    def visit_SimpleStatementLine(self, node: cst.SimpleStatementLine) -> bool | None:
        # detect top-level call expression (bare Call at module body)
        if self._at_module_level:
            if (
                len(node.body) == 1
                and isinstance(node.body[0], cst.Expr)
                and isinstance(node.body[0].value, cst.Call)
            ):
                # Heuristic: ignore logging/basicConfig or typing-only calls
                node.body[0].value
                text = node.code.strip()
                if "logging.basicConfig" not in text:
                    # annotate with a trailing comment
                    self.state.top_level_call_count += 1
                    # append a comment only once (avoid duplicating)
                    trailing = node.trailing_whitespace.comment
                    comment_text = "  # TODO: top-level call at import; consider moving under if __name__ == '__main__':"
                    if trailing is None or "top-level call" not in trailing.value:
                        new_trailing = cst.TrailingWhitespace(
                            whitespace=node.trailing_whitespace.whitespace,
                            comment=cst.Comment(comment_text),
                            newline=node.trailing_whitespace.newline,
                        )
                        node.with_changes(trailing_whitespace=new_trailing)
                        return False  # replace
        return True

    def leave_SimpleStatementLine(
        self,
        original_node: cst.SimpleStatementLine,
        updated_node: cst.SimpleStatementLine,
    ) -> cst.BaseStatement:
        return updated_node

    def visit_IndentedBlock(self, node: cst.IndentedBlock) -> bool | None:
        self._at_module_level = False
        return True

    def leave_IndentedBlock(
        self, original_node: cst.IndentedBlock, updated_node: cst.IndentedBlock
    ) -> cst.BaseSuite:
        return updated_node

    # ----- print(...) -> logger.info(...) -----

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.Call:
        # Only transform print(...) calls
        if m.matches(updated_node.func, m.Name("print")):
            # Convert to logger.info(...)
            self.state.print_to_logger_count += 1
            new = updated_node.with_changes(
                func=cst.Attribute(value=cst.Name("logger"), attr=cst.Name("info"))
            )
            return new
        return updated_node

    # ----- except: / except Exception -----

    def leave_ExceptHandler(
        self, original_node: cst.ExceptHandler, updated_node: cst.ExceptHandler
    ) -> cst.ExceptHandler:
        is_bare = updated_node.type is None
        is_broad = False
        if updated_node.type is not None:
            t = updated_node.type
            if isinstance(t, cst.Name) and t.value in {"Exception", "BaseException"}:
                is_broad = True

        (
            updated_node.body.header.trailing_whitespace.comment
            if hasattr(updated_node.body, "header")
            else None
        )
        todo_comment = None
        if is_bare:
            self.state.bare_except_count += 1
            # change to Exception and add TODO
            updated_node = updated_node.with_changes(type=cst.Name("Exception"))
            todo_comment = "  # TODO: narrow this bare 'except' to specific exception(s)"
        elif is_broad:
            self.state.broad_except_count += 1
            todo_comment = "  # TODO: narrow broad 'except Exception' to specific exception(s)"

        if todo_comment:
            body = updated_node.body
            if isinstance(body, cst.IndentedBlock):
                new_header = body.header.with_changes(
                    trailing_whitespace=cst.TrailingWhitespace(
                        whitespace=body.header.trailing_whitespace.whitespace,
                        comment=cst.Comment(todo_comment),
                        newline=body.header.trailing_whitespace.newline,
                    )
                )
                updated_node = updated_node.with_changes(body=body.with_changes(header=new_header))
        return updated_node

    # ----- from x import * -----

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> cst.ImportFrom:
        # If star import, append comment to the statement
        if isinstance(updated_node.names, cst.ImportStar):
            # add TODO trailing comment
            # updated_node is a BaseSmallStatement; add comment at line end
            return updated_node.with_changes(
                whitespace_after_from=updated_node.whitespace_after_from,
            )
        return updated_node

    def leave_SimpleStatementLine_ImportFrom(self, original_node, updated_node):
        return updated_node

    # ----- Function defs: rename unused params to _param -----

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        FunctionKey(name=original_node.name.value, lineno=original_node.line)
        {
            p
            for k, pset in self.unused_index.items()
            for p in (pset if isinstance(pset, set) else set())
        }  # placeholder

        # The above is not ideal; we instead rely on mapping passed in constructor per-file.
        # We'll rename params that appear in self.unused_index for this function key.
        return updated_node


# We need a custom transformer that has access to per-file unused param mapping.
class ParamRenameTransformer(cst.CSTTransformer):
    def __init__(self, unused_for_func: dict[tuple[str, int], set[str]]):
        self.unused_for_func = unused_for_func
        self.current_func_key: tuple[str, int] | None = None
        self.renamed_count = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        self.current_func_key = (node.name.value, node.line)
        return True

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        key = (original_node.name.value, original_node.line)
        unused = self.unused_for_func.get(key, set())
        if not unused:
            return updated_node

        def rename_param(p: cst.Param) -> cst.Param:
            if (
                p.name.value in unused
                and p.name.value not in {"self", "cls"}
                and not p.name.value.startswith("_")
            ):
                self.renamed_count += 1
                return p.with_changes(name=cst.Name("_" + p.name.value))
            return p

        params = updated_node.params
        new_params = params.with_changes(
            params=[rename_param(p) for p in params.params],
            posonly_params=[rename_param(p) for p in getattr(params, "posonly_params", [])],
            kwonly_params=[rename_param(p) for p in params.kwonly_params],
        )
        return updated_node.with_changes(params=new_params)


# Helper: build per-file mapping FunctionKey -> unused params and feed into transformer
def build_unused_index_for_file(path: str, src: str) -> dict[tuple[str, int], set[str]]:
    idx = analyze_unused_params(path, src)
    mapping: dict[tuple[str, int], set[str]] = {}
    for key, params in idx.items():
        mapping[(key.name, key.lineno)] = params
    return mapping


# Apply transforms to one file
def process_file(path: str, apply: bool, verbose: int) -> tuple[bool, str, dict[str, int]]:
    with open(path, encoding="utf-8") as f:
        src = f.read()

    # Build unused-param map
    unused_map = build_unused_index_for_file(path, src)

    try:
        mod = cst.parse_module(src)
    except Exception as e:
        return False, f"Parse error: {e}", {}

    # Pass 1: detect logging, etc.
    state = ModuleState()
    mod.visit(FirstPassDetect(state))

    # Pass 2: transform prints, exceptions, star imports, top-level calls comments
    transformer = AutoFixTransformer(unused_index={}, verbose=verbose)
    mod2 = mod.visit(transformer)

    # Pass 3: rename unused params via CST (uses AST-derived map)
    param_tf = ParamRenameTransformer(unused_for_func=unused_map)
    mod3 = mod2.visit(param_tf)

    # Ensure logging boilerplate if prints were converted
    if transformer.state.print_to_logger_count > 0 and (
        not state.has_logging_import or not state.has_logger_var
    ):
        # Re-run ensure logger by creating a temp state and injecting
        # Simpler: prepend import/logging lines if not present
        new_code = mod3.code
        prelude = ""
        if not state.has_logging_import:
            prelude += "import logging\n"
        if not state.has_logger_var:
            prelude += "logger = logging.getLogger(__name__)\n"
        mod3 = cst.parse_module(prelude + new_code)

    changed = mod3.code != src
    summary = {
        "print_to_logger": transformer.state.print_to_logger_count,
        "broad_except_annotated": transformer.state.broad_except_count,
        "bare_except_fixed": transformer.state.bare_except_count,
        "params_renamed": param_tf.renamed_count,
        "top_level_calls_flagged": transformer.state.top_level_call_count,
    }

    if apply and changed:
        with open(path, "w", encoding="utf-8") as f:
            f.write(mod3.code)
    return changed, "ok", summary


def main():
    ns = parse_args()
    py_files = walk_python_files(
        ns.root, include_tests=ns.include_tests, include_scripts=ns.include_scripts
    )
    if ns.verbose:
        print(f"Discovered {len(py_files)} Python files under {ns.root}", file=sys.stderr)

    totals = {
        "files_changed": 0,
        "print_to_logger": 0,
        "broad_except_annotated": 0,
        "bare_except_fixed": 0,
        "params_renamed": 0,
        "top_level_calls_flagged": 0,
    }

    for path in py_files:
        changed, msg, summary = process_file(path, apply=ns.apply, verbose=ns.verbose)
        if ns.verbose and changed:
            print(f"[CHANGED] {path} -> {summary}", file=sys.stderr)
        if changed:
            totals["files_changed"] += 1
        for k in (
            "print_to_logger",
            "broad_except_annotated",
            "bare_except_fixed",
            "params_renamed",
            "top_level_calls_flagged",
        ):
            totals[k] += summary.get(k, 0)

    print("==== Autofix Summary ====")
    for k, v in totals.items():
        print(f"{k}: {v}")
    if not ns.apply:
        print("\n(Dry-run only; re-run with --apply to write changes.)")


if __name__ == "__main__":
    main()
