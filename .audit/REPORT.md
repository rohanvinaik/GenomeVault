# Audit Summary

## Syntax Errors
None

## Ruff Issues Before
check_lint_status.py:48:29: E741 Ambiguous variable name: `l`
devtools/debug_genomevault.py:32:20: F401 `pydantic` imported but unused; consider using `importlib.util.find_spec` to test for availability
devtools/debug_genomevault.py:38:24: F401 `pydantic_settings` imported but unused; consider using `importlib.util.find_spec` to test for availability
devtools/diagnose_failures.py:17:12: F401 `torch` imported but unused; consider using `importlib.util.find_spec` to test for availability
enhanced_cleanup.py:286:9: E722 Do not use bare `except`
enhanced_cleanup.py:490:13: E722 Do not use bare `except`
enhanced_cleanup.py:685:13: E722 Do not use bare `except`
examples/basic_usage.py:241:9: F823 Local variable `logger` referenced before assignment
examples/demo_hypervector_encoding.py:12:1: E402 Module level import not at top of file
examples/demo_hypervector_encoding.py:14:1: E402 Module level import not at top of file
examples/demo_hypervector_encoding.py:15:1: E402 Module level import not at top of file
examples/demo_hypervector_encoding.py:23:1: E402 Module level import not at top of file
examples/demo_hypervector_encoding.py:24:1: E402 Module level import not at top of file
examples/demo_hypervector_encoding.py:25:1: E402 Module level import not at top of file
examples/demo_hypervector_encoding.py:26:1: E402 Module level import not at top of file
examples/diabetes_risk_demo.py:9:1: E402 Module level import not at top of file
examples/diabetes_risk_demo.py:11:1: E402 Module level import not at top of file
examples/example_usage.py:10:1: E402 Module level import not at top of file
examples/example_usage.py:12:1: E402 Module level import not at top of file
examples/example_usage.py:14:1: E402 Module level import not at top of file
examples/example_usage.py:15:1: E402 Module level import not at top of file
examples/example_usage.py:19:1: E402 Module level import not at top of file
examples/example_usage.py:20:1: E402 Module level import not at top of file
examples/example_usage.py:23:1: E402 Module level import not at top of file
examples/example_usage.py:24:1: E402 Module level import not at top of file
examples/example_usage.py:25:1: E402 Module level import not at top of file
examples/hdc_error_tuning_example.py:8:1: E402 Module level import not at top of file
examples/hdc_error_tuning_example.py:10:1: E402 Module level import not at top of file
examples/hdc_pir_integration_demo.py:9:1: E402 Module level import not at top of file
examples/hdc_pir_integration_demo.py:10:1: E402 Module level import not at top of file
examples/hdc_pir_integration_demo.py:12:1: E402 Module level import not at top of file
examples/hdc_pir_integration_demo.py:14:1: E402 Module level import not at top of file
examples/hdc_pir_integration_demo.py:15:1: E402 Module level import not at top of file
examples/integration_example.py:11:1: E402 Module level import not at top of file
examples/integration_example.py:13:1: E402 Module level import not at top of file
examples/integration_example.py:15:1: E402 Module level import not at top of file
examples/integration_example.py:16:1: E402 Module level import not at top of file
examples/integration_example.py:19:1: E402 Module level import not at top of file
examples/integration_example.py:20:1: E402 Module level import not at top of file
examples/integration_example.py:21:1: E402 Module level import not at top of file
examples/integration_example.py:24:1: E402 Module level import not at top of file
examples/integration_example.py:25:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:12:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:13:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:14:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:16:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:19:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:22:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:23:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:24:1: E402 Module level import not at top of file

## Ruff Issues After
check_lint_status.py:48:29: E741 Ambiguous variable name: `l`
devtools/debug_genomevault.py:32:20: F401 `pydantic` imported but unused; consider using `importlib.util.find_spec` to test for availability
devtools/debug_genomevault.py:38:24: F401 `pydantic_settings` imported but unused; consider using `importlib.util.find_spec` to test for availability
devtools/diagnose_failures.py:17:12: F401 `torch` imported but unused; consider using `importlib.util.find_spec` to test for availability
enhanced_cleanup.py:286:9: E722 Do not use bare `except`
enhanced_cleanup.py:490:13: E722 Do not use bare `except`
enhanced_cleanup.py:685:13: E722 Do not use bare `except`
examples/basic_usage.py:241:9: F823 Local variable `logger` referenced before assignment
examples/demo_hypervector_encoding.py:12:1: E402 Module level import not at top of file
examples/demo_hypervector_encoding.py:14:1: E402 Module level import not at top of file
examples/demo_hypervector_encoding.py:15:1: E402 Module level import not at top of file
examples/demo_hypervector_encoding.py:23:1: E402 Module level import not at top of file
examples/demo_hypervector_encoding.py:24:1: E402 Module level import not at top of file
examples/demo_hypervector_encoding.py:25:1: E402 Module level import not at top of file
examples/demo_hypervector_encoding.py:26:1: E402 Module level import not at top of file
examples/diabetes_risk_demo.py:9:1: E402 Module level import not at top of file
examples/diabetes_risk_demo.py:11:1: E402 Module level import not at top of file
examples/example_usage.py:10:1: E402 Module level import not at top of file
examples/example_usage.py:12:1: E402 Module level import not at top of file
examples/example_usage.py:14:1: E402 Module level import not at top of file
examples/example_usage.py:15:1: E402 Module level import not at top of file
examples/example_usage.py:19:1: E402 Module level import not at top of file
examples/example_usage.py:20:1: E402 Module level import not at top of file
examples/example_usage.py:23:1: E402 Module level import not at top of file
examples/example_usage.py:24:1: E402 Module level import not at top of file
examples/example_usage.py:25:1: E402 Module level import not at top of file
examples/hdc_error_tuning_example.py:8:1: E402 Module level import not at top of file
examples/hdc_error_tuning_example.py:10:1: E402 Module level import not at top of file
examples/hdc_pir_integration_demo.py:9:1: E402 Module level import not at top of file
examples/hdc_pir_integration_demo.py:10:1: E402 Module level import not at top of file
examples/hdc_pir_integration_demo.py:12:1: E402 Module level import not at top of file
examples/hdc_pir_integration_demo.py:14:1: E402 Module level import not at top of file
examples/hdc_pir_integration_demo.py:15:1: E402 Module level import not at top of file
examples/integration_example.py:11:1: E402 Module level import not at top of file
examples/integration_example.py:13:1: E402 Module level import not at top of file
examples/integration_example.py:15:1: E402 Module level import not at top of file
examples/integration_example.py:16:1: E402 Module level import not at top of file
examples/integration_example.py:19:1: E402 Module level import not at top of file
examples/integration_example.py:20:1: E402 Module level import not at top of file
examples/integration_example.py:21:1: E402 Module level import not at top of file
examples/integration_example.py:24:1: E402 Module level import not at top of file
examples/integration_example.py:25:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:12:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:13:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:14:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:16:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:19:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:22:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:23:1: E402 Module level import not at top of file
examples/orphan_disease_workflow.py:24:1: E402 Module level import not at top of file

## Mypy (light)
genomevault/hypervector/encoding/sparse_projection.py: error: Source file found twice under different module names: "genomevault.genomevault.hypervector.encoding.sparse_projection" and "genomevault.hypervector.encoding.sparse_projection"
genomevault/hypervector/encoding/sparse_projection.py: note: See https://mypy.readthedocs.io/en/stable/running_mypy.html#mapping-file-paths-to-modules for more info
genomevault/hypervector/encoding/sparse_projection.py: note: Common resolutions include: a) adding `__init__.py` somewhere, b) using `--explicit-package-bases` or adjusting MYPYPATH
Found 1 error in 1 file (errors prevented further checking)

## Statistics
- Total Ruff issues after fixes:      804 lines
- Mypy issues:        4 lines
