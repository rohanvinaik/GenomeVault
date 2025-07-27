# GenomeVault Test Suite Linting Report

## Summary

- **flake8**: 807 issues in 79 files
- **pylint**: 401 issues in 90 files
- **mypy**: 322 issues in 90 files

## Flake8 Issues

### bridge_simple.py
Path: `/Users/rohanvinaik/experiments/bridge_simple.py`

- /Users/rohanvinaik/experiments/bridge_simple.py:8:1: F401 'time' imported but unused
- /Users/rohanvinaik/experiments/bridge_simple.py:10:1: F401 'threading' imported but unused
- /Users/rohanvinaik/experiments/bridge_simple.py:12:1: F401 'pathlib.Path' imported but unused
- /Users/rohanvinaik/experiments/bridge_simple.py:35:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/bridge_simple.py:43:1: E305 expected 2 blank lines after class or function definition, found 1
- /Users/rohanvinaik/experiments/bridge_simple.py:60:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/bridge_simple.py:66:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/bridge_simple.py:81:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/bridge_simple.py:93:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/bridge_simple.py:97:1: E305 expected 2 blank lines after class or function definition, found 1

### command_reader.py
Path: `/Users/rohanvinaik/experiments/command_reader.py`

- /Users/rohanvinaik/experiments/command_reader.py:20:1: E722 do not use bare 'except'

### debug_launcher.py
Path: `/Users/rohanvinaik/experiments/debug_launcher.py`

- /Users/rohanvinaik/experiments/debug_launcher.py:5:1: F401 'sys' imported but unused
- /Users/rohanvinaik/experiments/debug_launcher.py:8:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/debug_launcher.py:10:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/debug_launcher.py:12:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/debug_launcher.py:20:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/debug_launcher.py:22:66: W291 trailing whitespace
- /Users/rohanvinaik/experiments/debug_launcher.py:23:32: E128 continuation line under-indented for visual indent
- /Users/rohanvinaik/experiments/debug_launcher.py:27:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/debug_launcher.py:33:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/debug_launcher.py:36:1: W293 blank line contains whitespace
- ... and 7 more issues

### diabetes_analysis_demo.py
Path: `/Users/rohanvinaik/experiments/diabetes_analysis_demo.py`

- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:7:1: F401 'numpy as np' imported but unused
- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:8:1: F401 'pathlib.Path' imported but unused
- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:9:1: F401 'json' imported but unused
- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:13:1: F401 'genomics.privacy_genomics.PrivacyPreservingAnalysis' imported but unused
- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:15:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:20:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:25:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:31:36: E261 at least two spaces before inline comment
- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:33:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:35:1: W293 blank line contains whitespace
- ... and 18 more issues

### efficiency_methods.py
Path: `/Users/rohanvinaik/experiments/efficiency_methods.py`

- /Users/rohanvinaik/experiments/efficiency_methods.py:11:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/efficiency_methods.py:13:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/efficiency_methods.py:17:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/efficiency_methods.py:22:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/efficiency_methods.py:24:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/efficiency_methods.py:27:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/efficiency_methods.py:31:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/efficiency_methods.py:34:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/efficiency_methods.py:37:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/efficiency_methods.py:40:36: E226 missing whitespace around arithmetic operator
- ... and 33 more issues

### basic_usage_fixed.py
Path: `/Users/rohanvinaik/experiments/fixes/basic_usage_fixed.py`

- /Users/rohanvinaik/experiments/fixes/basic_usage_fixed.py:8:1: F401 'os' imported but unused
- /Users/rohanvinaik/experiments/fixes/basic_usage_fixed.py:9:1: F401 'datetime.datetime' imported but unused
- /Users/rohanvinaik/experiments/fixes/basic_usage_fixed.py:10:1: F401 'pathlib.Path' imported but unused
- /Users/rohanvinaik/experiments/fixes/basic_usage_fixed.py:13:1: F401 'genomevault.core.config.Config' imported but unused
- /Users/rohanvinaik/experiments/fixes/basic_usage_fixed.py:14:1: F401 'genomevault.local_processing.TranscriptomicsProcessor' imported but unused
- /Users/rohanvinaik/experiments/fixes/basic_usage_fixed.py:37:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/basic_usage_fixed.py:38:17: F541 f-string is missing placeholders
- /Users/rohanvinaik/experiments/fixes/basic_usage_fixed.py:47:5: F841 local variable 'processor' is assigned to but never used
- /Users/rohanvinaik/experiments/fixes/basic_usage_fixed.py:76:33: W291 trailing whitespace
- /Users/rohanvinaik/experiments/fixes/basic_usage_fixed.py:77:32: W291 trailing whitespace
- ... and 7 more issues

### bridge_simple_fixed.py
Path: `/Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py`

- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:10:1: F401 'threading' imported but unused
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:12:1: F401 'pathlib.Path' imported but unused
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:55:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:62:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:69:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:79:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:84:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:88:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:91:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:107:1: W293 blank line contains whitespace
- ... and 9 more issues

### diagnose_genomevault_import.py
Path: `/Users/rohanvinaik/experiments/fixes/diagnose_genomevault_import.py`

- /Users/rohanvinaik/experiments/fixes/diagnose_genomevault_import.py:10:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/diagnose_genomevault_import.py:12:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/diagnose_genomevault_import.py:14:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/diagnose_genomevault_import.py:17:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/diagnose_genomevault_import.py:21:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/diagnose_genomevault_import.py:34:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/diagnose_genomevault_import.py:48:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/diagnose_genomevault_import.py:57:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/diagnose_genomevault_import.py:65:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/diagnose_genomevault_import.py:68:1: W293 blank line contains whitespace
- ... and 7 more issues

### fix_cupy_issue.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_cupy_issue.py`

- /Users/rohanvinaik/experiments/fixes/fix_cupy_issue.py:9:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/fix_cupy_issue.py:11:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_cupy_issue.py:13:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_cupy_issue.py:17:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_cupy_issue.py:20:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_cupy_issue.py:25:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_cupy_issue.py:29:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_cupy_issue.py:33:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_cupy_issue.py:39:19: F541 f-string is missing placeholders
- /Users/rohanvinaik/experiments/fixes/fix_cupy_issue.py:43:19: F541 f-string is missing placeholders
- ... and 11 more issues

### fix_genomevault_comprehensive.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py`

- /Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py:10:1: F401 're' imported but unused
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py:12:1: C901 'apply_fixes' is too complex (19)
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py:12:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py:14:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py:16:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py:19:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py:23:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py:27:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py:32:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py:35:1: W293 blank line contains whitespace
- ... and 35 more issues

### fix_genomevault_logging_comprehensive.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_genomevault_logging_comprehensive.py`

- /Users/rohanvinaik/experiments/fixes/fix_genomevault_logging_comprehensive.py:9:1: C901 'fix_logging_file' is too complex (31)
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_logging_comprehensive.py:9:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_logging_comprehensive.py:11:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_logging_comprehensive.py:13:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_logging_comprehensive.py:17:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_logging_comprehensive.py:21:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_logging_comprehensive.py:27:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_logging_comprehensive.py:30:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_logging_comprehensive.py:36:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_logging_comprehensive.py:41:1: W293 blank line contains whitespace
- ... and 17 more issues

### fix_mac_gpu_compatibility.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py`

- /Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py:8:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py:10:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py:12:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py:15:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py:20:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py:24:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py:30:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py:33:5: F841 local variable 'import_section_end' is assigned to but never used
- /Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py:34:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py:42:1: W293 blank line contains whitespace
- ... and 19 more issues

### fix_pir_circular_import.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py`

- /Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py:8:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py:10:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py:13:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py:16:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py:20:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py:23:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py:28:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py:32:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py:42:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py:53:1: W293 blank line contains whitespace
- ... and 35 more issues

### fix_processors_error.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_processors_error.py`

- /Users/rohanvinaik/experiments/fixes/fix_processors_error.py:8:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/fix_processors_error.py:10:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_processors_error.py:12:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_processors_error.py:15:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_processors_error.py:19:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_processors_error.py:24:32: E226 missing whitespace around arithmetic operator
- /Users/rohanvinaik/experiments/fixes/fix_processors_error.py:25:24: E226 missing whitespace around arithmetic operator
- /Users/rohanvinaik/experiments/fixes/fix_processors_error.py:28:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_processors_error.py:31:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_processors_error.py:36:1: W293 blank line contains whitespace
- ... and 5 more issues

### fix_remaining_issues.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py`

- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:12:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:15:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:23:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:30:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:38:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:51:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:54:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:56:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:60:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:63:1: W293 blank line contains whitespace
- ... and 38 more issues

### genomevault_import_fixer.py
Path: `/Users/rohanvinaik/experiments/fixes/genomevault_import_fixer.py`

- /Users/rohanvinaik/experiments/fixes/genomevault_import_fixer.py:46:6: E999 IndentationError: expected an indented block after function definition on line 45

### install_all_dependencies.py
Path: `/Users/rohanvinaik/experiments/fixes/install_all_dependencies.py`

- /Users/rohanvinaik/experiments/fixes/install_all_dependencies.py:11:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/install_all_dependencies.py:13:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/install_all_dependencies.py:16:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/install_all_dependencies.py:25:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/install_all_dependencies.py:30:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/install_all_dependencies.py:34:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/install_all_dependencies.py:39:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/install_all_dependencies.py:43:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/install_all_dependencies.py:46:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/install_all_dependencies.py:50:1: W293 blank line contains whitespace
- ... and 29 more issues

### install_genomevault_deps.py
Path: `/Users/rohanvinaik/experiments/fixes/install_genomevault_deps.py`

- /Users/rohanvinaik/experiments/fixes/install_genomevault_deps.py:10:1: C901 'install_genomevault_dependencies' is too complex (12)
- /Users/rohanvinaik/experiments/fixes/install_genomevault_deps.py:10:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/install_genomevault_deps.py:12:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/install_genomevault_deps.py:15:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/install_genomevault_deps.py:22:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/install_genomevault_deps.py:25:18: W291 trailing whitespace
- /Users/rohanvinaik/experiments/fixes/install_genomevault_deps.py:30:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/install_genomevault_deps.py:34:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/install_genomevault_deps.py:39:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/install_genomevault_deps.py:43:1: W293 blank line contains whitespace
- ... and 15 more issues

### quick_fix_logging.py
Path: `/Users/rohanvinaik/experiments/fixes/quick_fix_logging.py`

- /Users/rohanvinaik/experiments/fixes/quick_fix_logging.py:9:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/quick_fix_logging.py:11:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/quick_fix_logging.py:13:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/quick_fix_logging.py:17:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/quick_fix_logging.py:21:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/quick_fix_logging.py:27:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/quick_fix_logging.py:30:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/quick_fix_logging.py:42:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/quick_fix_logging.py:46:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/quick_fix_logging.py:49:1: W293 blank line contains whitespace
- ... and 5 more issues

### run_comprehensive_linting.py
Path: `/Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py`

- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:9:1: F401 'json' imported but unused
- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:11:1: C901 'run_linters' is too complex (18)
- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:11:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:13:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:16:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:19:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:29:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:45:5: E722 do not use bare 'except'
- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:49:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:59:1: W293 blank line contains whitespace
- ... and 34 more issues

### tail_chasing_fixer.py
Path: `/Users/rohanvinaik/experiments/fixes/tail_chasing_fixer.py`

- /Users/rohanvinaik/experiments/fixes/tail_chasing_fixer.py:49:6: E999 IndentationError: expected an indented block after function definition on line 48

### test_system.py
Path: `/Users/rohanvinaik/experiments/fixes/test_system.py`

- /Users/rohanvinaik/experiments/fixes/test_system.py:9:1: F401 'time' imported but unused
- /Users/rohanvinaik/experiments/fixes/test_system.py:14:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/test_system.py:21:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/test_system.py:31:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/test_system.py:38:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/test_system.py:56:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/test_system.py:78:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/test_system.py:81:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/test_system.py:89:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/test_system.py:97:1: W293 blank line contains whitespace
- ... and 32 more issues

### unified_bridge.py
Path: `/Users/rohanvinaik/experiments/fixes/unified_bridge.py`

- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:15:1: F401 'pathlib.Path' imported but unused
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:63:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:71:5: E722 do not use bare 'except'
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:74:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:88:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:95:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:107:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:108:1: C901 'send_command' is too complex (14)
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:112:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:116:1: W293 blank line contains whitespace
- ... and 21 more issues

### update_menu_pythonpath.py
Path: `/Users/rohanvinaik/experiments/fixes/update_menu_pythonpath.py`

- /Users/rohanvinaik/experiments/fixes/update_menu_pythonpath.py:8:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/update_menu_pythonpath.py:10:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/update_menu_pythonpath.py:12:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/update_menu_pythonpath.py:16:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/update_menu_pythonpath.py:19:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/update_menu_pythonpath.py:26:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/update_menu_pythonpath.py:29:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/update_menu_pythonpath.py:34:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/update_menu_pythonpath.py:38:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/update_menu_pythonpath.py:41:1: W293 blank line contains whitespace
- ... and 5 more issues

### validate_genomevault.py
Path: `/Users/rohanvinaik/experiments/fixes/validate_genomevault.py`

- /Users/rohanvinaik/experiments/fixes/validate_genomevault.py:10:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/fixes/validate_genomevault.py:12:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/validate_genomevault.py:16:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/validate_genomevault.py:26:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/validate_genomevault.py:37:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/validate_genomevault.py:48:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/validate_genomevault.py:64:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/validate_genomevault.py:67:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/validate_genomevault.py:74:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/fixes/validate_genomevault.py:88:1: W293 blank line contains whitespace
- ... and 2 more issues

### genomevault_hd_experiment.py
Path: `/Users/rohanvinaik/experiments/genomevault_hd_experiment.py`

- /Users/rohanvinaik/experiments/genomevault_hd_experiment.py:25:6: E999 IndentationError: expected an indented block after function definition on line 24

### genomevault_menu.py
Path: `/Users/rohanvinaik/experiments/genomevault_menu.py`

- /Users/rohanvinaik/experiments/genomevault_menu.py:48:6: E999 IndentationError: expected an indented block after function definition on line 47

### gpu_benchmark.py
Path: `/Users/rohanvinaik/experiments/gpu_benchmark.py`

- /Users/rohanvinaik/experiments/gpu_benchmark.py:30:6: E999 IndentationError: expected an indented block after function definition on line 29

### how_to_guide.py
Path: `/Users/rohanvinaik/experiments/how_to_guide.py`

- /Users/rohanvinaik/experiments/how_to_guide.py:64:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/how_to_guide.py:69:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/how_to_guide.py:119:29: W291 trailing whitespace

### monitor_server.py
Path: `/Users/rohanvinaik/experiments/monitor_server.py`

- /Users/rohanvinaik/experiments/monitor_server.py:18:6: E999 IndentationError: expected an indented block after function definition on line 17

### pipe_reader.py
Path: `/Users/rohanvinaik/experiments/pipe_reader.py`

- /Users/rohanvinaik/experiments/pipe_reader.py:7:1: F401 'sys' imported but unused
- /Users/rohanvinaik/experiments/pipe_reader.py:16:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/pipe_reader.py:24:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/pipe_reader.py:27:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/pipe_reader.py:31:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/pipe_reader.py:44:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/pipe_reader.py:48:1: E305 expected 2 blank lines after class or function definition, found 1

### power_control.py
Path: `/Users/rohanvinaik/experiments/power_control.py`

- /Users/rohanvinaik/experiments/power_control.py:14:1: F401 'signal' imported but unused
- /Users/rohanvinaik/experiments/power_control.py:30:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/power_control.py:42:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/power_control.py:51:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/power_control.py:56:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/power_control.py:61:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/power_control.py:66:13: E306 expected 1 blank line before a nested definition, found 0
- /Users/rohanvinaik/experiments/power_control.py:76:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/power_control.py:81:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/power_control.py:87:35: W291 trailing whitespace
- ... and 35 more issues

### project_launcher.py
Path: `/Users/rohanvinaik/experiments/project_launcher.py`

- /Users/rohanvinaik/experiments/project_launcher.py:105:6: E999 IndentationError: expected an indented block after function definition on line 104

### run_experiment.py
Path: `/Users/rohanvinaik/experiments/run_experiment.py`

- /Users/rohanvinaik/experiments/run_experiment.py:36:6: E999 IndentationError: expected an indented block after function definition on line 35

### server_api.py
Path: `/Users/rohanvinaik/experiments/server_api.py`

- /Users/rohanvinaik/experiments/server_api.py:9:1: F401 'json' imported but unused
- /Users/rohanvinaik/experiments/server_api.py:19:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/server_api.py:27:1: E305 expected 2 blank lines after class or function definition, found 1
- /Users/rohanvinaik/experiments/server_api.py:30:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/server_api.py:35:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/server_api.py:40:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/server_api.py:46:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/server_api.py:51:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/server_api.py:59:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/server_api.py:61:1: W293 blank line contains whitespace
- ... and 42 more issues

### terminal_bridge.py
Path: `/Users/rohanvinaik/experiments/terminal_bridge.py`

- /Users/rohanvinaik/experiments/terminal_bridge.py:8:1: F401 'sys' imported but unused
- /Users/rohanvinaik/experiments/terminal_bridge.py:13:1: F401 'pathlib.Path' imported but unused
- /Users/rohanvinaik/experiments/terminal_bridge.py:15:1: F401 'subprocess' imported but unused
- /Users/rohanvinaik/experiments/terminal_bridge.py:32:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/terminal_bridge.py:39:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/terminal_bridge.py:44:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/terminal_bridge.py:48:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/terminal_bridge.py:72:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/terminal_bridge.py:87:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/terminal_bridge.py:98:1: W293 blank line contains whitespace
- ... and 16 more issues

### terminal_bridge_v2.py
Path: `/Users/rohanvinaik/experiments/terminal_bridge_v2.py`

- /Users/rohanvinaik/experiments/terminal_bridge_v2.py:90:6: E999 IndentationError: expected an indented block after function definition on line 89

### test_experiment.py
Path: `/Users/rohanvinaik/experiments/test_experiment.py`

- /Users/rohanvinaik/experiments/test_experiment.py:7:1: F401 'json' imported but unused
- /Users/rohanvinaik/experiments/test_experiment.py:8:1: F401 'time' imported but unused
- /Users/rohanvinaik/experiments/test_experiment.py:25:5: F401 'psutil' imported but unused

### test_pipe.py
Path: `/Users/rohanvinaik/experiments/test_pipe.py`

- /Users/rohanvinaik/experiments/test_pipe.py:7:1: F401 'time' imported but unused
- /Users/rohanvinaik/experiments/test_pipe.py:9:1: E302 expected 2 blank lines, found 1
- /Users/rohanvinaik/experiments/test_pipe.py:11:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/test_pipe.py:13:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/test_pipe.py:19:1: W293 blank line contains whitespace
- /Users/rohanvinaik/experiments/test_pipe.py:28:1: E305 expected 2 blank lines after class or function definition, found 1

### test_hdc_adversarial.py
Path: `/Users/rohanvinaik/genomevault/tests/adversarial/test_hdc_adversarial.py`

- /Users/rohanvinaik/genomevault/tests/adversarial/test_hdc_adversarial.py:30:6: E999 IndentationError: expected an indented block after function definition on line 29

### test_pir_adversarial.py
Path: `/Users/rohanvinaik/genomevault/tests/adversarial/test_pir_adversarial.py`

- /Users/rohanvinaik/genomevault/tests/adversarial/test_pir_adversarial.py:28:6: E999 IndentationError: expected an indented block after function definition on line 27

### test_zk_adversarial.py
Path: `/Users/rohanvinaik/genomevault/tests/adversarial/test_zk_adversarial.py`

- /Users/rohanvinaik/genomevault/tests/adversarial/test_zk_adversarial.py:17:6: E999 IndentationError: expected an indented block after function definition on line 16

### conftest.py
Path: `/Users/rohanvinaik/genomevault/tests/conftest.py`

- /Users/rohanvinaik/genomevault/tests/conftest.py:187:5: E999 IndentationError: expected an indented block after function definition on line 186

### test_pir_e2e.py
Path: `/Users/rohanvinaik/genomevault/tests/e2e/test_pir_e2e.py`

- /Users/rohanvinaik/genomevault/tests/e2e/test_pir_e2e.py:26:6: E999 IndentationError: expected an indented block after function definition on line 25

### test_zk_e2e.py
Path: `/Users/rohanvinaik/genomevault/tests/e2e/test_zk_e2e.py`

- /Users/rohanvinaik/genomevault/tests/e2e/test_zk_e2e.py:21:6: E999 IndentationError: expected an indented block after function definition on line 20

### test_hipaa_governance.py
Path: `/Users/rohanvinaik/genomevault/tests/integration/test_hipaa_governance.py`

- /Users/rohanvinaik/genomevault/tests/integration/test_hipaa_governance.py:33:6: E999 IndentationError: expected an indented block after function definition on line 32

### test_proof_of_training.py
Path: `/Users/rohanvinaik/genomevault/tests/integration/test_proof_of_training.py`

- /Users/rohanvinaik/genomevault/tests/integration/test_proof_of_training.py:27:6: E999 IndentationError: expected an indented block after function definition on line 26

### test_pir_protocol.py
Path: `/Users/rohanvinaik/genomevault/tests/pir/test_pir_protocol.py`

- /Users/rohanvinaik/genomevault/tests/pir/test_pir_protocol.py:23:6: E999 IndentationError: expected an indented block after function definition on line 22

### test_hdc_properties.py
Path: `/Users/rohanvinaik/genomevault/tests/property/test_hdc_properties.py`

- /Users/rohanvinaik/genomevault/tests/property/test_hdc_properties.py:82:6: E999 IndentationError: expected an indented block after function definition on line 81

### test_zk_properties.py
Path: `/Users/rohanvinaik/genomevault/tests/property/test_zk_properties.py`

- /Users/rohanvinaik/genomevault/tests/property/test_zk_properties.py:25:6: E999 IndentationError: expected an indented block after function definition on line 24

### test_advanced_implementations.py
Path: `/Users/rohanvinaik/genomevault/tests/test_advanced_implementations.py`

- /Users/rohanvinaik/genomevault/tests/test_advanced_implementations.py:10:1: F401 'typing.Any' imported but unused
- /Users/rohanvinaik/genomevault/tests/test_advanced_implementations.py:10:1: F401 'typing.Dict' imported but unused
- /Users/rohanvinaik/genomevault/tests/test_advanced_implementations.py:169:11: F541 f-string is missing placeholders
- /Users/rohanvinaik/genomevault/tests/test_advanced_implementations.py:249:11: F541 f-string is missing placeholders
- /Users/rohanvinaik/genomevault/tests/test_advanced_implementations.py:259:11: F541 f-string is missing placeholders

### test_basic.py
Path: `/Users/rohanvinaik/genomevault/tests/test_basic.py`

- /Users/rohanvinaik/genomevault/tests/test_basic.py:1:1: F401 'typing.Any' imported but unused
- /Users/rohanvinaik/genomevault/tests/test_basic.py:1:1: F401 'typing.Dict' imported but unused
- /Users/rohanvinaik/genomevault/tests/test_basic.py:8:1: F401 'pytest' imported but unused

### test_client.py
Path: `/Users/rohanvinaik/genomevault/tests/test_client.py`

- /Users/rohanvinaik/genomevault/tests/test_client.py:14:6: E999 IndentationError: expected an indented block after function definition on line 13

### test_compression.py
Path: `/Users/rohanvinaik/genomevault/tests/test_compression.py`

- /Users/rohanvinaik/genomevault/tests/test_compression.py:22:6: E999 IndentationError: expected an indented block after function definition on line 21

### test_hdc_error_handling.py
Path: `/Users/rohanvinaik/genomevault/tests/test_hdc_error_handling.py`

- /Users/rohanvinaik/genomevault/tests/test_hdc_error_handling.py:22:6: E999 IndentationError: expected an indented block after function definition on line 21

### test_hdc_implementation.py
Path: `/Users/rohanvinaik/genomevault/tests/test_hdc_implementation.py`

- /Users/rohanvinaik/genomevault/tests/test_hdc_implementation.py:47:6: E999 IndentationError: expected an indented block after function definition on line 46

### test_hdc_pir_integration.py
Path: `/Users/rohanvinaik/genomevault/tests/test_hdc_pir_integration.py`

- /Users/rohanvinaik/genomevault/tests/test_hdc_pir_integration.py:31:6: E999 IndentationError: expected an indented block after function definition on line 30

### test_hypervector.py
Path: `/Users/rohanvinaik/genomevault/tests/test_hypervector.py`

- /Users/rohanvinaik/genomevault/tests/test_hypervector.py:18:6: E999 IndentationError: expected an indented block after function definition on line 17

### test_infrastructure.py
Path: `/Users/rohanvinaik/genomevault/tests/test_infrastructure.py`

- /Users/rohanvinaik/genomevault/tests/test_infrastructure.py:67:6: E999 IndentationError: expected an indented block after function definition on line 66

### test_it_pir.py
Path: `/Users/rohanvinaik/genomevault/tests/test_it_pir.py`

- /Users/rohanvinaik/genomevault/tests/test_it_pir.py:14:6: E999 IndentationError: expected an indented block after function definition on line 13

### test_it_pir_protocol.py
Path: `/Users/rohanvinaik/genomevault/tests/test_it_pir_protocol.py`

- /Users/rohanvinaik/genomevault/tests/test_it_pir_protocol.py:14:6: E999 IndentationError: expected an indented block after function definition on line 13

### test_packed_hypervector.py
Path: `/Users/rohanvinaik/genomevault/tests/test_packed_hypervector.py`

- /Users/rohanvinaik/genomevault/tests/test_packed_hypervector.py:19:6: E999 IndentationError: expected an indented block after function definition on line 18

### test_refactored_circuits.py
Path: `/Users/rohanvinaik/genomevault/tests/test_refactored_circuits.py`

- /Users/rohanvinaik/genomevault/tests/test_refactored_circuits.py:32:6: E999 IndentationError: expected an indented block after function definition on line 31

### test_robust_it_pir.py
Path: `/Users/rohanvinaik/genomevault/tests/test_robust_it_pir.py`

- /Users/rohanvinaik/genomevault/tests/test_robust_it_pir.py:14:6: E999 IndentationError: expected an indented block after function definition on line 13

### test_simple.py
Path: `/Users/rohanvinaik/genomevault/tests/test_simple.py`

- /Users/rohanvinaik/genomevault/tests/test_simple.py:1:1: F401 'typing.Any' imported but unused
- /Users/rohanvinaik/genomevault/tests/test_simple.py:1:1: F401 'typing.Dict' imported but unused

### test_smoke.py
Path: `/Users/rohanvinaik/genomevault/tests/test_smoke.py`

- /Users/rohanvinaik/genomevault/tests/test_smoke.py:1:1: F401 'typing.Any' imported but unused
- /Users/rohanvinaik/genomevault/tests/test_smoke.py:1:1: F401 'typing.Dict' imported but unused
- /Users/rohanvinaik/genomevault/tests/test_smoke.py:15:5: F401 'cryptography' imported but unused
- /Users/rohanvinaik/genomevault/tests/test_smoke.py:16:5: F401 'fastapi' imported but unused
- /Users/rohanvinaik/genomevault/tests/test_smoke.py:17:5: F401 'numpy as np' imported but unused

### test_version.py
Path: `/Users/rohanvinaik/genomevault/tests/test_version.py`

- /Users/rohanvinaik/genomevault/tests/test_version.py:14:6: E999 IndentationError: expected an indented block after function definition on line 13

### test_zk_median_circuit.py
Path: `/Users/rohanvinaik/genomevault/tests/test_zk_median_circuit.py`

- /Users/rohanvinaik/genomevault/tests/test_zk_median_circuit.py:22:6: E999 IndentationError: expected an indented block after function definition on line 21

### test_config.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_config.py`

- /Users/rohanvinaik/genomevault/tests/unit/test_config.py:32:6: E999 IndentationError: expected an indented block after function definition on line 31

### test_diabetes_pilot.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_diabetes_pilot.py`

- /Users/rohanvinaik/genomevault/tests/unit/test_diabetes_pilot.py:21:6: E999 IndentationError: expected an indented block after function definition on line 20

### test_enhanced_pir.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_enhanced_pir.py`

- /Users/rohanvinaik/genomevault/tests/unit/test_enhanced_pir.py:33:6: E999 IndentationError: expected an indented block after function definition on line 32

### test_hdc_hypervector.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_hdc_hypervector.py`

- /Users/rohanvinaik/genomevault/tests/unit/test_hdc_hypervector.py:25:6: E999 IndentationError: expected an indented block after function definition on line 24

### test_hdc_hypervector_encoding.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_hdc_hypervector_encoding.py`

- /Users/rohanvinaik/genomevault/tests/unit/test_hdc_hypervector_encoding.py:45:6: E999 IndentationError: expected an indented block after function definition on line 44

### test_hipaa.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_hipaa.py`

- /Users/rohanvinaik/genomevault/tests/unit/test_hipaa.py:26:6: E999 IndentationError: expected an indented block after function definition on line 25

### test_monitoring.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_monitoring.py`

- /Users/rohanvinaik/genomevault/tests/unit/test_monitoring.py:31:6: E999 IndentationError: expected an indented block after function definition on line 30

### test_multi_omics.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_multi_omics.py`

- /Users/rohanvinaik/genomevault/tests/unit/test_multi_omics.py:49:6: E999 IndentationError: expected an indented block after function definition on line 48

### test_pir_basic.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_pir_basic.py`

- /Users/rohanvinaik/genomevault/tests/unit/test_pir_basic.py:23:6: E999 IndentationError: expected an indented block after function definition on line 22

### test_zk_basic.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_zk_basic.py`

- /Users/rohanvinaik/genomevault/tests/unit/test_zk_basic.py:17:6: E999 IndentationError: expected an indented block after function definition on line 16

### test_zk_property_circuits.py
Path: `/Users/rohanvinaik/genomevault/tests/zk/test_zk_property_circuits.py`

- /Users/rohanvinaik/genomevault/tests/zk/test_zk_property_circuits.py:74:6: E999 IndentationError: expected an indented block after function definition on line 73

## Pylint Issues

### bridge_simple.py
Path: `/Users/rohanvinaik/experiments/bridge_simple.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 87, Column: 13
- **logging-fstring-interpolation**: Use lazy % formatting in logging functions
  Line: 89, Column: 8
- **logging-fstring-interpolation**: Use lazy % formatting in logging functions
  Line: 92, Column: 8
- **wrong-import-order**: standard import "pathlib.Path" should be placed before third party import "flask.Flask"
  Line: 12, Column: 0

### command_reader.py
Path: `/Users/rohanvinaik/experiments/command_reader.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **bare-except**: No exception type(s) specified
  Line: 20, Column: 0
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 12, Column: 13

### debug_launcher.py
Path: `/Users/rohanvinaik/experiments/debug_launcher.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 22, Column: 17
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 53, Column: 22

### diabetes_analysis_demo.py
Path: `/Users/rohanvinaik/experiments/diabetes_analysis_demo.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **undefined-variable**: Undefined variable 'datetime'
  Line: 87, Column: 25

### efficiency_methods.py
Path: `/Users/rohanvinaik/experiments/efficiency_methods.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 13, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 17, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 24, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 27, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 31, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 34, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 37, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 43, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 49, Column: 0
- ... and 26 more issues

### basic_usage_fixed.py
Path: `/Users/rohanvinaik/experiments/fixes/basic_usage_fixed.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Cannot import 'genomevault.local_processing.phenotypes' due to 'unindent does not match any outer indentation level (genomevault.local_processing.phenotypes, line 55)'
  Line: 17, Column: 0
- **syntax-error**: Cannot import 'genomevault.utils.config' due to 'unindent does not match any outer indentation level (genomevault.utils.config, line 169)'
  Line: 18, Column: 0
- **syntax-error**: Cannot import 'genomevault.utils.encryption' due to 'unindent does not match any outer indentation level (genomevault.utils.encryption, line 66)'
  Line: 19, Column: 0
- **f-string-without-interpolation**: Using an f-string that does not have any interpolated variables
  Line: 33, Column: 16
- **f-string-without-interpolation**: Using an f-string that does not have any interpolated variables
  Line: 175, Column: 16
- **unused-variable**: Unused variable 'nonce'
  Line: 182, Column: 20
- **unused-variable**: Unused variable 'tag'
  Line: 182, Column: 27

### bridge_simple_fixed.py
Path: `/Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 66, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 116, Column: 13

### diagnose_genomevault_import.py
Path: `/Users/rohanvinaik/experiments/fixes/diagnose_genomevault_import.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 25, Column: 13
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 50, Column: 13
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 57, Column: 13
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 67, Column: 18
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 85, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 105, Column: 9

### fix_cupy_issue.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_cupy_issue.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 19, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 24, Column: 9
- **f-string-without-interpolation**: Using an f-string that does not have any interpolated variables
  Line: 40, Column: 18
- **f-string-without-interpolation**: Using an f-string that does not have any interpolated variables
  Line: 44, Column: 18
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 62, Column: 9
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 78, Column: 13

### fix_genomevault_comprehensive.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 25, Column: 13
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 30, Column: 13
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 36, Column: 13
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 78, Column: 13
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 112, Column: 17
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 115, Column: 17
- **unused-variable**: Unused variable 'result'
  Line: 62, Column: 12
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 255, Column: 9

### fix_genomevault_logging_comprehensive.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_genomevault_logging_comprehensive.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 20, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 25, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 135, Column: 9
- **bare-except**: No exception type(s) specified
  Line: 161, Column: 8

### fix_mac_gpu_compatibility.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 14, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 19, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 56, Column: 9
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 72, Column: 13
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 86, Column: 18
- **bare-except**: No exception type(s) specified
  Line: 123, Column: 4
- **bare-except**: No exception type(s) specified
  Line: 117, Column: 12

### fix_pir_circular_import.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 22, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 27, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 56, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 63, Column: 13
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 99, Column: 17
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 193, Column: 9

### fix_processors_error.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_processors_error.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 14, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 35, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 39, Column: 9
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 48, Column: 13

### fix_remaining_issues.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 19, Column: 17
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 32, Column: 22
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 63, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 68, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 105, Column: 13
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 163, Column: 13
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 239, Column: 9

### genomevault_import_fixer.py
Path: `/Users/rohanvinaik/experiments/fixes/genomevault_import_fixer.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 47 (fixes.genomevault_import_fixer, line 48)'
  Line: 48, Column: 5

### install_all_dependencies.py
Path: `/Users/rohanvinaik/experiments/fixes/install_all_dependencies.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 76, Column: 23
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 86, Column: 17
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 125, Column: 17
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 201, Column: 9
- **f-string-without-interpolation**: Using an f-string that does not have any interpolated variables
  Line: 206, Column: 10
- **f-string-without-interpolation**: Using an f-string that does not have any interpolated variables
  Line: 207, Column: 10

### install_genomevault_deps.py
Path: `/Users/rohanvinaik/experiments/fixes/install_genomevault_deps.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **f-string-without-interpolation**: Using an f-string that does not have any interpolated variables
  Line: 82, Column: 14
- **unused-variable**: Unused variable 'result'
  Line: 66, Column: 12

### quick_fix_logging.py
Path: `/Users/rohanvinaik/experiments/fixes/quick_fix_logging.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 20, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 25, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 45, Column: 9

### run_comprehensive_linting.py
Path: `/Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **bare-except**: No exception type(s) specified
  Line: 45, Column: 4
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 23, Column: 17
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 33, Column: 25
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 47, Column: 8
- **bare-except**: No exception type(s) specified
  Line: 80, Column: 4
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 64, Column: 25
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 82, Column: 8
- **bare-except**: No exception type(s) specified
  Line: 110, Column: 4
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 88, Column: 17
- ... and 4 more issues

### tail_chasing_fixer.py
Path: `/Users/rohanvinaik/experiments/fixes/tail_chasing_fixer.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 50 (fixes.tail_chasing_fixer, line 51)'
  Line: 51, Column: 5

### test_system.py
Path: `/Users/rohanvinaik/experiments/fixes/test_system.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **missing-timeout**: Missing timeout argument for method 'requests.get' can cause your program to hang indefinitely
  Line: 43, Column: 30
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 113, Column: 21
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 145, Column: 17
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 161, Column: 18
- **bare-except**: No exception type(s) specified
  Line: 210, Column: 4
- **missing-timeout**: Missing timeout argument for method 'requests.delete' can cause your program to hang indefinitely
  Line: 208, Column: 8
- **missing-timeout**: Missing timeout argument for method 'requests.post' can cause your program to hang indefinitely
  Line: 215, Column: 19
- **missing-timeout**: Missing timeout argument for method 'requests.get' can cause your program to hang indefinitely
  Line: 223, Column: 30

### unified_bridge.py
Path: `/Users/rohanvinaik/experiments/fixes/unified_bridge.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **bare-except**: No exception type(s) specified
  Line: 72, Column: 4
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 70, Column: 13
- **unused-variable**: Unused variable 'project'
  Line: 78, Column: 8
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 141, Column: 13
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 157, Column: 21
- **bare-except**: No exception type(s) specified
  Line: 221, Column: 8
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 219, Column: 17

### update_menu_pythonpath.py
Path: `/Users/rohanvinaik/experiments/fixes/update_menu_pythonpath.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 15, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 33, Column: 9
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 37, Column: 9
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 48, Column: 13

### validate_genomevault.py
Path: `/Users/rohanvinaik/experiments/fixes/validate_genomevault.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 25, Column: 13
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 36, Column: 13
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 47, Column: 13
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 60, Column: 17
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 79, Column: 21

### genomevault_hd_experiment.py
Path: `/Users/rohanvinaik/experiments/genomevault_hd_experiment.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 28 (genomevault_hd_experiment, line 29)'
  Line: 29, Column: 5

### genomevault_menu.py
Path: `/Users/rohanvinaik/experiments/genomevault_menu.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 50 (genomevault_menu, line 51)'
  Line: 51, Column: 5

### gpu_benchmark.py
Path: `/Users/rohanvinaik/experiments/gpu_benchmark.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 33 (gpu_benchmark, line 34)'
  Line: 34, Column: 5

### how_to_guide.py
Path: `/Users/rohanvinaik/experiments/how_to_guide.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0

### monitor_server.py
Path: `/Users/rohanvinaik/experiments/monitor_server.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 21 (monitor_server, line 22)'
  Line: 22, Column: 5

### pipe_reader.py
Path: `/Users/rohanvinaik/experiments/pipe_reader.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **too-many-nested-blocks**: Too many nested blocks (6/5)
  Line: 19, Column: 4

### power_control.py
Path: `/Users/rohanvinaik/experiments/power_control.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 61, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 81, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 97, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 107, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 126, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 131, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 139, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 587, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 590, Column: 0
- ... and 12 more issues

### project_launcher.py
Path: `/Users/rohanvinaik/experiments/project_launcher.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 107 (project_launcher, line 108)'
  Line: 108, Column: 5

### restart_server.py
Path: `/Users/rohanvinaik/experiments/restart_server.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 14, Column: 9
- **subprocess-run-check**: 'subprocess.run' used without explicitly defining the value for 'check'.
  Line: 28, Column: 0

### run_experiment.py
Path: `/Users/rohanvinaik/experiments/run_experiment.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 41 (run_experiment, line 42)'
  Line: 42, Column: 5

### server_api.py
Path: `/Users/rohanvinaik/experiments/server_api.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 35, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 40, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 51, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 59, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 61, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 73, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 83, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 90, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 93, Column: 0
- ... and 43 more issues

### terminal_bridge.py
Path: `/Users/rohanvinaik/experiments/terminal_bridge.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 44, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 48, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 72, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 87, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 98, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 106, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 118, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 125, Column: 0
- **trailing-whitespace**: Trailing whitespace
  Line: 131, Column: 0
- ... and 9 more issues

### terminal_bridge_v2.py
Path: `/Users/rohanvinaik/experiments/terminal_bridge_v2.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 89 (terminal_bridge_v2, line 90)'
  Line: 90, Column: 5

### test_direct_pipe.py
Path: `/Users/rohanvinaik/experiments/test_direct_pipe.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 14, Column: 5

### test_experiment.py
Path: `/Users/rohanvinaik/experiments/test_experiment.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0

### test_pipe.py
Path: `/Users/rohanvinaik/experiments/test_pipe.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **unspecified-encoding**: Using open without explicitly specifying an encoding
  Line: 21, Column: 13

### __init__.py
Path: `/Users/rohanvinaik/genomevault/experiments/__init__.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/__init__.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/adversarial/__init__.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0

### test_hdc_adversarial.py
Path: `/Users/rohanvinaik/genomevault/tests/adversarial/test_hdc_adversarial.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 29 (genomevault.tests.adversarial.test_hdc_adversarial, line 30)'
  Line: 30, Column: 5

### test_pir_adversarial.py
Path: `/Users/rohanvinaik/genomevault/tests/adversarial/test_pir_adversarial.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 27 (genomevault.tests.adversarial.test_pir_adversarial, line 28)'
  Line: 28, Column: 5

### test_zk_adversarial.py
Path: `/Users/rohanvinaik/genomevault/tests/adversarial/test_zk_adversarial.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 16 (genomevault.tests.adversarial.test_zk_adversarial, line 17)'
  Line: 17, Column: 5

### conftest.py
Path: `/Users/rohanvinaik/genomevault/tests/conftest.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 186 (genomevault.tests.conftest, line 187)'
  Line: 187, Column: 4

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/e2e/__init__.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0

### test_pir_e2e.py
Path: `/Users/rohanvinaik/genomevault/tests/e2e/test_pir_e2e.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 25 (genomevault.tests.e2e.test_pir_e2e, line 26)'
  Line: 26, Column: 5

### test_zk_e2e.py
Path: `/Users/rohanvinaik/genomevault/tests/e2e/test_zk_e2e.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 20 (genomevault.tests.e2e.test_zk_e2e, line 21)'
  Line: 21, Column: 5

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/integration/__init__.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0

### test_hipaa_governance.py
Path: `/Users/rohanvinaik/genomevault/tests/integration/test_hipaa_governance.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 32 (genomevault.tests.integration.test_hipaa_governance, line 33)'
  Line: 33, Column: 5

### test_proof_of_training.py
Path: `/Users/rohanvinaik/genomevault/tests/integration/test_proof_of_training.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 26 (genomevault.tests.integration.test_proof_of_training, line 27)'
  Line: 27, Column: 5

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/pir/__init__.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0

### test_pir_protocol.py
Path: `/Users/rohanvinaik/genomevault/tests/pir/test_pir_protocol.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 22 (genomevault.tests.pir.test_pir_protocol, line 23)'
  Line: 23, Column: 5

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/property/__init__.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0

### test_hdc_properties.py
Path: `/Users/rohanvinaik/genomevault/tests/property/test_hdc_properties.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 81 (genomevault.tests.property.test_hdc_properties, line 82)'
  Line: 82, Column: 5

### test_zk_properties.py
Path: `/Users/rohanvinaik/genomevault/tests/property/test_zk_properties.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 24 (genomevault.tests.property.test_zk_properties, line 25)'
  Line: 25, Column: 5

### test_advanced_implementations.py
Path: `/Users/rohanvinaik/genomevault/tests/test_advanced_implementations.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **pointless-string-statement**: String statement has no effect
  Line: 12, Column: 0
- **wrong-import-position**: Import "import os" should be placed at the top of the module
  Line: 14, Column: 0
- **wrong-import-position**: Import "import sys" should be placed at the top of the module
  Line: 15, Column: 0
- **f-string-without-interpolation**: Using an f-string that does not have any interpolated variables
  Line: 169, Column: 10
- **f-string-without-interpolation**: Using an f-string that does not have any interpolated variables
  Line: 249, Column: 10
- **f-string-without-interpolation**: Using an f-string that does not have any interpolated variables
  Line: 259, Column: 10
- **wrong-import-order**: third party import "numpy" should be placed before first party imports "genomevault.zk_proofs.prover.Prover", "genomevault.zk_proofs.advanced.stark_prover.PostQuantumVerifier", "genomevault.zk_proofs.advanced.recursive_snark.RecursiveSNARKProver", "genomevault.zk_proofs.advanced.catalytic_proof.CatalyticProofEngine", "genomevault.pir.advanced.it_pir.InformationTheoreticPIR", "genomevault.hypervector_transform.advanced_compression.AdvancedHierarchicalCompressor" 
  Line: 7, Column: 0
- **wrong-import-order**: standard import "time" should be placed before third party import "numpy" and first party imports "genomevault.zk_proofs.prover.Prover", "genomevault.zk_proofs.advanced.stark_prover.PostQuantumVerifier", "genomevault.zk_proofs.advanced.recursive_snark.RecursiveSNARKProver", "genomevault.zk_proofs.advanced.catalytic_proof.CatalyticProofEngine", "genomevault.pir.advanced.it_pir.InformationTheoreticPIR", "genomevault.hypervector_transform.advanced_compression.AdvancedHierarchicalCompressor" 
  Line: 8, Column: 0
- **wrong-import-order**: standard import "hashlib" should be placed before third party import "numpy" and first party imports "genomevault.zk_proofs.prover.Prover", "genomevault.zk_proofs.advanced.stark_prover.PostQuantumVerifier", "genomevault.zk_proofs.advanced.recursive_snark.RecursiveSNARKProver", "genomevault.zk_proofs.advanced.catalytic_proof.CatalyticProofEngine", "genomevault.pir.advanced.it_pir.InformationTheoreticPIR", "genomevault.hypervector_transform.advanced_compression.AdvancedHierarchicalCompressor" 
  Line: 9, Column: 0
- ... and 3 more issues

### test_basic.py
Path: `/Users/rohanvinaik/genomevault/tests/test_basic.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **pointless-string-statement**: String statement has no effect
  Line: 3, Column: 0
- **wrong-import-position**: Import "import sys" should be placed at the top of the module
  Line: 5, Column: 0
- **wrong-import-position**: Import "from pathlib import Path" should be placed at the top of the module
  Line: 6, Column: 0
- **wrong-import-position**: Import "import pytest" should be placed at the top of the module
  Line: 8, Column: 0
- **no-member**: Module 'genomevault' has no '__version__' member
  Line: 19, Column: 11

### test_client.py
Path: `/Users/rohanvinaik/genomevault/tests/test_client.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 13 (genomevault.tests.test_client, line 14)'
  Line: 14, Column: 5

### test_compression.py
Path: `/Users/rohanvinaik/genomevault/tests/test_compression.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 21 (genomevault.tests.test_compression, line 22)'
  Line: 22, Column: 5

### test_hdc_error_handling.py
Path: `/Users/rohanvinaik/genomevault/tests/test_hdc_error_handling.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 21 (genomevault.tests.test_hdc_error_handling, line 22)'
  Line: 22, Column: 5

### test_hdc_implementation.py
Path: `/Users/rohanvinaik/genomevault/tests/test_hdc_implementation.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 46 (genomevault.tests.test_hdc_implementation, line 47)'
  Line: 47, Column: 5

### test_hdc_pir_integration.py
Path: `/Users/rohanvinaik/genomevault/tests/test_hdc_pir_integration.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 30 (genomevault.tests.test_hdc_pir_integration, line 31)'
  Line: 31, Column: 5

### test_hypervector.py
Path: `/Users/rohanvinaik/genomevault/tests/test_hypervector.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 17 (genomevault.tests.test_hypervector, line 18)'
  Line: 18, Column: 5

### test_infrastructure.py
Path: `/Users/rohanvinaik/genomevault/tests/test_infrastructure.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 66 (genomevault.tests.test_infrastructure, line 67)'
  Line: 67, Column: 5

### test_it_pir.py
Path: `/Users/rohanvinaik/genomevault/tests/test_it_pir.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 13 (genomevault.tests.test_it_pir, line 14)'
  Line: 14, Column: 5

### test_it_pir_protocol.py
Path: `/Users/rohanvinaik/genomevault/tests/test_it_pir_protocol.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 13 (genomevault.tests.test_it_pir_protocol, line 14)'
  Line: 14, Column: 5

### test_packed_hypervector.py
Path: `/Users/rohanvinaik/genomevault/tests/test_packed_hypervector.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 18 (genomevault.tests.test_packed_hypervector, line 19)'
  Line: 19, Column: 5

### test_refactored_circuits.py
Path: `/Users/rohanvinaik/genomevault/tests/test_refactored_circuits.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 31 (genomevault.tests.test_refactored_circuits, line 32)'
  Line: 32, Column: 5

### test_robust_it_pir.py
Path: `/Users/rohanvinaik/genomevault/tests/test_robust_it_pir.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 13 (genomevault.tests.test_robust_it_pir, line 14)'
  Line: 14, Column: 5

### test_simple.py
Path: `/Users/rohanvinaik/genomevault/tests/test_simple.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **pointless-string-statement**: String statement has no effect
  Line: 3, Column: 0

### test_smoke.py
Path: `/Users/rohanvinaik/genomevault/tests/test_smoke.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **pointless-string-statement**: String statement has no effect
  Line: 3, Column: 0
- **wrong-import-position**: Import "import pytest" should be placed at the top of the module
  Line: 5, Column: 0
- **redefined-builtin**: Redefining built-in 'input'
  Line: 43, Column: 27

### test_version.py
Path: `/Users/rohanvinaik/genomevault/tests/test_version.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 13 (genomevault.tests.test_version, line 14)'
  Line: 14, Column: 5

### test_zk_median_circuit.py
Path: `/Users/rohanvinaik/genomevault/tests/test_zk_median_circuit.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 21 (genomevault.tests.test_zk_median_circuit, line 22)'
  Line: 22, Column: 5

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/__init__.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0

### test_config.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_config.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 31 (genomevault.tests.unit.test_config, line 32)'
  Line: 32, Column: 5

### test_diabetes_pilot.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_diabetes_pilot.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 20 (genomevault.tests.unit.test_diabetes_pilot, line 21)'
  Line: 21, Column: 5

### test_enhanced_pir.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_enhanced_pir.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 32 (genomevault.tests.unit.test_enhanced_pir, line 33)'
  Line: 33, Column: 5

### test_hdc_hypervector.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_hdc_hypervector.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 24 (genomevault.tests.unit.test_hdc_hypervector, line 25)'
  Line: 25, Column: 5

### test_hdc_hypervector_encoding.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_hdc_hypervector_encoding.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 44 (genomevault.tests.unit.test_hdc_hypervector_encoding, line 45)'
  Line: 45, Column: 5

### test_hipaa.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_hipaa.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 25 (genomevault.tests.unit.test_hipaa, line 26)'
  Line: 26, Column: 5

### test_monitoring.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_monitoring.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 30 (genomevault.tests.unit.test_monitoring, line 31)'
  Line: 31, Column: 5

### test_multi_omics.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_multi_omics.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 48 (genomevault.tests.unit.test_multi_omics, line 49)'
  Line: 49, Column: 5

### test_pir_basic.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_pir_basic.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 22 (genomevault.tests.unit.test_pir_basic, line 23)'
  Line: 23, Column: 5

### test_zk_basic.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_zk_basic.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 16 (genomevault.tests.unit.test_zk_basic, line 17)'
  Line: 17, Column: 5

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/zk/__init__.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0

### test_zk_property_circuits.py
Path: `/Users/rohanvinaik/genomevault/tests/zk/test_zk_property_circuits.py`

- **unrecognized-option**: Unrecognized option found: async-contextmanager-decorators, no-space-check
  Line: 1, Column: 0
- **syntax-error**: Parsing failed: 'expected an indented block after function definition on line 73 (genomevault.tests.zk.test_zk_property_circuits, line 74)'
  Line: 74, Column: 5

## Mypy Issues

### bridge_simple.py
Path: `/Users/rohanvinaik/experiments/bridge_simple.py`

- /Users/rohanvinaik/experiments/bridge_simple.py:32: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/bridge_simple.py:59: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/bridge_simple.py:62: error: Item "None" of "Optional[Any]" has no attribute "get"  [union-attr]
- /Users/rohanvinaik/experiments/bridge_simple.py:63: error: Item "None" of "Optional[Any]" has no attribute "get"  [union-attr]
- /Users/rohanvinaik/experiments/bridge_simple.py:86: error: Function is missing a return type annotation  [no-untyped-def]
- Found 5 errors in 1 file (checked 1 source file)

### command_reader.py
Path: `/Users/rohanvinaik/experiments/command_reader.py`

- Success: no issues found in 1 source file

### debug_launcher.py
Path: `/Users/rohanvinaik/experiments/debug_launcher.py`

- /Users/rohanvinaik/experiments/debug_launcher.py:8: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/debug_launcher.py:8: note: Use "-> None" if function does not return a value
- Found 1 error in 1 file (checked 1 source file)

### diabetes_analysis_demo.py
Path: `/Users/rohanvinaik/experiments/diabetes_analysis_demo.py`

- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:12: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:75: error: Argument "key" to "max" has incompatible type overloaded function; expected "Callable[[str], Union[SupportsDunderLT[Any], SupportsDunderGT[Any]]]"  [arg-type]
- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:86: error: Name "datetime" is not defined  [name-defined]
- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:104: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:116: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/diabetes_analysis_demo.py:116: note: Use "-> None" if function does not return a value
- Found 5 errors in 1 file (checked 1 source file)

### efficiency_methods.py
Path: `/Users/rohanvinaik/experiments/efficiency_methods.py`

- /Users/rohanvinaik/experiments/efficiency_methods.py:17: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/efficiency_methods.py:17: note: Use "-> None" if function does not return a value
- /Users/rohanvinaik/experiments/efficiency_methods.py:21: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/efficiency_methods.py:28: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/efficiency_methods.py:31: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/efficiency_methods.py:71: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/efficiency_methods.py:74: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/efficiency_methods.py:115: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/efficiency_methods.py:118: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/efficiency_methods.py:152: error: Function is missing a type annotation  [no-untyped-def]
- ... and 1 more issues

### basic_usage_fixed.py
Path: `/Users/rohanvinaik/experiments/fixes/basic_usage_fixed.py`

- genomevault/local_processing/phenotypes.py:55: error: Unindent does not match any outer indentation level  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### bridge_simple_fixed.py
Path: `/Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py`

- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:56: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:63: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:72: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:85: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:91: error: Item "None" of "Optional[Any]" has no attribute "get"  [union-attr]
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:92: error: Item "None" of "Optional[Any]" has no attribute "get"  [union-attr]
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:98: error: Value of type "Collection[str]" is not indexable  [index]
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:113: error: Argument 2 to "rename" has incompatible type "Collection[str]"; expected "Union[str, bytes, PathLike[str], PathLike[bytes]]"  [arg-type]
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:132: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/bridge_simple_fixed.py:143: error: Argument 1 to "exists" has incompatible type "Collection[str]"; expected "Union[int, Union[str, bytes, PathLike[str], PathLike[bytes]]]"  [arg-type]
- ... and 4 more issues

### diagnose_genomevault_import.py
Path: `/Users/rohanvinaik/experiments/fixes/diagnose_genomevault_import.py`

- /Users/rohanvinaik/experiments/fixes/diagnose_genomevault_import.py:11: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/diagnose_genomevault_import.py:11: note: Use "-> None" if function does not return a value
- Found 1 error in 1 file (checked 1 source file)

### fix_cupy_issue.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_cupy_issue.py`

- /Users/rohanvinaik/experiments/fixes/fix_cupy_issue.py:10: error: Function is missing a return type annotation  [no-untyped-def]
- Found 1 error in 1 file (checked 1 source file)

### fix_genomevault_comprehensive.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py`

- /Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py:12: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py:169: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/fix_genomevault_comprehensive.py:169: note: Use "-> None" if function does not return a value
- Found 2 errors in 1 file (checked 1 source file)

### fix_genomevault_logging_comprehensive.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_genomevault_logging_comprehensive.py`

- /Users/rohanvinaik/experiments/fixes/fix_genomevault_logging_comprehensive.py:10: error: Function is missing a return type annotation  [no-untyped-def]
- Found 1 error in 1 file (checked 1 source file)

### fix_mac_gpu_compatibility.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py`

- /Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py:9: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py:102: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/fix_mac_gpu_compatibility.py:102: note: Use "-> None" if function does not return a value
- Found 2 errors in 1 file (checked 1 source file)

### fix_pir_circular_import.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py`

- /Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py:9: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py:75: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py:75: note: Use "-> None" if function does not return a value
- /Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py:111: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/fix_pir_circular_import.py:111: note: Use "-> None" if function does not return a value
- Found 3 errors in 1 file (checked 1 source file)

### fix_processors_error.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_processors_error.py`

- /Users/rohanvinaik/experiments/fixes/fix_processors_error.py:9: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/fix_processors_error.py:9: note: Use "-> None" if function does not return a value
- Found 1 error in 1 file (checked 1 source file)

### fix_remaining_issues.py
Path: `/Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py`

- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:13: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:51: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:128: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:128: note: Use "-> None" if function does not return a value
- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:171: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/fix_remaining_issues.py:171: note: Use "-> None" if function does not return a value
- Found 4 errors in 1 file (checked 1 source file)

### genomevault_import_fixer.py
Path: `/Users/rohanvinaik/experiments/fixes/genomevault_import_fixer.py`

- /Users/rohanvinaik/experiments/fixes/genomevault_import_fixer.py:48: error: Expected an indented block after function definition on line 47  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### install_all_dependencies.py
Path: `/Users/rohanvinaik/experiments/fixes/install_all_dependencies.py`

- /Users/rohanvinaik/experiments/fixes/install_all_dependencies.py:12: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/install_all_dependencies.py:125: error: Incompatible types in assignment (expression has type "CompletedProcess[bytes]", variable has type "CompletedProcess[str]")  [assignment]
- /Users/rohanvinaik/experiments/fixes/install_all_dependencies.py:135: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/install_all_dependencies.py:135: note: Use "-> None" if function does not return a value
- Found 3 errors in 1 file (checked 1 source file)

### install_genomevault_deps.py
Path: `/Users/rohanvinaik/experiments/fixes/install_genomevault_deps.py`

- /Users/rohanvinaik/experiments/fixes/install_genomevault_deps.py:11: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/install_genomevault_deps.py:11: note: Use "-> None" if function does not return a value
- Found 1 error in 1 file (checked 1 source file)

### quick_fix_logging.py
Path: `/Users/rohanvinaik/experiments/fixes/quick_fix_logging.py`

- /Users/rohanvinaik/experiments/fixes/quick_fix_logging.py:10: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/quick_fix_logging.py:58: error: Function is missing a return type annotation  [no-untyped-def]
- Found 2 errors in 1 file (checked 1 source file)

### run_comprehensive_linting.py
Path: `/Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py`

- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:11: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:106: error: Incompatible types in assignment (expression has type "int", target has type "str")  [assignment]
- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:146: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:179: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:208: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/run_comprehensive_linting.py:208: note: Use "-> None" if function does not return a value
- Found 5 errors in 1 file (checked 1 source file)

### tail_chasing_fixer.py
Path: `/Users/rohanvinaik/experiments/fixes/tail_chasing_fixer.py`

- /Users/rohanvinaik/experiments/fixes/tail_chasing_fixer.py:51: error: Expected an indented block after function definition on line 50  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_system.py
Path: `/Users/rohanvinaik/experiments/fixes/test_system.py`

- /Users/rohanvinaik/experiments/fixes/test_system.py:23: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/test_system.py:34: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/test_system.py:63: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/test_system.py:86: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/test_system.py:130: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/test_system.py:180: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/test_system.py:202: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/test_system.py:240: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/test_system.py:240: note: Use "-> None" if function does not return a value
- Found 8 errors in 1 file (checked 1 source file)

### unified_bridge.py
Path: `/Users/rohanvinaik/experiments/fixes/unified_bridge.py`

- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:64: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:76: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:76: note: Use "-> None" if function does not return a value
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:81: error: Argument 1 to "exists" has incompatible type "Collection[str]"; expected "Union[int, Union[str, bytes, PathLike[str], PathLike[bytes]]]"  [arg-type]
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:83: error: Argument 1 to "mkfifo" has incompatible type "Collection[str]"; expected "Union[str, bytes, PathLike[str], PathLike[bytes]]"  [arg-type]
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:84: error: Argument 1 to "chmod" has incompatible type "Collection[str]"; expected "Union[int, Union[str, bytes, PathLike[str], PathLike[bytes]]]"  [arg-type]
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:92: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:100: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:115: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/unified_bridge.py:121: error: Item "None" of "Optional[Any]" has no attribute "get"  [union-attr]
- ... and 34 more issues

### update_menu_pythonpath.py
Path: `/Users/rohanvinaik/experiments/fixes/update_menu_pythonpath.py`

- /Users/rohanvinaik/experiments/fixes/update_menu_pythonpath.py:9: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/update_menu_pythonpath.py:9: note: Use "-> None" if function does not return a value
- Found 1 error in 1 file (checked 1 source file)

### validate_genomevault.py
Path: `/Users/rohanvinaik/experiments/fixes/validate_genomevault.py`

- /Users/rohanvinaik/experiments/fixes/validate_genomevault.py:11: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/fixes/validate_genomevault.py:60: error: Module has no attribute "os"  [attr-defined]
- Found 2 errors in 1 file (checked 1 source file)

### genomevault_hd_experiment.py
Path: `/Users/rohanvinaik/experiments/genomevault_hd_experiment.py`

- /Users/rohanvinaik/experiments/genomevault_hd_experiment.py:29: error: Expected an indented block after function definition on line 28  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### genomevault_menu.py
Path: `/Users/rohanvinaik/experiments/genomevault_menu.py`

- /Users/rohanvinaik/experiments/genomevault_menu.py:51: error: Expected an indented block after function definition on line 50  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### gpu_benchmark.py
Path: `/Users/rohanvinaik/experiments/gpu_benchmark.py`

- /Users/rohanvinaik/experiments/gpu_benchmark.py:34: error: Expected an indented block after function definition on line 33  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### how_to_guide.py
Path: `/Users/rohanvinaik/experiments/how_to_guide.py`

- Success: no issues found in 1 source file

### monitor_server.py
Path: `/Users/rohanvinaik/experiments/monitor_server.py`

- /Users/rohanvinaik/experiments/monitor_server.py:22: error: Expected an indented block after function definition on line 21  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### pipe_reader.py
Path: `/Users/rohanvinaik/experiments/pipe_reader.py`

- /Users/rohanvinaik/experiments/pipe_reader.py:16: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/pipe_reader.py:16: note: Use "-> None" if function does not return a value
- Found 1 error in 1 file (checked 1 source file)

### power_control.py
Path: `/Users/rohanvinaik/experiments/power_control.py`

- /Users/rohanvinaik/experiments/power_control.py:28: error: Need type annotation for "power_activity_log" (hint: "power_activity_log: list[<type>] = ...")  [var-annotated]
- /Users/rohanvinaik/experiments/power_control.py:31: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/power_control.py:42: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/power_control.py:47: error: Name "start_time" is not defined  [name-defined]
- /Users/rohanvinaik/experiments/power_control.py:54: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/power_control.py:60: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/power_control.py:70: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/power_control.py:70: note: Use "-> None" if function does not return a value
- /Users/rohanvinaik/experiments/power_control.py:83: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/power_control.py:109: error: Function is missing a return type annotation  [no-untyped-def]
- ... and 10 more issues

### project_launcher.py
Path: `/Users/rohanvinaik/experiments/project_launcher.py`

- /Users/rohanvinaik/experiments/project_launcher.py:108: error: Expected an indented block after function definition on line 107  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### restart_server.py
Path: `/Users/rohanvinaik/experiments/restart_server.py`

- Success: no issues found in 1 source file

### run_experiment.py
Path: `/Users/rohanvinaik/experiments/run_experiment.py`

- /Users/rohanvinaik/experiments/run_experiment.py:42: error: Expected an indented block after function definition on line 41  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### server_api.py
Path: `/Users/rohanvinaik/experiments/server_api.py`

- /Users/rohanvinaik/experiments/server_api.py:21: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/server_api.py:29: error: Need type annotation for "experiment_queue"  [var-annotated]
- /Users/rohanvinaik/experiments/server_api.py:30: error: Need type annotation for "results_cache" (hint: "results_cache: dict[<type>, <type>] = ...")  [var-annotated]
- /Users/rohanvinaik/experiments/server_api.py:34: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/server_api.py:51: error: Function is missing a return type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/server_api.py:54: error: Value of type "Optional[Any]" is not indexable  [index]
- /Users/rohanvinaik/experiments/server_api.py:58: error: Value of type "Optional[Any]" is not indexable  [index]
- /Users/rohanvinaik/experiments/server_api.py:59: error: Item "None" of "Optional[Any]" has no attribute "get"  [union-attr]
- /Users/rohanvinaik/experiments/server_api.py:76: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/server_api.py:91: error: Function is missing a return type annotation  [no-untyped-def]
- ... and 14 more issues

### terminal_bridge.py
Path: `/Users/rohanvinaik/experiments/terminal_bridge.py`

- /Users/rohanvinaik/experiments/terminal_bridge.py:18: error: Need type annotation for "command_queues"  [var-annotated]
- /Users/rohanvinaik/experiments/terminal_bridge.py:27: error: Need type annotation for "terminal_status" (hint: "terminal_status: dict[<type>, <type>] = ...")  [var-annotated]
- /Users/rohanvinaik/experiments/terminal_bridge.py:32: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/terminal_bridge.py:40: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/terminal_bridge.py:46: error: Item "None" of "Optional[Any]" has no attribute "get"  [union-attr]
- /Users/rohanvinaik/experiments/terminal_bridge.py:47: error: Item "None" of "Optional[Any]" has no attribute "get"  [union-attr]
- /Users/rohanvinaik/experiments/terminal_bridge.py:93: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/terminal_bridge.py:108: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/terminal_bridge.py:123: error: Function is missing a type annotation  [no-untyped-def]
- /Users/rohanvinaik/experiments/terminal_bridge.py:140: error: Function is missing a type annotation  [no-untyped-def]
- ... and 1 more issues

### terminal_bridge_v2.py
Path: `/Users/rohanvinaik/experiments/terminal_bridge_v2.py`

- /Users/rohanvinaik/experiments/terminal_bridge_v2.py:94: error: Expected an indented block after function definition on line 93  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_direct_pipe.py
Path: `/Users/rohanvinaik/experiments/test_direct_pipe.py`

- Success: no issues found in 1 source file

### test_experiment.py
Path: `/Users/rohanvinaik/experiments/test_experiment.py`

- Success: no issues found in 1 source file

### test_pipe.py
Path: `/Users/rohanvinaik/experiments/test_pipe.py`

- /Users/rohanvinaik/experiments/test_pipe.py:9: error: Function is missing a type annotation  [no-untyped-def]
- Found 1 error in 1 file (checked 1 source file)

### __init__.py
Path: `/Users/rohanvinaik/genomevault/experiments/__init__.py`

- Success: no issues found in 1 source file

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/__init__.py`

- Success: no issues found in 1 source file

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/adversarial/__init__.py`

- Success: no issues found in 1 source file

### test_hdc_adversarial.py
Path: `/Users/rohanvinaik/genomevault/tests/adversarial/test_hdc_adversarial.py`

- tests/adversarial/test_hdc_adversarial.py:29: error: Expected an indented block after function definition on line 28  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_pir_adversarial.py
Path: `/Users/rohanvinaik/genomevault/tests/adversarial/test_pir_adversarial.py`

- tests/adversarial/test_pir_adversarial.py:28: error: Expected an indented block after function definition on line 27  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_zk_adversarial.py
Path: `/Users/rohanvinaik/genomevault/tests/adversarial/test_zk_adversarial.py`

- tests/adversarial/test_zk_adversarial.py:18: error: Expected an indented block after function definition on line 17  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### conftest.py
Path: `/Users/rohanvinaik/genomevault/tests/conftest.py`

- tests/conftest.py:188: error: Expected an indented block after function definition on line 187  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/e2e/__init__.py`

- Success: no issues found in 1 source file

### test_pir_e2e.py
Path: `/Users/rohanvinaik/genomevault/tests/e2e/test_pir_e2e.py`

- tests/e2e/test_pir_e2e.py:25: error: Expected an indented block after function definition on line 24  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_zk_e2e.py
Path: `/Users/rohanvinaik/genomevault/tests/e2e/test_zk_e2e.py`

- tests/e2e/test_zk_e2e.py:21: error: Expected an indented block after function definition on line 20  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/integration/__init__.py`

- Success: no issues found in 1 source file

### test_hipaa_governance.py
Path: `/Users/rohanvinaik/genomevault/tests/integration/test_hipaa_governance.py`

- tests/integration/test_hipaa_governance.py:34: error: Expected an indented block after function definition on line 33  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_proof_of_training.py
Path: `/Users/rohanvinaik/genomevault/tests/integration/test_proof_of_training.py`

- tests/integration/test_proof_of_training.py:28: error: Expected an indented block after function definition on line 27  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/pir/__init__.py`

- Success: no issues found in 1 source file

### test_pir_protocol.py
Path: `/Users/rohanvinaik/genomevault/tests/pir/test_pir_protocol.py`

- tests/pir/test_pir_protocol.py:22: error: Expected an indented block after function definition on line 21  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/property/__init__.py`

- Success: no issues found in 1 source file

### test_hdc_properties.py
Path: `/Users/rohanvinaik/genomevault/tests/property/test_hdc_properties.py`

- tests/property/test_hdc_properties.py:84: error: Expected an indented block after function definition on line 83  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_zk_properties.py
Path: `/Users/rohanvinaik/genomevault/tests/property/test_zk_properties.py`

- tests/property/test_zk_properties.py:26: error: Expected an indented block after function definition on line 25  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_advanced_implementations.py
Path: `/Users/rohanvinaik/genomevault/tests/test_advanced_implementations.py`

- genomevault/pir/advanced/it_pir.py:33: error: Unindent does not match any outer indentation level  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_basic.py
Path: `/Users/rohanvinaik/genomevault/tests/test_basic.py`

- Success: no issues found in 1 source file

### test_client.py
Path: `/Users/rohanvinaik/genomevault/tests/test_client.py`

- tests/test_client.py:16: error: Expected an indented block after function definition on line 15  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_compression.py
Path: `/Users/rohanvinaik/genomevault/tests/test_compression.py`

- tests/test_compression.py:23: error: Expected an indented block after function definition on line 22  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_hdc_error_handling.py
Path: `/Users/rohanvinaik/genomevault/tests/test_hdc_error_handling.py`

- tests/test_hdc_error_handling.py:23: error: Expected an indented block after function definition on line 22  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_hdc_implementation.py
Path: `/Users/rohanvinaik/genomevault/tests/test_hdc_implementation.py`

- tests/test_hdc_implementation.py:48: error: Expected an indented block after function definition on line 47  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_hdc_pir_integration.py
Path: `/Users/rohanvinaik/genomevault/tests/test_hdc_pir_integration.py`

- tests/test_hdc_pir_integration.py:32: error: Expected an indented block after function definition on line 31  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_hypervector.py
Path: `/Users/rohanvinaik/genomevault/tests/test_hypervector.py`

- tests/test_hypervector.py:19: error: Expected an indented block after function definition on line 18  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_infrastructure.py
Path: `/Users/rohanvinaik/genomevault/tests/test_infrastructure.py`

- tests/test_infrastructure.py:68: error: Expected an indented block after function definition on line 67  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_it_pir.py
Path: `/Users/rohanvinaik/genomevault/tests/test_it_pir.py`

- tests/test_it_pir.py:16: error: Expected an indented block after function definition on line 15  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_it_pir_protocol.py
Path: `/Users/rohanvinaik/genomevault/tests/test_it_pir_protocol.py`

- tests/test_it_pir_protocol.py:16: error: Expected an indented block after function definition on line 15  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_packed_hypervector.py
Path: `/Users/rohanvinaik/genomevault/tests/test_packed_hypervector.py`

- tests/test_packed_hypervector.py:20: error: Expected an indented block after function definition on line 19  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_refactored_circuits.py
Path: `/Users/rohanvinaik/genomevault/tests/test_refactored_circuits.py`

- tests/test_refactored_circuits.py:33: error: Expected an indented block after function definition on line 32  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_robust_it_pir.py
Path: `/Users/rohanvinaik/genomevault/tests/test_robust_it_pir.py`

- tests/test_robust_it_pir.py:16: error: Expected an indented block after function definition on line 15  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_simple.py
Path: `/Users/rohanvinaik/genomevault/tests/test_simple.py`

- Success: no issues found in 1 source file

### test_smoke.py
Path: `/Users/rohanvinaik/genomevault/tests/test_smoke.py`

- tests/test_smoke.py:38: error: Function is missing a type annotation for one or more arguments  [no-untyped-def]
- Found 1 error in 1 file (checked 1 source file)

### test_version.py
Path: `/Users/rohanvinaik/genomevault/tests/test_version.py`

- tests/test_version.py:16: error: Expected an indented block after function definition on line 15  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_zk_median_circuit.py
Path: `/Users/rohanvinaik/genomevault/tests/test_zk_median_circuit.py`

- tests/test_zk_median_circuit.py:21: error: Expected an indented block after function definition on line 20  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/__init__.py`

- Success: no issues found in 1 source file

### test_config.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_config.py`

- tests/unit/test_config.py:33: error: Expected an indented block after function definition on line 32  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_diabetes_pilot.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_diabetes_pilot.py`

- tests/unit/test_diabetes_pilot.py:22: error: Expected an indented block after function definition on line 21  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_enhanced_pir.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_enhanced_pir.py`

- tests/unit/test_enhanced_pir.py:34: error: Expected an indented block after function definition on line 33  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_hdc_hypervector.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_hdc_hypervector.py`

- tests/unit/test_hdc_hypervector.py:26: error: Expected an indented block after function definition on line 25  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_hdc_hypervector_encoding.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_hdc_hypervector_encoding.py`

- tests/unit/test_hdc_hypervector_encoding.py:44: error: Expected an indented block after function definition on line 43  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_hipaa.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_hipaa.py`

- tests/unit/test_hipaa.py:27: error: Expected an indented block after function definition on line 26  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_monitoring.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_monitoring.py`

- tests/unit/test_monitoring.py:31: error: Expected an indented block after function definition on line 30  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_multi_omics.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_multi_omics.py`

- tests/unit/test_multi_omics.py:50: error: Expected an indented block after function definition on line 49  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_pir_basic.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_pir_basic.py`

- tests/unit/test_pir_basic.py:24: error: Expected an indented block after function definition on line 23  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### test_zk_basic.py
Path: `/Users/rohanvinaik/genomevault/tests/unit/test_zk_basic.py`

- tests/unit/test_zk_basic.py:18: error: Expected an indented block after function definition on line 17  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

### __init__.py
Path: `/Users/rohanvinaik/genomevault/tests/zk/__init__.py`

- Success: no issues found in 1 source file

### test_zk_property_circuits.py
Path: `/Users/rohanvinaik/genomevault/tests/zk/test_zk_property_circuits.py`

- tests/zk/test_zk_property_circuits.py:74: error: Expected an indented block after function definition on line 73  [syntax]
- Found 1 error in 1 file (errors prevented further checking)

## Priority Fixes

Based on the analysis, here are the priority fixes:

### Most Common Issues

- **W293**: 531 occurrences
- **E302**: 94 occurrences
- **expected**: 45 occurrences
- **F401**: 43 occurrences
- **E305**: 26 occurrences
- **E722**: 14 occurrences
- **F841**: 12 occurrences
- **F541**: 11 occurrences
- **W291**: 10 occurrences
- **E226**: 10 occurrences
