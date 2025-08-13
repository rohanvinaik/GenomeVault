# Code Analysis Report - genomevault
*Generated: 2025-08-13T13:23:36.023601*

## Executive Summary
- **Total Issues Found**: 538
- **Fixable Issues**: 537 (99.8%)
- **Risk Score**: 21.62
- **Affected Modules**: 160
- **Analysis Time**: 0.00s

## Issues by Category

### Duplicate & Semantic Duplicates (151 issues)
**genomevault/api/models/updates.py**
- Line 15: Exact duplicate: dict_for_update identical to dict_for_update in vectors.py
- Line 185: Semantic duplicate: merge_with_existing similar to merge_with_existing in config.py (similarity: 0.89)

**genomevault/api/models/vectors.py**
- Line 56: Exact duplicate: dict_for_update identical to dict_for_update in vectors.py

**genomevault/experimental/pir_advanced/it_pir.py**
- Line 36: Exact duplicate: get_server_query identical to get_server_query in it_pir.py
- Line 53: Exact duplicate: is_complete identical to is_complete in it_pir.py
- Line 92: Exact duplicate: generate_query identical to generate_query in it_pir.py
- Line 161: Exact duplicate: process_server_query identical to process_server_query in it_pir.py
- Line 194: Exact duplicate: reconstruct_response identical to reconstruct_response in it_pir.py
- ... and 2 more

**genomevault/experimental/pir_advanced/robust_it_pir.py**
- Line 13: Exact duplicate: get identical to get in robust_it_pir.py
- Line 53: Exact duplicate: get identical to get in robust_it_pir.py

**genomevault/experimental/zk_circuits/catalytic_proof.py**
- Line 115: Exact duplicate: get_usage_stats identical to get_usage_stats in catalytic_proof.py
- Line 39: Exact duplicate: space_efficiency identical to space_efficiency in catalytic_proof.py
- Line 73: Exact duplicate: read identical to read in catalytic_proof.py
- Line 81: Exact duplicate: write identical to write in catalytic_proof.py
- Line 89: Exact duplicate: reset identical to reset in catalytic_proof.py
- ... and 2 more

**genomevault/experimental/zk_circuits/recursive_snark.py**
- Line 37: Exact duplicate: proof_count identical to proof_count in recursive_snark.py

### LLM-Generated Artifacts (143 issues)
**benchmarks/benchmark_packed_hypervector.py**
- Line 20: Suspiciously uniform pattern: all_same_length

**comprehensive_fix.py**
- Line 440: Suspiciously uniform pattern: reverse_alphabetical
- Line 137: JSON data contains placeholder/filler content

**devtools/comprehensive_cleanup.py**
- Line 686: Suspiciously uniform pattern: reverse_alphabetical
- Line 824: Suspiciously uniform pattern: alphabetical
- Line 110: Suspiciously uniform pattern: reverse_alphabetical
- Line 67: Suspiciously uniform pattern: alphabetical
- Line 69: Suspiciously uniform pattern: all_same_length

**devtools/comprehensive_status.py**
- Line 112: Suspiciously uniform pattern: reverse_alphabetical
- Line 148: Suspiciously uniform pattern: reverse_alphabetical
- Line 95: Suspiciously uniform pattern: reverse_alphabetical

**devtools/enhanced_cleanup.py**
- Line 854: Suspiciously uniform pattern: alphabetical
- Line 884: Suspiciously uniform pattern: alphabetical
- Line 1186: Suspiciously uniform pattern: alphabetical
- Line 1426: Suspiciously uniform pattern: alphabetical
- Line 874: Suspiciously uniform pattern: reverse_alphabetical
- ... and 4 more

### Missing & Undefined Symbols (127 issues)
**devtools/cleanup_root_directory.py**
- Line 16: Reference to undefined symbol '__file__'

**devtools/debug_genomevault.py**
- Line 18: Reference to undefined symbol '__file__'

**devtools/diagnostic.py**
- Line 13: Reference to undefined symbol '__file__'

**devtools/setup_dev.py**
- Line 122: Reference to undefined symbol '__file__'
- Line 173: Reference to undefined symbol '__file__'
- Line 234: Reference to undefined symbol '__file__'
- Line 263: Reference to undefined symbol '__file__'
- Line 296: Reference to undefined symbol '__file__'

**examples/minimal_verification.py**
- Line 1: Reference to undefined symbol '__file__'

**examples/simple_test.py**
- Line 15: Reference to undefined symbol '__file__'

**genomevault/api/routers/metrics.py**
- Line 1: Reference to undefined symbol 'Info'

**genomevault/blockchain/hipaa/integration.py**
- Line 208: Reference to undefined symbol '_hipaa_committee'
- Line 233: Reference to undefined symbol '_hipaa_proposals'
- Line 250: Reference to undefined symbol '_verifier'
- Line 250: Reference to undefined symbol '_governance'
- Line 269: Reference to undefined symbol '_credentials'
- ... and 1 more

**genomevault/clinical/eval/harness.py**
- Line 52: Reference to undefined symbol 'calibrator'

**genomevault/config/paths.py**
- Line 3: Reference to undefined symbol '__file__'

**genomevault/core/exceptions.py**
- Line 18: Reference to undefined symbol 'details'

### Placeholder & Stub Functions (104 issues)
**devtools/genomevault_autofix.py**
- Line 56: Function 'parse_args' implies logic but has complexity 1 in parse_args

**devtools/green_toolchain_impl.py**
- Line 27: Function 'validate_current_state' implies logic but has complexity 1 in GreenToolchainImplementer.validate_current_state

**devtools/test_autofix_example.py**
- Line 17: P1_FUNCTIONAL: risky_operation
- Line 29: P1_FUNCTIONAL: another_operation
- Line 49: P1_FUNCTIONAL: initialize_something
- Line 29: Function contains only 'pass' in another_operation
- Line 40: Function 'process_data' implies logic but has complexity 1 in process_data
- ... and 1 more

**genomevault/blockchain/node/base_node.py**
- Line 21: P1_FUNCTIONAL: BaseNode.handle

**genomevault/experimental/zk_circuits/stark_prover.py**
- Line 956: P1_FUNCTIONAL: STARKProver._get_primitive_root

**genomevault/local_processing/sequencing.py**
- Line 75: P1_FUNCTIONAL: Variant.get_id (Functional requirement: id_generators)

**genomevault/logging.py**
- Line 3: P1_FUNCTIONAL: setup_logging

**genomevault/logging_utils.py**
- Line 3: P1_FUNCTIONAL: get_logger (Functional requirement: data_access)

**genomevault/pir/server/pir_server.py**
- Line 469: P0_SECURITY: TrustedSignatoryServer._verify_hsm (Security risk: crypto_verify)

**genomevault/utils/backup.py**
- Line 534: P1_FUNCTIONAL: DisasterRecoveryOrchestrator._restore_component_data

**genomevault/utils/config.py**
- Line 345: P1_FUNCTIONAL: _StubSecrets.get

**genomevault/utils/post_quantum_crypto.py**
- Line 89: P0_SECURITY: MockDilithium.verify (Security risk: crypto_verify)

**genomevault/zk_proofs/circuits/base_circuits.py**
- Line 40: P0_SECURITY: BaseCircuit.public_statement (Security-related file path)
- Line 44: P0_SECURITY: BaseCircuit.witness (Security-related file path)

**scripts/test_stub_fixes.py**
- Line 90: P1_FUNCTIONAL: test_function

### Context Window Thrashing (11 issues)
**genomevault/api/models/hv.py**
- Line 197: Context-window thrashing cluster: 2 similar functions (primary: validate_dimension)

**genomevault/clinical/calibration/calibrators.py**
- Line 57: Context-window thrashing: predict_proba() at lines 57 and 123 (77.5% similar, 66 lines apart)

**genomevault/hypervector_transform/hierarchical.py**
- Line 179: Context-window thrashing cluster: 7 similar functions (primary: __init__)
- Line 231: Context-window thrashing cluster: 6 similar functions (primary: forward)

**genomevault/local_processing/drift_detection.py**
- Line 526: Context-window thrashing cluster: 6 similar functions (primary: __init__)
- Line 596: Context-window thrashing cluster: 6 similar functions (primary: detect_drift)

**genomevault/pir/it_pir_protocol.py**
- Line 39: Context-window thrashing: __init__() at lines 39 and 266 (71.8% similar, 227 lines apart)

**genomevault/zk_proofs/circuits/base_genomic.py**
- Line 115: Context-window thrashing: generate_witness() at lines 115 and 166 (82.1% similar, 51 lines apart)

**genomevault/zk_proofs/circuits/biological/multi_omics.py**
- Line 36: Context-window thrashing cluster: 3 similar functions (primary: __init__)
- Line 183: Context-window thrashing cluster: 6 similar functions (primary: setup)
- Line 197: Context-window thrashing cluster: 6 similar functions (primary: generate_constraints)

### Other Issues (1 issues)
**genomevault/api/app.py**
- Line None: Suspicious chain of 3 fixes in 3 file(s) over 4 hours - possible tail-chasing

### Code Quality & Complexity (1 issues)
**devtools/comprehensive_cleanup.py**
- Line 33: High coupling risk between ComprehensiveCleanup.__init__ and ComprehensiveCleanup.log_fix

## Most Affected Files

1. **genomevault/local_processing/proteomics.py**: 24 issues
1. **genomevault/utils/monitoring.py**: 23 issues
1. **genomevault/utils/backup.py**: 22 issues
1. **genomevault/utils/post_quantum_crypto.py**: 19 issues
1. **genomevault/utils/common.py**: 13 issues
1. **devtools/setup_dev.py**: 12 issues
1. **genomevault/zk_proofs/prover.py**: 11 issues
1. **devtools/enhanced_cleanup.py**: 10 issues
1. **genomevault/experimental/zk_circuits/catalytic_proof.py**: 9 issues
1. **genomevault/local_processing/sequencing.py**: 9 issues

## Priority Actions

### High Priority
1. Fix duplicate and placeholder functions
2. Remove unused imports and clean up import structure
3. Consolidate semantic duplicates

### Medium Priority
1. Refactor context window thrashing patterns
2. Implement missing function stubs
3. Clean up LLM-generated filler content

### Low Priority
1. Improve documentation and docstrings
2. Optimize code organization
3. Add comprehensive test coverage

## Next Steps

1. Review this report to understand issue patterns
2. Use `--generate-fixes` to create automated fix scripts
3. Start with high-priority issues for maximum impact
4. Run analysis regularly to track progress

---
*Analysis performed on: /Users/rohanvinaik/genomevault*
*TailChasingFixer Version: 1.0*
