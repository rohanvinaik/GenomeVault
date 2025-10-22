#!/usr/bin/env python3
"""
NEAT Diagnostic Patch - runner.py:194-240
Adds comprehensive logging and timeout handling to multiprocessing pool
"""

PATCH_CODE = '''
    if options.threads > 1:
        pool = mp.Pool(options.threads)
        results = pool.starmap_async(read_simulator_single, output_opts)

        _LOG.info(f"[GENOMEVAULT_DIAG] Launching multiprocess simulation with {len(output_opts)} tasks on {options.threads} workers")
        _LOG.info(f"[GENOMEVAULT_DIAG] PID: {mp.current_process().pid}")
        pool.close()

        # GENOMEVAULT_FIX: Add timeout and progress monitoring instead of indefinite join()
        import time
        start_time = time.time()
        timeout = 3600  # 1 hour timeout
        check_interval = 10  # Check every 10 seconds

        completed = False
        while not completed:
            try:
                # Wait with timeout instead of blocking forever
                if results.ready():
                    _LOG.info(f"[GENOMEVAULT_DIAG] All workers completed successfully")
                    completed = True
                    break

                # Check for timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    _LOG.error(f"[GENOMEVAULT_DIAG] TIMEOUT after {timeout}s - killing remaining workers")
                    pool.terminate()
                    break

                # Progress logging
                if int(elapsed) % 30 == 0:  # Every 30 seconds
                    # Count completed chunks
                    completed_count = sum(1 for opt_tuple in output_opts
                                        if (opt_tuple[2].temp_dir_path / opt_tuple[2].reference.stem / opt_tuple[2].fq1.name).exists())
                    _LOG.info(f"[GENOMEVAULT_DIAG] Progress: {completed_count}/{len(output_opts)} chunks after {int(elapsed)}s")

                time.sleep(check_interval)

            except KeyboardInterrupt:
                _LOG.warning(f"[GENOMEVAULT_DIAG] Interrupted - terminating pool")
                pool.terminate()
                break

        # Try to join with timeout
        try:
            pool.join(timeout=30)
            _LOG.info(f"[GENOMEVAULT_DIAG] Pool joined successfully")
        except Exception as e:
            _LOG.error(f"[GENOMEVAULT_DIAG] Pool join failed: {e}")
            pool.terminate()

        # GENOMEVAULT_FIX: Attempt to retrieve results even if some workers failed
        try:
            output_results = results.get(timeout=10)
            _LOG.info(f"[GENOMEVAULT_DIAG] Retrieved {len(output_results)} results")
            for thread_idx, local_contig, local_variants, files_written in output_results:
                all_variants[local_contig][thread_idx] = local_variants
                output_files.append((thread_idx, files_written))
        except Exception as e:
            _LOG.error(f"[GENOMEVAULT_DIAG] Failed to get results: {e}")
            _LOG.warning(f"[GENOMEVAULT_DIAG] Attempting to salvage partial outputs...")
            # Salvage what we can from disk
            for thread_idx, start, current_options, _, contig, _, _, _, _, _ in output_opts:
                current_output_dir = current_options.temp_dir_path / current_options.reference.stem
                fq1_path = current_output_dir / current_options.fq1.name
                if fq1_path.exists() and fq1_path.stat().st_size > 1024:
                    _LOG.info(f"[GENOMEVAULT_DIAG] Salvaged chunk {thread_idx}: {fq1_path}")
                    output_files.append((thread_idx, (fq1_path, current_output_dir / current_options.fq2.name if current_options.paired_ended else None)))
'''

print("Diagnostic patch created:")
print("Location: /Users/rohanvinaik/genomevault/scripts/neat_diagnostic_patch.py")
print("\nTo apply manually:")
print("1. Backup: cp runner.py runner.py.backup")
print("2. Edit: ~/miniconda3/envs/neat/lib/python3.10/site-packages/neat/read_simulator/runner.py")
print("3. Replace lines 193-199 with the PATCH_CODE above")
print("4. Re-run NEAT to see diagnostic output")
