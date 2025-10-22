"""
Batch Processing Utilities for Differential Encoding Migrations

Provides utilities for efficiently processing large batches of genomic data
with parallel execution, progress tracking, and error recovery.

Features:
- Parallel batch processing
- Progress tracking and visualization
- Error recovery and retry logic
- Resource usage monitoring
- Checkpoint and resume capability
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Callable, Any, Optional, Dict, Iterator
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# Batch Processing Configuration
# ==============================================================================

class ProcessingMode(Enum):
    """Processing modes for batch operations."""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    MULTIPROCESS = "multiprocess"


@dataclass
class BatchConfig:
    """
    Configuration for batch processing.

    Attributes:
        batch_size: Number of items per batch
        max_workers: Maximum number of parallel workers
        mode: Processing mode (sequential, threaded, multiprocess)
        retry_attempts: Number of retry attempts for failed items
        retry_delay_seconds: Delay between retries
        checkpoint_interval: Save checkpoint every N batches
        show_progress: Display progress bar
        fail_fast: Stop on first error
    """
    batch_size: int = 10
    max_workers: int = 4
    mode: ProcessingMode = ProcessingMode.THREADED
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    checkpoint_interval: int = 5
    show_progress: bool = True
    fail_fast: bool = False


@dataclass
class BatchResult:
    """
    Result of batch processing.

    Attributes:
        total_items: Total number of items processed
        successful_items: Number of successful items
        failed_items: Number of failed items
        total_time_seconds: Total processing time
        avg_time_per_item_seconds: Average time per item
        errors: List of error messages
        results: List of individual results
    """
    total_items: int
    successful_items: int
    failed_items: int
    total_time_seconds: float
    avg_time_per_item_seconds: float
    errors: List[str] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)


# ==============================================================================
# Progress Tracker
# ==============================================================================

class ProgressTracker:
    """
    Track and display batch processing progress.

    Provides real-time progress updates with estimated time remaining.
    """

    def __init__(
        self,
        total_items: int,
        description: str = "Processing",
        show_bar: bool = True,
    ):
        """
        Initialize progress tracker.

        Args:
            total_items: Total number of items to process
            description: Description to display
            show_bar: Whether to show progress bar
        """
        self.total_items = total_items
        self.description = description
        self.show_bar = show_bar
        self.completed = 0
        self.failed = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time

    def update(self, increment: int = 1, failed: bool = False) -> None:
        """
        Update progress.

        Args:
            increment: Number of items completed
            failed: Whether items failed
        """
        if failed:
            self.failed += increment
        else:
            self.completed += increment

        if self.show_bar:
            self._display_progress()

    def _display_progress(self) -> None:
        """Display progress bar."""
        total_processed = self.completed + self.failed
        if total_processed == 0:
            return

        # Calculate progress
        progress_pct = (total_processed / self.total_items) * 100
        elapsed = time.time() - self.start_time
        rate = total_processed / elapsed if elapsed > 0 else 0
        remaining = self.total_items - total_processed
        eta = remaining / rate if rate > 0 else 0

        # Create progress bar
        bar_length = 40
        filled_length = int(bar_length * total_processed / self.total_items)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        # Format output
        print(
            f'\r{self.description}: |{bar}| '
            f'{total_processed}/{self.total_items} '
            f'({progress_pct:.1f}%) '
            f'[{rate:.1f} items/s] '
            f'[ETA: {eta:.0f}s] '
            f'[Failed: {self.failed}]',
            end='',
            flush=True
        )

        # Newline when complete
        if total_processed >= self.total_items:
            print()

    def finish(self) -> None:
        """Finish progress tracking and display summary."""
        if self.show_bar:
            print()
            elapsed = time.time() - self.start_time
            logger.info(
                f"Completed {self.completed}/{self.total_items} items "
                f"in {elapsed:.1f}s ({self.failed} failed)"
            )


# ==============================================================================
# Batch Processor
# ==============================================================================

class BatchProcessor:
    """
    Process items in batches with parallel execution.

    Supports multiple processing modes, automatic retries, and progress tracking.
    """

    def __init__(self, config: BatchConfig):
        """
        Initialize batch processor.

        Args:
            config: Batch processing configuration
        """
        self.config = config
        self.checkpoint_data = {}

    def process_items(
        self,
        items: List[Any],
        processor_func: Callable[[Any], Any],
        description: str = "Processing items",
    ) -> BatchResult:
        """
        Process list of items in batches.

        Args:
            items: List of items to process
            processor_func: Function to process each item
            description: Description for progress tracking

        Returns:
            BatchResult with processing statistics

        Example:
            >>> def process_genome(genome_path):
            ...     # Process genome
            ...     return result
            >>> processor = BatchProcessor(BatchConfig(batch_size=10))
            >>> result = processor.process_items(genome_paths, process_genome)
        """
        logger.info(
            f"Processing {len(items)} items in batches of {self.config.batch_size} "
            f"(mode: {self.config.mode.value}, workers: {self.config.max_workers})"
        )

        start_time = time.time()
        results = []
        errors = []
        successful = 0
        failed = 0

        # Create progress tracker
        progress = ProgressTracker(
            total_items=len(items),
            description=description,
            show_bar=self.config.show_progress,
        )

        # Process items based on mode
        if self.config.mode == ProcessingMode.SEQUENTIAL:
            for item in items:
                result, error = self._process_with_retry(item, processor_func)
                results.append(result)
                if error:
                    errors.append(error)
                    failed += 1
                    progress.update(failed=True)
                    if self.config.fail_fast:
                        break
                else:
                    successful += 1
                    progress.update()

        elif self.config.mode == ProcessingMode.THREADED:
            results, errors, successful, failed = self._process_threaded(
                items, processor_func, progress
            )

        elif self.config.mode == ProcessingMode.MULTIPROCESS:
            results, errors, successful, failed = self._process_multiprocess(
                items, processor_func, progress
            )

        progress.finish()

        # Calculate statistics
        total_time = time.time() - start_time
        avg_time = total_time / len(items) if items else 0

        return BatchResult(
            total_items=len(items),
            successful_items=successful,
            failed_items=failed,
            total_time_seconds=total_time,
            avg_time_per_item_seconds=avg_time,
            errors=errors,
            results=results,
        )

    def _process_with_retry(
        self,
        item: Any,
        processor_func: Callable[[Any], Any],
    ) -> tuple[Any, Optional[str]]:
        """
        Process item with retry logic.

        Args:
            item: Item to process
            processor_func: Processing function

        Returns:
            Tuple of (result, error_message)
        """
        last_error = None

        for attempt in range(self.config.retry_attempts):
            try:
                result = processor_func(item)
                return result, None

            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                logger.warning(
                    f"Attempt {attempt + 1}/{self.config.retry_attempts} failed: {last_error}"
                )

                if attempt < self.config.retry_attempts - 1:
                    time.sleep(self.config.retry_delay_seconds)

        # All retries failed
        logger.error(f"All retry attempts failed for item: {last_error}")
        return None, last_error

    def _process_threaded(
        self,
        items: List[Any],
        processor_func: Callable[[Any], Any],
        progress: ProgressTracker,
    ) -> tuple[List[Any], List[str], int, int]:
        """Process items using thread pool."""
        results = []
        errors = []
        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._process_with_retry, item, processor_func): item
                for item in items
            }

            for future in as_completed(futures):
                result, error = future.result()
                results.append(result)

                if error:
                    errors.append(error)
                    failed += 1
                    progress.update(failed=True)
                    if self.config.fail_fast:
                        # Cancel remaining futures
                        for f in futures:
                            f.cancel()
                        break
                else:
                    successful += 1
                    progress.update()

        return results, errors, successful, failed

    def _process_multiprocess(
        self,
        items: List[Any],
        processor_func: Callable[[Any], Any],
        progress: ProgressTracker,
    ) -> tuple[List[Any], List[str], int, int]:
        """Process items using process pool."""
        results = []
        errors = []
        successful = 0
        failed = 0

        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._process_with_retry, item, processor_func): item
                for item in items
            }

            for future in as_completed(futures):
                result, error = future.result()
                results.append(result)

                if error:
                    errors.append(error)
                    failed += 1
                    progress.update(failed=True)
                    if self.config.fail_fast:
                        for f in futures:
                            f.cancel()
                        break
                else:
                    successful += 1
                    progress.update()

        return results, errors, successful, failed


# ==============================================================================
# Batch Iterator
# ==============================================================================

class BatchIterator:
    """
    Iterate over items in batches.

    Provides memory-efficient iteration over large datasets.
    """

    def __init__(
        self,
        items: List[Any],
        batch_size: int = 10,
    ):
        """
        Initialize batch iterator.

        Args:
            items: List of items to iterate
            batch_size: Number of items per batch
        """
        self.items = items
        self.batch_size = batch_size

    def __iter__(self) -> Iterator[List[Any]]:
        """Iterate over batches."""
        for i in range(0, len(self.items), self.batch_size):
            yield self.items[i:i + self.batch_size]

    def __len__(self) -> int:
        """Number of batches."""
        return (len(self.items) + self.batch_size - 1) // self.batch_size


# ==============================================================================
# Resource Monitor
# ==============================================================================

class ResourceMonitor:
    """
    Monitor resource usage during batch processing.

    Tracks CPU, memory, and I/O usage to optimize batch size and worker count.
    """

    def __init__(self):
        """Initialize resource monitor."""
        self.cpu_usage = []
        self.memory_usage = []
        self.start_time = time.time()

    def record_snapshot(self) -> Dict[str, Any]:
        """
        Record current resource usage snapshot.

        Returns:
            Dictionary with resource metrics
        """
        import psutil

        # Get current process
        process = psutil.Process()

        # CPU usage
        cpu_pct = process.cpu_percent(interval=0.1)
        self.cpu_usage.append(cpu_pct)

        # Memory usage
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / (1024 * 1024)
        self.memory_usage.append(mem_mb)

        snapshot = {
            'timestamp': time.time() - self.start_time,
            'cpu_percent': cpu_pct,
            'memory_mb': mem_mb,
            'memory_percent': process.memory_percent(),
        }

        return snapshot

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of resource usage.

        Returns:
            Dictionary with summary statistics
        """
        if not self.cpu_usage or not self.memory_usage:
            return {}

        return {
            'cpu': {
                'mean': np.mean(self.cpu_usage),
                'max': np.max(self.cpu_usage),
                'min': np.min(self.cpu_usage),
            },
            'memory_mb': {
                'mean': np.mean(self.memory_usage),
                'max': np.max(self.memory_usage),
                'min': np.min(self.memory_usage),
            },
            'duration_seconds': time.time() - self.start_time,
        }


# ==============================================================================
# Utility Functions
# ==============================================================================

def auto_tune_batch_size(
    sample_items: List[Any],
    processor_func: Callable[[Any], Any],
    target_memory_mb: float = 1000.0,
    max_batch_size: int = 100,
) -> int:
    """
    Automatically determine optimal batch size based on memory usage.

    Args:
        sample_items: Sample items for testing
        processor_func: Processing function
        target_memory_mb: Target memory usage in MB
        max_batch_size: Maximum batch size to test

    Returns:
        Recommended batch size
    """
    import psutil

    logger.info("Auto-tuning batch size...")

    # Test with small batch
    test_size = min(5, len(sample_items))
    process = psutil.Process()

    # Measure baseline memory
    baseline_mem = process.memory_info().rss / (1024 * 1024)

    # Process sample batch
    for item in sample_items[:test_size]:
        processor_func(item)

    # Measure memory after processing
    after_mem = process.memory_info().rss / (1024 * 1024)

    # Calculate memory per item
    mem_per_item = (after_mem - baseline_mem) / test_size

    # Calculate optimal batch size
    if mem_per_item > 0:
        optimal_size = int(target_memory_mb / mem_per_item)
        optimal_size = min(optimal_size, max_batch_size)
        optimal_size = max(optimal_size, 1)
    else:
        optimal_size = max_batch_size

    logger.info(
        f"Auto-tuned batch size: {optimal_size} "
        f"(~{mem_per_item:.1f} MB per item, target: {target_memory_mb} MB)"
    )

    return optimal_size
