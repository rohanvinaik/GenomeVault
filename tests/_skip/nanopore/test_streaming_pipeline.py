import numpy as np

from genomevault.nanopore.streaming import NanoporeSlice, NanoporeStreamProcessor


def generate_signal(n_slices=8, length=256, seed=11):
    """Generate signal.

    Yields items from the function's iteration.

    Args:        n_slices: N slices parameter.        length: Length parameter.        seed: Seed parameter.

    Yields:        Items from the iteration.

    Example:        >>> result = generate_signal()        >>> print(result)
    """
    rng = np.random.default_rng(seed)
    for i in range(n_slices):
        yield NanoporeSlice(index=i, raw=rng.normal(0, 1, size=length).astype("float32"))


def test_streaming_smoke():
    """Test streaming smoke.

    Example:        >>> result = test_streaming_smoke()        >>> print(result)
    """
    proc = NanoporeStreamProcessor(batch_size=2, use_gpu=False)
    stats = None
    for sl in generate_signal():
        stats = proc._cpu_process_batch([sl])
    assert stats is None or hasattr(stats, "processed_slices")
