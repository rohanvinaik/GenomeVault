import argparse
import time

import numpy as np

from genomevault.hypervector.encoding.sparse_projection import \
    SparseRandomProjection
from genomevault.observability.logging import configure_logging

logger = configure_logging()


def run(variants: int, features: int, dim: int, seed: int):
    X = np.random.default_rng(seed).standard_normal((variants, features))
    proj = SparseRandomProjection(n_components=dim, density=0.1, seed=seed).fit(
        n_features=features
    )
    t0 = time.time()
    Y = proj.transform(X)
    dt = (time.time() - t0) * 1000.0
    logger.info(
        "Encoded %dx%d -> dim=%d in %.2f ms (shape=%s)",
        variants,
        features,
        dim,
        dt,
        Y.shape,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=10000)
    ap.add_argument("--features", type=int, default=100)
    ap.add_argument("--dim", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.variants, args.features, args.dim, args.seed)
