"""Training Proof Cli module."""
from __future__ import annotations

import argparse

from genomevault.zk_proofs.circuits.implementations.plonk_circuits import (

    prove_training_sum_over_threshold,
    verify_training_sum_over_threshold,
)


def main(argv=None) -> int:
    """Main.

    Args:
        argv: Argv.

    Returns:
        Integer result.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--sum", type=float, required=True)
    p.add_argument("--threshold", type=float, required=True)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args(argv)

    proof = prove_training_sum_over_threshold(args.sum, args.threshold)
    if args.verify:
        ok = verify_training_sum_over_threshold(args.threshold, proof)
        print("OK" if ok else "NO")
    else:
        print(proof.hex())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
