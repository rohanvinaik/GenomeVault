# GenomeVault ZK Backend (Circom + snarkjs)

This repo uses a **real Groth16 backend** via Circom/snarkjs for the `sum64` circuit (a+b=c).

## Toolchain
- Node.js (>=16)
- circom (2.x): `npm i -g circom`
- snarkjs (latest): `npm i -g snarkjs`

Verify:
```bash
which node && which circom && which snarkjs
```

## Build
```bash
bash scripts/zk_build_sum64.sh
```

Artifacts are written under `genomevault/zk/circuits/sum64/build/`.

## Use
- **Engine:** `genomevault.zk.real_engine.RealZKEngine`
- **API:** `/proofs/create` and `/proofs/verify` with `circuit_type="sum64"`

## Tests
```bash
pytest -q tests/zk/test_real_backend_sum64.py
pytest -q tests/api/test_proofs_real_backend.py
```

> CI note: skip these tests if toolchain is unavailable.
